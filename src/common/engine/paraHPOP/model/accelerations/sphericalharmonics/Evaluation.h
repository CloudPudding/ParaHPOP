#pragma once

#include "paraHPOP/model/accelerations/sphericalharmonics/DegreeLoop.h"
#include "paraHPOP/model/accelerations/sphericalharmonics/LegendreRecursor.h"
#include "paraHPOP/model/accelerations/sphericalharmonics/PseudoPhasor.h"

#include "paraHPOP/model/environment/Environment.h"
#include "interface/config/model/sphericalharmonics/Coefficients.h"

namespace paraHPOP {
namespace model {
namespace accelerations {
namespace sphericalharmonics {

/** @brief Spherical harmonics recursive evaluation class.
 *
 * Computes the gravitational acceleration due to non-spherical mass
 * distribution using a spherical-coordinate recursion for the associated
 * Legendre functions. This avoids the large V/W coefficient matrix of the
 * traditional Montenbruck formulation, keeping the memory footprint
 * proportional to the coefficient count alone.
 *
 * The algorithm structure is:
 *   1. Convert Cartesian position → (r, φ, λ)
 *   2. Order loop: m = 0…maxOrder
 *      2a. Degree loop: n = m…maxDegree (Legendre vertical recurrence)
 *          - accumulate radial, latitudinal, longitudinal components
 *      2b. Advance diagonal P_{m,m} and phasor cos/sin(mλ)
 *   3. Scale by GM/r²
 *   4. Transform spherical acceleration → Cartesian
 *
 * The same algorithm is shared by host and device; on device the helpers
 * are force-inlined into the calling TU for zero call overhead.
 */
template<bool work>
class Evaluation {
    using Self = Evaluation;
    using CoefficientsT
        = interface::config::model::sphericalharmonics::Coefficients::Ref<work>;
    using VecT      = feta::vector::Item<Real, 3>;
    using LegendreT = LegendreRecursor;

public:
    /** @brief Evaluate the SH acceleration (device/host).
     *
     * @param position  Body-fixed Cartesian position
     * @param coeffs    SH coefficients reference
     * @param MaxDegree Maximum degree (-1 → use coeffs.maxDegree())
     * @param MaxOrder  Maximum order  (-1 → use coeffs.maxOrder())
     * @return Acceleration in the body-fixed frame
     */
    FORCEINLINE()
    DEVICEHOST() static VecT
        eval(const VecT& position, const CoefficientsT& coeffs,
            const int& MaxDegree = -1, const int& MaxOrder = -1)
    {
        idx_t maxDegree = (MaxDegree == -1) ? coeffs.maxDegree()
                                            : static_cast<idx_t>(MaxDegree);
        idx_t maxOrder  = (MaxOrder == -1) ? coeffs.maxOrder()
                                           : static_cast<idx_t>(MaxOrder);
        return coreEval_(position, coeffs, maxDegree, maxOrder);
    }

    /** @brief Device-only ``eval`` overload that accumulates the
     * spherical-acceleration sum through a per-thread shmem-backed
     * Vec3R handle, AND moves the 4-double PseudoPhasor recurrence
     * state into a parallel per-thread shmem slot.  Same
     * shmem-dependency-chain trick that broke the
     * parm::adaptive::advance over-parallelisation (see
     * [[parm-advance-shmem-done]]): moving the running ``acc`` AND the
     * persistent (sinL, cosL, sinML, cosML) phasor out of the register
     * file frees up to 56 B of pressure across the SH eval inner
     * loops, which compete with the Legendre recurrence state at the
     * 64-reg cap on sm_61.
     *
     * The caller provides ``accHandle`` (per-thread shmem Vec3R slot)
     * and ``phasorHandle`` (per-thread shmem Vec4R slot).
     * ``coreEvalDevice_`` writes the final spherical-to-Cartesian
     * acceleration to ``accHandle[i]``; the caller reads it back for
     * the inverse rotation. ``useRadialCache`` permits the bounded
     * degree <= 96 radial-prefix cache; false retains the original
     * per-order radial recurrence for controlled comparisons. */
    template<typename AccHandleT, typename PhasorHandleT>
    FORCEINLINE() __device__ static void evalDevice(
        const SampleIndex& sampleIdx, const VecT& position,
        const CoefficientsT& coeffs, AccHandleT& accHandle,
        PhasorHandleT& phasorHandle,
        const int& MaxDegree = -1, const int& MaxOrder = -1,
        bool useRadialCache = true, bool useCorrectedDivision = false)
    {
        idx_t maxDegree = (MaxDegree == -1) ? coeffs.maxDegree()
                                            : static_cast<idx_t>(MaxDegree);
        idx_t maxOrder  = (MaxOrder == -1) ? coeffs.maxOrder()
                                           : static_cast<idx_t>(MaxOrder);
        coreEvalDevice_(sampleIdx, position, coeffs, maxDegree, maxOrder,
            accHandle, phasorHandle, useRadialCache, useCorrectedDivision);
    }

protected:
    // -----------------------------------------------------------------
    //  Core evaluation — unified device/host
    // -----------------------------------------------------------------

    /** @brief Full SH evaluation: position → acceleration.
     *
     * Force-inlined on device so the entire SH double-loop compiles into
     * the kernel body with zero CAL/RET overhead.
     */
    FORCEINLINE()
    DEVICEHOST() static VecT
        coreEval_(const VecT& position, const CoefficientsT& coeffs,
            const idx_t& maxDegree, const idx_t& maxOrder)
    {
        const Real rn = position.norm();
        const PhiState phi = PhiState::fromPosition(position, rn);
        PseudoPhasor phasor = PseudoPhasor::make(position);

        /* Save sin/cos(λ) before the order loop modifies sinML/cosML.
         * sinL_/cosL_ are invariant across the loop, but saving them
         * explicitly allows the compiler to free the phasor earlier. */
        const Real sinL = phasor.sinL();
        const Real cosL = phasor.cosL();

        VecT acc = orderLoop_(
            phi, phasor, coeffs, maxDegree, maxOrder, coeffs.bodyRadius() / rn);

        acc *= coeffs.gm() / (rn * rn);
        return sphericalToCartesian_(acc, phi, sinL, cosL);
    }

    // -----------------------------------------------------------------
    //  Order loop
    // -----------------------------------------------------------------

    /** @brief Accumulate over all orders m = 0…maxOrder. */
    FORCEINLINE()
    DEVICEHOST() static VecT orderLoop_(const PhiState& phi,
        PseudoPhasor& phasor, const CoefficientsT& coeffs,
        const idx_t& maxDegree, const idx_t& maxOrder, const Real& alpha)
    {
        using DegreeLoopT = DegreeLoop<work>;

        VecT acc;
        Real alpham = 1.0;
        DiagonalState diag;

        for (idx_t m = 0; m <= maxOrder; m++) {
            acc += DegreeLoopT::accumulate(
                diag, phi, phasor, coeffs, m, maxDegree, alpha, alpham);

            if (m < maxOrder) {
                ++phasor;
                const Real cm = coeffs.factor(m + 1, m + 1, 0);
                diag.advance(cm, phi);
                alpham *= alpha;
            }
        }
        return acc;
    }

    /** @brief Device-only ``orderLoop_`` that accumulates into a
     * per-thread shmem-backed Vec3R handle AND routes the
     * PseudoPhasor recurrence through a parallel per-thread shmem
     * Vec4R slot via a @ref RefPseudoPhasor view.
     *
     * The phasor view holds no register-resident copy of the
     * (sinL, cosL, sinML, cosML) state — every @c phasor.phases() or
     * @c ++phasor goes through the handle, so the whole 32 B of
     * persistent recurrence state lives in shmem instead of registers.
     * (Whether the compiler ultimately keeps those values in shmem
     * proper or CSEs them back to registers depends on the volatility
     * of @c PhasorHandleT: a plain ``WRef::HandleT`` permits CSE; a
     * ``VolatileRef::HandleT`` forbids it.) */
    template<typename AccHandleT, typename PhasorT>
    FORCEINLINE() __device__ static void orderLoopDevice_(
        const SampleIndex& sampleIdx, const PhiState& phi,
        const CoefficientsT& coeffs,
        const idx_t& maxDegree, const idx_t& maxOrder, const Real& alpha,
        AccHandleT& accHandle, PhasorT& phasor, bool useRadialCache)
    {
        constexpr idx_t maxCachedDegree = 96;
        if (useRadialCache && maxDegree <= maxCachedDegree) {
            /* The old alpham/alphan updates both generate this exact
             * left-associated FP64 multiplication chain. Cache its values
             * and the existing radial formula's (n + 1) * alphan prefix;
             * all remaining products and accumulation order stay intact. */
            RadialPower radialPowers[maxCachedDegree + 1];
            Real alphan = 1.0;
            for (idx_t n = 0; n <= maxDegree; ++n) {
                radialPowers[n] = { alphan, (n + 1) * alphan };
                if (n < maxDegree)
                    alphan *= alpha;
            }
            orderLoopDeviceImpl_<true>(sampleIdx, phi, coeffs, maxDegree,
                maxOrder, alpha, accHandle, phasor, radialPowers);
        } else {
            orderLoopDeviceImpl_<false>(sampleIdx, phi, coeffs, maxDegree,
                maxOrder, alpha, accHandle, phasor, nullptr);
        }
    }

    template<bool cachedRadial, typename AccHandleT, typename PhasorT>
    FORCEINLINE() __device__ static void orderLoopDeviceImpl_(
        const SampleIndex& sampleIdx, const PhiState& phi,
        const CoefficientsT& coeffs,
        const idx_t& maxDegree, const idx_t& maxOrder, const Real& alpha,
        AccHandleT& accHandle, PhasorT& phasor,
        const RadialPower* radialPowers)
    {
        using DegreeLoopT = DegreeLoop<work>;

        accHandle[sampleIdx] = VecT(0);
        Real alpham          = 1.0;
        DiagonalState diag;

        for (idx_t m = 0; m <= maxOrder; m++) {
            accHandle[sampleIdx] += DegreeLoopT::template accumulate<cachedRadial>(
                diag, phi, phasor, coeffs, m, maxDegree, alpha, alpham,
                radialPowers);

            if (m < maxOrder) {
                ++phasor;
                const Real cm = coeffs.factor(m + 1, m + 1, 0);
                diag.advance(cm, phi);
                if constexpr (!cachedRadial)
                    alpham *= alpha;
            }
        }
    }

    /** @brief Device-only ``coreEval_`` writing through shmem-backed
     * accumulator handle AND routing the PseudoPhasor recurrence
     * through a parallel per-thread shmem slot via @ref RefPseudoPhasor. */
    template<typename AccHandleT, typename PhasorHandleT>
    FORCEINLINE() __device__ static void coreEvalDevice_(
        const SampleIndex& sampleIdx, const VecT& position,
        const CoefficientsT& coeffs, const idx_t& maxDegree,
        const idx_t& maxOrder, AccHandleT& accHandle,
        PhasorHandleT& phasorHandle, bool useRadialCache,
        bool useCorrectedDivision)
    {
        using PhasorT = RefPseudoPhasor<PhasorHandleT>;

        const Real rn = position.norm();
        const PhiState phi = PhiState::fromPosition(position, rn, useCorrectedDivision);

        /* Initialise the shmem slot, then construct the view once.
         * Subsequent reads go through phasor.sinL()/cosL() — no
         * direct slot.get<>() calls leak into this outer code. */
        PhasorT::init(phasorHandle, sampleIdx, position);
        PhasorT phasor{ phasorHandle, sampleIdx };

        orderLoopDevice_(sampleIdx, phi, coeffs, maxDegree, maxOrder,
            coeffs.bodyRadius() / rn, accHandle, phasor, useRadialCache);

        /* Apply the GM/r² scaling in place via the shmem handle.
         * ``sinL``/``cosL`` are read from the phasor's shmem slot at
         * the call site (2 LDS reads, lifetime confined to the
         * call) — capturing them into pre-loop locals would force
         * 16 B of register pressure across the SH eval body, which
         * is exactly what the shmem-backed phasor is meant to avoid. */
        const Real gmOverR2 = coeffs.gm() / (rn * rn);
        accHandle[sampleIdx] = sphericalToCartesian_(
            VecT(accHandle[sampleIdx]) * gmOverR2, phi,
            phasor.sinL(), phasor.cosL());
    }

    // -----------------------------------------------------------------
    //  Coordinate transform
    // -----------------------------------------------------------------

    /** @brief Convert spherical acceleration components to Cartesian.
     *
     * Basis vectors:
     *   e_r   = [ cφ·cλ,  cφ·sλ,  sφ ]
     *   e_φ   = [-sφ·cλ, -sφ·sλ,  cφ ]
     *   e_λ   = [   -sλ,     cλ,   0 ]
     *
     * a_cart = a_r · e_r + a_φ · e_φ + a_λ · e_λ
     */
    FORCEINLINE()
    DEVICEHOST() static VecT sphericalToCartesian_(const VecT& acc,
        const PhiState& phi, const Real& sinL, const Real& cosL)
    {
        VecT out;
        out.template get<0>() = phi.cosPhi * cosL * acc.template get<0>()
            - phi.sinPhi * cosL * acc.template get<1>()
            - sinL * acc.template get<2>();
        out.template get<1>() = phi.cosPhi * sinL * acc.template get<0>()
            - phi.sinPhi * sinL * acc.template get<1>()
            + cosL * acc.template get<2>();
        out.template get<2>() = phi.sinPhi * acc.template get<0>()
            + phi.cosPhi * acc.template get<1>();
        return out;
    }
};

} // namespace sphericalharmonics
} // namespace accelerations
} // namespace model
} // namespace paraHPOP
