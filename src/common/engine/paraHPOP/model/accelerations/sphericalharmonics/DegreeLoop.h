#pragma once

#include "paraHPOP/model/accelerations/sphericalharmonics/LegendreRecursor.h"
#include "paraHPOP/model/accelerations/sphericalharmonics/PseudoPhasor.h"
#include "interface/config/model/sphericalharmonics/Coefficients.h"

namespace paraHPOP {
namespace model {
namespace accelerations {
namespace sphericalharmonics {

/** @brief Reusable prefixes of the original radial contribution formula. */
struct RadialPower {
    Real alphan;
    Real radial;
};

/** @brief Degree loop computation for the spherical harmonics inner loop.
 *
 * Iterates over degrees n = m..maxDegree for a fixed order m, accumulating
 * the radial, latitudinal, and longitudinal acceleration components.
 *
 * The diagonal state and phi state are taken as inputs so the compiler can
 * keep the recursor in registers during the inner loop while spilling the
 * outer-loop state between m iterations.
 */
template<bool work>
struct DegreeLoop {
    using CoefficientsT
        = interface::config::model::sphericalharmonics::Coefficients::Ref<work>;
    using FactorPairT
        = interface::config::model::sphericalharmonics::FactorPair;
    using VecT      = feta::vector::Item<Real, 3>;
    using LegendreT = LegendreRecursor;

    /** @brief Accumulate spherical acceleration components for a fixed
     * order m over all degrees n.
     *
     * Unified device/host path — on device this is force-inlined into the
     * calling TU; on host the extern template instantiation provides the
     * symbol.
     *
     * Templated on ``PhasorT`` so the same body compiles against the
     * register-backed @ref PseudoPhasor (host + device legacy path)
     * and the shmem-backed @ref RefPseudoPhasor view (device path with
     * per-thread shmem slot).  Only the @c phases() method is invoked
     * on the phasor here.
     */
    template<bool cachedRadial = false, typename PhasorT>
    FORCEINLINE()
    DEVICEHOST() static VecT accumulate(const DiagonalState& diag,
        const PhiState& phi, const PhasorT& phasor,
        const CoefficientsT& coeffs, const idx_t& m, const idx_t& maxDegree,
        const Real& alpha, Real alphan,
        const RadialPower* radialPowers = nullptr)
    {
        LegendreT legendre = LegendreT::fromDiagonal(diag);
        VecT result;
        result.setZero();

        for (idx_t n = m; n <= maxDegree; n++) {
            if constexpr (cachedRadial)
                alphan = radialPowers[n].alphan;
            applyTerms_<cachedRadial>(
                result, legendre, phi, phasor, coeffs, n, m, alphan,
                radialPowers);
            advanceLegendre_<cachedRadial>(
                legendre, phi, coeffs, alphan, alpha, n, m, maxDegree);
        }

        return result;
    }

private:
    /** @brief Extract Legendre values and phase coefficients, then
     * accumulate the three spherical acceleration components.
     *
     * Combines extraction and accumulation in a single step to minimise
     * temporary live-variable pressure.
     */
    template<bool cachedRadial, typename PhasorT>
    FORCEINLINE()
    DEVICEHOST() static void applyTerms_(VecT& result,
        LegendreT& legendre, const PhiState& phi, const PhasorT& phasor,
        const CoefficientsT& coeffs, const idx_t& n, const idx_t& m,
        const Real& alphan, const RadialPower* radialPowers)
    {
        LegendreValues lv   = legendre.extractValues(m, phi);
        const Phases phases = phasor.phases(coeffs, n, m);

        // Cold path: at the literal pole (cosφ exactly 0, reachable only
        // for inputs with x = y = 0 exactly) the m = 1 longitudinal term
        // is the only one with a finite non-zero limit that the recurrence
        // produces as 0/0.  Substitute the analytical value here.  Branch
        // is loop-invariant in cosφ and never taken off-pole, so the
        // compiler can spill the cold-path registers and pay essentially
        // nothing in the hot loop.
        if (m == 1 && phi.cosPhi == 0.0) {
            const Real sign = (phi.sinPhi > 0.0)
                ? 1.0
                : ((n & 1u) ? 1.0 : -1.0);
            lv.divCosPnm = sign * coeffs.polarLimitM1(n);
        }

        if constexpr (cachedRadial)
            result.template get<0>()
                -= (radialPowers[n].radial * lv.Pnm * phases.C);
        else
            result.template get<0>() -= ((n + 1) * alphan * lv.Pnm * phases.C);
        result.template get<1>() += (alphan * lv.dPnmDphi * phases.C);
        result.template get<2>() -= (m * alphan * lv.divCosPnm * phases.S);
    }

    /** @brief Advance the Legendre recursor to the next degree.
     *
     * Uses factorPair() to load both recursion coefficients in a single
     * address calculation (a_{n+1,m} and b_{n+1,m} are stored contiguously).
     */
    template<bool cachedRadial>
    FORCEINLINE()
    DEVICEHOST() static void advanceLegendre_(LegendreT& legendre,
        const PhiState& phi, const CoefficientsT& coeffs, Real& alphan,
        const Real& alpha, const idx_t& n, const idx_t& m,
        const idx_t& maxDegree)
    {
        if (n < maxDegree) {
            const FactorPairT fp = coeffs.factorPair(n + 1, m);
            legendre.advance(fp.anm, fp.bnm, phi);
            if constexpr (!cachedRadial)
                alphan *= alpha;
        }
    }
};

} // namespace sphericalharmonics
} // namespace accelerations
} // namespace model
} // namespace paraHPOP
