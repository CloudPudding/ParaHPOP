#pragma once

#include "paraHPOP/typedefs.h"
#include "paraHPOP/util.h"
#include "interface/config/model/sphericalharmonics/Coefficients.h"

namespace paraHPOP {
namespace model {
namespace accelerations {
namespace sphericalharmonics {

/** @brief Phases */
struct Phases {
    Real C; /**< Phase C value */
    Real S; /**< Phase S value */
};

/** @brief Pseudo-phasor for the spherical harmonics computation */
class PseudoPhasor {
    using Self = PseudoPhasor;
    using VecT = feta::vector::Item<Real, 3>;

public:
    /** @brief Factory method - guarding against near-polar cases */
    DEVICEHOST()
    static PseudoPhasor make(const VecT& position, const Real& eps = 1e-14)
    {
        Real x    = position.get<0>();
        Real y    = position.get<1>();
        Real rho  = position.head<2>().norm();
        Real sinL = (rho > eps) ? (y / rho) : 0.0;
        Real cosL = (rho > eps) ? (x / rho) : 1.0;
        return PseudoPhasor(sinL, cosL);
    }

    /** @brief Construct from the given sin and cos of lambda */
    DEVICEHOST()
    PseudoPhasor(const Real& sinL, const Real& cosL)
        : sinL_{ sinL }
        , cosL_{ cosL }
    {
    }

    /** @brief Expose the sine of Lambda */
    DEVICEHOST() inline Real& sinL() { return sinL_; }
    DEVICEHOST() inline const Real& sinL() const { return sinL_; }

    /** @brief Expose the cosine of Lambda */
    DEVICEHOST() inline Real& cosL() { return cosL_; }
    DEVICEHOST() inline const Real& cosL() const { return cosL_; }

    /** @brief Expose the sine of m*Lambda */
    DEVICEHOST() inline Real& sinML() { return sinML_; }
    DEVICEHOST() inline const Real& sinML() const { return sinML_; }

    /** @brief Expose the cosine of m*Lambda */
    DEVICEHOST() inline Real& cosML() { return cosML_; }
    DEVICEHOST() inline const Real& cosML() const { return cosML_; }

    /** @brief Update phasor for the next m */
    FORCEINLINE() DEVICEHOST() Self& operator++()
    {
        Real newSinML = cosL_ * sinML_ + sinL_ * cosML_;
        Real newCosML = cosL_ * cosML_ - sinL_ * sinML_;
        sinML_        = newSinML;
        cosML_        = newCosML;
        return *this;
    }


    /** @brief From the given coefficients, extract the PhaseC value */
    template<typename CoefficientsT>
    DEVICEHOST()
    Real phaseC(
        const CoefficientsT& coeffs, const idx_t& n, const idx_t& m) const
    {
        using namespace interface::config::model::sphericalharmonics;
        Real c = coeffs.template at<Coefficient::C>(n, m);
        Real s = (m > 0) ? coeffs.template at<Coefficient::S>(n, m) : 0.0;
        return c * cosML_ + s * sinML_;
    }

    /** @brief From the given coefficients, extract the PhaseS value */
    template<typename CoefficientsT>
    DEVICEHOST()
    Real phaseS(
        const CoefficientsT& coeffs, const idx_t& n, const idx_t& m) const
    {
        using namespace interface::config::model::sphericalharmonics;
        Real c = coeffs.template at<Coefficient::C>(n, m);
        Real s = (m > 0) ? coeffs.template at<Coefficient::S>(n, m) : 0.0;
        return c * sinML_ - s * cosML_;
    }

    /** @brief From the given coefficients, extract both phaseC and phaseS
     * values */
    template<typename CoefficientsT>
    FORCEINLINE()
    DEVICEHOST() Phases phases(
        const CoefficientsT& coeffs, const idx_t& n, const idx_t& m) const
    {
        using namespace interface::config::model::sphericalharmonics;
        Real c = coeffs.template at<Coefficient::C>(n, m);
        Real s = (m > 0) ? coeffs.template at<Coefficient::S>(n, m) : 0.0;
        Phases phases;
        phases.C = c * cosML_ + s * sinML_;
        phases.S = c * sinML_ - s * cosML_;
        return phases;
    }

private:
    Real sinL_;        /**< sin(lambda) */
    Real cosL_;        /**< cos(lambda) */
    Real sinML_ = 0.0; /**< sin(m*lambda) */
    Real cosML_ = 1.0; /**< cos(m*lambda) */
};

/** @brief Shmem-backed PseudoPhasor view — operates directly on a
 * per-thread 4-double slot via the supplied handle, holding no
 * register-resident copy of the recurrence state.
 *
 * Slot layout (per-thread, SoA across the block):
 *   slot[0] = sinL,  slot[1] = cosL  (invariant across the order loop)
 *   slot[2] = sinML, slot[3] = cosML (advanced by ``operator++``)
 *
 * Every access (``sinL()``, ``phases()``, ``operator++``, ...) issues a
 * fresh read/write through ``slot_``.  No local-state mirror is held by
 * the view — so a ``RefPseudoPhasor`` object itself is just two
 * references (~16 B of references on the stack as ABI scaffolding).
 * Whether the compiler keeps the slot's contents in shmem proper or
 * CSEs the reads back into registers depends on ``HandleT``: a plain
 * ``WRef::HandleT`` permits CSE, a ``VolatileRef::HandleT`` forbids it.
 *
 * Provides the same public API as @ref PseudoPhasor (``sinL``,
 * ``cosL``, ``sinML``, ``cosML``, ``phases``, ``operator++``) so it can
 * be used as a drop-in template argument to @ref DegreeLoop::accumulate.
 */
template<typename HandleT>
class RefPseudoPhasor {
    using Self = RefPseudoPhasor;
    using VecT = feta::vector::Item<Real, 3>;

    HandleT& slot_;
    const SampleIndex& i_;

public:
    FORCEINLINE() __device__
    RefPseudoPhasor(HandleT& slot, const SampleIndex& i)
        : slot_{ slot }
        , i_{ i }
    {
    }

    /** @brief Initialise the per-thread slot in shmem from
     * ``position`` (sinL, cosL derived from xy; sinML=0, cosML=1).
     * Call once, before constructing a @ref RefPseudoPhasor over the
     * slot. */
    FORCEINLINE() __device__ static void init(
        HandleT& slot, const SampleIndex& i, const VecT& position,
        const Real& eps = 1e-14)
    {
        Real x    = position.template get<0>();
        Real y    = position.template get<1>();
        Real rho  = position.template head<2>().norm();
        Real sinL = (rho > eps) ? (y / rho) : 0.0;
        Real cosL = (rho > eps) ? (x / rho) : 1.0;
        slot.template get<0>(i) = sinL;
        slot.template get<1>(i) = cosL;
        slot.template get<2>(i) = 0.0;
        slot.template get<3>(i) = 1.0;
    }

    FORCEINLINE() __device__ Real sinL() const
    {
        return slot_.template get<0>(i_);
    }
    FORCEINLINE() __device__ Real cosL() const
    {
        return slot_.template get<1>(i_);
    }
    FORCEINLINE() __device__ Real sinML() const
    {
        return slot_.template get<2>(i_);
    }
    FORCEINLINE() __device__ Real cosML() const
    {
        return slot_.template get<3>(i_);
    }

    /** @brief Advance the phasor for the next m.  Reads all four
     * fields once, writes the two advancing fields back through the
     * handle. */
    FORCEINLINE() __device__ Self& operator++()
    {
        const Real sL  = slot_.template get<0>(i_);
        const Real cL  = slot_.template get<1>(i_);
        const Real sML = slot_.template get<2>(i_);
        const Real cML = slot_.template get<3>(i_);
        slot_.template get<2>(i_) = cL * sML + sL * cML;
        slot_.template get<3>(i_) = cL * cML - sL * sML;
        return *this;
    }

    /** @brief Compute both phaseC and phaseS for the given (n, m). */
    template<typename CoefficientsT>
    FORCEINLINE() __device__ Phases phases(
        const CoefficientsT& coeffs, const idx_t& n, const idx_t& m) const
    {
        using namespace interface::config::model::sphericalharmonics;
        const Real c
            = coeffs.template at<Coefficient::C>(n, m);
        const Real s
            = (m > 0) ? coeffs.template at<Coefficient::S>(n, m) : 0.0;
        const Real sML = slot_.template get<2>(i_);
        const Real cML = slot_.template get<3>(i_);
        Phases phases;
        phases.C = c * cML + s * sML;
        phases.S = c * sML - s * cML;
        return phases;
    }
};

} // namespace sphericalharmonics
} // namespace accelerations
} // namespace model
} // namespace paraHPOP