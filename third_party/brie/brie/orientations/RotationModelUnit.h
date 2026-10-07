#pragma once

#include "brie/orientations/detail/IERS2000Model.h"
#include "brie/orientations/detail/LagrangeInterp.h"

#include <feta/vector/Item.h>

namespace brie {
namespace orientations {

/**
 * @brief Baseline model-based orientation unit (brie `unit_type = 4`).
 *
 * A model-based orientation family that separates *how tabular inputs are
 * interpolated* (`InterpPolicy`) from *how the interpolated values + rates
 * become a rotation* (`ModelPolicy`). Both are compile-time policies — no
 * device virtuals, in keeping with the rest of brie.
 *
 *  - `InterpPolicy<UseTexture, work>`: owns the flat `Real` payload and the
 *    per-table grid search; supplies `iEvalAll` / `iDEvalAll` / `iEvalDEvalAll`,
 *    which fill a typed `Item<Real, NCHANNELS>` of every channel's value and/or
 *    per-key rate. `UseTexture`/`work` thread through *only* this data layer.
 *  - `ModelPolicy` (static, stateless): `composeMatrix` / `compose` map the
 *    interpolated channels to a rotation matrix (+ derivative) / quaternion
 *    + angular velocity, and `epochToModelTime` maps the query epoch to the
 *    model's own time variable.
 *
 * Built now: `IERS2000Unit` = `RotationModelUnit<LagrangeInterp,
 * IERS2000Model>` (native IPF → ITRF). The empty future slot is a genuine
 * NAIF type-4 (difference-array) model — a different `ModelPolicy` over the
 * same `compose(q, ω, nodes, rates, t)` contract.
 *
 * @note The unit is dispatched through `getQuaternionDirect` /
 *       `getAngularVelocityDirect` (a direct quaternion + ω, never the
 *       RA/Dec/PM round-trip): the composed matrix already yields the
 *       quaternion via Shepperd and ω from the analytic matrix derivative,
 *       so routing through Euler angles would re-introduce a pole
 *       singularity (Earth Dec ≈ 90°) and discard the exact ω.
 */
template<template<bool, bool> class InterpPolicy, class ModelPolicy,
    bool UseTexture, bool work>
class RotationModelUnit {

public:
    /** @brief The interpolation policy instantiation. */
    using InterpT = InterpPolicy<UseTexture, work>;
    /** @brief Feta quaternion / vector types. */
    using QuatT = feta::Vec4T<Real>;
    using Vec3R = feta::Vec3T<Real>;

    /** @brief The interpolator (owns the payload references). */
    InterpT interp_;

    /** @brief Factory from the whole-array scalar data reference, this
     *  unit's data offset, and the two table node counts. */
    DEVICEHOST()
    static RotationModelUnit make(const typename InterpT::ScalarRefT& data,
        const idx_t& baseOffset, const idx_t& nNut, const idx_t& nErp)
    {
        RotationModelUnit out;
        out.interp_ = InterpT::make(data, baseOffset, nNut, nErp);
        return out;
    }

    /** @brief Rotation matrix `R` (ICRF→target) and its per-second time
     *  derivative `Ṙ` at TT model time `ttMjd` (test / introspection hook;
     *  pass `Rdot == nullptr` to skip the derivative).
     *  @note Introspection-only precision: the ERA phase here is fed the
     *  day-fraction `ttMjd − floor(ttMjd)`, quantized to ULP(ttMjd) → a ~2.5e-11
     *  rad ERA floor (no TDB epoch is available at this entry to recover the
     *  high-precision fraction the production paths get from `epochToModelTime`);
     *  this is what `RotationModelTest`'s ~2e-11 matrix tolerance is set against,
     *  so do not use `matrices()` for production-grade phase. When
     *  `Rdot == nullptr`, `rates` stays default-zero and is unused (composeMatrix
     *  reads the rate channels only behind the `Rdot` guard). */
    DEVICEHOST()
    void matrices(Real* const R, Real* const Rdot, const Real& ttMjd) const
    {
        feta::vector::Item<Real, ModelPolicy::NCHANNELS> nodes, rates;
        if (Rdot != nullptr)
            interp_.iEvalDEvalAll(nodes, rates, ttMjd); /* fused: values + rates */
        else
            interp_.iEvalAll(nodes, ttMjd); /* values only */
        ModelPolicy::composeMatrix(
            R, Rdot, nodes, rates, ttMjd, ttMjd - feta::math::floor(ttMjd));
    }

    /** @brief Quaternion (ICRF→target) + angular velocity ω (of target
     *  w.r.t. ICRF, inertial components, rad/s) at TT model time `ttMjd`. */
    DEVICEHOST()
    void quatAndOmegaModelTime(
        QuatT& q, Vec3R& w, const Real& ttMjd) const
    {
        feta::vector::Item<Real, ModelPolicy::NCHANNELS> nodes, rates;
        interp_.iEvalDEvalAll(nodes, rates, ttMjd); /* fused single sweep */
        ModelPolicy::compose(
            q, w, nodes, rates, ttMjd, ttMjd - feta::math::floor(ttMjd));
    }

    /** @brief Quaternion + ω at the query epoch `tdbEpoch` (TDB SPICE
     *  seconds past J2000); converts to the model's time variable first.
     *  Carries the high-precision O(1) day-fraction (`epochToModelTime`'s
     *  out-param) into `compose` so the ERA phase is not quantized to
     *  ULP(model-time) — the model-time double itself only feeds interp and
     *  the slow ERA polynomial term, which tolerate the magnitude. */
    DEVICEHOST()
    void quatAndOmega(QuatT& q, Vec3R& w, const Real& tdbEpoch) const
    {
        Real dayFrac;
        const Real ttMjd = ModelPolicy::epochToModelTime(tdbEpoch, &dayFrac);
        feta::vector::Item<Real, ModelPolicy::NCHANNELS> nodes, rates;
        interp_.iEvalDEvalAll(nodes, rates, ttMjd); /* fused single sweep */
        ModelPolicy::compose(q, w, nodes, rates, ttMjd, dayFrac);
    }

    /** @brief Quaternion only (ICRF→target) at the query epoch. */
    DEVICEHOST()
    QuatT getQuaternion(const Real& tdbEpoch) const
    {
        QuatT q;
        Vec3R w;
        quatAndOmega(q, w, tdbEpoch);
        return q;
    }

    /** @brief Angular velocity only (rad/s, inertial) at the query epoch —
     *  the PRODUCTION cudaj `omegaResolveKernel` path (via
     *  `RefEphUnit::getAngularVelocityDirect`). The register-spill fix (#28)
     *  is the FUSED single-sweep interpolation (`iEvalDEvalAll`), which avoids
     *  the cross-pass liveness of separate value/derivative sweeps; the
     *  rotation itself uses the full `compose` (exact ω = vee(Ṙᵀ R)), the
     *  quaternion discarded (DCE'd). */
    DEVICEHOST()
    Vec3R getAngularVelocity(const Real& tdbEpoch) const
    {
        Real dayFrac;
        const Real ttMjd = ModelPolicy::epochToModelTime(tdbEpoch, &dayFrac);
        feta::vector::Item<Real, ModelPolicy::NCHANNELS> nodes, rates;
        interp_.iEvalDEvalAll(nodes, rates, ttMjd); /* fused single sweep */
        QuatT q;
        Vec3R w;
        /* quaternion-native compose: no dense Ṙ materialization → no register
         * spill (the q is discarded/DCE'd here; ω is the exact rotation's
         * angular velocity, consistent with the cached quaternion). */
        ModelPolicy::composeQuat(q, w, nodes, rates, ttMjd, dayFrac);
        return w;
    }

    DEVICEHOST() RotationModelUnit clone() const { return *this; }
};

/** @brief The native IPF → ITRF (IERS2000) orientation unit built now. */
template<bool UseTexture, bool work>
using IERS2000Unit = RotationModelUnit<detail::LagrangeInterp,
    detail::IERS2000Model, UseTexture, work>;

/* Future slot (not built): a genuine NAIF type-4 difference-array model is
 * another ModelPolicy over the same compose(q, ω, nodes, rates, t) contract:
 *   template<bool U, bool w>
 *   using Type4Unit = RotationModelUnit<SomeInterp, Type4Model, U, w>; */

} // namespace orientations
} // namespace brie
