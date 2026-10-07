#pragma once

#include "brie/typedefs.h"

#include <feta/vector/Item.h>
#include <parm/interpolate/lagrange.h>

namespace brie {
namespace orientations {
namespace detail {

/**
 * @brief Dual-table uniform-grid Lagrange interpolation policy — the
 * InterpPolicy specialization of @ref RotationModelUnit for the native IPF
 * path.
 *
 * Wraps two parm Lagrange interpolators over one flat `Real` payload:
 *   - nutation: 3 channels (X, Y, s) on a *uniform* 0.5-day grid, degree 5
 *               (uniform fast-path: `GlobalInterpolator` with x0/h)
 *   - ERP:      5 channels (xp, yp, ΔUT1, δX, δY) on a nominal 1.0-day grid,
 *               degree 3 — but the operational product is sampled uniformly
 *               in UTC, so leap seconds make the TT-MJD2000 grid *non-uniform*.
 *               It therefore uses `NonUniformInterpolator` over the actual
 *               node keys (a real interval search, no x0/h assumption), which
 *               reproduces GODOT's backend bit-for-bit across leap seconds.
 * sharing a single TT-MJD2000 key. This is a deliberate, documented
 * superset of the single-table `Type2BodyUnit` interp layer (only this
 * data-touching layer threads `UseTexture`/`work`).
 *
 * Payload layout (component-major, from the body unit's data offset):
 *   [ X[N1] Y[N1] s[N1] ][ xp[N2] yp[N2] ΔUT1[N2] δX[N2] δY[N2] ]
 *   [ x0_nut, h_nut, x0_erp, h_erp ][ ekeys[N2] ]
 * so the payload is self-describing given (N1, N2) from the metadata. The
 * ERP node keys are appended after the four grid constants (x0_erp/h_erp are
 * kept for the envelope / back-compat but no longer drive the ERP interp).
 */
template<bool UseTexture, bool work>
struct LagrangeInterp {
    /** @brief Channel layout. */
    static constexpr idx_t VD_NUT  = 3;
    static constexpr idx_t VD_ERP  = 5;
    static constexpr idx_t DEG_NUT = 5;
    static constexpr idx_t DEG_ERP = 3;
    /** @brief Total channel count. MUST equal the ModelPolicy's `NCHANNELS` —
     *  enforced at the `RotationModelUnit` call sites, which hand the same
     *  `Item<Real, NCH>` to both this policy and `ModelPolicy::compose*`, so a
     *  width mismatch is a compile error. */
    static constexpr idx_t NCH = VD_NUT + VD_ERP;

    /** @brief Per-table node-value reference types. */
    using NutVecT =
        typename feta::vector::texture::Array<Real, VD_NUT,
            UseTexture>::template Ref<work>;
    using ErpVecT =
        typename feta::vector::texture::Array<Real, VD_ERP,
            UseTexture>::template Ref<work>;
    /** @brief Whole-array scalar reference type (carries tex + base). */
    using ScalarRefT =
        typename feta::scalar::texture::Array<Real,
            UseTexture>::template Ref<work>;

    using NutInterpT =
        parm::interpolate::lagrange::GlobalInterpolator<NutVecT, DEG_NUT>;
    /* ERP grid is non-uniform (leap seconds) → search the actual node keys. */
    using ErpInterpT =
        parm::interpolate::lagrange::NonUniformInterpolator<ErpVecT, ScalarRefT,
            DEG_ERP>;

    NutInterpT nut_;
    ErpInterpT erp_;

    /** @brief Build a vector node-value reference into the flat payload. */
    template<typename VecT>
    DEVICEHOST()
    static VecT makeVecRef(
        const ScalarRefT& data, const idx_t& offset, const idx_t& n)
    {
        VecT v;
        v.data_      = data.data() + offset;
        v.nVecs_     = n;
        v.dimOffset_ = n;
        if constexpr (UseTexture) {
            v.tex_       = data.tex();
            v.texOffset_ = data.texOffset() + offset;
        } else {
            v.tex_       = 0;
            v.texOffset_ = 0;
        }
        return v;
    }

    /** @brief Build a scalar node-key reference into the flat payload (the
     *  scalar analogue of @ref makeVecRef, for the non-uniform ERP grid). */
    DEVICEHOST()
    static ScalarRefT makeKeysRef(
        const ScalarRefT& data, const idx_t& offset, const idx_t& n)
    {
        ScalarRefT k;
        k.data_ = data.data() + offset;
        k.size_ = n;
        if constexpr (UseTexture) {
            k.tex_       = data.tex();
            k.texOffset_ = data.texOffset() + offset;
        } else {
            k.tex_       = 0;
            k.texOffset_ = 0;
        }
        return k;
    }

    /**
     * @brief Construct the dual-table interpolator from the whole-array
     * scalar data reference, this unit's `baseOffset`, and the node counts.
     */
    DEVICEHOST()
    static LagrangeInterp make(const ScalarRefT& data, const idx_t& baseOffset,
        const idx_t& nNut, const idx_t& nErp)
    {
        const idx_t erpOff   = baseOffset + VD_NUT * nNut;
        const idx_t constOff = erpOff + VD_ERP * nErp;
        const idx_t keyOff   = constOff + 4; /* ERP node keys block */

        const Real x0Nut = data[constOff + 0];
        const Real hNut  = data[constOff + 1];
        /* constOff+2,3 = x0_erp, h_erp: kept for the envelope / back-compat but
         * no longer drive the ERP interp (its grid is non-uniform). */

        LagrangeInterp out;
        out.nut_ = NutInterpT::make(
            makeVecRef<NutVecT>(data, baseOffset, nNut), x0Nut, hNut);
        out.erp_ = ErpInterpT::make(
            makeVecRef<ErpVecT>(data, erpOff, nErp),
            makeKeysRef(data, keyOff, nErp));
        return out;
    }

    /** @brief Interpolate all 8 channels at TT model time `ttMjd` into the typed
     *  channel buffer: nutation → `head<VD_NUT>`, ERP → `tail<VD_ERP>`. */
    DEVICEHOST()
    void iEvalAll(feta::vector::Item<Real, NCH>& nodes, const Real& ttMjd) const
    {
        nodes.template head<VD_NUT>() = nut_.eval(ttMjd);
        nodes.template tail<VD_ERP>() = erp_.eval(ttMjd);
    }

    /** @brief Interpolate all 8 channel rates (per day) at `ttMjd` into the typed
     *  channel buffer: nutation → `head<VD_NUT>`, ERP → `tail<VD_ERP>`. */
    DEVICEHOST()
    void iDEvalAll(feta::vector::Item<Real, NCH>& rates, const Real& ttMjd) const
    {
        rates.template head<VD_NUT>() = nut_.dEval(ttMjd);
        rates.template tail<VD_ERP>() = erp_.dEval(ttMjd);
    }

    /** @brief Interpolate all 8 channel values AND rates in a SINGLE fused
     *  sweep per table (one stencil search each), mirroring the Chebyshev
     *  combined-Clenshaw pattern. This is the register-lean path for the ω
     *  kernel: it avoids the second binary search and the cross-pass liveness
     *  of `iEvalAll` then a separate `iDEvalAll`. Bit-identical to
     *  `iEvalAll` + `iDEvalAll` — the `evaldEval` value head is the same
     *  `sweep_<Deriv=true>` value accumulation as `eval`, the tail the same
     *  derivative as `dEval`. Item layout: head = value, tail = rate. */
    DEVICEHOST()
    void iEvalDEvalAll(feta::vector::Item<Real, NCH>& nodes,
        feta::vector::Item<Real, NCH>& rates, const Real& ttMjd) const
    {
        const feta::vector::Item<Real, 2 * VD_NUT> n = nut_.evaldEval(ttMjd);
        const feta::vector::Item<Real, 2 * VD_ERP> e = erp_.evaldEval(ttMjd);
        /* head = value, tail = rate (per table); scatter nut → [0, VD_NUT),
         * ERP → [VD_NUT, NCH) for both the value and rate halves. */
        nodes.template head<VD_NUT>() = n.template head<VD_NUT>();
        nodes.template tail<VD_ERP>() = e.template head<VD_ERP>();
        rates.template head<VD_NUT>() = n.template tail<VD_NUT>();
        rates.template tail<VD_ERP>() = e.template tail<VD_ERP>();
    }
};

} // namespace detail
} // namespace orientations
} // namespace brie
