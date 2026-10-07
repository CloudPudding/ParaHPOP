#pragma once

#include "parm/typedefs.h"
#include "parm/util.h"

#include <feta/core/simd/Packet.h>
#include <feta/math/math.h>
#include <feta/vector/Item.h>
#include <feta/vector/PacketItem.h>

namespace parm {
namespace interpolate {
namespace lagrange {

namespace detail {

/**
 * @brief Compile-time-unrolled per-channel Lagrange accumulate (shared by the
 * uniform and non-uniform evaluators).
 *
 * Reads channel ``From + d`` of the ``node``-th vector and folds in the shared
 * Lagrange weights (``wval`` for the value, ``wder`` for the derivative). The
 * weights are computed by the caller's stencil walk; this helper only depends
 * on ``coeffs`` and the two scalar weights, so it is identical for a uniform
 * (local integer coordinate) or non-uniform (raw key) grid.
 */
template<typename CoeffsT, idx_t From, idx_t HowMany, bool Deriv, idx_t d = 0>
DEVICEHOST()
void accumulateChannels(typename CoeffsT::ComponentT* const val,
    typename CoeffsT::ComponentT* const der,
    const typename CoeffsT::ComponentT& wval,
    const typename CoeffsT::ComponentT& wder, const idx_t& node,
    const CoeffsT& coeffs)
{
    if constexpr (d < HowMany) {
        const typename CoeffsT::ComponentT y = coeffs.template get<From + d>(node);
        val[d]                               = feta::math::fma(wval, y, val[d]);
        if constexpr (Deriv)
            der[d] = feta::math::fma(wder, y, der[d]);
        accumulateChannels<CoeffsT, From, HowMany, Deriv, d + 1>(
            val, der, wval, wder, node, coeffs);
    }
}

/**
 * @brief Shared single value(+derivative) Lagrange sweep over the ``NPts``
 * stencil — the common core of the uniform and non-uniform evaluators.
 *
 * For ``IsUniform`` the stencil nodes sit at the integer local coordinates
 * ``0..NPts-1`` (``query`` is the local coordinate ``u`` and the denominator
 * factor is the exact integer offset ``j - m``).  For a non-uniform grid the
 * nodes are the raw keys ``keys[lo + 0..NPts-1]`` (``query`` is the global
 * coordinate ``t`` and the denominator factor is the real key difference).
 * Everything else — the product/derivative recurrence (with its ``fma`` order),
 * the reciprocal-denominator weight, and the per-channel @ref accumulateChannels
 * fold — is identical and lives here once.  The derivative is written
 * *unscaled*: the uniform caller applies its ``du/dt = 1/h`` chain factor
 * afterwards, while the non-uniform derivative is already ``d/dt``.
 */
template<typename CoeffsT, idx_t Deg, bool IsUniform, idx_t From, idx_t HowMany,
    bool Deriv, typename KeysT = int>
DEVICEHOST()
void lagrangeSweep(typename CoeffsT::ComponentT* const val,
    typename CoeffsT::ComponentT* const der, const CoeffsT& coeffs,
    const idx_t lo, const typename CoeffsT::ComponentT query,
    const KeysT* keys = nullptr)
{
    using ComponentT     = typename CoeffsT::ComponentT;
    constexpr idx_t NPts = Deg + 1;
    for (idx_t d = 0; d < HowMany; ++d) {
        val[d] = ComponentT(0);
        if constexpr (Deriv)
            der[d] = ComponentT(0);
    }
    for (idx_t j = 0; j < NPts; ++j) {
        /* P_j(query) = prod_{m!=j}(query - node_m), its derivative, and the
         * denominator D_j = prod_{m!=j}(node_j - node_m). */
        ComponentT prod  = ComponentT(1);
        ComponentT dprod = ComponentT(0);
        ComponentT denom = ComponentT(1);
        for (idx_t m = 0; m < NPts; ++m) {
            if (m == j)
                continue;
            ComponentT f;
            ComponentT dfac;
            if constexpr (IsUniform) {
                f = query - ComponentT(m);
                /* (j - m) is a signed offset; idx_t is unsigned, so form the
                 * difference in signed arithmetic before promoting. */
                dfac = ComponentT((long)j - (long)m);
            } else {
                const ComponentT xm = (*keys)[lo + m];
                f                   = query - xm;
                dfac                = (*keys)[lo + j] - xm;
            }
            if constexpr (Deriv)
                dprod = feta::math::fma(dprod, f, prod);
            prod *= f;
            denom *= dfac;
        }
        const ComponentT invD = ComponentT(1) / denom;
        const ComponentT wval = prod * invD;
        const ComponentT wder = Deriv ? dprod * invD : ComponentT(0);
        accumulateChannels<CoeffsT, From, HowMany, Deriv, 0>(
            val, der, wval, wder, lo + j, coeffs);
    }
}

/**
 * @brief Compile-time-unrolled per-component packet accumulate (shared by both
 * evaluators' ``evalPacket``):
 * ``out.packet<d>() += sum_j wval[j] * coeffs.get<From + d>(lo + j)``.
 */
template<typename CoeffsT, idx_t Deg, idx_t W, idx_t From, idx_t HowMany,
    idx_t d = 0>
void accumulatePacket(
    feta::vector::PacketItem<typename CoeffsT::ComponentT, HowMany, W>& out,
    const CoeffsT& coeffs,
    const feta::simd::Packet<typename CoeffsT::ComponentT, W> (&wval)[Deg + 1],
    const idx_t lo)
{
    using ComponentT     = typename CoeffsT::ComponentT;
    using PacketT        = feta::simd::Packet<ComponentT, W>;
    constexpr idx_t NPts = Deg + 1;
    if constexpr (d < HowMany) {
        PacketT acc = out.template packet<d>();
        for (idx_t j = 0; j < NPts; ++j)
            acc = acc
                + wval[j]
                    * PacketT::broadcast(coeffs.template get<From + d>(lo + j));
        out.template packet<d>() = acc;
        accumulatePacket<CoeffsT, Deg, W, From, HowMany, d + 1>(
            out, coeffs, wval, lo);
    }
}

/**
 * @brief Shared host-SIMD (W-lane) value evaluation — the common core of the
 * uniform and non-uniform ``evalPacket``.  The product Lagrange weights depend
 * only on the (lane-shared) node positions; ``IsUniform`` selects integer local
 * nodes vs. raw keys exactly as @ref lagrangeSweep does.  Accumulates into
 * ``out`` (the caller zeroes it).
 */
template<typename CoeffsT, idx_t Deg, bool IsUniform, idx_t W, idx_t From,
    idx_t HowMany, typename KeysT = int>
void lagrangePacketEval(
    feta::vector::PacketItem<typename CoeffsT::ComponentT, HowMany, W>& out,
    const CoeffsT& coeffs, const idx_t lo,
    const feta::simd::Packet<typename CoeffsT::ComponentT, W>& qPkt,
    const KeysT* keys = nullptr)
{
    using ComponentT     = typename CoeffsT::ComponentT;
    using PacketT        = feta::simd::Packet<ComponentT, W>;
    constexpr idx_t NPts = Deg + 1;
    PacketT wval[NPts];
    for (idx_t j = 0; j < NPts; ++j) {
        PacketT prod     = PacketT::broadcast(ComponentT(1));
        ComponentT denom = ComponentT(1);
        for (idx_t m = 0; m < NPts; ++m) {
            if (m == j)
                continue;
            if constexpr (IsUniform) {
                prod  = prod * (qPkt - PacketT::broadcast(ComponentT(m)));
                denom = denom * ComponentT((long)j - (long)m);
            } else {
                const ComponentT xm = (*keys)[lo + m];
                prod                = prod * (qPkt - PacketT::broadcast(xm));
                denom               = denom * ((*keys)[lo + j] - xm);
            }
        }
        wval[j] = prod * PacketT::broadcast(ComponentT(1) / denom);
    }
    accumulatePacket<CoeffsT, Deg, W, From, HowMany, 0>(out, coeffs, wval, lo);
}

} // namespace detail

/**
 * @brief Fixed-degree Lagrange evaluator on a uniform node grid.
 *
 * Evaluates the unique degree-``Deg`` polynomial through ``NPts = Deg + 1``
 * equally spaced nodes.  The query is expressed in a *local* coordinate
 * ``u`` in which the stencil nodes sit at the integer positions
 * ``u = 0, 1, ..., NPts - 1``.  The first (product) Lagrange form is used,
 *
 *     L_j(u) = prod_{m != j} (u - m) / (j - m),
 *
 * which — unlike the second (barycentric) form — never divides by
 * ``(u - x_j)`` and is therefore *node-safe*: a query landing exactly on a
 * node produces a clean value, with no 0/0.  The denominators
 * ``D_j = prod_{m != j}(j - m)`` are exact small integers
 * (``|D_j| <= (NPts - 1)!``), represented exactly in double precision.
 *
 * The derivative shares the same product walk via the product rule
 * (``d/du prod = sum_k prod_{m != k}``), so a combined value+derivative
 * evaluation costs a single ``O(NPts^2)`` sweep.  ``dEval`` returns the
 * derivative with respect to the interpolation variable ``t`` (i.e. it
 * applies the ``du/dt = 1/h`` chain-rule factor); for the IPF node tables
 * (keyed in days) the rate is therefore *per day*.
 *
 * @note Stencil selection matches GODOT's ``BlockDataInterpolatorBackend``
 *       for interior queries: with ``HalfPts = NPts / 2`` half-points, the
 *       window is the ``NPts`` nodes centred on the bracketing interval.
 *       At the absolute array ends the window is clamped to keep a full
 *       ``NPts`` stencil (a simpler, standard policy than GODOT's
 *       end-of-block point reduction; the usable epoch range never reaches
 *       within ``HalfPts`` nodes of the raw array ends).
 */
template<typename CoeffsT, idx_t Deg>
struct Evaluator {
    /** @brief Scalar component type. */
    using ComponentT = typename CoeffsT::ComponentT;
    /** @brief Number of stencil nodes. */
    static constexpr idx_t NPts = Deg + 1;
    /** @brief Half-points (GODOT ``numHalfpoints``). */
    static constexpr idx_t HalfPts = NPts / 2;

    /** @brief Single value(+derivative) sweep over the ``NPts`` stencil.
     *
     *  Writes (overwrites) ``HowMany`` components into ``val`` (and, when
     *  ``Deriv``, into ``der``, already scaled by ``dScale = 1/h``). */
    template<idx_t From, idx_t HowMany, bool Deriv>
    DEVICEHOST()
    void sweep_(ComponentT* const val, ComponentT* const der,
        const CoeffsT& coeffs, const ComponentT& dScale) const
    {
        detail::lagrangeSweep<CoeffsT, Deg, true, From, HowMany, Deriv>(
            val, der, coeffs, lo, u);
        if constexpr (Deriv)
            for (idx_t d = 0; d < HowMany; ++d)
                der[d] *= dScale;
    }

    /** @brief Evaluate the interpolant (overwrites ``out``). */
    template<idx_t From, idx_t HowMany>
    DEVICEHOST()
    void eval(feta::vector::Item<ComponentT, HowMany>& out,
        const CoeffsT& coeffs) const
    {
        sweep_<From, HowMany, false>(out.data(), nullptr, coeffs, ComponentT(0));
    }

    /** @brief Evaluate the derivative w.r.t. ``t`` (overwrites ``out``). */
    template<idx_t From, idx_t HowMany>
    DEVICEHOST()
    void dEval(feta::vector::Item<ComponentT, HowMany>& out,
        const CoeffsT& coeffs, const ComponentT& dScale) const
    {
        ComponentT scratch[HowMany];
        sweep_<From, HowMany, true>(scratch, out.data(), coeffs, dScale);
    }

    /** @brief Evaluate value and derivative together (overwrites ``out``;
     *  head = value, tail = derivative w.r.t. ``t``). */
    template<idx_t From, idx_t HowMany>
    DEVICEHOST()
    void evaldEval(feta::vector::Item<ComponentT, 2 * HowMany>& out,
        const CoeffsT& coeffs, const ComponentT& dScale) const
    {
        sweep_<From, HowMany, true>(
            out.data(), out.data() + HowMany, coeffs, dScale);
    }

    /* Vector array handle types (method-level, parameterized by HowMany —
     * this Evaluator templates From/HowMany per method, not at struct
     * scope). */
    template<bool work, idx_t HowMany>
    using VecArrT = typename feta::vector::Array<ComponentT,
        HowMany>::template Ref<work>::HandleT;
    template<bool work, idx_t HowMany>
    using TwoVecArrT = typename feta::vector::Array<ComponentT,
        2 * HowMany>::template Ref<work>::HandleT;

    /** @brief Array-based value evaluation (accumulates ``sign *`` value into
     *  ``out[index]``; reuses the scalar ``sweep_``). */
    template<bool work, idx_t From, idx_t HowMany>
    DEVICEHOST()
    void eval(const SampleIndex& index, VecArrT<work, HowMany>& out,
        const CoeffsT& coeffs, const Real& sign) const
    {
        feta::vector::Item<ComponentT, HowMany> v;
        sweep_<From, HowMany, false>(v.data(), nullptr, coeffs, ComponentT(0));
        out[index] += sign * v;
    }

    /** @brief Array-based derivative evaluation w.r.t. ``t`` (accumulates).
     *  ``dScale = 1/h`` is folded inside ``sweep_`` — do NOT re-apply it. */
    template<bool work, idx_t From, idx_t HowMany>
    DEVICEHOST()
    void dEval(const SampleIndex& index, VecArrT<work, HowMany>& out,
        const CoeffsT& coeffs, const ComponentT& dScale, const Real& sign) const
    {
        ComponentT scratch[HowMany];
        feta::vector::Item<ComponentT, HowMany> d;
        sweep_<From, HowMany, true>(scratch, d.data(), coeffs, dScale);
        out[index] += sign * d;
    }

    /** @brief Array-based value+derivative (head = value, tail = rate). */
    template<bool work, idx_t From, idx_t HowMany>
    DEVICEHOST()
    void evaldEval(const SampleIndex& index, TwoVecArrT<work, HowMany>& out,
        const CoeffsT& coeffs, const ComponentT& dScale, const Real& sign) const
    {
        feta::vector::Item<ComponentT, 2 * HowMany> vd;
        sweep_<From, HowMany, true>(
            vd.data(), vd.data() + HowMany, coeffs, dScale);
        out[index].template head<HowMany>()
            += sign * vd.template head<HowMany>();
        out[index].template tail<HowMany>()
            += sign * vd.template tail<HowMany>();
    }

    /** @brief Packet (W-lane) value evaluation — host SIMD analogue of
     *  ``eval``.  All W lanes share this evaluator's stencil ``lo``; only the
     *  local coordinate differs per lane and is supplied as ``uPkt``.
     *
     *  The first (product) Lagrange weights depend only on the local
     *  coordinate (not on the component), so the ``NPts`` per-lane weights are
     *  computed once and reused across components — cheaper than the
     *  coefficient-coupled per-component recurrence the Chebyshev packet path
     *  needs.  Lane-for-lane this reproduces the scalar ``sweep_`` value pass
     *  (same per-component summation order; the only difference is fused vs
     *  unfused multiply-add, well within the evaluator's accuracy contract).
     *
     *  Accumulates into ``out`` (the caller zeroes it), matching the Chebyshev
     *  ``ievalPacket`` accumulate contract; the scalar ``eval`` overwrites. */
    template<idx_t W, idx_t From, idx_t HowMany>
    void evalPacket(feta::vector::PacketItem<ComponentT, HowMany, W>& out,
        const CoeffsT& coeffs,
        const feta::simd::Packet<ComponentT, W>& uPkt) const
    {
        detail::lagrangePacketEval<CoeffsT, Deg, true, W, From, HowMany>(
            out, coeffs, lo, uPkt);
    }

    /** @brief Stencil start node index. */
    const idx_t lo;
    /** @brief Local coordinate; stencil nodes sit at integers ``0..NPts-1``. */
    const ComponentT u;
};

/**
 * @brief Global Lagrange interpolator over a uniform node grid.
 *
 * Owns a flat vector array of node values (``VecDims`` channels per node)
 * plus the grid origin ``x0`` and spacing ``h``.  Locates the centred
 * ``NPts``-node stencil for a query time and evaluates value / derivative /
 * combined via @ref Evaluator.  Callable from host and device.
 *
 * @tparam CoeffsT  Node-value array type (a FETA vector array reference)
 *                  exposing ``ComponentT``, ``ItemT``, ``VecDims``,
 *                  ``size()`` and ``get<dim>(idx)``.
 * @tparam Deg      Polynomial degree (``NPts = Deg + 1`` stencil nodes).
 */
template<typename CoeffsT, idx_t Deg>
class GlobalInterpolator {
    using Self = GlobalInterpolator;

    /** @brief Compile-time-unrolled per-component packet load + accumulate.
     *  Helper for ``ievalPacket``'s mixed-stencil fallback: walks ``D`` from
     *  ``HowMany - 1`` down to ``0``, loading each lane-major row of
     *  ``compBuf[D][W]`` as a packet and adding it into ``out.packet<D>()``. */
    template<idx_t W, idx_t HowMany, dims_t D>
    static inline void loadAndAccumulate_(
        feta::vector::PacketItem<typename CoeffsT::ComponentT, HowMany, W>& out,
        const typename CoeffsT::ComponentT (&compBuf)[HowMany][W])
    {
        using PacketT = feta::simd::Packet<typename CoeffsT::ComponentT, W>;
        out.template packet<D>()
            = out.template packet<D>() + PacketT::load(compBuf[D]);
        if constexpr (D >= 1)
            loadAndAccumulate_<W, HowMany, D - 1>(out, compBuf);
    }

public:
    /** @brief Scalar component type. */
    using ComponentT = typename CoeffsT::ComponentT;
    /** @brief The vector item type. */
    using ItemT = typename CoeffsT::ItemT;
    /** @brief Number of channels per node. */
    static constexpr idx_t VecDims = CoeffsT::VecDims;
    /** @brief Number of stencil nodes. */
    static constexpr idx_t NPts = Deg + 1;
    /** @brief Half-points. */
    static constexpr idx_t HalfPts = NPts / 2;
    /** @brief Evaluator type. */
    using EvaluatorT = Evaluator<CoeffsT, Deg>;
    /** @brief Templated item type. */
    template<idx_t HM>
    using TItemT = feta::vector::Item<ComponentT, HM>;
    /** @brief Vector / double-vector array handle types (for the
     *  array-accumulate overloads). */
    template<bool work, idx_t HowMany = VecDims>
    using VecArrT = typename feta::vector::Array<ComponentT,
        HowMany>::template Ref<work>::HandleT;
    template<bool work, idx_t HowMany = VecDims>
    using TwoVecArrT = typename feta::vector::Array<ComponentT,
        2 * HowMany>::template Ref<work>::HandleT;

    /** @brief Factory from node values + uniform grid (origin, spacing). */
    DEVICEHOST()
    static GlobalInterpolator make(
        const CoeffsT& values, const ComponentT& x0, const ComponentT& h)
    {
        GlobalInterpolator g;
        g.values_ = values;
        g.x0_     = x0;
        g.h_      = h;
        return g;
    }

    /** @brief Number of grid nodes. */
    DEVICEHOST() idx_t numNodes() const { return values_.size(); }

    /** @brief Locate the centred stencil and local coordinate for ``t``. */
    DEVICEHOST() EvaluatorT findStencil(const ComponentT& t) const
    {
        /* node[fIdx] <= t < node[fIdx+1] on the uniform grid.  Use signed
         * arithmetic for the stencil placement so out-of-range queries
         * clamp correctly (idx_t is unsigned). */
        const ComponentT ratio = (t - x0_) / h_;
        long fIdx              = (long)feta::math::floor(ratio);
        /* centre the NPts-node window on the bracketing interval */
        long lo          = fIdx - (long)HalfPts + 1;
        const long maxLo = (long)values_.size() - (long)NPts;
        if (lo < 0)
            lo = 0;
        if (lo > maxLo)
            lo = maxLo;
        const ComponentT u = (t - (x0_ + (ComponentT)lo * h_)) / h_;
        return EvaluatorT{ (idx_t)lo, u };
    }

    /** @brief Evaluate the interpolation. */
    template<idx_t From = 0, idx_t HowMany = VecDims>
    DEVICEHOST() TItemT<HowMany> eval(const ComponentT& t) const
    {
        TItemT<HowMany> out;
        ieval<From, HowMany>(out, t);
        return out;
    }
    template<idx_t From = 0, idx_t HowMany = VecDims>
    DEVICEHOST() void ieval(TItemT<HowMany>& out, const ComponentT& t) const
    {
        findStencil(t).template eval<From, HowMany>(out, values_);
    }

    /** @brief Evaluate the derivative w.r.t. the interpolation variable. */
    template<idx_t From = 0, idx_t HowMany = VecDims>
    DEVICEHOST() TItemT<HowMany> dEval(const ComponentT& t) const
    {
        TItemT<HowMany> out;
        idEval<From, HowMany>(out, t);
        return out;
    }
    template<idx_t From = 0, idx_t HowMany = VecDims>
    DEVICEHOST() void idEval(TItemT<HowMany>& out, const ComponentT& t) const
    {
        findStencil(t).template dEval<From, HowMany>(
            out, values_, ComponentT(1) / h_);
    }

    /** @brief Evaluate both value and derivative (head = value, tail = rate). */
    template<idx_t From = 0, idx_t HowMany = VecDims>
    DEVICEHOST() TItemT<2 * HowMany> evaldEval(const ComponentT& t) const
    {
        TItemT<2 * HowMany> out;
        iEvaldEval<From, HowMany>(out, t);
        return out;
    }
    template<idx_t From = 0, idx_t HowMany = VecDims>
    DEVICEHOST()
    void iEvaldEval(TItemT<2 * HowMany>& out, const ComponentT& t) const
    {
        findStencil(t).template evaldEval<From, HowMany>(
            out, values_, ComponentT(1) / h_);
    }

    /** @brief Array-accumulate value: out[index] += sign * value(t). */
    template<bool work, idx_t From = 0, idx_t HowMany = VecDims>
    DEVICEHOST()
    void ieval(const SampleIndex& index, VecArrT<work, HowMany>& out,
        const ComponentT& t, const Real& sign) const
    {
        findStencil(t).template eval<work, From, HowMany>(
            index, out, values_, sign);
    }
    /** @brief Array-accumulate derivative: out[index] += sign * d/dt(t). */
    template<bool work, idx_t From = 0, idx_t HowMany = VecDims>
    DEVICEHOST()
    void idEval(const SampleIndex& index, VecArrT<work, HowMany>& out,
        const ComponentT& t, const Real& sign) const
    {
        findStencil(t).template dEval<work, From, HowMany>(
            index, out, values_, ComponentT(1) / h_, sign);
    }
    /** @brief Array-accumulate value+derivative (head = value, tail = rate). */
    template<bool work, idx_t From = 0, idx_t HowMany = VecDims>
    DEVICEHOST()
    void iEvaldEval(const SampleIndex& index, TwoVecArrT<work, HowMany>& out,
        const ComponentT& t, const Real& sign) const
    {
        findStencil(t).template evaldEval<work, From, HowMany>(
            index, out, values_, ComponentT(1) / h_, sign);
    }

    /** @brief Packet (W-lane) value interpolation — host SIMD evaluation.
     *
     *  Per-lane query times in ``tPkt`` may map to the same or to different
     *  ``NPts``-node stencils.  Mirrors the Chebyshev
     *  ``GlobalInterpolator::ievalPacket`` structure:
     *  - Per-lane ``findStencil`` + a shared-stencil predicate.
     *  - If all lanes share the same stencil start ``lo`` (the common case for
     *    sub-step queries inside a node window), pack the per-lane local
     *    coordinates and run the packet Lagrange sweep
     *    (``Evaluator::evalPacket``) once.
     *  - Otherwise (lanes straddling a stencil shift) fall back to per-lane
     *    scalar ``eval``, transposing the lane-major scalar results into
     *    component-major packets.  Correctness-equivalent to the scalar path;
     *    loses SIMD on the rare boundary packets.
     *
     *  Accumulates into ``out`` (the caller is responsible for zeroing it),
     *  matching the Chebyshev packet contract and the brie packet-ephemeris
     *  consumer.  Value-only, mirroring the Chebyshev packet path.
     *
     *  @tparam W        Packet width (lane count).
     *  @tparam From     Component start (default 0).
     *  @tparam HowMany  Component count (default VecDims).
     */
    template<idx_t W, idx_t From = 0, idx_t HowMany = VecDims>
    inline void ievalPacket(
        feta::vector::PacketItem<ComponentT, HowMany, W>& out,
        const feta::simd::Packet<ComponentT, W>& tPkt) const
    {
        using PacketT = feta::simd::Packet<ComponentT, W>;

        alignas(64) ComponentT tBuf[W];
        PacketT::store(tBuf, tPkt);

        /* Per-lane stencil lookup + shared-stencil test. */
        idx_t loBuf[W];
        alignas(64) ComponentT uBuf[W];
        EvaluatorT e0   = findStencil(tBuf[0]);
        loBuf[0]        = e0.lo;
        uBuf[0]         = e0.u;
        bool uniformLo  = true;
        for (idx_t k = 1; k < W; ++k) {
            const EvaluatorT ek = findStencil(tBuf[k]);
            loBuf[k]            = ek.lo;
            uBuf[k]             = ek.u;
            uniformLo &= (loBuf[k] == loBuf[0]);
        }

        if (uniformLo) {
            const PacketT uPkt = PacketT::load(uBuf);
            /* shared stencil; the scalar ``u`` member is unused on this path */
            const EvaluatorT e{ loBuf[0], ComponentT(0) };
            e.template evalPacket<W, From, HowMany>(out, values_, uPkt);
            return;
        }

        /* Mixed-stencil fallback: per-lane scalar evaluator, transpose
         * lane-major scalar results into component-major packets. */
        alignas(64) ComponentT compBuf[HowMany][W] = {};
        for (idx_t k = 0; k < W; ++k) {
            const EvaluatorT eLane{ loBuf[k], uBuf[k] };
            TItemT<HowMany> scalarOut; /* scalar eval overwrites */
            eLane.template eval<From, HowMany>(scalarOut, values_);
            const ComponentT* sd = scalarOut.data();
            for (idx_t d = 0; d < HowMany; ++d)
                compBuf[d][k] = sd[d];
        }
        loadAndAccumulate_<W, HowMany, HowMany - 1>(out, compBuf);
    }

    /** @brief Expose the node values. */
    DEVICEHOST() const CoeffsT& values() const { return values_; }
    /** @brief Expose the grid origin. */
    DEVICEHOST() const ComponentT& x0() const { return x0_; }
    /** @brief Expose the grid spacing. */
    DEVICEHOST() const ComponentT& h() const { return h_; }

    /** @brief Data made public for PODification. */
    CoeffsT values_   = CoeffsT{};
    ComponentT x0_    = ComponentT(0);
    ComponentT h_     = ComponentT(0);
};

/**
 * @brief Fixed-degree Lagrange evaluator on a *non-uniform* node grid.
 *
 * The stencil nodes sit at arbitrary, strictly increasing positions
 * ``keys[lo], ..., keys[lo + Deg]``. Unlike the uniform @ref Evaluator (which
 * works in a local integer coordinate and uses exact integer denominators),
 * this evaluator forms the first (product) Lagrange form directly in the
 * *global* interpolation variable on the raw keys,
 *
 *     L_j(t) = prod_{m != j} (t - keys[m]) / (keys[j] - keys[m]),
 *
 * which is node-safe (no division by ``t - keys[j]``) and reproduces the exact
 * arithmetic GODOT's ``IpfReader`` backend uses for the operational ERP grid
 * (whose leap-second gaps make it non-uniform). Because the variable is ``t``
 * itself, the derivative from the product-rule walk is already ``d/dt`` — there
 * is *no* ``1/h`` chain-rule factor (contrast the uniform evaluator's
 * ``dScale``).
 *
 * @note Working in global coordinates intentionally matches GODOT bit-for-bit;
 *       a local-origin shift would be more accurate than GODOT and therefore
 *       *break* the fidelity gate.
 */
template<typename CoeffsT, typename KeysT, idx_t Deg>
struct NonUniformEvaluator {
    /** @brief Scalar component type. */
    using ComponentT = typename CoeffsT::ComponentT;
    /** @brief Number of stencil nodes. */
    static constexpr idx_t NPts = Deg + 1;
    /** @brief Half-points (window centring). */
    static constexpr idx_t HalfPts = NPts / 2;

    /** @brief Single value(+derivative) sweep over the ``NPts`` stencil in
     *  global coordinates (raw keys); the derivative is w.r.t. ``t`` directly
     *  (no ``1/h`` scale). */
    template<idx_t From, idx_t HowMany, bool Deriv>
    DEVICEHOST()
    void sweep_(ComponentT* const val, ComponentT* const der,
        const CoeffsT& coeffs) const
    {
        detail::lagrangeSweep<CoeffsT, Deg, false, From, HowMany, Deriv, KeysT>(
            val, der, coeffs, lo, t, &keys_);
    }

    /** @brief Evaluate the interpolant (overwrites ``out``). */
    template<idx_t From, idx_t HowMany>
    DEVICEHOST()
    void eval(feta::vector::Item<ComponentT, HowMany>& out,
        const CoeffsT& coeffs) const
    {
        sweep_<From, HowMany, false>(out.data(), nullptr, coeffs);
    }

    /** @brief Evaluate the derivative w.r.t. ``t`` (overwrites ``out``). */
    template<idx_t From, idx_t HowMany>
    DEVICEHOST()
    void dEval(feta::vector::Item<ComponentT, HowMany>& out,
        const CoeffsT& coeffs) const
    {
        ComponentT scratch[HowMany];
        sweep_<From, HowMany, true>(scratch, out.data(), coeffs);
    }

    /** @brief Evaluate value and derivative together (head = value, tail =
     *  derivative w.r.t. ``t``). */
    template<idx_t From, idx_t HowMany>
    DEVICEHOST()
    void evaldEval(feta::vector::Item<ComponentT, 2 * HowMany>& out,
        const CoeffsT& coeffs) const
    {
        sweep_<From, HowMany, true>(out.data(), out.data() + HowMany, coeffs);
    }

    /* Vector array handle types (method-level, parameterized by HowMany). */
    template<bool work, idx_t HowMany>
    using VecArrT = typename feta::vector::Array<ComponentT,
        HowMany>::template Ref<work>::HandleT;
    template<bool work, idx_t HowMany>
    using TwoVecArrT = typename feta::vector::Array<ComponentT,
        2 * HowMany>::template Ref<work>::HandleT;

    /** @brief Array-based value evaluation (accumulates into ``out[index]``). */
    template<bool work, idx_t From, idx_t HowMany>
    DEVICEHOST()
    void eval(const SampleIndex& index, VecArrT<work, HowMany>& out,
        const CoeffsT& coeffs, const Real& sign) const
    {
        feta::vector::Item<ComponentT, HowMany> v;
        sweep_<From, HowMany, false>(v.data(), nullptr, coeffs);
        out[index] += sign * v;
    }

    /** @brief Array-based derivative evaluation w.r.t. ``t`` (no ``1/h``
     *  scale — global-coordinate derivative; accumulates). */
    template<bool work, idx_t From, idx_t HowMany>
    DEVICEHOST()
    void dEval(const SampleIndex& index, VecArrT<work, HowMany>& out,
        const CoeffsT& coeffs, const Real& sign) const
    {
        ComponentT scratch[HowMany];
        feta::vector::Item<ComponentT, HowMany> d;
        sweep_<From, HowMany, true>(scratch, d.data(), coeffs);
        out[index] += sign * d;
    }

    /** @brief Array-based value+derivative (head = value, tail = d/dt rate). */
    template<bool work, idx_t From, idx_t HowMany>
    DEVICEHOST()
    void evaldEval(const SampleIndex& index, TwoVecArrT<work, HowMany>& out,
        const CoeffsT& coeffs, const Real& sign) const
    {
        feta::vector::Item<ComponentT, 2 * HowMany> vd;
        sweep_<From, HowMany, true>(vd.data(), vd.data() + HowMany, coeffs);
        out[index].template head<HowMany>()
            += sign * vd.template head<HowMany>();
        out[index].template tail<HowMany>()
            += sign * vd.template tail<HowMany>();
    }

    /** @brief Packet (W-lane) value evaluation on a non-uniform grid — host
     *  SIMD analogue of ``eval``.  All W lanes share this evaluator's stencil
     *  ``lo``; the per-lane query is the RAW global coordinate ``tPkt`` (there
     *  is no local-coordinate shift, matching the scalar non-uniform sweep and
     *  GODOT's global-coordinate arithmetic).
     *
     *  The product Lagrange weights use the (lane-independent) node keys
     *  ``keys_[lo+m]`` and the shared denominator ``prod(keys[j]-keys[m])``;
     *  only ``t`` varies per lane.  Lane-for-lane this reproduces the scalar
     *  ``sweep_`` value pass.  Accumulates into ``out`` (caller zeroes it). */
    template<idx_t W, idx_t From, idx_t HowMany>
    void evalPacket(feta::vector::PacketItem<ComponentT, HowMany, W>& out,
        const CoeffsT& coeffs,
        const feta::simd::Packet<ComponentT, W>& tPkt) const
    {
        detail::lagrangePacketEval<CoeffsT, Deg, false, W, From, HowMany, KeysT>(
            out, coeffs, lo, tPkt, &keys_);
    }

    /** @brief Stencil start node index. */
    const idx_t lo;
    /** @brief Global query coordinate (the interpolation variable). */
    const ComponentT t;
    /** @brief Node-key reference (the actual, possibly non-uniform, grid). */
    const KeysT keys_;
};

/**
 * @brief Global Lagrange interpolator over a *non-uniform* node grid.
 *
 * Owns a flat vector array of node values (``VecDims`` channels per node) plus
 * a parallel scalar array of node keys (the actual, strictly increasing grid
 * positions). The bracketing interval is located by a real binary search over
 * the keys — *not* the uniform ``(t - x0) / h`` assumption — then the centred
 * ``NPts``-node stencil is evaluated by @ref NonUniformEvaluator. Public
 * surface mirrors @ref GlobalInterpolator so consumers swap the type in place
 * (the only difference: ``dEval`` / ``evaldEval`` take no ``1/h`` argument).
 *
 * @tparam CoeffsT  Node-value array reference (``ComponentT``, ``ItemT``,
 *                  ``VecDims``, ``size()``, ``get<dim>(idx)``).
 * @tparam KeysT    Scalar node-key array reference (``operator[](idx)``).
 * @tparam Deg      Polynomial degree (``NPts = Deg + 1`` stencil nodes).
 */
template<typename CoeffsT, typename KeysT, idx_t Deg>
class NonUniformInterpolator {
    using Self = NonUniformInterpolator;

    /** @brief Compile-time-unrolled per-component packet load + accumulate;
     *  helper for ``ievalPacket``'s mixed-stencil fallback (grid-agnostic —
     *  just transposes lane-major scalar results into ``out``). */
    template<idx_t W, idx_t HowMany, dims_t D>
    static inline void loadAndAccumulate_(
        feta::vector::PacketItem<typename CoeffsT::ComponentT, HowMany, W>& out,
        const typename CoeffsT::ComponentT (&compBuf)[HowMany][W])
    {
        using PacketT = feta::simd::Packet<typename CoeffsT::ComponentT, W>;
        out.template packet<D>()
            = out.template packet<D>() + PacketT::load(compBuf[D]);
        if constexpr (D >= 1)
            loadAndAccumulate_<W, HowMany, D - 1>(out, compBuf);
    }

public:
    /** @brief Scalar component type. */
    using ComponentT = typename CoeffsT::ComponentT;
    /** @brief The vector item type. */
    using ItemT = typename CoeffsT::ItemT;
    /** @brief Number of channels per node. */
    static constexpr idx_t VecDims = CoeffsT::VecDims;
    /** @brief Number of stencil nodes. */
    static constexpr idx_t NPts = Deg + 1;
    /** @brief Half-points. */
    static constexpr idx_t HalfPts = NPts / 2;
    /** @brief Evaluator type. */
    using EvaluatorT = NonUniformEvaluator<CoeffsT, KeysT, Deg>;
    /** @brief Templated item type. */
    template<idx_t HM>
    using TItemT = feta::vector::Item<ComponentT, HM>;
    /** @brief Vector / double-vector array handle types (for the
     *  array-accumulate overloads). */
    template<bool work, idx_t HowMany = VecDims>
    using VecArrT = typename feta::vector::Array<ComponentT,
        HowMany>::template Ref<work>::HandleT;
    template<bool work, idx_t HowMany = VecDims>
    using TwoVecArrT = typename feta::vector::Array<ComponentT,
        2 * HowMany>::template Ref<work>::HandleT;

    /** @brief Factory from node values + the parallel node-key array. */
    DEVICEHOST()
    static NonUniformInterpolator make(
        const CoeffsT& values, const KeysT& keys)
    {
        NonUniformInterpolator g;
        g.values_ = values;
        g.keys_   = keys;
        return g;
    }

    /** @brief Number of grid nodes. */
    DEVICEHOST() idx_t numNodes() const { return values_.size(); }

    /** @brief Locate the centred stencil for ``t`` by a real binary search
     *  over the node keys (largest ``i`` with ``keys[i] <= t``), then centre
     *  and clamp the ``NPts`` window. No uniform-grid assumption. */
    DEVICEHOST() EvaluatorT findStencil(const ComponentT& t) const
    {
        /* half-open [lo, hi) lower-bracket search; signed arithmetic because
         * idx_t is unsigned and the centred window start can go negative. */
        const long n = (long)values_.size();
        long lo      = 0;
        long hi      = n;
        while (lo < hi) {
            const long mid = lo + ((hi - lo) >> 1);
            if (keys_[(idx_t)mid] <= t)
                lo = mid + 1;
            else
                hi = mid;
        }
        const long fIdx     = lo - 1; /* last key <= t (may be -1) */
        long start          = fIdx - (long)HalfPts + 1;
        const long maxStart = n - (long)NPts;
        if (start < 0)
            start = 0;
        if (start > maxStart)
            start = maxStart;
        return EvaluatorT{ (idx_t)start, t, keys_ };
    }

    /** @brief Evaluate the interpolation. */
    template<idx_t From = 0, idx_t HowMany = VecDims>
    DEVICEHOST() TItemT<HowMany> eval(const ComponentT& t) const
    {
        TItemT<HowMany> out;
        ieval<From, HowMany>(out, t);
        return out;
    }
    template<idx_t From = 0, idx_t HowMany = VecDims>
    DEVICEHOST() void ieval(TItemT<HowMany>& out, const ComponentT& t) const
    {
        findStencil(t).template eval<From, HowMany>(out, values_);
    }

    /** @brief Evaluate the derivative w.r.t. the interpolation variable. */
    template<idx_t From = 0, idx_t HowMany = VecDims>
    DEVICEHOST() TItemT<HowMany> dEval(const ComponentT& t) const
    {
        TItemT<HowMany> out;
        idEval<From, HowMany>(out, t);
        return out;
    }
    template<idx_t From = 0, idx_t HowMany = VecDims>
    DEVICEHOST() void idEval(TItemT<HowMany>& out, const ComponentT& t) const
    {
        findStencil(t).template dEval<From, HowMany>(out, values_);
    }

    /** @brief Evaluate both value and derivative (head = value, tail = rate). */
    template<idx_t From = 0, idx_t HowMany = VecDims>
    DEVICEHOST() TItemT<2 * HowMany> evaldEval(const ComponentT& t) const
    {
        TItemT<2 * HowMany> out;
        iEvaldEval<From, HowMany>(out, t);
        return out;
    }
    template<idx_t From = 0, idx_t HowMany = VecDims>
    DEVICEHOST()
    void iEvaldEval(TItemT<2 * HowMany>& out, const ComponentT& t) const
    {
        findStencil(t).template evaldEval<From, HowMany>(out, values_);
    }

    /** @brief Array-accumulate value: out[index] += sign * value(t). */
    template<bool work, idx_t From = 0, idx_t HowMany = VecDims>
    DEVICEHOST()
    void ieval(const SampleIndex& index, VecArrT<work, HowMany>& out,
        const ComponentT& t, const Real& sign) const
    {
        findStencil(t).template eval<work, From, HowMany>(
            index, out, values_, sign);
    }
    /** @brief Array-accumulate derivative w.r.t. ``t`` (no ``1/h``). */
    template<bool work, idx_t From = 0, idx_t HowMany = VecDims>
    DEVICEHOST()
    void idEval(const SampleIndex& index, VecArrT<work, HowMany>& out,
        const ComponentT& t, const Real& sign) const
    {
        findStencil(t).template dEval<work, From, HowMany>(
            index, out, values_, sign);
    }
    /** @brief Array-accumulate value+derivative (head = value, tail = rate). */
    template<bool work, idx_t From = 0, idx_t HowMany = VecDims>
    DEVICEHOST()
    void iEvaldEval(const SampleIndex& index, TwoVecArrT<work, HowMany>& out,
        const ComponentT& t, const Real& sign) const
    {
        findStencil(t).template evaldEval<work, From, HowMany>(
            index, out, values_, sign);
    }

    /** @brief Packet (W-lane) value interpolation — host SIMD evaluation on a
     *  non-uniform grid.  Per-lane query times in ``tPkt`` may map to the same
     *  or to different ``NPts``-node stencils; mirrors the uniform
     *  ``GlobalInterpolator::ievalPacket``:
     *  - Per-lane ``findStencil`` (binary search over keys) + a shared-stencil
     *    predicate.
     *  - If all lanes share the stencil start ``lo``, run the packet sweep
     *    (``NonUniformEvaluator::evalPacket``) once on the RAW per-lane ``tPkt``
     *    (no ``u`` conversion — non-uniform works directly in the global
     *    coordinate).
     *  - Otherwise fall back to per-lane scalar ``eval``, transposing into
     *    component-major packets.
     *  Accumulates into ``out`` (the caller zeroes it).  Value-only. */
    template<idx_t W, idx_t From = 0, idx_t HowMany = VecDims>
    inline void ievalPacket(
        feta::vector::PacketItem<ComponentT, HowMany, W>& out,
        const feta::simd::Packet<ComponentT, W>& tPkt) const
    {
        using PacketT = feta::simd::Packet<ComponentT, W>;

        alignas(64) ComponentT tBuf[W];
        PacketT::store(tBuf, tPkt);

        /* Per-lane stencil lookup + shared-stencil test.  The evaluator
         * consumes the raw ``t`` directly, so only the stencil starts are
         * buffered (the raw t per lane is already in tBuf for the fallback). */
        idx_t loBuf[W];
        loBuf[0]       = findStencil(tBuf[0]).lo;
        bool uniformLo = true;
        for (idx_t k = 1; k < W; ++k) {
            loBuf[k] = findStencil(tBuf[k]).lo;
            uniformLo &= (loBuf[k] == loBuf[0]);
        }

        if (uniformLo) {
            /* shared stencil; the scalar ``t`` member is unused on this path
             * (evalPacket takes the per-lane tPkt). The 3-member aggregate
             * must still supply the key reference. */
            const EvaluatorT e{ loBuf[0], ComponentT(0), keys_ };
            e.template evalPacket<W, From, HowMany>(out, values_, tPkt);
            return;
        }

        /* Mixed-stencil fallback: per-lane scalar evaluator (raw t = tBuf[k]),
         * transpose lane-major scalar results into component-major packets. */
        alignas(64) ComponentT compBuf[HowMany][W] = {};
        for (idx_t k = 0; k < W; ++k) {
            const EvaluatorT eLane{ loBuf[k], tBuf[k], keys_ };
            TItemT<HowMany> scalarOut; /* scalar eval overwrites */
            eLane.template eval<From, HowMany>(scalarOut, values_);
            const ComponentT* sd = scalarOut.data();
            for (idx_t d = 0; d < HowMany; ++d)
                compBuf[d][k] = sd[d];
        }
        loadAndAccumulate_<W, HowMany, HowMany - 1>(out, compBuf);
    }

    /** @brief Expose the node values. */
    DEVICEHOST() const CoeffsT& values() const { return values_; }
    /** @brief Expose the node keys. */
    DEVICEHOST() const KeysT& keys() const { return keys_; }

    /** @brief Data made public for PODification. */
    CoeffsT values_ = CoeffsT{};
    KeysT keys_     = KeysT{};
};

} // namespace lagrange
} // namespace interpolate
} // namespace parm
