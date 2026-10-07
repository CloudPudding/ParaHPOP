#pragma once

#include "parm/interpolate/detail/Interpolators.h"

#include <feta/core/simd/Packet.h>
#include <feta/vector/PacketItem.h>

#ifdef PR6D_PERF_INSTRUMENT
#include <atomic>
#include <cstdio>
namespace parm {
namespace interpolate {
namespace chebyshev {
namespace pr6dperf {
inline std::atomic<unsigned long long>& uniformIdxHits()
{
    static std::atomic<unsigned long long> v{ 0 };
    return v;
}
inline std::atomic<unsigned long long>& mixedIdxFallbacks()
{
    static std::atomic<unsigned long long> v{ 0 };
    return v;
}
} // namespace pr6dperf
} // namespace chebyshev
} // namespace interpolate
} // namespace parm
#endif

namespace parm {
namespace interpolate {
namespace chebyshev {

/** @brief Chebyshev evaluator */
template<typename CoeffsT, idx_t From, idx_t HowMany>
struct Evaluator {
    using ComponentT = typename CoeffsT::ComponentT;
    using Self       = Evaluator<CoeffsT, From, HowMany>;
    /* Vector items */
    using VecT    = feta::vector::Item<ComponentT, HowMany>;
    using TwoVecT = feta::vector::Item<ComponentT, 2 * HowMany>;
    /* Vector array items */
    template<bool work>
    using VecArrT = typename feta::vector::Array<ComponentT,
        HowMany>::template Ref<work>::HandleT;
    template<bool work>
    using TwoVecArrT = typename feta::vector::Array<ComponentT,
        2 * HowMany>::template Ref<work>::HandleT;
    /** @brief Get the interpolation time */
    DEVICEHOST()
    static inline Real interpolationTime(const ComponentT& epoch,
        const ComponentT& intStarts, const ComponentT& intRadius)
    {
        return (epoch - intStarts) / intRadius - 1.0;
    }

    /** @brief Evaluate the Chebyshev series at the configured query point.
     *
     *  Standard textbook backward Clenshaw recurrence on degree-N
     *  coefficients: one `fma(s2, x, w)` per coefficient, then a final
     *  `fma(s, x, c[0] − w)` reconstruction. Roughly N FMAs per call.
     *
     *  **Precision contract**: the forward error of plain backward
     *  Clenshaw is O(N·ε) for stable Chebyshev arguments in [−1, 1].
     *  Empirically, the relative precision in double-precision is
     *  ~1e-13 — matching state-of-the-art production libraries
     *  (SPICE/NAIF CSPICE, GSL `gsl_cheb_eval`, Boost.Math
     *  `chebyshev_clenshaw_recurrence`). Tightening below this floor
     *  requires compensated Clenshaw (~4-5× FLOPs) or double-double
     *  arithmetic (~10-15× FLOPs); deferred unless a downstream consumer
     *  demonstrates a need for ULP-level precision from this path. */
    DEVICEHOST()
    void eval(VecT& out, const CoeffsT& coeffs) const
    {
        VecT w, x;

        for (idx_t i = 0; i < coeffs.highestDegree(); i++) {
            {
                idx_t ii = idx + coeffs.revOffset(i);
                w        = coeffs.template segment<From, HowMany>()[ii] - w;
            }
            w = exchange(x, feta::math::fma(s2, x, w));
        }
        out += feta::math::fma(
            s, x, coeffs.template segment<From, HowMany>()[idx] - w);
    }

    /** @brief Array-based evaluation */
    template<bool work>
    DEVICEHOST()
    void eval(const SampleIndex& index, VecArrT<work>& out,
        const CoeffsT& coeffs, const Real& sign) const
    {
        VecT w, x;

        for (idx_t i = 0; i < coeffs.highestDegree(); i++) {
            {
                idx_t ii = idx + coeffs.revOffset(i);
                w        = coeffs.template segment<From, HowMany>()[ii] - w;
            }
            w = exchange(x, feta::math::fma(s2, x, w));
        }
        out[index] += sign
            * feta::math::fma(
                s, x, coeffs.template segment<From, HowMany>()[idx] - w);
    }

    /** @brief Packet (W-lane) Clenshaw recurrence — uniform-interval
     *         variant.
     *
     *  When all W lanes' query epochs map to the same chebyshev
     *  interval (the >>99% case for sub-second integration steps
     *  inside multi-day chebyshev intervals), the interval index
     *  ``idx`` is shared across the packet, and the per-degree
     *  coefficient reads (``coeffs.data_.template get<dim>(...)``)
     *  return scalar values broadcast into all W lanes via
     *  ``Packet::broadcast``.  Only ``s`` and ``s2`` (the per-lane
     *  Chebyshev interpolation time and its 2× counterpart) carry
     *  per-lane state.
     *
     *  Mirror of ``eval_<dim>`` (the compile-time-unrolled scalar
     *  variant); produces a ``PacketItem<ComponentT, HowMany, W>``
     *  output via per-component recurrence.  The recurrence and
     *  final-sum match the scalar evaluator step-for-step — only
     *  the operand types lift.
     *
     *  Caller (``GlobalInterpolator::ievalPacket``) is responsible
     *  for the uniform-interval predicate and falling back to
     *  per-lane scalar ``ieval`` when lanes diverge.
     *
     *  @tparam W      Packet width (lane count).
     *  @tparam dim    Recursive component index (decreasing).
     *  @param  out    Packet output, accumulated by component.
     *  @param  coeffs Coefficients (scalar source).
     *  @param  sPkt   Packet of per-lane chebyshev interpolation times.
     *  @param  s2Pkt  Packet of 2 * sPkt.
     */
    template<idx_t W, dims_t dim>
    inline void evalPacket_(
        feta::vector::PacketItem<ComponentT, HowMany, W>& out,
        const CoeffsT& coeffs, const feta::simd::Packet<ComponentT, W>& sPkt,
        const feta::simd::Packet<ComponentT, W>& s2Pkt) const
    {
        using PacketT                 = feta::simd::Packet<ComponentT, W>;
        static constexpr idx_t NetDim = dim - From;

        if constexpr (dim > From)
            Self::template evalPacket_<W, dim - 1>(out, coeffs, sPkt, s2Pkt);

        /* This component's recurrence — same shape as ``eval_<dim>``
         * but operands are packets.  ``coeffs.template revGet<dim>``
         * + ``coeffs.data_.template get<dim>`` return scalars; we
         * broadcast at the boundary. */
        PacketT w = PacketT::zero();
        PacketT x = PacketT::zero();
        for (idx_t i = 0; i < coeffs.highestDegree(); i++) {
            const ComponentT cScalar = coeffs.template revGet<dim>(idx, i);
            w                        = PacketT::broadcast(cScalar) - w;
            /* w = exchange(x, fmadd(s2, x, w))
             *   new_x = s2 * x + w  (explicit packet ops; ``fma``
             *   helper above takes scalar ``a`` and won't match here)
             *   tmp_w = x; x = new_x; w = tmp_w. */
            const PacketT new_x = s2Pkt * x + w;
            const PacketT tmp   = x;
            x                   = new_x;
            w                   = tmp;
        }
        /* Final recursion step: out[NetDim] += s * x + (c[idx] - w). */
        const ComponentT cScalarFinal = coeffs.data_.template get<dim>(idx);
        const PacketT cFinal          = PacketT::broadcast(cScalarFinal);
        const PacketT contribute      = sPkt * x + (cFinal - w);
        out.template packet<NetDim>()
            = out.template packet<NetDim>() + contribute;
    }

    /** @brief Packet entry — initialises ``out`` to zero and runs
     *  the per-component recurrence top-down. */
    template<idx_t W>
    inline void evalPacket(
        feta::vector::PacketItem<ComponentT, HowMany, W>& out,
        const CoeffsT& coeffs, const feta::simd::Packet<ComponentT, W>& sPkt,
        const feta::simd::Packet<ComponentT, W>& s2Pkt) const
    {
        Self::template evalPacket_<W, From + HowMany - 1>(
            out, coeffs, sPkt, s2Pkt);
    }

    /** @brief Recursion for single component compile-time unrolled interpolant
     * evaluations */
    template<dims_t dim>
    DEVICEHOST()
    void eval_(ComponentT* const out, const CoeffsT& coeffs) const
    {
        static constexpr idx_t NetDim = dim - From;

        if constexpr (dim > From)
            Self::eval_<dim - 1>(out, coeffs);
        /* This step */
        {
            ComponentT w = 0, x = 0;
            /* main backward recursion */
            for (idx_t i = 0; i < coeffs.highestDegree(); i++) {
                w = coeffs.template revGet<dim>(idx, i) - w;
                w = exchange(x, feta::math::fma(s2, x, w));
            }
            /* final recursion step */
            out[NetDim] += feta::math::fma(
                s, x, coeffs.data_.template get<dim>(idx) - w);
        }
    }

    /** @brief Evaluate the derivative of the Chebyshev series.
     *
     *  Backward Clenshaw with two interleaved registers — `(w, x)` for
     *  the value, `(dw, v)` for the derivative — sharing the per-degree
     *  coefficient walk. Roughly 2N FMAs per call.
     *
     *  **Precision contract**: same ~1e-13 relative floor as `eval()`
     *  (see its docstring). The derivative path adds one extra FMA per
     *  coefficient but does not increase the error order — the
     *  derivative Clenshaw is itself a Clenshaw recurrence on a related
     *  coefficient set, with the same O(N·ε) forward error bound. */
    DEVICEHOST()
    void dEval(VecT& out, const CoeffsT& coeffs, const ComponentT dScale) const
    {
        VecT w, dw, x, v;

        for (idx_t i = 0; i < coeffs.highestDegree(); i++) {
            {
                idx_t ii = idx + coeffs.revOffset(i);
                w        = coeffs.template segment<From, HowMany>()[ii] - w;
            }
            w  = exchange(x, feta::math::fma(s2, x, w));
            dw = exchange(
                v, feta::math::fma(s2, v, feta::math::fma(2.0, w, -dw)));
        }
        out += dScale * feta::math::fma(s, v, x - dw);
    }

    /** @brief Array-based derivative evaluation */
    template<bool work>
    DEVICEHOST()
    void dEval(const SampleIndex& index, VecArrT<work>& out,
        const CoeffsT& coeffs, const ComponentT dScale, const Real& sign) const
    {
        VecT w, dw, x, v;

        for (idx_t i = 0; i < coeffs.highestDegree(); i++) {
            {
                idx_t ii = idx + coeffs.revOffset(i);
                w        = coeffs.template segment<From, HowMany>()[ii] - w;
            }
            w  = exchange(x, feta::math::fma(s2, x, w));
            dw = exchange(
                v, feta::math::fma(s2, v, feta::math::fma(2.0, w, -dw)));
        }
        out[index] += sign * dScale * feta::math::fma(s, v, x - dw);
    }

    /** @brief Recursion for the single component compile-time unrolled
     * interpolant derivative evaluations */
    template<dims_t dim>
    DEVICEHOST()
    void dEval_(ComponentT* const out, const CoeffsT& coeffs,
        const ComponentT& dScale) const
    {
        static constexpr idx_t NetDim = dim - From;

        if constexpr (dim > From)
            Self::dEval_<dim - 1>(out, coeffs, dScale);
        /* This step */
        {
            ComponentT w = 0, dw = 0, x = 0, v = 0;
            /* main backward recursion */
            for (idx_t i = 0; i < coeffs.highestDegree(); i++) {
                w  = coeffs.template revGet<dim>(idx, i) - w;
                w  = exchange(x, feta::math::fma(s2, x, w));
                dw = exchange(
                    v, feta::math::fma(s2, v, feta::math::fma(2.0, w, -dw)));
            }
            /* final recursion step and return */
            out[NetDim] += dScale * feta::math::fma(s, v, x - dw);
        }
    }

    /** @brief Evaluate both the interpolation and its derivative */
    DEVICEHOST()
    void evaldEval(
        TwoVecT& out, const CoeffsT& coeffs, const ComponentT dScale) const
    {
        VecT w, dw, x, v;
        for (idx_t i = 0; i < coeffs.highestDegree(); i++) {
            {
                idx_t ii = idx + coeffs.revOffset(i);
                w        = coeffs.template segment<From, HowMany>()[ii] - w;
            }
            w  = exchange(x, feta::math::fma(s2, x, w));
            dw = exchange(
                v, feta::math::fma(s2, v, feta::math::fma(2.0, w, -dw)));
        }
        out.template head<HowMany>() += feta::math::fma(
            s, x, coeffs.template segment<From, HowMany>()[idx] - w);
        out.template tail<HowMany>() += dScale * feta::math::fma(s, v, x - dw);
    }

    /** @brief Array-based evaluation */
    template<bool work>
    DEVICEHOST()
    void evaldEval(const SampleIndex& index, TwoVecArrT<work>& out,
        const CoeffsT& coeffs, const ComponentT dScale, const Real& sign) const
    {
        VecT w, dw, x, v;
        for (idx_t i = 0; i < coeffs.highestDegree(); i++) {
            {
                idx_t ii = idx + coeffs.revOffset(i);
                w        = coeffs.template segment<From, HowMany>()[ii] - w;
            }
            w  = exchange(x, feta::math::fma(s2, x, w));
            dw = exchange(
                v, feta::math::fma(s2, v, feta::math::fma(2.0, w, -dw)));
        }
        out[index].template head<HowMany>() += sign
            * feta::math::fma(
                s, x, coeffs.template segment<From, HowMany>()[idx] - w);
        out[index].template tail<HowMany>()
            += sign * dScale * feta::math::fma(s, v, x - dw);
    }

    /** @brief Recursion for the low-memory compile-time unrolled interpolant
     * and its derivative evaluations */
    template<dims_t dim>
    DEVICEHOST()
    void evaldEval_(ComponentT* const out, const CoeffsT& coeffs,
        const ComponentT& dScale) const
    {
        static constexpr idx_t VecDims = HowMany;
        static constexpr idx_t NetDim  = dim - From;

        if constexpr (dim > From) {
            Self::evaldEval_<dim - 1>(out, coeffs, dScale);
        }
        /* This step */
        {
            ComponentT w = 0, dw = 0, x = 0, v = 0;
            /* main backward recursion */
            for (idx_t i = 0; i < coeffs.highestDegree(); i++) {
                /* value */
                w = coeffs.template revGet<dim>(idx, i) - w;
                w = exchange(x, feta::math::fma(s2, x, w));
                /* derivative */
                dw = exchange(
                    v, feta::math::fma(s2, v, feta::math::fma(2.0, w, -dw)));
            }

            /* final recursion step: evaluate velocity before updating position
             * part
             */
            out[VecDims + NetDim] += dScale * feta::math::fma(s, v, x - dw);
            /* evaluate position */
            out[NetDim] += feta::math::fma(
                s, x, coeffs.data_.template get<dim>(idx) - w);
        }
    }

    /** @brief Exchange helper */
    template<typename T, typename U>
    DEVICEHOST()
    static T exchange(T& a, const U& b)
    {
        T old = a;
        a     = b;
        return old;
    }

    /** @brief All data made public for PODification */

    /** @brief The index to be targeted */
    const idx_t idx;
    /** @brief component buffer values */
    const ComponentT s;
    const ComponentT s2;
};

/**
 * @brief Global Chebyshev polynomial interpolator over a piecewise interval
 * partition.
 *
 * Locates the sub-interval containing a query time ``t``, maps it to the
 * Chebyshev reference interval ``[-1, 1]``, and evaluates the polynomial
 * via a backward Clenshaw recursion.  Value, derivative, and combined
 * evaluations are all supported.  Callable from both host and device code.
 *
 * @tparam CoeffsT     Coefficient array type exposing ``ComponentT``,
 *                     ``ItemT``, ``VecDims``, and ``highestDegree()``.
 * @tparam IntervalsT  Interval boundary array type.
 */
template<typename CoeffsT, typename IntervalsT>
class GlobalInterpolator
    : public detail::GlobalInterpolator<CoeffsT, IntervalsT> {

    /** @brief The parent type */
    using ParentT = detail::GlobalInterpolator<CoeffsT, IntervalsT>;

public:
    /** @brief The radius type */
    using RadiusT = typename CoeffsT::ComponentT;
    /** @brief the return type */
    using ItemT = typename CoeffsT::ItemT;
    /** @brief the data type */
    using ComponentT = typename CoeffsT::ComponentT;
    /** @brief The double return type */
    using TwoItemT = VecNT<ComponentT, 2 * ItemT::VecDims>;
    /** @brief Short hand for the vector coefficient dimensions */
    static constexpr idx_t VecDims = CoeffsT::VecDims;
    /** @brief Short-hand for templated item t */
    template<idx_t VDims>
    using TItemT = feta::vector::Item<ComponentT, VDims>;

    /* Vector array items */
    template<bool work, idx_t HowMany = VecDims>
    using VecArrT = typename feta::vector::Array<ComponentT,
        HowMany>::template Ref<work>::HandleT;
    template<bool work, idx_t HowMany = VecDims>
    using TwoVecArrT = typename feta::vector::Array<ComponentT,
        2 * HowMany>::template Ref<work>::HandleT;

    /** @brief Factory method to construct from Parent Object */
    DEVICEHOST() static GlobalInterpolator make(const ParentT& other)
    {
        return GlobalInterpolator{ other };
    }

    /** @brief Factory method to construct from coeffs, intervals, and radius */
    DEVICEHOST()
    static GlobalInterpolator make(const CoeffsT& coeffs,
        const IntervalsT& intervals, const RadiusT& radius)
    {
        return GlobalInterpolator{ ParentT::make(coeffs, intervals, radius) };
    }

private:
    /** @brief Compile-time-unrolled per-component packet load +
     *  accumulate.  Helper for ``ievalPacket``'s mixed-interval
     *  fallback: walks ``D`` from ``HowMany - 1`` down to ``0``,
     *  loading each row of ``compBuf[D][W]`` as a packet and adding
     *  it to ``out.packet<D>()``. */
    template<idx_t W, idx_t HowMany, dims_t D>
    static inline void loadAndAccumulate_(
        feta::vector::PacketItem<ComponentT, HowMany, W>& out,
        const ComponentT (&compBuf)[HowMany][W])
    {
        using PacketT = feta::simd::Packet<ComponentT, W>;
        out.template packet<D>()
            = out.template packet<D>() + PacketT::load(compBuf[D]);
        if constexpr (D >= 1)
            loadAndAccumulate_<W, HowMany, D - 1>(out, compBuf);
    }

public:
    /** @brief Find the interval and update the interpolation time */
    template<typename EvaluatorT>
    DEVICEHOST()
    EvaluatorT findInterval(const ComponentT& t) const
    {
        /* First we find the interval */
        idx_t idx = ParentT::findInterval(t);
        /* Then we compute the chebyshev time */
        Real s = EvaluatorT::interpolationTime(
            t, this->intervals()[idx], this->radius());
        return EvaluatorT{ idx, s, 2 * s };
    }

    /** @brief Evaluate the interpolation */
    template<idx_t From = 0, idx_t HowMany = VecDims>
    DEVICEHOST()
    TItemT<HowMany> eval(const ComponentT& t) const
    {
        TItemT<HowMany> out;
        ieval<From, HowMany>(out, t);
        return out;
    }
    template<idx_t From = 0, idx_t HowMany = VecDims>
    DEVICEHOST()
    void ieval(TItemT<HowMany>& out, const ComponentT& t) const
    {
        using EvaluatorT = Evaluator<CoeffsT, From, HowMany>;
        EvaluatorT e     = findInterval<EvaluatorT>(t);
        e.eval(out, this->coeffs_);
    }
    template<bool work, idx_t From = 0, idx_t HowMany = VecDims>
    DEVICEHOST()
    void ieval(const SampleIndex& index, VecArrT<work, HowMany>& out,
        const ComponentT& t, const Real& sign) const
    {
        using EvaluatorT = Evaluator<CoeffsT, From, HowMany>;
        EvaluatorT e     = findInterval<EvaluatorT>(t);
        e.template eval<work>(index, out, this->coeffs_, sign);
    }

    /** @brief Packet (W-lane) interpolation evaluation.
     *
     *  Per-lane query times in ``tPkt`` may map to the same or to
     *  different chebyshev intervals.  This method:
     *  - Computes per-lane interval indices via the existing scalar
     *    ``findInterval``.
     *  - If all lanes share the same interval (>>99% case for
     *    sub-second integration steps inside multi-day chebyshev
     *    intervals), packs ``s`` and ``s2 = 2 * s`` into packets and
     *    runs the packet Clenshaw recurrence (``Evaluator::evalPacket``)
     *    once.
     *  - Otherwise (mixed intervals at chebyshev-interval boundaries)
     *    falls back to per-lane scalar ``ieval``, packing the W
     *    scalar Items into a packet output via per-component packet
     *    loads.  Correctness-equivalent to the scalar path; loses
     *    SIMD on rare boundary packets.
     *
     *  @tparam W        Packet width (lane count).
     *  @tparam From     Component start (default 0).
     *  @tparam HowMany  Component count (default VecDims).
     *  @param  out      Packet output, accumulated.
     *  @param  tPkt     Per-lane chebyshev query times.
     */
    template<idx_t W, idx_t From = 0, idx_t HowMany = VecDims>
    inline void ievalPacket(
        feta::vector::PacketItem<ComponentT, HowMany, W>& out,
        const feta::simd::Packet<ComponentT, W>& tPkt) const
    {
        using PacketT    = feta::simd::Packet<ComponentT, W>;
        using EvaluatorT = Evaluator<CoeffsT, From, HowMany>;

        /* Extract per-lane t scalars. */
        alignas(64) ComponentT tBuf[W];
        PacketT::store(tBuf, tPkt);

        /* Per-lane interval lookup + uniformity test. */
        idx_t idxBuf[W];
        idxBuf[0]       = ParentT::findInterval(tBuf[0]);
        bool uniformIdx = true;
        for (idx_t k = 1; k < W; ++k) {
            idxBuf[k] = ParentT::findInterval(tBuf[k]);
            uniformIdx &= (idxBuf[k] == idxBuf[0]);
        }

#ifdef PR6D_PERF_INSTRUMENT
        if (uniformIdx)
            pr6dperf::uniformIdxHits().fetch_add(1, std::memory_order_relaxed);
        else
            pr6dperf::mixedIdxFallbacks().fetch_add(
                1, std::memory_order_relaxed);
#endif

        if (uniformIdx) {
            /* Pack per-lane chebyshev s; s2 = 2 * s. */
            alignas(64) ComponentT sBuf[W];
            for (idx_t k = 0; k < W; ++k)
                sBuf[k] = EvaluatorT::interpolationTime(
                    tBuf[k], this->intervals()[idxBuf[0]], this->radius());
            const PacketT sPkt  = PacketT::load(sBuf);
            const PacketT s2Pkt = sPkt + sPkt;

            /* Construct evaluator with shared idx; sPkt/s2Pkt are
             * passed explicitly to evalPacket since the struct's
             * scalar ``s``/``s2`` members are unused on this path. */
            EvaluatorT e{ idxBuf[0], ComponentT{ 0 }, ComponentT{ 0 } };
            e.template evalPacket<W>(out, this->coeffs_, sPkt, s2Pkt);
            return;
        }

        /* Mixed-interval fallback: per-lane scalar evaluator,
         * transpose lane-major scalar results into component-major
         * packets.  Reuses the per-lane interval indices we already
         * computed in ``idxBuf`` (avoids the ~40% ``findInterval``
         * cost vs naive ``this->ieval`` per lane).  PR-6δ phase
         * B-perf optimisation. */
        alignas(64) ComponentT compBuf[HowMany][W] = {};

        for (idx_t k = 0; k < W; ++k) {
            const ComponentT s_k = EvaluatorT::interpolationTime(
                tBuf[k], this->intervals()[idxBuf[k]], this->radius());
            EvaluatorT e_lane{ idxBuf[k], s_k, ComponentT{ 2 } * s_k };
            TItemT<HowMany> scalarOut;
            scalarOut.setZero();
            e_lane.eval(scalarOut, this->coeffs_);
            const ComponentT* sd = scalarOut.data();
            for (idx_t d = 0; d < HowMany; ++d)
                compBuf[d][k] = sd[d];
        }
        /* Compile-time-unrolled per-component packet add (mirrors
         * scalar ``ieval`` semantics — accumulates into out). */
        loadAndAccumulate_<W, HowMany, HowMany - 1>(out, compBuf);
    }

    /** @brief Evaluate the derivative of the interpolation */
    template<idx_t From = 0, idx_t HowMany = VecDims>
    DEVICEHOST()
    TItemT<HowMany> dEval(const ComponentT& t) const
    {
        TItemT<HowMany> out;
        idEval<From, HowMany>(out, t);
        return out;
    }
    template<idx_t From = 0, idx_t HowMany = VecDims>
    DEVICEHOST()
    void idEval(TItemT<HowMany>& out, const ComponentT& t) const
    {
        using EvaluatorT = Evaluator<CoeffsT, From, HowMany>;
        EvaluatorT e     = findInterval<EvaluatorT>(t);
        e.dEval(out, this->coeffs_, 1.0 / this->radius());
    }
    template<bool work, idx_t From = 0, idx_t HowMany = VecDims>
    DEVICEHOST()
    void idEval(const SampleIndex& index, VecArrT<work, HowMany>& out,
        const ComponentT& t, const Real& sign) const
    {
        using EvaluatorT = Evaluator<CoeffsT, From, HowMany>;
        EvaluatorT e     = findInterval<EvaluatorT>(t);
        e.template dEval<work>(
            index, out, this->coeffs_, 1.0 / this->radius(), sign);
    }

    /** @brief Evaluate both the interpolation and its derivative */
    template<idx_t From = 0, idx_t HowMany = VecDims>
    DEVICEHOST()
    TItemT<2 * HowMany> evaldEval(const ComponentT& t) const
    {
        TItemT<2 * HowMany> out;
        iEvaldEval<From, HowMany>(out, t);
        return out;
    }
    template<idx_t From = 0, idx_t HowMany = VecDims>
    DEVICEHOST()
    void iEvaldEval(TItemT<2 * HowMany>& out, const ComponentT& t) const
    {
        using EvaluatorT = Evaluator<CoeffsT, From, HowMany>;
        EvaluatorT e     = findInterval<EvaluatorT>(t);
        e.evaldEval(out, this->coeffs_, 1.0 / this->radius());
    }
    template<bool work, idx_t From = 0, idx_t HowMany = VecDims>
    DEVICEHOST()
    void iEvaldEval(const SampleIndex& index, TwoVecArrT<work, HowMany>& out,
        const ComponentT& t, const Real& sign) const
    {
        using EvaluatorT = Evaluator<CoeffsT, From, HowMany>;
        EvaluatorT e     = findInterval<EvaluatorT>(t);
        e.template evaldEval<work>(
            index, out, this->coeffs_, 1.0 / this->radius(), sign);
    }
};

} // namespace chebyshev
} // namespace interpolate
} // namespace parm