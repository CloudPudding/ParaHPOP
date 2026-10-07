#pragma once

/**
 * @file Reductions.h
 * @brief Packet in-place arithmetic, norms, and lane (un)packing —
 *        `packetIAdd`/`packetISub` (+ their recursive drivers), the
 *        `packetSquaredNorm`/`packetNorm`/.../`packetDot` reductions,
 *        `maskedPacketAssign`, `unpackLanes`/`packLanes`, and the
 *        `nPackets`/`tailWidth`/`nTiles` sizing helpers.
 *
 * Part of the `TiledEval.h` facade (block 2).
 */

#include "feta/core/simd/L2Cache.h"
#include "feta/typedefs.h"
#include "feta/vector/Item.h"
#include "feta/vector/expr/OperationsDetail.h"

#include <algorithm>
#include <omp.h>
#include <tuple>
#include "feta/vector/expr/cpu/PacketOps.h"
#include "feta/vector/expr/cpu/tiled/Capture.h"

namespace feta {
namespace cpu {

/**
 * @brief Per-component packet in-place add: ``target.packet<d>() +=
 *        packetGet<d>(expr, pi)`` for each ``d``.
 *
 *  Compile-time-recursive over components; complements ``packetAssign``
 *  which does ``=``.  Lets callers write
 *  ``packetIAdd(acc, factor * delta, pi)`` instead of W-component
 *  manual fan-out.
 *
 *  Target must expose ``packet<d>()`` (i.e. be a ``PacketItem``-like
 *  type); expr must be a packet-evaluable expression with matching
 *  ``VecDims``.
 */
namespace detail {

template<dims_t Dim, typename Target, typename Expr, idx_t W>
struct RecursivePacketIAdd {
    inline static void apply(Target& dst, const Expr& src,
        const PacketIndex<W>& pi)
    {
        dst.template packet<Dim>() = dst.template packet<Dim>()
            + packet::packetGet<Dim, W>(src, pi);
        if constexpr (Dim >= 1)
            RecursivePacketIAdd<Dim - 1, Target, Expr, W>::apply(dst, src, pi);
    }
};

} // namespace detail

template<typename Target, typename Expr, idx_t W>
inline void packetIAdd(Target& dst, const Expr& src,
    const PacketIndex<W>& pi)
{
    static_assert(Target::VecDims == Expr::VecDims,
        "packetIAdd: target and expression must have the same VecDims");
    detail::RecursivePacketIAdd<Target::VecDims - 1, Target, Expr, W>::apply(
        dst, src, pi);
}

/**
 * @brief Scaled packet in-place add: ``target.packet<d>() += factor *
 *        packetGet<d>(expr, pi)`` for each ``d``.
 *
 *  Convenient because feta's scalar ``operator*`` uses
 *  ``std::is_arithmetic`` to dispatch and does NOT match
 *  ``Packet<T, W> × VecExpr``: ``Packet`` isn't an arithmetic type,
 *  so ``factor * delta`` would not compile when ``factor`` is a
 *  packet.  This overload bypasses the issue by passing the packet
 *  factor explicitly and applying it per-component during the
 *  recursion.
 *
 *  Used by cudajectory's packet RHS (PR-6δ phase B-perf):
 *  ``packetIAdd<W>(acc, factor, delta, pi)`` for ``acc += factor *
 *  (pos - bodyPos)``.
 */
namespace detail {

template<dims_t Dim, typename Target, typename Expr, idx_t W>
struct RecursivePacketScaledIAdd {
    using DataT = typename Target::ComponentT;
    inline static void apply(Target& dst,
        const simd::Packet<DataT, W>& factor,
        const Expr& src, const PacketIndex<W>& pi)
    {
        dst.template packet<Dim>() = dst.template packet<Dim>()
            + factor * packet::packetGet<Dim, W>(src, pi);
        if constexpr (Dim >= 1)
            RecursivePacketScaledIAdd<Dim - 1, Target, Expr, W>::apply(
                dst, factor, src, pi);
    }
};

} // namespace detail

template<typename Target, typename Expr, idx_t W>
inline void packetIAdd(Target& dst,
    const simd::Packet<typename Target::ComponentT, W>& factor,
    const Expr& src, const PacketIndex<W>& pi)
{
    static_assert(Target::VecDims == Expr::VecDims,
        "packetIAdd (scaled): target and expression must have matching VecDims");
    detail::RecursivePacketScaledIAdd<Target::VecDims - 1, Target, Expr,
        W>::apply(dst, factor, src, pi);
}

/**
 * @brief Per-component packet in-place subtract: ``target.packet<d>()
 *        -= packetGet<d>(expr, pi)`` for each ``d``.
 */
namespace detail {

template<dims_t Dim, typename Target, typename Expr, idx_t W>
struct RecursivePacketISub {
    inline static void apply(Target& dst, const Expr& src,
        const PacketIndex<W>& pi)
    {
        dst.template packet<Dim>() = dst.template packet<Dim>()
            - packet::packetGet<Dim, W>(src, pi);
        if constexpr (Dim >= 1)
            RecursivePacketISub<Dim - 1, Target, Expr, W>::apply(dst, src, pi);
    }
};

} // namespace detail

template<typename Target, typename Expr, idx_t W>
inline void packetISub(Target& dst, const Expr& src,
    const PacketIndex<W>& pi)
{
    static_assert(Target::VecDims == Expr::VecDims,
        "packetISub: target and expression must have the same VecDims");
    detail::RecursivePacketISub<Target::VecDims - 1, Target, Expr, W>::apply(
        dst, src, pi);
}

// ─── Packet reductions (cross-component → single Packet<DataT, W>) ─────────
//
// Mirror feta's scalar ``Expression::squaredNorm`` / ``rCubedNorm`` /
// ``dot`` / ``maxNorm`` etc. but produce per-lane ``Packet<DataT, W>``
// results instead of scalar reductions.  Walk the expression via the
// existing ``packetGet<dim, W>`` free-function dispatch so any
// expression that's packet-composable (Sum, Sub, CWiseScale,
// PacketItem leaf, ComponentView, Item leaf, etc.) is handled
// uniformly without per-node specialisation.
//
// PR-6δ phase B-perf: enables clean evalPacket-style code:
//   auto delta  = pos - bodyPos;                       // lazy expr
//   PacketT r3i = packetRCubedNorm<W>(delta, pi);     // 1 / |Δ|³
//   packetIAdd<W>(acc, mulScalar(r3i, -gm) * delta, pi);

namespace detail {

/** @brief Compile-time recursive packet squared-norm. */
template<dims_t Dim, typename Expr, idx_t W>
struct PacketSquaredNorm {
    using DataT = typename Expr::ComponentT;
    inline static simd::Packet<DataT, W> eval(
        const Expr& expr, const PacketIndex<W>& pi)
    {
        const auto p = packet::packetGet<Dim, W>(expr, pi);
        if constexpr (Dim == 0) {
            return p * p;
        } else {
            return p * p
                + PacketSquaredNorm<Dim - 1, Expr, W>::eval(expr, pi);
        }
    }
};

/** @brief Compile-time recursive packet sum-of-components. */
template<dims_t Dim, typename Expr, idx_t W>
struct PacketSum {
    using DataT = typename Expr::ComponentT;
    inline static simd::Packet<DataT, W> eval(
        const Expr& expr, const PacketIndex<W>& pi)
    {
        const auto p = packet::packetGet<Dim, W>(expr, pi);
        if constexpr (Dim == 0) {
            return p;
        } else {
            return p + PacketSum<Dim - 1, Expr, W>::eval(expr, pi);
        }
    }
};

/** @brief Compile-time recursive packet max-norm (max |component|). */
template<dims_t Dim, typename Expr, idx_t W>
struct PacketMaxNorm {
    using DataT = typename Expr::ComponentT;
    inline static simd::Packet<DataT, W> eval(
        const Expr& expr, const PacketIndex<W>& pi)
    {
        const auto p = abs(packet::packetGet<Dim, W>(expr, pi));
        if constexpr (Dim == 0) {
            return p;
        } else {
            return max(p, PacketMaxNorm<Dim - 1, Expr, W>::eval(expr, pi));
        }
    }
};

/** @brief Compile-time recursive packet dot product. */
template<dims_t Dim, typename L, typename R, idx_t W>
struct PacketDot {
    using DataT = typename L::ComponentT;
    inline static simd::Packet<DataT, W> eval(
        const L& l, const R& r, const PacketIndex<W>& pi)
    {
        const auto pl = packet::packetGet<Dim, W>(l, pi);
        const auto pr = packet::packetGet<Dim, W>(r, pi);
        if constexpr (Dim == 0) {
            return pl * pr;
        } else {
            return pl * pr + PacketDot<Dim - 1, L, R, W>::eval(l, r, pi);
        }
    }
};

} // namespace detail

/** @brief Per-lane sum of squared components — packet ``squaredNorm``. */
template<typename Expr, idx_t W>
inline simd::Packet<typename Expr::ComponentT, W> packetSquaredNorm(
    const Expr& expr, const PacketIndex<W>& pi)
{
    static_assert(Expr::VecDims >= 1,
        "packetSquaredNorm: expression must have at least one component");
    return detail::PacketSquaredNorm<Expr::VecDims - 1, Expr, W>::eval(
        expr, pi);
}

/** @brief Per-lane Euclidean norm — ``sqrt(squaredNorm)``. */
template<typename Expr, idx_t W>
inline simd::Packet<typename Expr::ComponentT, W> packetNorm(
    const Expr& expr, const PacketIndex<W>& pi)
{
    return sqrt(packetSquaredNorm(expr, pi));
}

/** @brief Per-lane reciprocal of squared-norm — ``1 / |v|²``. */
template<typename Expr, idx_t W>
inline simd::Packet<typename Expr::ComponentT, W> packetRSquaredNorm(
    const Expr& expr, const PacketIndex<W>& pi)
{
    using DataT          = typename Expr::ComponentT;
    using PacketT        = simd::Packet<DataT, W>;
    const PacketT r2     = packetSquaredNorm(expr, pi);
    return PacketT::broadcast(DataT{ 1 }) / r2;
}

/** @brief Per-lane reciprocal norm — ``1 / |v|``. */
template<typename Expr, idx_t W>
inline simd::Packet<typename Expr::ComponentT, W> packetRNorm(
    const Expr& expr, const PacketIndex<W>& pi)
{
    using DataT          = typename Expr::ComponentT;
    using PacketT        = simd::Packet<DataT, W>;
    return PacketT::broadcast(DataT{ 1 }) / packetNorm(expr, pi);
}

/** @brief Per-lane reciprocal cubed norm — ``1 / |v|³``.  Hot in
 *  gravity / SRP packet evaluators. */
template<typename Expr, idx_t W>
inline simd::Packet<typename Expr::ComponentT, W> packetRCubedNorm(
    const Expr& expr, const PacketIndex<W>& pi)
{
    using DataT          = typename Expr::ComponentT;
    using PacketT        = simd::Packet<DataT, W>;
    const PacketT r2     = packetSquaredNorm(expr, pi);
    return PacketT::broadcast(DataT{ 1 }) / (r2 * sqrt(r2));
}

/** @brief Per-lane sum of components. */
template<typename Expr, idx_t W>
inline simd::Packet<typename Expr::ComponentT, W> packetSum(
    const Expr& expr, const PacketIndex<W>& pi)
{
    static_assert(Expr::VecDims >= 1,
        "packetSum: expression must have at least one component");
    return detail::PacketSum<Expr::VecDims - 1, Expr, W>::eval(expr, pi);
}

/** @brief Per-lane max-norm — ``max_d |v_d|``. */
template<typename Expr, idx_t W>
inline simd::Packet<typename Expr::ComponentT, W> packetMaxNorm(
    const Expr& expr, const PacketIndex<W>& pi)
{
    static_assert(Expr::VecDims >= 1,
        "packetMaxNorm: expression must have at least one component");
    return detail::PacketMaxNorm<Expr::VecDims - 1, Expr, W>::eval(expr, pi);
}

/** @brief Per-lane dot product of two packet-evaluable expressions. */
template<typename L, typename R, idx_t W>
inline simd::Packet<typename L::ComponentT, W> packetDot(
    const L& l, const R& r, const PacketIndex<W>& pi)
{
    static_assert(L::VecDims == R::VecDims,
        "packetDot: operands must have the same VecDims");
    static_assert(L::VecDims >= 1,
        "packetDot: operands must have at least one component");
    return detail::PacketDot<L::VecDims - 1, L, R, W>::eval(l, r, pi);
}

/**
 * @brief Conditional (masked) packet assignment — CUDA early-return analogue.
 *
 * Only writes to lanes where `mask` is true. Inactive lanes are no-ops,
 * equivalent to `if (condition) out[i] = expr;` in scalar code or
 * `if (i >= N) return;` in CUDA kernels.
 *
 * @tparam ExprL  Destination type.
 * @tparam ExprR  Source expression type.
 * @tparam W      Packet width.
 * @param dest    Destination array/ref.
 * @param expr    Expression to evaluate.
 * @param pi      Packet index.
 * @param mask    Lane mask (true = active, false = no-op).
 */
template<typename ExprL, typename ExprR, idx_t W>
inline void maskedPacketAssign(ExprL& dest, const ExprR& expr,
    const PacketIndex<W>& pi,
    const simd::PacketMask<typename ExprL::ComponentT, W>& mask)
{
    using DataT = typename ExprL::ComponentT;
    packet::MaskedRecursivePacketAssign<ExprL::VecDims - 1,
        ExprL, ExprR, W, DataT>::eval(pi, mask, dest, expr);
}

/**
 * @brief Unpack a packet into per-lane scalars.
 *
 * Stores the W lanes of ``src`` into ``dst[0..W-1]``.  Used by per-lane
 * scalar fallback paths inside packet kernels (e.g. when a lane-divergent
 * predicate forces a scalar dispatch).
 */
template<typename T, idx_t W>
inline void unpackLanes(const simd::Packet<T, W>& src, T (&dst)[W])
{
    simd::Packet<T, W>::store(dst, src);
}

/**
 * @brief Pack per-lane scalars into a packet.
 *
 * Inverse of ``unpackLanes(Packet, T*)``: loads ``src[0..W-1]`` back into
 * a single ``Packet<T, W>``.
 */
template<typename T, idx_t W>
inline void packLanes(simd::Packet<T, W>& dst, const T (&src)[W])
{
    dst = simd::Packet<T, W>::load(src);
}

namespace detail {

template<dims_t Dim, typename T, dims_t D, idx_t W>
struct UnpackLanesItemRec {
    inline static void apply(const vector::PacketItem<T, D, W>& src,
        vector::Item<T, D> (&dst)[W])
    {
        if constexpr (Dim > 0)
            UnpackLanesItemRec<Dim - 1, T, D, W>::apply(src, dst);
        alignas(64) T buf[W];
        simd::Packet<T, W>::store(buf, src.template packet<Dim>());
        for (idx_t k = 0; k < W; ++k) dst[k].data()[Dim] = buf[k];
    }
};

template<dims_t Dim, typename T, dims_t D, idx_t W>
struct PackLanesItemRec {
    inline static void apply(vector::PacketItem<T, D, W>& dst,
        const vector::Item<T, D> (&src)[W])
    {
        if constexpr (Dim > 0)
            PackLanesItemRec<Dim - 1, T, D, W>::apply(dst, src);
        alignas(64) T buf[W];
        for (idx_t k = 0; k < W; ++k) buf[k] = src[k].data()[Dim];
        dst.template packet<Dim>() = simd::Packet<T, W>::load(buf);
    }
};

} // namespace detail

/**
 * @brief Unpack a ``PacketItem<T, D, W>`` into ``W`` scalar
 *        ``Item<T, D>`` lanes.
 *
 * Stages each component packet into a buffer once, then transposes
 * lane-major.  Use to feed a per-lane scalar kernel from inside a
 * packet body — replaces hand-rolled ``Packet::store`` + index loops.
 */
template<typename T, dims_t D, idx_t W>
inline void unpackLanes(const vector::PacketItem<T, D, W>& src,
    vector::Item<T, D> (&dst)[W])
{
    detail::UnpackLanesItemRec<D - 1, T, D, W>::apply(src, dst);
}

/**
 * @brief Pack ``W`` scalar ``Item<T, D>`` lanes into a
 *        ``PacketItem<T, D, W>``.  Inverse of ``unpackLanes``.
 */
template<typename T, dims_t D, idx_t W>
inline void packLanes(vector::PacketItem<T, D, W>& dst,
    const vector::Item<T, D> (&src)[W])
{
    detail::PackLanesItemRec<D - 1, T, D, W>::apply(dst, src);
}

/**
 * @brief Query helpers for tile/packet decomposition.
 */
template<typename DataT>
inline idx_t nPackets(idx_t N)
{
    constexpr idx_t W = simd::PreferredWidth<DataT>;
    return (N + W - 1) / W;
}

template<typename DataT>
inline idx_t tailWidth(idx_t N)
{
    constexpr idx_t W = simd::PreferredWidth<DataT>;
    idx_t rem = N % W;
    return rem == 0 ? W : rem;
}

inline idx_t nTiles(idx_t N, idx_t tileSize = DEFAULT_TILE_SIZE)
{
    return (N + tileSize - 1) / tileSize;
}

} // namespace cpu
} // namespace feta
