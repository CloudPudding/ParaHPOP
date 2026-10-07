#pragma once

/**
 * @file LoadStore.h
 * @brief Packet element access — forward declarations, the expression-kind
 *        traits (`IsRefArrayLike`/`IsItemLike`/`IsMultiLoadExpr`), the
 *        RefArray/Handle load/store + prefetch primitives, and the
 *        `packetGet`/`packetStore` overload-resolution dispatchers.
 *
 * Part of the `PacketOps.h` facade.
 */

#include "feta/core/SampleIndex.h"
#include "feta/core/simd/simd.h"
#include "feta/vector/PacketItem.h"
#include "feta/vector/expr/Expression.h"
#include "feta/vector/expr/OperationsDetail.h"

namespace feta {
namespace cpu {
namespace packet {

using namespace feta::simd;

// ═════════════════════════════════════════════════════════════════════════════
//  Forward declarations — required by nvcc's EDG frontend, which resolves
//  dependent template calls at definition time rather than instantiation time.
// ═════════════════════════════════════════════════════════════════════════════

template<dims_t dim, idx_t W, typename DataT, dims_t VD, bool work, bool UseTex>
inline Packet<DataT, W> packetGet(
    const vector::texture::detail::RefArray<DataT, VD, work, UseTex>&,
    const PacketIndex<W>&);

template<dims_t dim, idx_t W, typename DataT, dims_t VD>
inline Packet<DataT, W> packetGet(
    const vector::Item<DataT, VD>&, const PacketIndex<W>&);

template<dims_t dim, idx_t W, typename DataT, dims_t VD>
inline Packet<DataT, W> packetGet(
    const vector::PacketItem<DataT, VD, W>&, const PacketIndex<W>&);

template<dims_t dim, idx_t W, typename DataT, dims_t VD>
inline Packet<DataT, W> packetGet(
    const vector::OnesNT<DataT, VD>&, const PacketIndex<W>&);

template<dims_t dim, idx_t W, typename DataT, dims_t VD>
inline Packet<DataT, W> packetGet(
    const vector::ZerosNT<DataT, VD>&, const PacketIndex<W>&);

template<dims_t dim, idx_t W, typename DataT>
inline Packet<DataT, W> packetGet(
    const vector::NeutralQuaternion<DataT>&, const PacketIndex<W>&);

template<dims_t dim, idx_t W, typename L, typename R, bool sub>
inline auto packetGet(
    const vector::expr::detail::Sum<L, R, sub>&, const PacketIndex<W>&);

template<dims_t dim, idx_t W, typename Expr, typename ScaleT>
inline auto packetGet(
    const vector::expr::detail::CWiseScale<Expr, ScaleT>&, const PacketIndex<W>&);

template<dims_t dim, idx_t W, typename Expr, typename ScaleT>
inline auto packetGet(
    const vector::expr::detail::CWiseRScale<Expr, ScaleT>&, const PacketIndex<W>&);

template<dims_t dim, idx_t W, typename Expr, typename ScaleT>
inline auto packetGet(
    const vector::expr::detail::CWiseInverse<Expr, ScaleT>&, const PacketIndex<W>&);

template<dims_t dim, idx_t W, typename L, typename R>
inline auto packetGet(
    const vector::expr::detail::CWiseMult<L, R>&, const PacketIndex<W>&);

template<dims_t dim, idx_t W, typename L, typename R>
inline auto packetGet(
    const vector::expr::detail::CWiseDiv<L, R>&, const PacketIndex<W>&);

template<dims_t dim, idx_t W, typename E>
inline auto packetGet(
    const vector::expr::detail::CWiseAbs<E>&, const PacketIndex<W>&);

template<dims_t dim, idx_t W, typename L, typename R>
inline auto packetGet(
    const vector::expr::detail::CWiseMax<L, R>&, const PacketIndex<W>&);

template<dims_t dim, idx_t W, typename L, typename R>
inline auto packetGet(
    const vector::expr::detail::Cross<L, R>&, const PacketIndex<W>&);

template<dims_t dim, idx_t W, typename E>
inline auto packetGet(
    const vector::expr::detail::Normalize<E>&, const PacketIndex<W>&);

template<dims_t dim, idx_t W, typename E>
inline auto packetGet(
    const vector::expr::detail::QuatConj<E>&, const PacketIndex<W>&);

template<dims_t dim, idx_t W, typename E>
inline auto packetGet(
    const vector::expr::detail::QuatAntiInvolute<E>&, const PacketIndex<W>&);

template<dims_t dim, idx_t W, typename L, typename R>
inline auto packetGet(
    const vector::expr::detail::QuatMul<L, R>&, const PacketIndex<W>&);

template<dims_t dim, idx_t W, typename Q, typename V>
inline auto packetGet(
    const vector::expr::detail::QuatRotate<Q, V>&, const PacketIndex<W>&);

template<dims_t dim, idx_t W, typename InnerT, dims_t dimOffset, dims_t nDims>
inline auto packetGet(
    const vector::expr::detail::BaseComponentView<InnerT, dimOffset, nDims>&,
    const PacketIndex<W>&);

template<dims_t dim, idx_t W, typename InnerT, dims_t frontExtra, dims_t backExtra>
inline auto packetGet(
    const vector::expr::detail::ExpandedView<InnerT, frontExtra, backExtra>&,
    const PacketIndex<W>&);

// ═════════════════════════════════════════════════════════════════════════════
//  packetGet — definitions for each expression node type
// ═════════════════════════════════════════════════════════════════════════════

// ─── Leaf: RefArray (the main SoA array reference) ──────────────────────────

namespace detail {

/**
 * @brief Detect whether `T` has `data_` and `dimOffset_` members (i.e. is
 * a RefArray-like type).
 */
template<typename T, typename = void>
struct IsRefArrayLike : std::false_type {};

template<typename T>
struct IsRefArrayLike<T, std::void_t<
    decltype(std::declval<T>().data_),
    decltype(std::declval<T>().dimOffset_)>> : std::true_type {};

/**
 * @brief Detect whether `T` has a `data_` array member (Item-like).
 */
template<typename T, typename = void>
struct IsItemLike : std::false_type {};

template<typename T>
struct IsItemLike<T, std::void_t<
    decltype(std::declval<T>().template get<0>())>> : std::true_type {};

/**
 * @brief Trait: true for expression nodes whose packetGet<dim> loads ALL
 * child components regardless of which dim is requested.  These benefit
 * from being materialised into a PacketItem before per-dim consumption.
 *
 * Marked types: Cross, QuatMul, QuatRotate, Normalize.
 */
template<typename T>
struct IsMultiLoadExpr : std::false_type {};

template<typename L, typename R>
struct IsMultiLoadExpr<vector::expr::detail::Cross<L, R>> : std::true_type {};

template<typename L, typename R>
struct IsMultiLoadExpr<vector::expr::detail::QuatMul<L, R>> : std::true_type {};

template<typename Q, typename V>
struct IsMultiLoadExpr<vector::expr::detail::QuatRotate<Q, V>> : std::true_type {};

template<typename E>
struct IsMultiLoadExpr<vector::expr::detail::Normalize<E>> : std::true_type {};

} // namespace detail


/**
 * @brief Load a packet from a RefArray-like expression at component `dim`.
 *
 * For full packets: aligned/unaligned SIMD load.
 * For tail packets: masked load (inactive lanes get zero).
 */
template<dims_t dim, typename DataT, idx_t W, typename ExprT>
inline auto packetGetRefArray(
    const ExprT& expr, const PacketIndex<W>& pi)
    -> Packet<DataT, W>
{
    const DataT* ptr = expr.data_ + expr.dimOffset_ * dim + pi.base_;
    if (pi.full())
        return Packet<DataT, W>::load(ptr);
    else
        return Packet<DataT, W>::maskLoad(
            ptr, PacketMask<DataT, W>::firstN(pi.active_));
}

/**
 * @brief Store a packet into a RefArray-like expression at component `dim`.
 */
template<dims_t dim, typename DataT, idx_t W, typename ExprT>
inline void packetStoreRefArray(
    ExprT& dest, const PacketIndex<W>& pi, const Packet<DataT, W>& val)
{
    DataT* ptr = dest.data_ + dest.dimOffset_ * dim + pi.base_;
    if (pi.full())
        Packet<DataT, W>::store(ptr, val);
    else
        Packet<DataT, W>::maskStore(
            ptr, PacketMask<DataT, W>::firstN(pi.active_), val);
}

/**
 * @brief Masked store into a RefArray.
 */
template<dims_t dim, typename DataT, idx_t W, typename ExprT>
inline void packetMaskedStoreRefArray(
    ExprT& dest, const PacketIndex<W>& pi,
    const Packet<DataT, W>& val, const PacketMask<DataT, W>& mask)
{
    DataT* ptr = dest.data_ + dest.dimOffset_ * dim + pi.base_;
    // Combine user mask with tail mask
    auto finalMask = pi.full() ? mask
        : (mask & PacketMask<DataT, W>::firstN(pi.active_));
    Packet<DataT, W>::maskStore(ptr, finalMask, val);
}

// ═════════════════════════════════════════════════════════════════════════════
//  Handle-specific load/store — Handle uses `ptr` and `dOffset` members
// ═════════════════════════════════════════════════════════════════════════════

template<dims_t dim, typename DataT, idx_t W, typename HandleT>
inline auto packetGetHandle(
    const HandleT& h, const PacketIndex<W>& pi)
    -> Packet<DataT, W>
{
    const DataT* p = h.ptr + h.dOffset * dim + pi.base_;
    if (pi.full())
        return Packet<DataT, W>::load(p);
    else
        return Packet<DataT, W>::maskLoad(
            p, PacketMask<DataT, W>::firstN(pi.active_));
}

template<dims_t dim, typename DataT, idx_t W, typename HandleT>
inline void packetStoreHandle(
    HandleT& h, const PacketIndex<W>& pi, const Packet<DataT, W>& val)
{
    DataT* p = h.ptr + h.dOffset * dim + pi.base_;
    if (pi.full())
        Packet<DataT, W>::store(p, val);
    else
        Packet<DataT, W>::maskStore(
            p, PacketMask<DataT, W>::firstN(pi.active_), val);
}

template<dims_t dim, typename DataT, idx_t W, typename HandleT>
inline void packetMaskedStoreHandle(
    HandleT& h, const PacketIndex<W>& pi,
    const Packet<DataT, W>& val, const PacketMask<DataT, W>& mask)
{
    DataT* p = h.ptr + h.dOffset * dim + pi.base_;
    auto finalMask = pi.full() ? mask
        : (mask & PacketMask<DataT, W>::firstN(pi.active_));
    Packet<DataT, W>::maskStore(p, finalMask, val);
}

/**
 * @brief Prefetch the next packet's memory for a RefArray at component `dim`.
 */
template<dims_t dim, typename ExprT>
inline void prefetchRefArray(const ExprT& expr, idx_t nextBase)
{
    const auto* ptr = expr.data_ + expr.dimOffset_ * dim + nextBase;
    simd::prefetchL1(ptr);
}

/**
 * @brief Prefetch into L2 for a RefArray at component `dim`.
 */
template<dims_t dim, typename ExprT>
inline void prefetchRefArrayL2(const ExprT& expr, idx_t nextBase)
{
    const auto* ptr = expr.data_ + expr.dimOffset_ * dim + nextBase;
    simd::prefetchL2(ptr);
}

/**
 * @brief Write-intent prefetch for a RefArray at component `dim`.
 */
template<dims_t dim, typename ExprT>
inline void prefetchRefArrayW(const ExprT& expr, idx_t nextBase)
{
    const auto* ptr = expr.data_ + expr.dimOffset_ * dim + nextBase;
    simd::prefetchL1W(ptr);
}


// ═════════════════════════════════════════════════════════════════════════════
//  Generic packetGet dispatcher — uses overload resolution
// ═════════════════════════════════════════════════════════════════════════════

/**
 * @brief Primary packetGet template — dispatches based on expression node type.
 *
 * Each expression node type has its own overload below. The primary template
 * handles RefArray-like leaves.
 */

// ─── RefArray leaf ──────────────────────────────────────────────────────────

template<dims_t dim, idx_t W, typename DataT, dims_t VD, bool work, bool UseTex>
inline Packet<DataT, W> packetGet(
    const vector::texture::detail::RefArray<DataT, VD, work, UseTex>& expr,
    const PacketIndex<W>& pi)
{
    return packetGetRefArray<dim, DataT, W>(expr, pi);
}

// ─── Handle leaf ────────────────────────────────────────────────────────────

template<dims_t dim, idx_t W, typename DataT, dims_t VD, bool work>
inline Packet<DataT, W> packetGet(
    const vector::texture::detail::Handle<DataT, VD, work>& h,
    const PacketIndex<W>& pi)
{
    return packetGetHandle<dim, DataT, W>(h, pi);
}

// ─── Item leaf (broadcast scalar value into all lanes) ──────────────────────

template<dims_t dim, idx_t W, typename DataT, dims_t VD>
inline Packet<DataT, W> packetGet(
    const vector::Item<DataT, VD>& item,
    const PacketIndex<W>&)
{
    return Packet<DataT, W>::broadcast(item.template get<dim>());
}

// ─── PacketItem leaf (return stored packet directly — zero overhead) ────────

template<dims_t dim, idx_t W, typename DataT, dims_t VD>
inline Packet<DataT, W> packetGet(
    const vector::PacketItem<DataT, VD, W>& item,
    const PacketIndex<W>&)
{
    return item.template packet<dim>();
}

// ─── OnesNT (broadcast 1) ──────────────────────────────────────────────────

template<dims_t dim, idx_t W, typename DataT, dims_t VD>
inline Packet<DataT, W> packetGet(
    const vector::OnesNT<DataT, VD>&,
    const PacketIndex<W>&)
{
    return Packet<DataT, W>::broadcast(DataT{1});
}

// ─── ZerosNT (broadcast 0) ─────────────────────────────────────────────────

template<dims_t dim, idx_t W, typename DataT, dims_t VD>
inline Packet<DataT, W> packetGet(
    const vector::ZerosNT<DataT, VD>&,
    const PacketIndex<W>&)
{
    return Packet<DataT, W>::zero();
}

// ─── NeutralQuaternion (1 at dim 0, 0 elsewhere) ───────────────────────────

template<dims_t dim, idx_t W, typename DataT>
inline Packet<DataT, W> packetGet(
    const vector::NeutralQuaternion<DataT>&,
    const PacketIndex<W>&)
{
    if constexpr (dim == 0)
        return Packet<DataT, W>::broadcast(DataT{1});
    else
        return Packet<DataT, W>::zero();
}

// ─── Sum / Diff ─────────────────────────────────────────────────────────────

template<dims_t dim, idx_t W, typename L, typename R, bool sub>
inline auto packetGet(
    const vector::expr::detail::Sum<L, R, sub>& expr,
    const PacketIndex<W>& pi)
{
    auto lp = packetGet<dim, W>(expr.l_, pi);
    auto rp = packetGet<dim, W>(expr.r_, pi);
    if constexpr (sub)
        return lp - rp;
    else
        return lp + rp;
}

// ─── CWiseScale (scalar * vector) ──────────────────────────────────────────

template<dims_t dim, idx_t W, typename Expr, typename ScaleT>
inline auto packetGet(
    const vector::expr::detail::CWiseScale<Expr, ScaleT>& expr,
    const PacketIndex<W>& pi)
{
    using DataT = typename Expr::ComponentT;
    auto p = packetGet<dim, W>(expr.expr_, pi);
    return Packet<DataT, W>::broadcast(static_cast<DataT>(expr.factor_)) * p;
}

// ─── CWiseRScale (vector / scalar) ─────────────────────────────────────────

template<dims_t dim, idx_t W, typename Expr, typename ScaleT>
inline auto packetGet(
    const vector::expr::detail::CWiseRScale<Expr, ScaleT>& expr,
    const PacketIndex<W>& pi)
{
    using DataT = typename Expr::ComponentT;
    auto p = packetGet<dim, W>(expr.expr_, pi);
    // factor_ already holds the reciprocal for floats
    return p * Packet<DataT, W>::broadcast(static_cast<DataT>(expr.factor_));
}

// ─── CWiseInverse (scalar / vector) ────────────────────────────────────────

template<dims_t dim, idx_t W, typename Expr, typename ScaleT>
inline auto packetGet(
    const vector::expr::detail::CWiseInverse<Expr, ScaleT>& expr,
    const PacketIndex<W>& pi)
{
    using DataT = typename Expr::ComponentT;
    auto p = packetGet<dim, W>(expr.expr_, pi);
    return Packet<DataT, W>::broadcast(static_cast<DataT>(expr.factor_)) / p;
}

// ─── CWiseMult (vector * vector) ───────────────────────────────────────────

template<dims_t dim, idx_t W, typename L, typename R>
inline auto packetGet(
    const vector::expr::detail::CWiseMult<L, R>& expr,
    const PacketIndex<W>& pi)
{
    return packetGet<dim, W>(expr.l_, pi) * packetGet<dim, W>(expr.r_, pi);
}

// ─── CWiseDiv (vector / vector) ────────────────────────────────────────────

template<dims_t dim, idx_t W, typename L, typename R>
inline auto packetGet(
    const vector::expr::detail::CWiseDiv<L, R>& expr,
    const PacketIndex<W>& pi)
{
    return packetGet<dim, W>(expr.l_, pi) / packetGet<dim, W>(expr.r_, pi);
}

// ─── CWiseAbs ──────────────────────────────────────────────────────────────

template<dims_t dim, idx_t W, typename E>
inline auto packetGet(
    const vector::expr::detail::CWiseAbs<E>& expr,
    const PacketIndex<W>& pi)
{
    return abs(packetGet<dim, W>(expr.expr_, pi));
}

// ─── CWiseMax ──────────────────────────────────────────────────────────────

template<dims_t dim, idx_t W, typename L, typename R>
inline auto packetGet(
    const vector::expr::detail::CWiseMax<L, R>& expr,
    const PacketIndex<W>& pi)
{
    return max(packetGet<dim, W>(expr.l_, pi),
               packetGet<dim, W>(expr.r_, pi));
}

// ─── Cross (3D) ────────────────────────────────────────────────────────────
// NOTE: The per-dim packetGet is kept for composability (e.g. when Cross
// appears as a sub-expression).  When Cross is the top-level RHS of an
// assignment, the specialised RecursivePacketAssign below eliminates
// redundant loads by computing all three output dimensions in one shot.

template<dims_t dim, idx_t W, typename L, typename R>
inline auto packetGet(
    const vector::expr::detail::Cross<L, R>& expr,
    const PacketIndex<W>& pi)
{
    auto l0 = packetGet<0, W>(expr.l_, pi);
    auto l1 = packetGet<1, W>(expr.l_, pi);
    auto l2 = packetGet<2, W>(expr.l_, pi);
    auto r0 = packetGet<0, W>(expr.r_, pi);
    auto r1 = packetGet<1, W>(expr.r_, pi);
    auto r2 = packetGet<2, W>(expr.r_, pi);

    if constexpr (dim == 0)
        return fmsub(l1, r2, l2 * r1);   // l1*r2 - l2*r1
    else if constexpr (dim == 1)
        return fmsub(l2, r0, l0 * r2);   // l2*r0 - l0*r2
    else
        return fmsub(l0, r1, l1 * r0);   // l0*r1 - l1*r0
}

// ─── Normalize ─────────────────────────────────────────────────────────────

template<dims_t dim, idx_t W, typename E>
inline auto packetGet(
    const vector::expr::detail::Normalize<E>& expr,
    const PacketIndex<W>& pi)
{
    using DataT = typename E::ComponentT;
    // Compute squared norm via packet accumulation over all dims
    auto sn = packetGet<0, W>(expr.e_, pi) * packetGet<0, W>(expr.e_, pi);
    auto accumSN = [&]<dims_t... Is>(std::integer_sequence<dims_t, Is...>) {
        ((sn = sn + packetGet<Is+1, W>(expr.e_, pi) * packetGet<Is+1, W>(expr.e_, pi)), ...);
    };
    accumSN(std::make_integer_sequence<dims_t, E::VecDims - 1>{});

    auto rn = Packet<DataT, W>::broadcast(DataT{1}) / sqrt(sn);
    return packetGet<dim, W>(expr.e_, pi) * rn;
}

// ─── QuatConj ──────────────────────────────────────────────────────────────

template<dims_t dim, idx_t W, typename E>
inline auto packetGet(
    const vector::expr::detail::QuatConj<E>& expr,
    const PacketIndex<W>& pi)
{
    auto p = packetGet<dim, W>(expr.e_, pi);
    if constexpr (dim == 0)
        return p;
    else
        return -p;
}

// ─── QuatAntiInvolute ──────────────────────────────────────────────────────

template<dims_t dim, idx_t W, typename E>
inline auto packetGet(
    const vector::expr::detail::QuatAntiInvolute<E>& expr,
    const PacketIndex<W>& pi)
{
    auto p = packetGet<dim, W>(expr.e_, pi);
    if constexpr (dim == 3)
        return -p;
    else
        return p;
}

// ─── QuatMul ───────────────────────────────────────────────────────────────

template<dims_t dim, idx_t W, typename L, typename R>
inline auto packetGet(
    const vector::expr::detail::QuatMul<L, R>& expr,
    const PacketIndex<W>& pi)
{
    auto l0 = packetGet<0, W>(expr.l_, pi);
    auto l1 = packetGet<1, W>(expr.l_, pi);
    auto l2 = packetGet<2, W>(expr.l_, pi);
    auto l3 = packetGet<3, W>(expr.l_, pi);
    auto r0 = packetGet<0, W>(expr.r_, pi);
    auto r1 = packetGet<1, W>(expr.r_, pi);
    auto r2 = packetGet<2, W>(expr.r_, pi);
    auto r3 = packetGet<3, W>(expr.r_, pi);

    if constexpr (dim == 0)
        // l0*r0 - l1*r1 - l2*r2 - l3*r3
        return fmsub(l0, r0, l1 * r1) - fmadd(l2, r2, l3 * r3);
    else if constexpr (dim == 1)
        // l0*r1 + l1*r0 + l2*r3 - l3*r2
        return fmadd(l0, r1, l1 * r0) + fmsub(l2, r3, l3 * r2);
    else if constexpr (dim == 2)
        // l0*r2 - l1*r3 + l2*r0 + l3*r1
        return fmsub(l0, r2, l1 * r3) + fmadd(l2, r0, l3 * r1);
    else
        // l0*r3 + l1*r2 - l2*r1 + l3*r0
        return fmadd(l0, r3, l1 * r2) + fmsub(l3, r0, l2 * r1);
}

// ─── QuatRotate (Rodrigues) ────────────────────────────────────────────────

template<dims_t dim, idx_t W, typename Q, typename V>
inline auto packetGet(
    const vector::expr::detail::QuatRotate<Q, V>& expr,
    const PacketIndex<W>& pi)
{
    using DataT = typename Q::ComponentT;
    auto w  = packetGet<0, W>(expr.q_, pi);
    auto ux = packetGet<1, W>(expr.q_, pi);
    auto uy = packetGet<2, W>(expr.q_, pi);
    auto uz = packetGet<3, W>(expr.q_, pi);

    auto vx = packetGet<0, W>(expr.v_, pi);
    auto vy = packetGet<1, W>(expr.v_, pi);
    auto vz = packetGet<2, W>(expr.v_, pi);

    auto two = Packet<DataT, W>::broadcast(DataT{2});
    auto tx = two * fmsub(uy, vz, uz * vy);
    auto ty = two * fmsub(uz, vx, ux * vz);
    auto tz = two * fmsub(ux, vy, uy * vx);

    if constexpr (dim == 0)
        return fmadd(w, tx, vx) + fmsub(uy, tz, uz * ty);
    else if constexpr (dim == 1)
        return fmadd(w, ty, vy) + fmsub(uz, tx, ux * tz);
    else
        return fmadd(w, tz, vz) + fmsub(ux, ty, uy * tx);
}

// ─── BaseComponentView (segment) ───────────────────────────────────────────

template<dims_t dim, idx_t W, typename InnerT, dims_t dimOffset, dims_t nDims>
inline auto packetGet(
    const vector::expr::detail::BaseComponentView<InnerT, dimOffset, nDims>& expr,
    const PacketIndex<W>& pi)
{
    return packetGet<dimOffset + dim, W>(expr.vecArray_, pi);
}

// ─── ExpandedView ──────────────────────────────────────────────────────────

template<dims_t dim, idx_t W, typename InnerT, dims_t frontExtra, dims_t backExtra>
inline auto packetGet(
    const vector::expr::detail::ExpandedView<InnerT, frontExtra, backExtra>& expr,
    const PacketIndex<W>& pi)
{
    using DataT = typename InnerT::ComponentT;
    constexpr dims_t TrueEnd = frontExtra + InnerT::VecDims;
    if constexpr (dim < frontExtra || dim >= TrueEnd)
        return Packet<DataT, W>::zero();
    else
        return packetGet<dim - frontExtra, W>(expr.vecArray_, pi);
}


// ═════════════════════════════════════════════════════════════════════════════
//  packetStore — dispatch for writable leaf nodes
// ═════════════════════════════════════════════════════════════════════════════

template<dims_t dim, idx_t W, typename DataT, dims_t VD, bool work, bool UseTex>
inline void packetStore(
    vector::texture::detail::RefArray<DataT, VD, work, UseTex>& dest,
    const PacketIndex<W>& pi,
    const Packet<DataT, W>& val)
{
    packetStoreRefArray<dim, DataT, W>(dest, pi, val);
}

template<dims_t dim, idx_t W, typename DataT, dims_t VD, bool work, bool UseTex>
inline void packetMaskedStore(
    vector::texture::detail::RefArray<DataT, VD, work, UseTex>& dest,
    const PacketIndex<W>& pi,
    const Packet<DataT, W>& val,
    const PacketMask<DataT, W>& mask)
{
    packetMaskedStoreRefArray<dim, DataT, W>(dest, pi, val, mask);
}

// ─── Handle store ───────────────────────────────────────────────────────────

template<dims_t dim, idx_t W, typename DataT, dims_t VD, bool work>
inline void packetStore(
    vector::texture::detail::Handle<DataT, VD, work>& dest,
    const PacketIndex<W>& pi,
    const Packet<DataT, W>& val)
{
    packetStoreHandle<dim, DataT, W>(dest, pi, val);
}

template<dims_t dim, idx_t W, typename DataT, dims_t VD, bool work>
inline void packetMaskedStore(
    vector::texture::detail::Handle<DataT, VD, work>& dest,
    const PacketIndex<W>& pi,
    const Packet<DataT, W>& val,
    const PacketMask<DataT, W>& mask)
{
    packetMaskedStoreHandle<dim, DataT, W>(dest, pi, val, mask);
}

// ─── BaseComponentView (segment) store ─────────────────────────────────────
//
// PR-6δ pass B kickoff: the read-side ``packetGet`` for ComponentView
// already exists above (line ~629); the matching store-side overload
// is the missing piece that lets writes flow through a
// ``Tile::slot<s>()`` (which returns a ``ComponentView``) when callers
// want to assign packet expressions back into the slot.  Both the
// unmasked and masked forms delegate to the inner array's existing
// store via the same dim-offset trick the ``packetGet`` overload uses.

template<dims_t dim, idx_t W, typename InnerT, dims_t dimOffset, dims_t nDims>
inline void packetStore(
    vector::expr::detail::BaseComponentView<InnerT, dimOffset, nDims>& dest,
    const PacketIndex<W>& pi,
    const Packet<typename InnerT::ComponentT, W>& val)
{
    packetStore<dimOffset + dim, W>(dest.vecArray_, pi, val);
}

template<dims_t dim, idx_t W, typename InnerT, dims_t dimOffset, dims_t nDims>
inline void packetMaskedStore(
    vector::expr::detail::BaseComponentView<InnerT, dimOffset, nDims>& dest,
    const PacketIndex<W>& pi,
    const Packet<typename InnerT::ComponentT, W>& val,
    const PacketMask<typename InnerT::ComponentT, W>& mask)
{
    packetMaskedStore<dimOffset + dim, W>(dest.vecArray_, pi, val, mask);
}

} // namespace packet
} // namespace cpu
} // namespace feta
