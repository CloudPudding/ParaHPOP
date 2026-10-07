#pragma once

/**
 * @file Prefetch.h
 * @brief Packet prefetch + sizing — `RecursivePrefetchLeaf`/`L2`/`W`, the
 *        `prefetchExpr`/`prefetchExprL2`/`prefetchExprW` overload set, and
 *        the `exprSize` overloads.
 *
 * Part of the `PacketOps.h` facade; depends on the load/store primitives.
 */

#include "feta/core/SampleIndex.h"
#include "feta/core/simd/simd.h"
#include "feta/vector/PacketItem.h"
#include "feta/vector/expr/Expression.h"
#include "feta/vector/expr/OperationsDetail.h"
#include "feta/vector/expr/cpu/packet/LoadStore.h"

namespace feta {
namespace cpu {
namespace packet {

using namespace feta::simd;

// ═════════════════════════════════════════════════════════════════════════════
//  RecursivePrefetch — issue prefetch for all dims of a RefArray
// ═════════════════════════════════════════════════════════════════════════════

/**
 * @brief Prefetch one dimension of a RefArray-like leaf.
 */
template<dims_t dim, typename ExprT>
struct RecursivePrefetchLeaf {
    inline static void eval(const ExprT& expr, idx_t nextBase)
    {
        prefetchRefArray<dim>(expr, nextBase);
        RecursivePrefetchLeaf<dim - 1, ExprT>::eval(expr, nextBase);
    }
};

template<typename ExprT>
struct RecursivePrefetchLeaf<0, ExprT> {
    inline static void eval(const ExprT& expr, idx_t nextBase)
    {
        prefetchRefArray<0>(expr, nextBase);
    }
};

/**
 * @brief L2 prefetch for all dims of a RefArray leaf (compile-time unrolled).
 */
template<dims_t dim, typename ExprT>
struct RecursivePrefetchLeafL2 {
    inline static void eval(const ExprT& expr, idx_t nextBase)
    {
        prefetchRefArrayL2<dim>(expr, nextBase);
        RecursivePrefetchLeafL2<dim - 1, ExprT>::eval(expr, nextBase);
    }
};

template<typename ExprT>
struct RecursivePrefetchLeafL2<0, ExprT> {
    inline static void eval(const ExprT& expr, idx_t nextBase)
    {
        prefetchRefArrayL2<0>(expr, nextBase);
    }
};

/**
 * @brief Write-intent prefetch for all dims of a RefArray leaf.
 */
template<dims_t dim, typename ExprT>
struct RecursivePrefetchLeafW {
    inline static void eval(const ExprT& expr, idx_t nextBase)
    {
        prefetchRefArrayW<dim>(expr, nextBase);
        RecursivePrefetchLeafW<dim - 1, ExprT>::eval(expr, nextBase);
    }
};

template<typename ExprT>
struct RecursivePrefetchLeafW<0, ExprT> {
    inline static void eval(const ExprT& expr, idx_t nextBase)
    {
        prefetchRefArrayW<0>(expr, nextBase);
    }
};

/**
 * @brief Prefetch the next packet for all leaf arrays in an expression tree.
 *
 * For RefArray leaves: issues `_mm_prefetch` for each dimension.
 * For Item / constant leaves: no-op (data is in registers / immediate).
 * For composite nodes: recurses into children.
 */
template<typename ExprT>
inline void prefetchExpr(const ExprT& expr, idx_t nextBase)
{
    if constexpr (detail::IsRefArrayLike<ExprT>::value) {
        RecursivePrefetchLeaf<ExprT::VecDims - 1, ExprT>::eval(expr, nextBase);
    }
    // Items and constants: nothing to prefetch
}

/**
 * @brief L2 prefetch for all leaf arrays in an expression tree.
 */
template<typename ExprT>
inline void prefetchExprL2(const ExprT& expr, idx_t nextBase)
{
    if constexpr (detail::IsRefArrayLike<ExprT>::value) {
        RecursivePrefetchLeafL2<ExprT::VecDims - 1, ExprT>::eval(expr, nextBase);
    }
}

/**
 * @brief Write-intent prefetch for all leaf arrays in a destination expression.
 */
template<typename ExprT>
inline void prefetchExprW(const ExprT& expr, idx_t nextBase)
{
    if constexpr (detail::IsRefArrayLike<ExprT>::value) {
        RecursivePrefetchLeafW<ExprT::VecDims - 1, ExprT>::eval(expr, nextBase);
    }
}

// Overloads for composite nodes that recurse into children:

template<typename L, typename R, bool sub>
inline void prefetchExpr(
    const vector::expr::detail::Sum<L, R, sub>& expr, idx_t nextBase)
{
    prefetchExpr(expr.l_, nextBase);
    prefetchExpr(expr.r_, nextBase);
}

template<typename Expr, typename ScaleT>
inline void prefetchExpr(
    const vector::expr::detail::CWiseScale<Expr, ScaleT>& expr, idx_t nextBase)
{
    prefetchExpr(expr.expr_, nextBase);
}

template<typename Expr, typename ScaleT>
inline void prefetchExpr(
    const vector::expr::detail::CWiseRScale<Expr, ScaleT>& expr, idx_t nextBase)
{
    prefetchExpr(expr.expr_, nextBase);
}

template<typename Expr, typename ScaleT>
inline void prefetchExpr(
    const vector::expr::detail::CWiseInverse<Expr, ScaleT>& expr, idx_t nextBase)
{
    prefetchExpr(expr.expr_, nextBase);
}

template<typename L, typename R>
inline void prefetchExpr(
    const vector::expr::detail::CWiseMult<L, R>& expr, idx_t nextBase)
{
    prefetchExpr(expr.l_, nextBase);
    prefetchExpr(expr.r_, nextBase);
}

template<typename L, typename R>
inline void prefetchExpr(
    const vector::expr::detail::CWiseDiv<L, R>& expr, idx_t nextBase)
{
    prefetchExpr(expr.l_, nextBase);
    prefetchExpr(expr.r_, nextBase);
}

template<typename E>
inline void prefetchExpr(
    const vector::expr::detail::CWiseAbs<E>& expr, idx_t nextBase)
{
    prefetchExpr(expr.expr_, nextBase);
}

template<typename L, typename R>
inline void prefetchExpr(
    const vector::expr::detail::CWiseMax<L, R>& expr, idx_t nextBase)
{
    prefetchExpr(expr.l_, nextBase);
    prefetchExpr(expr.r_, nextBase);
}

template<typename L, typename R>
inline void prefetchExpr(
    const vector::expr::detail::Cross<L, R>& expr, idx_t nextBase)
{
    prefetchExpr(expr.l_, nextBase);
    prefetchExpr(expr.r_, nextBase);
}

template<typename E>
inline void prefetchExpr(
    const vector::expr::detail::Normalize<E>& expr, idx_t nextBase)
{
    prefetchExpr(expr.e_, nextBase);
}

template<typename E>
inline void prefetchExpr(
    const vector::expr::detail::QuatConj<E>& expr, idx_t nextBase)
{
    prefetchExpr(expr.e_, nextBase);
}

template<typename E>
inline void prefetchExpr(
    const vector::expr::detail::QuatAntiInvolute<E>& expr, idx_t nextBase)
{
    prefetchExpr(expr.e_, nextBase);
}

template<typename L, typename R>
inline void prefetchExpr(
    const vector::expr::detail::QuatMul<L, R>& expr, idx_t nextBase)
{
    prefetchExpr(expr.l_, nextBase);
    prefetchExpr(expr.r_, nextBase);
}

template<typename Q, typename V>
inline void prefetchExpr(
    const vector::expr::detail::QuatRotate<Q, V>& expr, idx_t nextBase)
{
    prefetchExpr(expr.q_, nextBase);
    prefetchExpr(expr.v_, nextBase);
}

// No-op for Item, OnesNT, ZerosNT, NeutralQuaternion (data is not in memory)
template<typename DataT, dims_t VD>
inline void prefetchExpr(const vector::Item<DataT, VD>&, idx_t) {}

template<typename DataT, dims_t VD>
inline void prefetchExpr(const vector::OnesNT<DataT, VD>&, idx_t) {}

template<typename DataT, dims_t VD>
inline void prefetchExpr(const vector::ZerosNT<DataT, VD>&, idx_t) {}

template<typename DataT>
inline void prefetchExpr(const vector::NeutralQuaternion<DataT>&, idx_t) {}

template<typename DataT, dims_t VD, idx_t W>
inline void prefetchExpr(const vector::PacketItem<DataT, VD, W>&, idx_t) {}

// ─── L2 prefetch composite overloads ─────────────────────────────────────────

template<typename L, typename R, bool sub>
inline void prefetchExprL2(
    const vector::expr::detail::Sum<L, R, sub>& expr, idx_t nextBase)
{ prefetchExprL2(expr.l_, nextBase); prefetchExprL2(expr.r_, nextBase); }

template<typename Expr, typename ScaleT>
inline void prefetchExprL2(
    const vector::expr::detail::CWiseScale<Expr, ScaleT>& expr, idx_t nextBase)
{ prefetchExprL2(expr.expr_, nextBase); }

template<typename Expr, typename ScaleT>
inline void prefetchExprL2(
    const vector::expr::detail::CWiseRScale<Expr, ScaleT>& expr, idx_t nextBase)
{ prefetchExprL2(expr.expr_, nextBase); }

template<typename Expr, typename ScaleT>
inline void prefetchExprL2(
    const vector::expr::detail::CWiseInverse<Expr, ScaleT>& expr, idx_t nextBase)
{ prefetchExprL2(expr.expr_, nextBase); }

template<typename L, typename R>
inline void prefetchExprL2(
    const vector::expr::detail::CWiseMult<L, R>& expr, idx_t nextBase)
{ prefetchExprL2(expr.l_, nextBase); prefetchExprL2(expr.r_, nextBase); }

template<typename L, typename R>
inline void prefetchExprL2(
    const vector::expr::detail::CWiseDiv<L, R>& expr, idx_t nextBase)
{ prefetchExprL2(expr.l_, nextBase); prefetchExprL2(expr.r_, nextBase); }

template<typename E>
inline void prefetchExprL2(
    const vector::expr::detail::CWiseAbs<E>& expr, idx_t nextBase)
{ prefetchExprL2(expr.expr_, nextBase); }

template<typename L, typename R>
inline void prefetchExprL2(
    const vector::expr::detail::CWiseMax<L, R>& expr, idx_t nextBase)
{ prefetchExprL2(expr.l_, nextBase); prefetchExprL2(expr.r_, nextBase); }

template<typename L, typename R>
inline void prefetchExprL2(
    const vector::expr::detail::Cross<L, R>& expr, idx_t nextBase)
{ prefetchExprL2(expr.l_, nextBase); prefetchExprL2(expr.r_, nextBase); }

template<typename E>
inline void prefetchExprL2(
    const vector::expr::detail::Normalize<E>& expr, idx_t nextBase)
{ prefetchExprL2(expr.e_, nextBase); }

template<typename E>
inline void prefetchExprL2(
    const vector::expr::detail::QuatConj<E>& expr, idx_t nextBase)
{ prefetchExprL2(expr.e_, nextBase); }

template<typename E>
inline void prefetchExprL2(
    const vector::expr::detail::QuatAntiInvolute<E>& expr, idx_t nextBase)
{ prefetchExprL2(expr.e_, nextBase); }

template<typename L, typename R>
inline void prefetchExprL2(
    const vector::expr::detail::QuatMul<L, R>& expr, idx_t nextBase)
{ prefetchExprL2(expr.l_, nextBase); prefetchExprL2(expr.r_, nextBase); }

template<typename Q, typename V>
inline void prefetchExprL2(
    const vector::expr::detail::QuatRotate<Q, V>& expr, idx_t nextBase)
{ prefetchExprL2(expr.q_, nextBase); prefetchExprL2(expr.v_, nextBase); }

// L2 no-ops for register-resident / constant leaves
template<typename DataT, dims_t VD>
inline void prefetchExprL2(const vector::Item<DataT, VD>&, idx_t) {}
template<typename DataT, dims_t VD>
inline void prefetchExprL2(const vector::OnesNT<DataT, VD>&, idx_t) {}
template<typename DataT, dims_t VD>
inline void prefetchExprL2(const vector::ZerosNT<DataT, VD>&, idx_t) {}
template<typename DataT>
inline void prefetchExprL2(const vector::NeutralQuaternion<DataT>&, idx_t) {}
template<typename DataT, dims_t VD, idx_t W>
inline void prefetchExprL2(const vector::PacketItem<DataT, VD, W>&, idx_t) {}

// ─── Write-intent prefetch composite overloads ───────────────────────────────
// These walk the expression tree the same way but issue write-intent prefetches.
// Primarily used for destination arrays.

template<typename L, typename R, bool sub>
inline void prefetchExprW(
    const vector::expr::detail::Sum<L, R, sub>& expr, idx_t nextBase)
{ prefetchExprW(expr.l_, nextBase); prefetchExprW(expr.r_, nextBase); }

template<typename Expr, typename ScaleT>
inline void prefetchExprW(
    const vector::expr::detail::CWiseScale<Expr, ScaleT>& expr, idx_t nextBase)
{ prefetchExprW(expr.expr_, nextBase); }

template<typename Expr, typename ScaleT>
inline void prefetchExprW(
    const vector::expr::detail::CWiseRScale<Expr, ScaleT>& expr, idx_t nextBase)
{ prefetchExprW(expr.expr_, nextBase); }

template<typename Expr, typename ScaleT>
inline void prefetchExprW(
    const vector::expr::detail::CWiseInverse<Expr, ScaleT>& expr, idx_t nextBase)
{ prefetchExprW(expr.expr_, nextBase); }

template<typename L, typename R>
inline void prefetchExprW(
    const vector::expr::detail::CWiseMult<L, R>& expr, idx_t nextBase)
{ prefetchExprW(expr.l_, nextBase); prefetchExprW(expr.r_, nextBase); }

template<typename L, typename R>
inline void prefetchExprW(
    const vector::expr::detail::CWiseDiv<L, R>& expr, idx_t nextBase)
{ prefetchExprW(expr.l_, nextBase); prefetchExprW(expr.r_, nextBase); }

template<typename E>
inline void prefetchExprW(
    const vector::expr::detail::CWiseAbs<E>& expr, idx_t nextBase)
{ prefetchExprW(expr.expr_, nextBase); }

template<typename L, typename R>
inline void prefetchExprW(
    const vector::expr::detail::CWiseMax<L, R>& expr, idx_t nextBase)
{ prefetchExprW(expr.l_, nextBase); prefetchExprW(expr.r_, nextBase); }

template<typename L, typename R>
inline void prefetchExprW(
    const vector::expr::detail::Cross<L, R>& expr, idx_t nextBase)
{ prefetchExprW(expr.l_, nextBase); prefetchExprW(expr.r_, nextBase); }

template<typename E>
inline void prefetchExprW(
    const vector::expr::detail::Normalize<E>& expr, idx_t nextBase)
{ prefetchExprW(expr.e_, nextBase); }

template<typename E>
inline void prefetchExprW(
    const vector::expr::detail::QuatConj<E>& expr, idx_t nextBase)
{ prefetchExprW(expr.e_, nextBase); }

template<typename E>
inline void prefetchExprW(
    const vector::expr::detail::QuatAntiInvolute<E>& expr, idx_t nextBase)
{ prefetchExprW(expr.e_, nextBase); }

template<typename L, typename R>
inline void prefetchExprW(
    const vector::expr::detail::QuatMul<L, R>& expr, idx_t nextBase)
{ prefetchExprW(expr.l_, nextBase); prefetchExprW(expr.r_, nextBase); }

template<typename Q, typename V>
inline void prefetchExprW(
    const vector::expr::detail::QuatRotate<Q, V>& expr, idx_t nextBase)
{ prefetchExprW(expr.q_, nextBase); prefetchExprW(expr.v_, nextBase); }

// Write-intent no-ops for register-resident / constant leaves
template<typename DataT, dims_t VD>
inline void prefetchExprW(const vector::Item<DataT, VD>&, idx_t) {}
template<typename DataT, dims_t VD>
inline void prefetchExprW(const vector::OnesNT<DataT, VD>&, idx_t) {}
template<typename DataT, dims_t VD>
inline void prefetchExprW(const vector::ZerosNT<DataT, VD>&, idx_t) {}
template<typename DataT>
inline void prefetchExprW(const vector::NeutralQuaternion<DataT>&, idx_t) {}
template<typename DataT, dims_t VD, idx_t W>
inline void prefetchExprW(const vector::PacketItem<DataT, VD, W>&, idx_t) {}


// ═════════════════════════════════════════════════════════════════════════════
//  Size propagation utilities
// ═════════════════════════════════════════════════════════════════════════════

/**
 * @brief Extract the runtime size (number of samples) from an expression.
 *
 * Walks the tree to find a RefArray leaf with a known size.
 * For Item/constant expressions, returns 0 (caller should provide N).
 */
template<typename ExprT>
inline idx_t exprSize(const ExprT& expr)
{
    if constexpr (detail::IsRefArrayLike<ExprT>::value)
        return expr.size();
    else
        return 0;
}

template<typename L, typename R, bool sub>
inline idx_t exprSize(const vector::expr::detail::Sum<L, R, sub>& expr)
{
    idx_t ls = exprSize(expr.l_);
    return (ls > 0) ? ls : exprSize(expr.r_);
}

template<typename Expr, typename ScaleT>
inline idx_t exprSize(const vector::expr::detail::CWiseScale<Expr, ScaleT>& expr)
{
    return exprSize(expr.expr_);
}

template<typename Expr, typename ScaleT>
inline idx_t exprSize(const vector::expr::detail::CWiseRScale<Expr, ScaleT>& expr)
{
    return exprSize(expr.expr_);
}

template<typename Expr, typename ScaleT>
inline idx_t exprSize(const vector::expr::detail::CWiseInverse<Expr, ScaleT>& expr)
{
    return exprSize(expr.expr_);
}

template<typename L, typename R>
inline idx_t exprSize(const vector::expr::detail::CWiseMult<L, R>& expr)
{
    idx_t ls = exprSize(expr.l_);
    return (ls > 0) ? ls : exprSize(expr.r_);
}

template<typename L, typename R>
inline idx_t exprSize(const vector::expr::detail::CWiseDiv<L, R>& expr)
{
    idx_t ls = exprSize(expr.l_);
    return (ls > 0) ? ls : exprSize(expr.r_);
}

template<typename L, typename R>
inline idx_t exprSize(const vector::expr::detail::CWiseMax<L, R>& expr)
{
    idx_t ls = exprSize(expr.l_);
    return (ls > 0) ? ls : exprSize(expr.r_);
}

template<typename L, typename R>
inline idx_t exprSize(const vector::expr::detail::Cross<L, R>& expr)
{
    idx_t ls = exprSize(expr.l_);
    return (ls > 0) ? ls : exprSize(expr.r_);
}

template<typename Q, typename V>
inline idx_t exprSize(const vector::expr::detail::QuatRotate<Q, V>& expr)
{
    idx_t qs = exprSize(expr.q_);
    return (qs > 0) ? qs : exprSize(expr.v_);
}

template<typename L, typename R>
inline idx_t exprSize(const vector::expr::detail::QuatMul<L, R>& expr)
{
    idx_t ls = exprSize(expr.l_);
    return (ls > 0) ? ls : exprSize(expr.r_);
}

template<typename E>
inline idx_t exprSize(const vector::expr::detail::Normalize<E>& expr)
{
    return exprSize(expr.e_);
}

template<typename E>
inline idx_t exprSize(const vector::expr::detail::CWiseAbs<E>& expr)
{
    return exprSize(expr.expr_);
}

template<typename E>
inline idx_t exprSize(const vector::expr::detail::QuatConj<E>& expr)
{
    return exprSize(expr.e_);
}

template<typename E>
inline idx_t exprSize(const vector::expr::detail::QuatAntiInvolute<E>& expr)
{
    return exprSize(expr.e_);
}

// Items and constants have no runtime size
template<typename DataT, dims_t VD>
inline idx_t exprSize(const vector::Item<DataT, VD>&) { return 0; }

template<typename DataT, dims_t VD>
inline idx_t exprSize(const vector::OnesNT<DataT, VD>&) { return 0; }

template<typename DataT, dims_t VD>
inline idx_t exprSize(const vector::ZerosNT<DataT, VD>&) { return 0; }

template<typename DataT>
inline idx_t exprSize(const vector::NeutralQuaternion<DataT>&) { return 0; }

template<typename DataT, dims_t VD, idx_t W>
inline idx_t exprSize(const vector::PacketItem<DataT, VD, W>&) { return 0; }

} // namespace packet
} // namespace cpu
} // namespace feta
