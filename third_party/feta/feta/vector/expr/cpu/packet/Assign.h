#pragma once

/**
 * @file Assign.h
 * @brief Packet assignment machinery — `materializePacket`, the generic
 *        `RecursivePacketAssign`/`MaskedRecursivePacketAssign`, and the
 *        per-node multi-load specializations (Cross/Normalize/QuatMul/
 *        QuatRotate/Sum/CWiseScale + the `SumAssignImpl` helper).
 *
 * Part of the `PacketOps.h` facade; depends on the load/store dispatchers.
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
//  materializePacket — capture an expression into a register-resident PacketItem
// ═════════════════════════════════════════════════════════════════════════════

namespace detail {

/**
 * @brief Recursively capture an expression into a PacketItem (compile-time
 * unrolled from dim down to 0).
 */
template<dims_t dim, typename ExprT, idx_t W>
struct MaterializeHelper {
    using DataT = typename ExprT::ComponentT;
    inline static void eval(
        const PacketIndex<W>& pi, const ExprT& expr,
        vector::PacketItem<DataT, ExprT::VecDims, W>& out)
    {
        out.template packet<dim>() = packetGet<dim, W>(expr, pi);
        MaterializeHelper<dim - 1, ExprT, W>::eval(pi, expr, out);
    }
};

template<typename ExprT, idx_t W>
struct MaterializeHelper<0, ExprT, W> {
    using DataT = typename ExprT::ComponentT;
    inline static void eval(
        const PacketIndex<W>& pi, const ExprT& expr,
        vector::PacketItem<DataT, ExprT::VecDims, W>& out)
    {
        out.template packet<0>() = packetGet<0, W>(expr, pi);
    }
};

} // namespace detail

/**
 * @brief Materialise an expression into a register-resident PacketItem.
 *
 * If the expression is already cheap (leaf / PacketItem), this is compiled
 * to a few register copies.  For expensive multi-load nodes (Cross, QuatMul,
 * QuatRotate, Normalize) it evaluates once and caches all components.
 */
template<typename ExprT, idx_t W>
inline auto materializePacket(const ExprT& expr, const PacketIndex<W>& pi)
{
    using DataT = typename ExprT::ComponentT;
    vector::PacketItem<DataT, ExprT::VecDims, W> item;
    detail::MaterializeHelper<ExprT::VecDims - 1, ExprT, W>::eval(pi, expr, item);
    return item;
}


// ═════════════════════════════════════════════════════════════════════════════
//  RecursivePacketAssign — compile-time unrolled packet assignment
// ═════════════════════════════════════════════════════════════════════════════

/**
 * @brief Unmasked packet assign: evaluates `from` and stores into `to`
 * for components `dim` down to 0.
 */
template<dims_t dim, typename ExprL, typename ExprR, idx_t W>
struct RecursivePacketAssign {
    inline static void eval(
        const PacketIndex<W>& pi, ExprL& to, const ExprR& from)
    {
        auto pkt = packetGet<dim, W>(from, pi);
        packetStore<dim, W>(to, pi, pkt);
        RecursivePacketAssign<dim - 1, ExprL, ExprR, W>::eval(pi, to, from);
    }
};

template<typename ExprL, typename ExprR, idx_t W>
struct RecursivePacketAssign<0, ExprL, ExprR, W> {
    inline static void eval(
        const PacketIndex<W>& pi, ExprL& to, const ExprR& from)
    {
        auto pkt = packetGet<0, W>(from, pi);
        packetStore<0, W>(to, pi, pkt);
    }
};

/**
 * @brief Masked packet assign: only stores to lanes where `mask` is true.
 * Analogous to CUDA early-return: inactive lanes are no-ops.
 */
template<dims_t dim, typename ExprL, typename ExprR, idx_t W, typename DataT>
struct MaskedRecursivePacketAssign {
    inline static void eval(
        const PacketIndex<W>& pi, const PacketMask<DataT, W>& mask,
        ExprL& to, const ExprR& from)
    {
        auto pkt = packetGet<dim, W>(from, pi);
        packetMaskedStore<dim, W>(to, pi, pkt, mask);
        MaskedRecursivePacketAssign<dim - 1, ExprL, ExprR, W, DataT>::eval(
            pi, mask, to, from);
    }
};

template<typename ExprL, typename ExprR, idx_t W, typename DataT>
struct MaskedRecursivePacketAssign<0, ExprL, ExprR, W, DataT> {
    inline static void eval(
        const PacketIndex<W>& pi, const PacketMask<DataT, W>& mask,
        ExprL& to, const ExprR& from)
    {
        auto pkt = packetGet<0, W>(from, pi);
        packetMaskedStore<0, W>(to, pi, pkt, mask);
    }
};


// ═════════════════════════════════════════════════════════════════════════════
//  Specialised RecursivePacketAssign — multi-output ops computed in one shot
//  to eliminate redundant sub-expression loads.
// ═════════════════════════════════════════════════════════════════════════════

// ─── MULTI-LOAD NODE REGISTRY ─── one node ⇒ up to three recursive forms ────
// Each multi-load node (Cross, Normalize, QuatMul, QuatRotate, Sum, CWiseScale)
// loads every child once and writes all outputs together. The three forms live
// in separate facades on purpose — the scalar form must stay device-light, the
// packet forms pull <omp.h> + x86 SIMD, so they cannot share a header (F1 audit):
//   1. scalar / GPU     RecursiveAssign<dim, ExprL, Node>
//                         @ feta/vector/expr/nodes/AssignMultiLoad.h
//   2. CPU-SIMD assign   RecursivePacketAssign<dim, ExprL, Node, W>
//                         @ feta/vector/expr/cpu/packet/Assign.h          (this file)
//   3. CPU-SIMD capture  RecursivePacketCapture<dim, Node, W>
//                         @ feta/vector/expr/cpu/tiled/Capture.h
// TO ADD A NODE: add its spec under a matching "<Node>" banner in ALL THREE.
// ───────────────────────────────────────────────────────────────────────────

// ─── Cross (3D): 6 loads, 3 FMA stores instead of 18 loads ─────────────────

template<typename ExprL, typename L, typename R, idx_t W>
struct RecursivePacketAssign<2, ExprL, vector::expr::detail::Cross<L, R>, W> {
    inline static void eval(
        const PacketIndex<W>& pi, ExprL& to,
        const vector::expr::detail::Cross<L, R>& from)
    {
        auto l0 = packetGet<0, W>(from.l_, pi);
        auto l1 = packetGet<1, W>(from.l_, pi);
        auto l2 = packetGet<2, W>(from.l_, pi);
        auto r0 = packetGet<0, W>(from.r_, pi);
        auto r1 = packetGet<1, W>(from.r_, pi);
        auto r2 = packetGet<2, W>(from.r_, pi);

        packetStore<0, W>(to, pi, fmsub(l1, r2, l2 * r1));
        packetStore<1, W>(to, pi, fmsub(l2, r0, l0 * r2));
        packetStore<2, W>(to, pi, fmsub(l0, r1, l1 * r0));
    }
};

// ─── Normalize (3D): 1 sqrt instead of 3 redundant sqrts ───────────────────

template<typename ExprL, typename E, idx_t W>
struct RecursivePacketAssign<2, ExprL, vector::expr::detail::Normalize<E>, W> {
    inline static void eval(
        const PacketIndex<W>& pi, ExprL& to,
        const vector::expr::detail::Normalize<E>& from)
    {
        using DataT = typename E::ComponentT;
        auto c0 = packetGet<0, W>(from.e_, pi);
        auto c1 = packetGet<1, W>(from.e_, pi);
        auto c2 = packetGet<2, W>(from.e_, pi);

        auto sn = fmadd(c0, c0, c1 * c1);
        sn = fmadd(c2, c2, sn);
        auto rn = Packet<DataT, W>::broadcast(DataT{1}) / sqrt(sn);

        packetStore<0, W>(to, pi, c0 * rn);
        packetStore<1, W>(to, pi, c1 * rn);
        packetStore<2, W>(to, pi, c2 * rn);
    }
};

// ─── Normalize (4D — quaternions): 1 sqrt instead of 4 redundant sqrts ─────

template<typename ExprL, typename E, idx_t W>
struct RecursivePacketAssign<3, ExprL, vector::expr::detail::Normalize<E>, W> {
    inline static void eval(
        const PacketIndex<W>& pi, ExprL& to,
        const vector::expr::detail::Normalize<E>& from)
    {
        using DataT = typename E::ComponentT;
        auto c0 = packetGet<0, W>(from.e_, pi);
        auto c1 = packetGet<1, W>(from.e_, pi);
        auto c2 = packetGet<2, W>(from.e_, pi);
        auto c3 = packetGet<3, W>(from.e_, pi);

        auto sn = fmadd(c0, c0, c1 * c1);
        sn = fmadd(c2, c2, sn);
        sn = fmadd(c3, c3, sn);
        auto rn = Packet<DataT, W>::broadcast(DataT{1}) / sqrt(sn);

        packetStore<0, W>(to, pi, c0 * rn);
        packetStore<1, W>(to, pi, c1 * rn);
        packetStore<2, W>(to, pi, c2 * rn);
        packetStore<3, W>(to, pi, c3 * rn);
    }
};

// ─── QuatMul: 8 loads, 4 FMA stores instead of 32 loads ────────────────────

template<typename ExprL, typename L, typename R, idx_t W>
struct RecursivePacketAssign<3, ExprL, vector::expr::detail::QuatMul<L, R>, W> {
    inline static void eval(
        const PacketIndex<W>& pi, ExprL& to,
        const vector::expr::detail::QuatMul<L, R>& from)
    {
        auto l0 = packetGet<0, W>(from.l_, pi);
        auto l1 = packetGet<1, W>(from.l_, pi);
        auto l2 = packetGet<2, W>(from.l_, pi);
        auto l3 = packetGet<3, W>(from.l_, pi);
        auto r0 = packetGet<0, W>(from.r_, pi);
        auto r1 = packetGet<1, W>(from.r_, pi);
        auto r2 = packetGet<2, W>(from.r_, pi);
        auto r3 = packetGet<3, W>(from.r_, pi);

        // w: l0*r0 - l1*r1 - l2*r2 - l3*r3
        packetStore<0, W>(to, pi,
            fmsub(l0, r0, l1 * r1) - fmadd(l2, r2, l3 * r3));
        // x: l0*r1 + l1*r0 + l2*r3 - l3*r2
        packetStore<1, W>(to, pi,
            fmadd(l0, r1, l1 * r0) + fmsub(l2, r3, l3 * r2));
        // y: l0*r2 - l1*r3 + l2*r0 + l3*r1
        packetStore<2, W>(to, pi,
            fmsub(l0, r2, l1 * r3) + fmadd(l2, r0, l3 * r1));
        // z: l0*r3 + l1*r2 - l2*r1 + l3*r0
        packetStore<3, W>(to, pi,
            fmadd(l0, r3, l1 * r2) + fmsub(l3, r0, l2 * r1));
    }
};

// ─── QuatRotate: 7 loads, 3 stores instead of 21 loads ─────────────────────

template<typename ExprL, typename Q, typename V, idx_t W>
struct RecursivePacketAssign<2, ExprL, vector::expr::detail::QuatRotate<Q, V>, W> {
    inline static void eval(
        const PacketIndex<W>& pi, ExprL& to,
        const vector::expr::detail::QuatRotate<Q, V>& from)
    {
        using DataT = typename Q::ComponentT;
        auto w  = packetGet<0, W>(from.q_, pi);
        auto ux = packetGet<1, W>(from.q_, pi);
        auto uy = packetGet<2, W>(from.q_, pi);
        auto uz = packetGet<3, W>(from.q_, pi);

        auto vx = packetGet<0, W>(from.v_, pi);
        auto vy = packetGet<1, W>(from.v_, pi);
        auto vz = packetGet<2, W>(from.v_, pi);

        auto two = Packet<DataT, W>::broadcast(DataT{2});
        auto tx = two * fmsub(uy, vz, uz * vy);
        auto ty = two * fmsub(uz, vx, ux * vz);
        auto tz = two * fmsub(ux, vy, uy * vx);

        packetStore<0, W>(to, pi, fmadd(w, tx, vx) + fmsub(uy, tz, uz * ty));
        packetStore<1, W>(to, pi, fmadd(w, ty, vy) + fmsub(uz, tx, ux * tz));
        packetStore<2, W>(to, pi, fmadd(w, tz, vz) + fmsub(ux, ty, uy * tx));
    }
};


// ─── Sum with expensive child: auto-materialize then per-dim store ──────────
// When either child of Sum is a multi-load expression (Cross, QuatMul,
// QuatRotate, Normalize), materialise the expensive child into a PacketItem
// first so its sub-expression loads happen once, not VecDims times.

namespace detail {

/**
 * @brief Evaluate a Sum expression where at least one child is expensive,
 * materialising expensive children into PacketItems before per-dim store.
 */
template<dims_t VD, typename ExprL, typename L, typename R, bool sub, idx_t W>
struct SumAssignImpl {
    using SumT = vector::expr::detail::Sum<L, R, sub>;
    using DataT = typename L::ComponentT;

    inline static void eval(
        const PacketIndex<W>& pi, ExprL& to, const SumT& from)
    {
        // Materialise expensive children; cheap ones stay as-is.
        // constexpr dispatch: only materialise if IsMultiLoadExpr.
        if constexpr (IsMultiLoadExpr<L>::value && IsMultiLoadExpr<R>::value) {
            auto lm = materializePacket(from.l_, pi);
            auto rm = materializePacket(from.r_, pi);
            storeSumDims(pi, to, lm, rm,
                std::make_integer_sequence<dims_t, VD>{});
        } else if constexpr (IsMultiLoadExpr<L>::value) {
            auto lm = materializePacket(from.l_, pi);
            storeSumDims(pi, to, lm, from.r_,
                std::make_integer_sequence<dims_t, VD>{});
        } else {
            auto rm = materializePacket(from.r_, pi);
            storeSumDims(pi, to, from.l_, rm,
                std::make_integer_sequence<dims_t, VD>{});
        }
    }

private:
    template<typename LExpr, typename RExpr, dims_t... Is>
    inline static void storeSumDims(
        const PacketIndex<W>& pi, ExprL& to,
        const LExpr& l, const RExpr& r,
        std::integer_sequence<dims_t, Is...>)
    {
        ((storeSumDim<Is>(pi, to, l, r)), ...);
    }

    template<dims_t d, typename LExpr, typename RExpr>
    inline static void storeSumDim(
        const PacketIndex<W>& pi, ExprL& to,
        const LExpr& l, const RExpr& r)
    {
        auto lp = packetGet<d, W>(l, pi);
        auto rp = packetGet<d, W>(r, pi);
        if constexpr (sub)
            packetStore<d, W>(to, pi, lp - rp);
        else
            packetStore<d, W>(to, pi, lp + rp);
    }
};

} // namespace detail

// 3D Sum with expensive right child (e.g. v + cross(a,b))
template<typename ExprL, typename L, typename R, bool sub, idx_t W>
    requires(detail::IsMultiLoadExpr<R>::value && !detail::IsMultiLoadExpr<L>::value
             && R::VecDims == 3)
struct RecursivePacketAssign<2, ExprL, vector::expr::detail::Sum<L, R, sub>, W> {
    inline static void eval(
        const PacketIndex<W>& pi, ExprL& to,
        const vector::expr::detail::Sum<L, R, sub>& from)
    {
        detail::SumAssignImpl<3, ExprL, L, R, sub, W>::eval(pi, to, from);
    }
};

// 3D Sum with expensive left child (e.g. cross(a,b) + v)
template<typename ExprL, typename L, typename R, bool sub, idx_t W>
    requires(detail::IsMultiLoadExpr<L>::value && !detail::IsMultiLoadExpr<R>::value
             && L::VecDims == 3)
struct RecursivePacketAssign<2, ExprL, vector::expr::detail::Sum<L, R, sub>, W> {
    inline static void eval(
        const PacketIndex<W>& pi, ExprL& to,
        const vector::expr::detail::Sum<L, R, sub>& from)
    {
        detail::SumAssignImpl<3, ExprL, L, R, sub, W>::eval(pi, to, from);
    }
};

// 3D Sum with both children expensive (e.g. cross(a,b) + quatRotate(q,v))
template<typename ExprL, typename L, typename R, bool sub, idx_t W>
    requires(detail::IsMultiLoadExpr<L>::value && detail::IsMultiLoadExpr<R>::value
             && L::VecDims == 3)
struct RecursivePacketAssign<2, ExprL, vector::expr::detail::Sum<L, R, sub>, W> {
    inline static void eval(
        const PacketIndex<W>& pi, ExprL& to,
        const vector::expr::detail::Sum<L, R, sub>& from)
    {
        detail::SumAssignImpl<3, ExprL, L, R, sub, W>::eval(pi, to, from);
    }
};

// 4D Sum with expensive right child
template<typename ExprL, typename L, typename R, bool sub, idx_t W>
    requires(detail::IsMultiLoadExpr<R>::value && !detail::IsMultiLoadExpr<L>::value
             && R::VecDims == 4)
struct RecursivePacketAssign<3, ExprL, vector::expr::detail::Sum<L, R, sub>, W> {
    inline static void eval(
        const PacketIndex<W>& pi, ExprL& to,
        const vector::expr::detail::Sum<L, R, sub>& from)
    {
        detail::SumAssignImpl<4, ExprL, L, R, sub, W>::eval(pi, to, from);
    }
};

// 4D Sum with expensive left child
template<typename ExprL, typename L, typename R, bool sub, idx_t W>
    requires(detail::IsMultiLoadExpr<L>::value && !detail::IsMultiLoadExpr<R>::value
             && L::VecDims == 4)
struct RecursivePacketAssign<3, ExprL, vector::expr::detail::Sum<L, R, sub>, W> {
    inline static void eval(
        const PacketIndex<W>& pi, ExprL& to,
        const vector::expr::detail::Sum<L, R, sub>& from)
    {
        detail::SumAssignImpl<4, ExprL, L, R, sub, W>::eval(pi, to, from);
    }
};

// 4D Sum with both children expensive
template<typename ExprL, typename L, typename R, bool sub, idx_t W>
    requires(detail::IsMultiLoadExpr<L>::value && detail::IsMultiLoadExpr<R>::value
             && L::VecDims == 4)
struct RecursivePacketAssign<3, ExprL, vector::expr::detail::Sum<L, R, sub>, W> {
    inline static void eval(
        const PacketIndex<W>& pi, ExprL& to,
        const vector::expr::detail::Sum<L, R, sub>& from)
    {
        detail::SumAssignImpl<4, ExprL, L, R, sub, W>::eval(pi, to, from);
    }
};

// ─── CWiseScale with expensive child: scalar * cross(a,b) etc. ─────────────

template<typename ExprL, typename Expr, typename ScaleT, idx_t W>
    requires(detail::IsMultiLoadExpr<Expr>::value && Expr::VecDims == 3)
struct RecursivePacketAssign<2, ExprL, vector::expr::detail::CWiseScale<Expr, ScaleT>, W> {
    inline static void eval(
        const PacketIndex<W>& pi, ExprL& to,
        const vector::expr::detail::CWiseScale<Expr, ScaleT>& from)
    {
        using DataT = typename Expr::ComponentT;
        auto mat = materializePacket(from.expr_, pi);
        auto s = Packet<DataT, W>::broadcast(static_cast<DataT>(from.factor_));
        packetStore<0, W>(to, pi, s * mat.template packet<0>());
        packetStore<1, W>(to, pi, s * mat.template packet<1>());
        packetStore<2, W>(to, pi, s * mat.template packet<2>());
    }
};

template<typename ExprL, typename Expr, typename ScaleT, idx_t W>
    requires(detail::IsMultiLoadExpr<Expr>::value && Expr::VecDims == 4)
struct RecursivePacketAssign<3, ExprL, vector::expr::detail::CWiseScale<Expr, ScaleT>, W> {
    inline static void eval(
        const PacketIndex<W>& pi, ExprL& to,
        const vector::expr::detail::CWiseScale<Expr, ScaleT>& from)
    {
        using DataT = typename Expr::ComponentT;
        auto mat = materializePacket(from.expr_, pi);
        auto s = Packet<DataT, W>::broadcast(static_cast<DataT>(from.factor_));
        packetStore<0, W>(to, pi, s * mat.template packet<0>());
        packetStore<1, W>(to, pi, s * mat.template packet<1>());
        packetStore<2, W>(to, pi, s * mat.template packet<2>());
        packetStore<3, W>(to, pi, s * mat.template packet<3>());
    }
};

} // namespace packet
} // namespace cpu
} // namespace feta
