#pragma once

/**
 * @file Capture.h
 * @brief Packet capture / assign — the `RecursivePacketCapture` driver +
 *        its multi-load specializations (Cross/QuatMul/QuatRotate/Normalize/
 *        Sum + `SumCaptureHelper`), and the `packetCapture`/`packetAssign`
 *        front-ends.
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
#include "feta/vector/expr/cpu/tiled/PacketEval.h"

namespace feta {
namespace cpu {

// ─── PacketItem capture / assign ─────────────────────────────────────────────

namespace detail {

/**
 * @brief Compile-time unrolled capture: evaluates packetGet<dim> for each
 * component and stores into a PacketItem.
 */
template<dims_t dim, typename Expr, idx_t W>
struct RecursivePacketCapture {
    using DataT = typename Expr::ComponentT;
    inline static void eval(
        const PacketIndex<W>& pi,
        vector::PacketItem<DataT, Expr::VecDims, W>& out,
        const Expr& from)
    {
        out.template packet<dim>() = packet::packetGet<dim, W>(from, pi);
        RecursivePacketCapture<dim - 1, Expr, W>::eval(pi, out, from);
    }
};

template<typename Expr, idx_t W>
struct RecursivePacketCapture<0, Expr, W> {
    using DataT = typename Expr::ComponentT;
    inline static void eval(
        const PacketIndex<W>& pi,
        vector::PacketItem<DataT, Expr::VecDims, W>& out,
        const Expr& from)
    {
        out.template packet<0>() = packet::packetGet<0, W>(from, pi);
    }
};

// ─── MULTI-LOAD NODE REGISTRY ─── one node ⇒ up to three recursive forms ────
// Each multi-load node (Cross, Normalize, QuatMul, QuatRotate, Sum, CWiseScale)
// loads every child once and writes all outputs together. The three forms live
// in separate facades on purpose — the scalar form must stay device-light, the
// packet forms pull <omp.h> + x86 SIMD, so they cannot share a header (F1 audit):
//   1. scalar / GPU     RecursiveAssign<dim, ExprL, Node>
//                         @ feta/vector/expr/nodes/AssignMultiLoad.h
//   2. CPU-SIMD assign   RecursivePacketAssign<dim, ExprL, Node, W>
//                         @ feta/vector/expr/cpu/packet/Assign.h
//   3. CPU-SIMD capture  RecursivePacketCapture<dim, Node, W>
//                         @ feta/vector/expr/cpu/tiled/Capture.h         (this file)
// TO ADD A NODE: add its spec under a matching "<Node>" banner in ALL THREE.
// ───────────────────────────────────────────────────────────────────────────

// ─── Specialised captures: compute all dims in one shot ─────────────────────

// Cross (3D): 6 loads instead of 18
template<typename L, typename R, idx_t W>
struct RecursivePacketCapture<2, vector::expr::detail::Cross<L, R>, W> {
    using Expr = vector::expr::detail::Cross<L, R>;
    using DataT = typename Expr::ComponentT;
    inline static void eval(
        const PacketIndex<W>& pi,
        vector::PacketItem<DataT, 3, W>& out,
        const Expr& from)
    {
        auto l0 = packet::packetGet<0, W>(from.l_, pi);
        auto l1 = packet::packetGet<1, W>(from.l_, pi);
        auto l2 = packet::packetGet<2, W>(from.l_, pi);
        auto r0 = packet::packetGet<0, W>(from.r_, pi);
        auto r1 = packet::packetGet<1, W>(from.r_, pi);
        auto r2 = packet::packetGet<2, W>(from.r_, pi);
        out.template packet<0>() = fmsub(l1, r2, l2 * r1);
        out.template packet<1>() = fmsub(l2, r0, l0 * r2);
        out.template packet<2>() = fmsub(l0, r1, l1 * r0);
    }
};

// QuatMul (4D): 8 loads instead of 32
template<typename L, typename R, idx_t W>
struct RecursivePacketCapture<3, vector::expr::detail::QuatMul<L, R>, W> {
    using Expr = vector::expr::detail::QuatMul<L, R>;
    using DataT = typename Expr::ComponentT;
    inline static void eval(
        const PacketIndex<W>& pi,
        vector::PacketItem<DataT, 4, W>& out,
        const Expr& from)
    {
        auto l0 = packet::packetGet<0, W>(from.l_, pi);
        auto l1 = packet::packetGet<1, W>(from.l_, pi);
        auto l2 = packet::packetGet<2, W>(from.l_, pi);
        auto l3 = packet::packetGet<3, W>(from.l_, pi);
        auto r0 = packet::packetGet<0, W>(from.r_, pi);
        auto r1 = packet::packetGet<1, W>(from.r_, pi);
        auto r2 = packet::packetGet<2, W>(from.r_, pi);
        auto r3 = packet::packetGet<3, W>(from.r_, pi);
        out.template packet<0>() = fmsub(l0, r0, l1 * r1) - fmadd(l2, r2, l3 * r3);
        out.template packet<1>() = fmadd(l0, r1, l1 * r0) + fmsub(l2, r3, l3 * r2);
        out.template packet<2>() = fmsub(l0, r2, l1 * r3) + fmadd(l2, r0, l3 * r1);
        out.template packet<3>() = fmadd(l0, r3, l1 * r2) + fmsub(l3, r0, l2 * r1);
    }
};

// QuatRotate (3D): 7 loads instead of 21
template<typename Q, typename V, idx_t W>
struct RecursivePacketCapture<2, vector::expr::detail::QuatRotate<Q, V>, W> {
    using Expr = vector::expr::detail::QuatRotate<Q, V>;
    using DataT = typename Expr::ComponentT;
    inline static void eval(
        const PacketIndex<W>& pi,
        vector::PacketItem<DataT, 3, W>& out,
        const Expr& from)
    {
        auto w  = packet::packetGet<0, W>(from.q_, pi);
        auto ux = packet::packetGet<1, W>(from.q_, pi);
        auto uy = packet::packetGet<2, W>(from.q_, pi);
        auto uz = packet::packetGet<3, W>(from.q_, pi);
        auto vx = packet::packetGet<0, W>(from.v_, pi);
        auto vy = packet::packetGet<1, W>(from.v_, pi);
        auto vz = packet::packetGet<2, W>(from.v_, pi);

        auto two = simd::Packet<DataT, W>::broadcast(DataT{2});
        auto tx = two * fmsub(uy, vz, uz * vy);
        auto ty = two * fmsub(uz, vx, ux * vz);
        auto tz = two * fmsub(ux, vy, uy * vx);

        out.template packet<0>() = fmadd(w, tx, vx) + fmsub(uy, tz, uz * ty);
        out.template packet<1>() = fmadd(w, ty, vy) + fmsub(uz, tx, ux * tz);
        out.template packet<2>() = fmadd(w, tz, vz) + fmsub(ux, ty, uy * tx);
    }
};

// Normalize (3D): 1 sqrt instead of 3
template<typename E, idx_t W>
struct RecursivePacketCapture<2, vector::expr::detail::Normalize<E>, W> {
    using Expr = vector::expr::detail::Normalize<E>;
    using DataT = typename Expr::ComponentT;
    inline static void eval(
        const PacketIndex<W>& pi,
        vector::PacketItem<DataT, 3, W>& out,
        const Expr& from)
    {
        auto c0 = packet::packetGet<0, W>(from.e_, pi);
        auto c1 = packet::packetGet<1, W>(from.e_, pi);
        auto c2 = packet::packetGet<2, W>(from.e_, pi);
        auto sn = fmadd(c0, c0, c1 * c1);
        sn = fmadd(c2, c2, sn);
        auto rn = simd::Packet<DataT, W>::broadcast(DataT{1}) / sqrt(sn);
        out.template packet<0>() = c0 * rn;
        out.template packet<1>() = c1 * rn;
        out.template packet<2>() = c2 * rn;
    }
};

// Normalize (4D): 1 sqrt instead of 4
template<typename E, idx_t W>
struct RecursivePacketCapture<3, vector::expr::detail::Normalize<E>, W> {
    using Expr = vector::expr::detail::Normalize<E>;
    using DataT = typename Expr::ComponentT;
    inline static void eval(
        const PacketIndex<W>& pi,
        vector::PacketItem<DataT, 4, W>& out,
        const Expr& from)
    {
        auto c0 = packet::packetGet<0, W>(from.e_, pi);
        auto c1 = packet::packetGet<1, W>(from.e_, pi);
        auto c2 = packet::packetGet<2, W>(from.e_, pi);
        auto c3 = packet::packetGet<3, W>(from.e_, pi);
        auto sn = fmadd(c0, c0, c1 * c1);
        sn = fmadd(c2, c2, sn);
        sn = fmadd(c3, c3, sn);
        auto rn = simd::Packet<DataT, W>::broadcast(DataT{1}) / sqrt(sn);
        out.template packet<0>() = c0 * rn;
        out.template packet<1>() = c1 * rn;
        out.template packet<2>() = c2 * rn;
        out.template packet<3>() = c3 * rn;
    }
};

// ─── Sum with expensive child: materialise expensive side first ─────────────

template<typename SumExpr, idx_t W, dims_t VD>
struct SumCaptureHelper {
    using DataT = typename SumExpr::ComponentT;
    using L = std::remove_cvref_t<decltype(std::declval<SumExpr>().l_)>;
    using R = std::remove_cvref_t<decltype(std::declval<SumExpr>().r_)>;
    static constexpr bool sub = SumExpr::isSub;

    inline static void eval(
        const PacketIndex<W>& pi,
        vector::PacketItem<DataT, VD, W>& out,
        const SumExpr& from)
    {
        if constexpr (packet::detail::IsMultiLoadExpr<L>::value &&
                      packet::detail::IsMultiLoadExpr<R>::value) {
            auto lm = packet::materializePacket(from.l_, pi);
            auto rm = packet::materializePacket(from.r_, pi);
            storeDims(pi, out, lm, rm, std::make_integer_sequence<dims_t, VD>{});
        } else if constexpr (packet::detail::IsMultiLoadExpr<L>::value) {
            auto lm = packet::materializePacket(from.l_, pi);
            storeDims(pi, out, lm, from.r_, std::make_integer_sequence<dims_t, VD>{});
        } else {
            auto rm = packet::materializePacket(from.r_, pi);
            storeDims(pi, out, from.l_, rm, std::make_integer_sequence<dims_t, VD>{});
        }
    }

private:
    template<typename LE, typename RE, dims_t... Is>
    inline static void storeDims(
        const PacketIndex<W>& pi,
        vector::PacketItem<DataT, VD, W>& out,
        const LE& l, const RE& r,
        std::integer_sequence<dims_t, Is...>)
    {
        ((storeDim<Is>(pi, out, l, r)), ...);
    }

    template<dims_t d, typename LE, typename RE>
    inline static void storeDim(
        const PacketIndex<W>& pi,
        vector::PacketItem<DataT, VD, W>& out,
        const LE& l, const RE& r)
    {
        auto lp = packet::packetGet<d, W>(l, pi);
        auto rp = packet::packetGet<d, W>(r, pi);
        if constexpr (sub)
            out.template packet<d>() = lp - rp;
        else
            out.template packet<d>() = lp + rp;
    }
};

// 3D Sum capture with expensive right child
template<typename L, typename R, bool sub, idx_t W>
    requires(packet::detail::IsMultiLoadExpr<R>::value
             && !packet::detail::IsMultiLoadExpr<L>::value)
struct RecursivePacketCapture<2, vector::expr::detail::Sum<L, R, sub>, W> {
    using Expr = vector::expr::detail::Sum<L, R, sub>;
    using DataT = typename Expr::ComponentT;
    inline static void eval(
        const PacketIndex<W>& pi,
        vector::PacketItem<DataT, 3, W>& out,
        const Expr& from)
    {
        SumCaptureHelper<Expr, W, 3>::eval(pi, out, from);
    }
};

// 3D Sum capture with expensive left child
template<typename L, typename R, bool sub, idx_t W>
    requires(packet::detail::IsMultiLoadExpr<L>::value
             && !packet::detail::IsMultiLoadExpr<R>::value)
struct RecursivePacketCapture<2, vector::expr::detail::Sum<L, R, sub>, W> {
    using Expr = vector::expr::detail::Sum<L, R, sub>;
    using DataT = typename Expr::ComponentT;
    inline static void eval(
        const PacketIndex<W>& pi,
        vector::PacketItem<DataT, 3, W>& out,
        const Expr& from)
    {
        SumCaptureHelper<Expr, W, 3>::eval(pi, out, from);
    }
};

// 3D Sum capture with both children expensive
template<typename L, typename R, bool sub, idx_t W>
    requires(packet::detail::IsMultiLoadExpr<L>::value
             && packet::detail::IsMultiLoadExpr<R>::value)
struct RecursivePacketCapture<2, vector::expr::detail::Sum<L, R, sub>, W> {
    using Expr = vector::expr::detail::Sum<L, R, sub>;
    using DataT = typename Expr::ComponentT;
    inline static void eval(
        const PacketIndex<W>& pi,
        vector::PacketItem<DataT, 3, W>& out,
        const Expr& from)
    {
        SumCaptureHelper<Expr, W, 3>::eval(pi, out, from);
    }
};

// 4D Sum capture with expensive right child
template<typename L, typename R, bool sub, idx_t W>
    requires(packet::detail::IsMultiLoadExpr<R>::value
             && !packet::detail::IsMultiLoadExpr<L>::value)
struct RecursivePacketCapture<3, vector::expr::detail::Sum<L, R, sub>, W> {
    using Expr = vector::expr::detail::Sum<L, R, sub>;
    using DataT = typename Expr::ComponentT;
    inline static void eval(
        const PacketIndex<W>& pi,
        vector::PacketItem<DataT, 4, W>& out,
        const Expr& from)
    {
        SumCaptureHelper<Expr, W, 4>::eval(pi, out, from);
    }
};

// 4D Sum capture with expensive left child
template<typename L, typename R, bool sub, idx_t W>
    requires(packet::detail::IsMultiLoadExpr<L>::value
             && !packet::detail::IsMultiLoadExpr<R>::value)
struct RecursivePacketCapture<3, vector::expr::detail::Sum<L, R, sub>, W> {
    using Expr = vector::expr::detail::Sum<L, R, sub>;
    using DataT = typename Expr::ComponentT;
    inline static void eval(
        const PacketIndex<W>& pi,
        vector::PacketItem<DataT, 4, W>& out,
        const Expr& from)
    {
        SumCaptureHelper<Expr, W, 4>::eval(pi, out, from);
    }
};

// 4D Sum capture with both children expensive
template<typename L, typename R, bool sub, idx_t W>
    requires(packet::detail::IsMultiLoadExpr<L>::value
             && packet::detail::IsMultiLoadExpr<R>::value)
struct RecursivePacketCapture<3, vector::expr::detail::Sum<L, R, sub>, W> {
    using Expr = vector::expr::detail::Sum<L, R, sub>;
    using DataT = typename Expr::ComponentT;
    inline static void eval(
        const PacketIndex<W>& pi,
        vector::PacketItem<DataT, 4, W>& out,
        const Expr& from)
    {
        SumCaptureHelper<Expr, W, 4>::eval(pi, out, from);
    }
};

} // namespace detail

/**
 * @brief Materialise a lazy expression into a register-local PacketItem.
 *
 * The returned PacketItem holds one SIMD Packet per vector component,
 * all in registers. This is the SIMD analogue of `cpu::capture` and the
 * key to single-pass fused evaluation: the intermediate stays in
 * registers rather than being written to memory.
 *
 * @par Example
 * @code
 *   feta::cpu::packetTiledFor<Real>(0, N, [&](auto pi) {
 *       auto vw   = feta::cpu::packetCapture<Real>(qR.quatRotate(vR), pi);
 *       auto expr = vw + omR.cross(vw);
 *       feta::cpu::packetAssign(outR, expr, pi);
 *   });
 * @endcode
 *
 * @tparam DataT  Scalar type (e.g. `double`).
 * @tparam Expr   Expression type (deduced).
 * @tparam W      SIMD width (deduced from PacketIndex).
 * @param expr    Lazy expression to evaluate.
 * @param pi      Packet index describing which samples to evaluate.
 * @return `PacketItem<DataT, Expr::VecDims, W>` with the materialised packets.
 */
template<typename DataT, typename Expr, idx_t W>
inline vector::PacketItem<DataT, Expr::VecDims, W> packetCapture(
    const Expr& expr, const PacketIndex<W>& pi)
{
    vector::PacketItem<DataT, Expr::VecDims, W> item;
    detail::RecursivePacketCapture<Expr::VecDims - 1, Expr, W>::eval(
        pi, item, expr);
    return item;
}

/**
 * @brief Store a packet expression into a destination array at a PacketIndex.
 *
 * Convenience wrapper around RecursivePacketAssign for use in
 * `packetTiledFor` lambdas.
 *
 * @tparam ExprL  Destination type.
 * @tparam ExprR  Source expression type.
 * @tparam W      SIMD width (deduced from PacketIndex).
 * @param dest    Destination array/ref.
 * @param expr    Expression to evaluate and store.
 * @param pi      Packet index.
 */
template<typename ExprL, typename ExprR, idx_t W>
inline void packetAssign(ExprL& dest, const ExprR& expr,
    const PacketIndex<W>& pi)
{
    static_assert(ExprL::VecDims == ExprR::VecDims,
        "Destination and expression must have the same number of dimensions!");
    packet::RecursivePacketAssign<ExprL::VecDims - 1, ExprL, ExprR, W>::eval(
        pi, dest, expr);
}

} // namespace cpu
} // namespace feta
