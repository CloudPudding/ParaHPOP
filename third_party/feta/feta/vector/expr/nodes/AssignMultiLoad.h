#pragma once

/**
 * @file AssignMultiLoad.h
 * @brief Specialized `RecursiveAssign` for the multi-load nodes (`Cross`,
 *        `Normalize`, `QuatMul`, `QuatRotate`) — load each child once and
 *        compute all outputs together (the generic driver would reload
 *        every child per output dimension).
 *
 * Part of the `OperationsDetail.h` facade; depends on the generic
 * `RecursiveAssign` and the node definitions.
 */

#include <cmath>

#include "feta/math/math.h"
#include "feta/vector/expr/Expression.h"
#include "feta/vector/expr/Reduce.h"
#include "feta/vector/expr/nodes/Recursion.h"
#include "feta/vector/expr/nodes/Geometric.h"
#include "feta/vector/expr/nodes/Quaternion.h"

namespace feta {
namespace vector {
namespace expr {
namespace detail {

// ---------------------------------------------------------------------------
// Specialized RecursiveAssign for multi-load expressions (GPU / scalar path).
//
// The generic RecursiveAssign calls get<dim>() once per output dimension.
// For Cross, QuatMul, QuatRotate and Normalize each get<dim>() independently
// loads ALL child components — causing 3–4× redundant loads/sqrts.  The
// specializations below load children once and compute all outputs together.
// ---------------------------------------------------------------------------

// ─── MULTI-LOAD NODE REGISTRY ─── one node ⇒ up to three recursive forms ────
// Each multi-load node (Cross, Normalize, QuatMul, QuatRotate, Sum, CWiseScale)
// loads every child once and writes all outputs together. The three forms live
// in separate facades on purpose — the scalar form must stay device-light, the
// packet forms pull <omp.h> + x86 SIMD, so they cannot share a header (F1 audit):
//   1. scalar / GPU     RecursiveAssign<dim, ExprL, Node>
//                         @ feta/vector/expr/nodes/AssignMultiLoad.h   (this file)
//   2. CPU-SIMD assign   RecursivePacketAssign<dim, ExprL, Node, W>
//                         @ feta/vector/expr/cpu/packet/Assign.h
//   3. CPU-SIMD capture  RecursivePacketCapture<dim, Node, W>
//                         @ feta/vector/expr/cpu/tiled/Capture.h
// TO ADD A NODE: add its spec under a matching "<Node>" banner in ALL THREE.
// ───────────────────────────────────────────────────────────────────────────

// --- Cross (3D) -----------------------------------------------------------

template<typename ExprL, typename L, typename R>
struct RecursiveAssign<2, ExprL, Cross<L, R>> {
    DEVICEHOST()
    inline static void eval(
        const SampleIndex& i, ExprL& to, const Cross<L, R>& from)
    {
        const auto l0         = from.l_.template get<0>(i);
        const auto l1         = from.l_.template get<1>(i);
        const auto l2         = from.l_.template get<2>(i);
        const auto r0         = from.r_.template get<0>(i);
        const auto r1         = from.r_.template get<1>(i);
        const auto r2         = from.r_.template get<2>(i);
        to.template get<0>(i) = l1 * r2 - l2 * r1;
        to.template get<1>(i) = l2 * r0 - l0 * r2;
        to.template get<2>(i) = l0 * r1 - l1 * r0;
    }

    DEVICEHOST()
    inline static void eval(const idx_t i, ExprL& to, const Cross<L, R>& from)
    {
        const auto l0         = from.l_.template get<0>(i);
        const auto l1         = from.l_.template get<1>(i);
        const auto l2         = from.l_.template get<2>(i);
        const auto r0         = from.r_.template get<0>(i);
        const auto r1         = from.r_.template get<1>(i);
        const auto r2         = from.r_.template get<2>(i);
        to.template get<0>(i) = l1 * r2 - l2 * r1;
        to.template get<1>(i) = l2 * r0 - l0 * r2;
        to.template get<2>(i) = l0 * r1 - l1 * r0;
    }

    DEVICEHOST()
    inline static void eval(ExprL& to, const Cross<L, R>& from)
    {
        const auto l0        = from.l_.template get<0>();
        const auto l1        = from.l_.template get<1>();
        const auto l2        = from.l_.template get<2>();
        const auto r0        = from.r_.template get<0>();
        const auto r1        = from.r_.template get<1>();
        const auto r2        = from.r_.template get<2>();
        to.template get<0>() = l1 * r2 - l2 * r1;
        to.template get<1>() = l2 * r0 - l0 * r2;
        to.template get<2>() = l0 * r1 - l1 * r0;
    }
};

// --- Normalize (3D) -------------------------------------------------------

template<typename ExprL, typename E>
    requires(E::VecDims == 3)
struct RecursiveAssign<2, ExprL, Normalize<E>> {
    DEVICEHOST()
    inline static void eval(
        const SampleIndex& i, ExprL& to, const Normalize<E>& from)
    {
        const auto c0 = from.e_.template get<0>(i);
        const auto c1 = from.e_.template get<1>(i);
        const auto c2 = from.e_.template get<2>(i);
        const auto sn = c0 * c0 + c1 * c1 + c2 * c2;
        const auto rn = typename E::ComponentT{ 1 } / feta::math::sqrt(sn);
        to.template get<0>(i) = c0 * rn;
        to.template get<1>(i) = c1 * rn;
        to.template get<2>(i) = c2 * rn;
    }

    DEVICEHOST()
    inline static void eval(const idx_t i, ExprL& to, const Normalize<E>& from)
    {
        const auto c0 = from.e_.template get<0>(i);
        const auto c1 = from.e_.template get<1>(i);
        const auto c2 = from.e_.template get<2>(i);
        const auto sn = c0 * c0 + c1 * c1 + c2 * c2;
        const auto rn = typename E::ComponentT{ 1 } / feta::math::sqrt(sn);
        to.template get<0>(i) = c0 * rn;
        to.template get<1>(i) = c1 * rn;
        to.template get<2>(i) = c2 * rn;
    }

    DEVICEHOST()
    inline static void eval(ExprL& to, const Normalize<E>& from)
    {
        const auto c0 = from.e_.template get<0>();
        const auto c1 = from.e_.template get<1>();
        const auto c2 = from.e_.template get<2>();
        const auto sn = c0 * c0 + c1 * c1 + c2 * c2;
        const auto rn = typename E::ComponentT{ 1 } / feta::math::sqrt(sn);
        to.template get<0>() = c0 * rn;
        to.template get<1>() = c1 * rn;
        to.template get<2>() = c2 * rn;
    }
};

// --- Normalize (4D) -------------------------------------------------------

template<typename ExprL, typename E>
    requires(E::VecDims == 4)
struct RecursiveAssign<3, ExprL, Normalize<E>> {
    DEVICEHOST()
    inline static void eval(
        const SampleIndex& i, ExprL& to, const Normalize<E>& from)
    {
        const auto c0 = from.e_.template get<0>(i);
        const auto c1 = from.e_.template get<1>(i);
        const auto c2 = from.e_.template get<2>(i);
        const auto c3 = from.e_.template get<3>(i);
        const auto sn = c0 * c0 + c1 * c1 + c2 * c2 + c3 * c3;
        const auto rn = typename E::ComponentT{ 1 } / feta::math::sqrt(sn);
        to.template get<0>(i) = c0 * rn;
        to.template get<1>(i) = c1 * rn;
        to.template get<2>(i) = c2 * rn;
        to.template get<3>(i) = c3 * rn;
    }

    DEVICEHOST()
    inline static void eval(const idx_t i, ExprL& to, const Normalize<E>& from)
    {
        const auto c0 = from.e_.template get<0>(i);
        const auto c1 = from.e_.template get<1>(i);
        const auto c2 = from.e_.template get<2>(i);
        const auto c3 = from.e_.template get<3>(i);
        const auto sn = c0 * c0 + c1 * c1 + c2 * c2 + c3 * c3;
        const auto rn = typename E::ComponentT{ 1 } / feta::math::sqrt(sn);
        to.template get<0>(i) = c0 * rn;
        to.template get<1>(i) = c1 * rn;
        to.template get<2>(i) = c2 * rn;
        to.template get<3>(i) = c3 * rn;
    }

    DEVICEHOST()
    inline static void eval(ExprL& to, const Normalize<E>& from)
    {
        const auto c0 = from.e_.template get<0>();
        const auto c1 = from.e_.template get<1>();
        const auto c2 = from.e_.template get<2>();
        const auto c3 = from.e_.template get<3>();
        const auto sn = c0 * c0 + c1 * c1 + c2 * c2 + c3 * c3;
        const auto rn = typename E::ComponentT{ 1 } / feta::math::sqrt(sn);
        to.template get<0>() = c0 * rn;
        to.template get<1>() = c1 * rn;
        to.template get<2>() = c2 * rn;
        to.template get<3>() = c3 * rn;
    }
};

// --- QuatMul (4D) ---------------------------------------------------------

template<typename ExprL, typename L, typename R>
struct RecursiveAssign<3, ExprL, QuatMul<L, R>> {
    DEVICEHOST()
    inline static void eval(
        const SampleIndex& i, ExprL& to, const QuatMul<L, R>& from)
    {
        const auto l0         = from.l_.template get<0>(i);
        const auto l1         = from.l_.template get<1>(i);
        const auto l2         = from.l_.template get<2>(i);
        const auto l3         = from.l_.template get<3>(i);
        const auto r0         = from.r_.template get<0>(i);
        const auto r1         = from.r_.template get<1>(i);
        const auto r2         = from.r_.template get<2>(i);
        const auto r3         = from.r_.template get<3>(i);
        to.template get<0>(i) = l0 * r0 - l1 * r1 - l2 * r2 - l3 * r3;
        to.template get<1>(i) = l0 * r1 + l1 * r0 + l2 * r3 - l3 * r2;
        to.template get<2>(i) = l0 * r2 - l1 * r3 + l2 * r0 + l3 * r1;
        to.template get<3>(i) = l0 * r3 + l1 * r2 - l2 * r1 + l3 * r0;
    }

    DEVICEHOST()
    inline static void eval(const idx_t i, ExprL& to, const QuatMul<L, R>& from)
    {
        const auto l0         = from.l_.template get<0>(i);
        const auto l1         = from.l_.template get<1>(i);
        const auto l2         = from.l_.template get<2>(i);
        const auto l3         = from.l_.template get<3>(i);
        const auto r0         = from.r_.template get<0>(i);
        const auto r1         = from.r_.template get<1>(i);
        const auto r2         = from.r_.template get<2>(i);
        const auto r3         = from.r_.template get<3>(i);
        to.template get<0>(i) = l0 * r0 - l1 * r1 - l2 * r2 - l3 * r3;
        to.template get<1>(i) = l0 * r1 + l1 * r0 + l2 * r3 - l3 * r2;
        to.template get<2>(i) = l0 * r2 - l1 * r3 + l2 * r0 + l3 * r1;
        to.template get<3>(i) = l0 * r3 + l1 * r2 - l2 * r1 + l3 * r0;
    }

    DEVICEHOST()
    inline static void eval(ExprL& to, const QuatMul<L, R>& from)
    {
        const auto l0        = from.l_.template get<0>();
        const auto l1        = from.l_.template get<1>();
        const auto l2        = from.l_.template get<2>();
        const auto l3        = from.l_.template get<3>();
        const auto r0        = from.r_.template get<0>();
        const auto r1        = from.r_.template get<1>();
        const auto r2        = from.r_.template get<2>();
        const auto r3        = from.r_.template get<3>();
        to.template get<0>() = l0 * r0 - l1 * r1 - l2 * r2 - l3 * r3;
        to.template get<1>() = l0 * r1 + l1 * r0 + l2 * r3 - l3 * r2;
        to.template get<2>() = l0 * r2 - l1 * r3 + l2 * r0 + l3 * r1;
        to.template get<3>() = l0 * r3 + l1 * r2 - l2 * r1 + l3 * r0;
    }
};

// --- QuatRotate (3D output) -----------------------------------------------

template<typename ExprL, typename Q, typename V>
struct RecursiveAssign<2, ExprL, QuatRotate<Q, V>> {
    DEVICEHOST()
    inline static void eval(
        const SampleIndex& i, ExprL& to, const QuatRotate<Q, V>& from)
    {
        const auto w          = from.q_.template get<0>(i);
        const auto ux         = from.q_.template get<1>(i);
        const auto uy         = from.q_.template get<2>(i);
        const auto uz         = from.q_.template get<3>(i);
        const auto vx         = from.v_.template get<0>(i);
        const auto vy         = from.v_.template get<1>(i);
        const auto vz         = from.v_.template get<2>(i);
        using T               = typename Q::ComponentT;
        const auto two        = T{ 2 };
        const auto tx         = two * (uy * vz - uz * vy);
        const auto ty         = two * (uz * vx - ux * vz);
        const auto tz         = two * (ux * vy - uy * vx);
        to.template get<0>(i) = vx + w * tx + (uy * tz - uz * ty);
        to.template get<1>(i) = vy + w * ty + (uz * tx - ux * tz);
        to.template get<2>(i) = vz + w * tz + (ux * ty - uy * tx);
    }

    DEVICEHOST()
    inline static void eval(
        const idx_t i, ExprL& to, const QuatRotate<Q, V>& from)
    {
        const auto w          = from.q_.template get<0>(i);
        const auto ux         = from.q_.template get<1>(i);
        const auto uy         = from.q_.template get<2>(i);
        const auto uz         = from.q_.template get<3>(i);
        const auto vx         = from.v_.template get<0>(i);
        const auto vy         = from.v_.template get<1>(i);
        const auto vz         = from.v_.template get<2>(i);
        using T               = typename Q::ComponentT;
        const auto two        = T{ 2 };
        const auto tx         = two * (uy * vz - uz * vy);
        const auto ty         = two * (uz * vx - ux * vz);
        const auto tz         = two * (ux * vy - uy * vx);
        to.template get<0>(i) = vx + w * tx + (uy * tz - uz * ty);
        to.template get<1>(i) = vy + w * ty + (uz * tx - ux * tz);
        to.template get<2>(i) = vz + w * tz + (ux * ty - uy * tx);
    }

    DEVICEHOST()
    inline static void eval(ExprL& to, const QuatRotate<Q, V>& from)
    {
        const auto w         = from.q_.template get<0>();
        const auto ux        = from.q_.template get<1>();
        const auto uy        = from.q_.template get<2>();
        const auto uz        = from.q_.template get<3>();
        const auto vx        = from.v_.template get<0>();
        const auto vy        = from.v_.template get<1>();
        const auto vz        = from.v_.template get<2>();
        using T              = typename Q::ComponentT;
        const auto two       = T{ 2 };
        const auto tx        = two * (uy * vz - uz * vy);
        const auto ty        = two * (uz * vx - ux * vz);
        const auto tz        = two * (ux * vy - uy * vx);
        to.template get<0>() = vx + w * tx + (uy * tz - uz * ty);
        to.template get<1>() = vy + w * ty + (uz * tx - ux * tz);
        to.template get<2>() = vz + w * tz + (ux * ty - uy * tx);
    }
};

} // namespace detail
} // namespace expr
} // namespace vector
} // namespace feta
