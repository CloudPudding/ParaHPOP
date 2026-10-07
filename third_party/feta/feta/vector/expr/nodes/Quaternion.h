#pragma once

/**
 * @file Quaternion.h
 * @brief Quaternion expression nodes (`QuatConj`, `QuatAntiInvolute`,
 *        `QuatMul`, `QuatRotate`).
 *
 * Part of the `OperationsDetail.h` facade.
 */

#include <cmath>

#include "feta/math/math.h"
#include "feta/vector/expr/Expression.h"
#include "feta/vector/expr/Reduce.h"

namespace feta {
namespace vector {
namespace expr {
namespace detail {

/** @brief Quaternion conjugation */
template<typename E>
class QuatConj : public Expression<QuatConj<E>, typename E::ComponentT> {
public:
    typename std::conditional<E::isLeaf, const E&, const E>::type e_;

    static constexpr bool isLeaf     = false;
    static constexpr dims_t VecDims  = E::VecDims;
    static constexpr bool work       = false;
    static constexpr ExprKind kind   = ExprKind::Unary;
    static constexpr dims_t depth    = E::depth + 1;
    static constexpr idx_t arraySize = E::arraySize;

    DEVICEHOST()
    QuatConj(const E& e)
        : e_{ e }
    {
        static_assert(E::VecDims == 4,
            "Only 4D vector expressions support the quaternion "
            "conjugate!");
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) inline get(const SampleIndex& i) const
    {
        if constexpr (dim == 0)
            return e_.template get<dim>(i);
        else
            return -e_.template get<dim>(i);
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) inline get(const idx_t i) const
    {
        if constexpr (dim == 0)
            return e_.template get<dim>(i);
        else
            return -e_.template get<dim>(i);
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) inline get() const
    {
        if constexpr (dim == 0)
            return e_.template get<dim>();
        else
            return -e_.template get<dim>();
    }
};

/** @brief Quaternion anti-involute (star-conjugate) */
template<typename E>
class QuatAntiInvolute
    : public Expression<QuatAntiInvolute<E>, typename E::ComponentT> {
public:
    typename std::conditional<E::isLeaf, const E&, const E>::type e_;

    static constexpr bool isLeaf     = false;
    static constexpr dims_t VecDims  = E::VecDims;
    static constexpr bool work       = false;
    static constexpr ExprKind kind   = ExprKind::Unary;
    static constexpr dims_t depth    = E::depth + 1;
    static constexpr idx_t arraySize = E::arraySize;

    DEVICEHOST()
    QuatAntiInvolute(const E& e)
        : e_{ e }
    {
        static_assert(E::VecDims == 4,
            "Only 4D vector expressions support the quaternion "
            "anti-involute!");
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) inline get(const SampleIndex& i) const
    {
        if constexpr (dim == 3)
            return -e_.template get<dim>(i);
        else
            return e_.template get<dim>(i);
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) inline get(const idx_t i) const
    {
        if constexpr (dim == 3)
            return -e_.template get<dim>(i);
        else
            return e_.template get<dim>(i);
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) inline get() const
    {
        if constexpr (dim == 3)
            return -e_.template get<dim>();
        else
            return e_.template get<dim>();
    }
};

/** @brief Quaternion product between two 3D vector expressions */
template<typename L, typename R>
class QuatMul : public Expression<QuatMul<L, R>, typename L::ComponentT> {
public:
    typename std::conditional<L::isLeaf, const L&, const L>::type l_;
    typename std::conditional<R::isLeaf, const R&, const R>::type r_;

    static constexpr bool isLeaf    = false;
    static constexpr dims_t VecDims = L::VecDims;
    static constexpr bool work      = false;
    static constexpr ExprKind kind  = ExprKind::Binary;
    static constexpr dims_t depth
        = (L::depth > R::depth ? L::depth : R::depth) + 1;
    static constexpr idx_t arraySize
        = (L::arraySize != 0) ? L::arraySize : R::arraySize;

    DEVICEHOST()
    QuatMul(const L& l, const R& r)
        : l_{ l }
        , r_{ r }
    {
        static_assert(L::VecDims == 4,
            "Only 4D vector expressions support the quaternion "
            "product!");
        static_assert(
            L::VecDims == R::VecDims, "Vectors must have same dimensions!");
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) inline get(const SampleIndex& i) const
    {
        return eval_<dim>(i);
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) inline get(const idx_t i) const
    {
        return eval_<dim>(i);
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) inline get() const
    {
        return eval_noindex_<dim>();
    }

private:
    /** Quaternion multiply for indexed access (SampleIndex or idx_t). */
    template<dims_t dim, typename Idx>
    DEVICEHOST()
    inline auto eval_(const Idx& i) const -> typename L::ComponentT
    {
        const auto l0 = l_.template get<0>(i);
        const auto l1 = l_.template get<1>(i);
        const auto l2 = l_.template get<2>(i);
        const auto l3 = l_.template get<3>(i);
        const auto r0 = r_.template get<0>(i);
        const auto r1 = r_.template get<1>(i);
        const auto r2 = r_.template get<2>(i);
        const auto r3 = r_.template get<3>(i);

        if constexpr (dim == 0) {
            return l0 * r0 - l1 * r1 - l2 * r2 - l3 * r3;
        } else if constexpr (dim == 1) {
            return l0 * r1 + l1 * r0 + l2 * r3 - l3 * r2;
        } else if constexpr (dim == 2) {
            return l0 * r2 - l1 * r3 + l2 * r0 + l3 * r1;
        } else {
            return l0 * r3 + l1 * r2 - l2 * r1 + l3 * r0;
        }
    }

    /** Quaternion multiply for direct (no-index) access. */
    template<dims_t dim>
    DEVICEHOST()
    inline auto eval_noindex_() const -> typename L::ComponentT
    {
        const auto l0 = l_.template get<0>();
        const auto l1 = l_.template get<1>();
        const auto l2 = l_.template get<2>();
        const auto l3 = l_.template get<3>();
        const auto r0 = r_.template get<0>();
        const auto r1 = r_.template get<1>();
        const auto r2 = r_.template get<2>();
        const auto r3 = r_.template get<3>();

        if constexpr (dim == 0) {
            return l0 * r0 - l1 * r1 - l2 * r2 - l3 * r3;
        } else if constexpr (dim == 1) {
            return l0 * r1 + l1 * r0 + l2 * r3 - l3 * r2;
        } else if constexpr (dim == 2) {
            return l0 * r2 - l1 * r3 + l2 * r0 + l3 * r1;
        } else {
            return l0 * r3 + l1 * r2 - l2 * r1 + l3 * r0;
        }
    }
};


/**
 * @brief Rodrigues-formula quaternion rotation of a 3D vector.
 *
 * Given a unit quaternion q = (w, x, y, z) and a 3D vector v, computes:
 *   t  = 2 * (u × v)        where u = (x, y, z)
 *   v' = v + w*t + u × t
 *
 * This uses 30 FLOPs (vs ~56 for the naive q⊗(0,v)⊗q* sandwich product)
 * and produces a 3D result directly without the intermediate 4D expansion.
 *
 * @tparam Q  Quaternion expression type (VecDims == 4).
 * @tparam V  Vector expression type (VecDims == 3).
 */
template<typename Q, typename V>
class QuatRotate : public Expression<QuatRotate<Q, V>, typename Q::ComponentT> {
public:
    typename std::conditional<Q::isLeaf, const Q&, const Q>::type q_;
    typename std::conditional<V::isLeaf, const V&, const V>::type v_;

    static constexpr bool isLeaf    = false;
    static constexpr dims_t VecDims = 3;
    static constexpr bool work      = false;
    static constexpr ExprKind kind  = ExprKind::Binary;
    static constexpr dims_t depth
        = (Q::depth > V::depth ? Q::depth : V::depth) + 1;
    static constexpr idx_t arraySize
        = (Q::arraySize != 0) ? Q::arraySize : V::arraySize;

    DEVICEHOST()
    QuatRotate(const Q& q, const V& v)
        : q_{ q }
        , v_{ v }
    {
        static_assert(
            Q::VecDims == 4, "QuatRotate requires a 4D quaternion expression!");
        static_assert(
            V::VecDims == 3, "QuatRotate requires a 3D vector expression!");
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) inline get(const SampleIndex& i) const
    {
        return eval_<dim>(i);
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) inline get(const idx_t i) const
    {
        return eval_<dim>(i);
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) inline get() const
    {
        return eval_noindex_<dim>();
    }

private:
    /**
     * Rodrigues rotation for indexed access (SampleIndex or idx_t).
     *
     *   u = q.vec = (q1, q2, q3)
     *   t = 2 * (u × v)
     *   result = v + w*t + u × t
     */
    template<dims_t dim, typename Idx>
    DEVICEHOST()
    inline auto eval_(const Idx& i) const -> typename Q::ComponentT
    {
        // Quaternion components: w = q0, u = (q1, q2, q3)
        const auto w  = q_.template get<0>(i);
        const auto ux = q_.template get<1>(i);
        const auto uy = q_.template get<2>(i);
        const auto uz = q_.template get<3>(i);

        const auto vx = v_.template get<0>(i);
        const auto vy = v_.template get<1>(i);
        const auto vz = v_.template get<2>(i);

        // t = 2 * (u × v)
        const auto tx
            = static_cast<typename Q::ComponentT>(2) * (uy * vz - uz * vy);
        const auto ty
            = static_cast<typename Q::ComponentT>(2) * (uz * vx - ux * vz);
        const auto tz
            = static_cast<typename Q::ComponentT>(2) * (ux * vy - uy * vx);

        // result = v + w*t + u × t
        if constexpr (dim == 0)
            return vx + w * tx + (uy * tz - uz * ty);
        else if constexpr (dim == 1)
            return vy + w * ty + (uz * tx - ux * tz);
        else
            return vz + w * tz + (ux * ty - uy * tx);
    }

    /** Rodrigues rotation for direct (no-index) access. */
    template<dims_t dim>
    DEVICEHOST()
    inline auto eval_noindex_() const -> typename Q::ComponentT
    {
        const auto w  = q_.template get<0>();
        const auto ux = q_.template get<1>();
        const auto uy = q_.template get<2>();
        const auto uz = q_.template get<3>();

        const auto vx = v_.template get<0>();
        const auto vy = v_.template get<1>();
        const auto vz = v_.template get<2>();

        const auto tx
            = static_cast<typename Q::ComponentT>(2) * (uy * vz - uz * vy);
        const auto ty
            = static_cast<typename Q::ComponentT>(2) * (uz * vx - ux * vz);
        const auto tz
            = static_cast<typename Q::ComponentT>(2) * (ux * vy - uy * vx);

        if constexpr (dim == 0)
            return vx + w * tx + (uy * tz - uz * ty);
        else if constexpr (dim == 1)
            return vy + w * ty + (uz * tx - ux * tz);
        else
            return vz + w * tz + (ux * ty - uy * tx);
    }
};

} // namespace detail
} // namespace expr
} // namespace vector
} // namespace feta
