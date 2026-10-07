#pragma once

/**
 * @file Geometric.h
 * @brief Geometric vector expression nodes (`Cross`, `Normalize`).
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

/** @brief Cross product between two 3D vector expressions */
template<typename L, typename R>
class Cross : public Expression<Cross<L, R>, typename L::ComponentT> {
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
    Cross(const L& l, const R& r)
        : l_{ l }
        , r_{ r }
    {
        static_assert(L::VecDims == 3,
            "Only 3D vector expressions support the cross "
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
    template<dims_t dim, typename Idx>
    DEVICEHOST()
    inline auto eval_(const Idx& i) const -> typename L::ComponentT
    {
        const auto l0 = l_.template get<0>(i);
        const auto l1 = l_.template get<1>(i);
        const auto l2 = l_.template get<2>(i);
        const auto r0 = r_.template get<0>(i);
        const auto r1 = r_.template get<1>(i);
        const auto r2 = r_.template get<2>(i);

        if constexpr (dim == 0)
            return l1 * r2 - l2 * r1;
        else if constexpr (dim == 1)
            return l2 * r0 - l0 * r2;
        else
            return l0 * r1 - l1 * r0;
    }

    template<dims_t dim>
    DEVICEHOST()
    inline auto eval_noindex_() const -> typename L::ComponentT
    {
        const auto l0 = l_.template get<0>();
        const auto l1 = l_.template get<1>();
        const auto l2 = l_.template get<2>();
        const auto r0 = r_.template get<0>();
        const auto r1 = r_.template get<1>();
        const auto r2 = r_.template get<2>();

        if constexpr (dim == 0)
            return l1 * r2 - l2 * r1;
        else if constexpr (dim == 1)
            return l2 * r0 - l0 * r2;
        else
            return l0 * r1 - l1 * r0;
    }
};

/** @brief Normalize the vector expression, returning the components of the
 * corresponding unit vector */
template<typename E>
class Normalize : public Expression<Normalize<E>, typename E::ComponentT> {
public:
    typename std::conditional<E::isLeaf, const E&, const E>::type e_;

    static constexpr bool isLeaf     = false;
    static constexpr dims_t VecDims  = E::VecDims;
    static constexpr bool work       = false;
    static constexpr ExprKind kind   = ExprKind::Unary;
    static constexpr dims_t depth    = E::depth + 1;
    static constexpr idx_t arraySize = E::arraySize;

    DEVICEHOST()
    Normalize(const E& e)
        : e_{ e }
    {
    }

    // NOTE: The per-dim get<dim>() below recomputes rNorm for each
    // component on the CUDA/scalar path.  On the CPU SIMD path,
    // specialised RecursivePacketAssign / RecursivePacketCapture
    // (PacketOps.h, TiledEval.h) compute rNorm once for all dims.
    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) inline get(const SampleIndex& i) const
    {
        return e_.template get<dim>(i) * e_.rNorm(i);
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) inline get(const idx_t i) const
    {
        return e_.template get<dim>(i) * e_.rNorm(i);
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) inline get() const
    {
        return e_.template get<dim>() * e_.rNorm();
    }
};

} // namespace detail
} // namespace expr
} // namespace vector
} // namespace feta
