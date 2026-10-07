#pragma once

/**
 * @file Arithmetic.h
 * @brief Component-wise arithmetic expression nodes (`Sum`, `AtomicSum`,
 *        `CWiseScale`/`CWiseRScale`/`CWiseInverse`/`CWiseMult`/`CWiseDiv`,
 *        the `mult`/`div` dispatch helpers, `CWiseAbs`, `CWiseMax`).
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

/**
 * @brief Expression template representing vector addition/subtraction
 * (component-wise).
 */
template<typename L, typename R, bool sub>
class Sum : public Expression<Sum<L, R, sub>, typename L::ComponentT> {
public:
    typename std::conditional<L::isLeaf, const L&, const L>::type l_;
    typename std::conditional<R::isLeaf, const R&, const R>::type r_;

    static constexpr bool isLeaf    = false;
    static constexpr bool isSub     = sub;
    static constexpr dims_t VecDims = L::VecDims;
    static constexpr bool work      = false;
    static constexpr ExprKind kind  = ExprKind::Binary;
    static constexpr dims_t depth
        = (L::depth > R::depth ? L::depth : R::depth) + 1;
    static constexpr idx_t arraySize
        = (L::arraySize != 0) ? L::arraySize : R::arraySize;

    /* Constructor is a function template so its signature is lazily
     * parsed.  When L or R is a packet expression (carrying SIMD-register
     * fields), only host call sites instantiate it — nvcc never parses
     * the device-side signature, avoiding "vector type not supported in
     * device code" errors.  Defaults preserve the canonical form for
     * scalar callers. */
    template<typename L_ = L, typename R_ = R>
    DEVICEHOST()
    Sum(const L_& l, const R_& r)
        : l_{ l }
        , r_{ r }
    {
        static_assert(
            L::VecDims == R::VecDims, "Vectors must have same dimensions!");
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get(const SampleIndex& i) const
    {
        if constexpr (sub) {
            return l_.template get<dim>(i) - r_.template get<dim>(i);
        } else {
            return l_.template get<dim>(i) + r_.template get<dim>(i);
        }
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get(const idx_t i) const
    {
        if constexpr (sub) {
            return l_.template get<dim>(i) - r_.template get<dim>(i);
        } else {
            return l_.template get<dim>(i) + r_.template get<dim>(i);
        }
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get() const
    {
        if constexpr (sub) {
            return l_.template get<dim>() - r_.template get<dim>();
        } else {
            return l_.template get<dim>() + r_.template get<dim>();
        }
    }
};

template<typename E1, typename E2>
using Diff = Sum<E1, E2, true>;

/** @brief Atomic add */
template<typename L, typename R>
class AtomicSum : public Expression<AtomicSum<L, R>, typename L::ComponentT> {
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
    AtomicSum(const L& l, const R& r)
        : l_{ l }
        , r_{ r }
    {
        static_assert(
            L::VecDims == R::VecDims, "Vectors must have same dimensions!");
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get(const SampleIndex& i) const
    {
#ifdef __CUDA_ARCH__
        return atomicAdd(&l_.template get<dim>(i), r_.template get<dim>(i));
#else
        return l_.template get<dim>(i) += r_.template get<dim>(i);
#endif
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get(const idx_t i) const
    {
#ifdef __CUDA_ARCH__
        return atomicAdd(&l_.template get<dim>(i), r_.template get<dim>(i));
#else
        return l_.template get<dim>(i) += r_.template get<dim>(i);
#endif
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get() const
    {
#ifdef __CUDA_ARCH__
        return atomicAdd(&l_.template get<dim>(), r_.template get<dim>());
#else
        return l_.template get<dim>() += r_.template get<dim>();
#endif
    }
};

/**
 * @brief Expression template representing multiplication of a vector with a
 * scalar.
 */
template<typename Expr, typename ScaleT>
class CWiseScale
    : public Expression<CWiseScale<Expr, ScaleT>, typename Expr::ComponentT> {
public:
    ScaleT factor_;
    typename std::conditional<Expr::isLeaf, const Expr&, const Expr>::type
        expr_;

    static constexpr bool isLeaf     = false;
    static constexpr dims_t VecDims  = Expr::VecDims;
    static constexpr bool work       = false;
    static constexpr ExprKind kind   = ExprKind::Unary;
    static constexpr dims_t depth    = Expr::depth + 1;
    static constexpr idx_t arraySize = Expr::arraySize;

    DEVICEHOST()
    CWiseScale(const Expr& expr, const ScaleT& factor)
        : factor_{ factor }
        , expr_{ expr }
    {
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get(const SampleIndex& i) const
    {
        return factor_ * expr_.template get<dim>(i);
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get(const idx_t i) const
    {
        return factor_ * expr_.template get<dim>(i);
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get() const
    {
        return factor_ * expr_.template get<dim>();
    }
};

/**
 * @brief Expression template representing division of a vector by a scalar.
 */
template<typename Expr, typename ScaleT>
class CWiseRScale
    : public Expression<CWiseRScale<Expr, ScaleT>, typename Expr::ComponentT> {
public:
    static constexpr bool useReciprocal_ = std::is_floating_point_v<ScaleT>;
    ScaleT factor_;
    typename std::conditional<Expr::isLeaf, const Expr&, const Expr>::type
        expr_;

    static constexpr bool isLeaf     = false;
    static constexpr dims_t VecDims  = Expr::VecDims;
    static constexpr bool work       = false;
    static constexpr ExprKind kind   = ExprKind::Unary;
    static constexpr dims_t depth    = Expr::depth + 1;
    static constexpr idx_t arraySize = Expr::arraySize;

    DEVICEHOST()
    CWiseRScale(const Expr& expr, const ScaleT& factor)
        : factor_{ useReciprocal_ ? ScaleT{ 1 } / factor : factor }
        , expr_{ expr }
    {
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get(const SampleIndex& i) const
    {
        if constexpr (useReciprocal_)
            return expr_.template get<dim>(i) * factor_;
        else
            return expr_.template get<dim>(i) / factor_;
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get(const idx_t i) const
    {
        if constexpr (useReciprocal_)
            return expr_.template get<dim>(i) * factor_;
        else
            return expr_.template get<dim>(i) / factor_;
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get() const
    {
        if constexpr (useReciprocal_)
            return expr_.template get<dim>() * factor_;
        else
            return expr_.template get<dim>() / factor_;
    }
};

/**
 * @brief Expression template representing component-wise inverse + scaling
 * of a vector.
 */
template<typename Expr, typename ScaleT>
class CWiseInverse
    : public Expression<CWiseInverse<Expr, ScaleT>, typename Expr::ComponentT> {
public:
    ScaleT factor_;
    typename std::conditional<Expr::isLeaf, const Expr&, const Expr>::type
        expr_;

    static constexpr bool isLeaf     = false;
    static constexpr dims_t VecDims  = Expr::VecDims;
    static constexpr bool work       = false;
    static constexpr ExprKind kind   = ExprKind::Unary;
    static constexpr dims_t depth    = Expr::depth + 1;
    static constexpr idx_t arraySize = Expr::arraySize;

    DEVICEHOST()
    CWiseInverse(const Expr& expr, const ScaleT& factor)
        : factor_{ factor }
        , expr_{ expr }
    {
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get(const SampleIndex& i) const
    {
        return factor_ / expr_.template get<dim>(i);
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get(const idx_t i) const
    {
        return factor_ / expr_.template get<dim>(i);
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get() const
    {
        return factor_ / expr_.template get<dim>();
    }
};


/** @brief Component-wise product between vector expressions */
template<typename L, typename R>
class CWiseMult : public Expression<CWiseMult<L, R>, typename L::ComponentT> {
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
    CWiseMult(const L& l, const R& r)
        : l_{ l }
        , r_{ r }
    {
        static_assert(
            L::VecDims == R::VecDims, "Vectors must have same dimensions!");
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get(const SampleIndex& i) const
    {
        return l_.template get<dim>(i) * r_.template get<dim>(i);
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get(const idx_t i) const
    {
        return l_.template get<dim>(i) * r_.template get<dim>(i);
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get() const
    {
        return l_.template get<dim>() * r_.template get<dim>();
    }
};


/** @brief Component-wise division between vector expressions */
template<typename L, typename R>
class CWiseDiv : public Expression<CWiseDiv<L, R>, typename L::ComponentT> {
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
    CWiseDiv(const L& l, const R& r)
        : l_{ l }
        , r_{ r }
    {
        static_assert(
            L::VecDims == R::VecDims, "Vectors must have same dimensions!");
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get(const SampleIndex& i) const
    {
        return l_.template get<dim>(i) / r_.template get<dim>(i);
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get(const idx_t i) const
    {
        return l_.template get<dim>(i) / r_.template get<dim>(i);
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get() const
    {
        return l_.template get<dim>() / r_.template get<dim>();
    }
};


/**
 * @brief Helper used to disambiguate multiplication cases: scalar-vec,
 * vec-scalar, vec-vec (component-wise)
 */
template<typename L, typename R, bool isLScalar, bool isRScalar, bool isLVec,
    bool isRVec>
struct mult {
    DEVICEHOST()
    static constexpr decltype(auto) eval(const L& l, const R& r);
};

template<typename L, typename R>
struct mult<L, R, true, false, false, true> {
    DEVICEHOST()
    static constexpr decltype(auto) eval(const L& l, const R& r)
    {
        // l is scalar, r is vec
        return detail::CWiseScale<R, L>(r, l);
    }
};

template<typename L, typename R>
struct mult<L, R, false, true, true, false> {
    DEVICEHOST()
    static constexpr decltype(auto) eval(const L& l, const R& r)
    {
        // l is vec, r is scalar
        return detail::CWiseScale<L, R>(l, r);
    }
};

template<typename L, typename R>
struct mult<L, R, false, false, true, true> {
    DEVICEHOST()
    static constexpr decltype(auto) eval(const L& l, const R& r)
    {
        // l is vec, r is vec
        return CWiseMult<L, R>(l, r);
    }
};


/**
 * @brief Helper used to disambiguate division cases: scalar-vec,
 * vec-scalar, vec-vec (component-wise)
 */
template<typename L, typename R, bool isLScalar, bool isRScalar, bool isLVec,
    bool isRVec>
struct div {
    DEVICEHOST()
    static constexpr decltype(auto) eval(const L& l, const R& r);
};

template<typename L, typename R>
struct div<L, R, true, false, false, true> {
    DEVICEHOST()
    static constexpr decltype(auto) eval(const L& l, const R& r)
    {
        // l is scalar, r is vec
        return detail::CWiseInverse<R, L>(r, l);
    }
};

template<typename L, typename R>
struct div<L, R, false, true, true, false> {
    DEVICEHOST()
    static constexpr decltype(auto) eval(const L& l, const R& r)
    {
        // l is vec, r is scalar
        return detail::CWiseRScale<L, R>(l, r);
    }
};

template<typename L, typename R>
struct div<L, R, false, false, true, true> {
    DEVICEHOST()
    static constexpr decltype(auto) eval(const L& l, const R& r)
    {
        // l is vec, r is vec
        return CWiseDiv<L, R>(l, r);
    }
};


/** @brief Component-wise absolute value of an expression. */
template<typename E>
class CWiseAbs : public Expression<CWiseAbs<E>, typename E::ComponentT> {
public:
    typename std::conditional<E::isLeaf, const E&, const E>::type expr_;

    static constexpr bool isLeaf     = false;
    static constexpr dims_t VecDims  = E::VecDims;
    static constexpr bool work       = false;
    static constexpr ExprKind kind   = ExprKind::Unary;
    static constexpr dims_t depth    = E::depth + 1;
    static constexpr idx_t arraySize = E::arraySize;

    DEVICEHOST()
    CWiseAbs(const E& e)
        : expr_{ e }
    {
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get(const SampleIndex& i) const
    {
        return feta::math::abs(expr_.template get<dim>(i));
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get(const idx_t i) const
    {
        return feta::math::abs(expr_.template get<dim>(i));
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get() const
    {
        return feta::math::abs(expr_.template get<dim>());
    }
};


/**
 * @brief Component-wise maximum between two vector expressions.
 */
template<typename L, typename R>
class CWiseMax : public Expression<CWiseMax<L, R>, typename L::ComponentT> {
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
    CWiseMax(const L& l, const R& r)
        : l_{ l }
        , r_{ r }
    {
        static_assert(
            L::VecDims == R::VecDims, "Vectors must have same dimensions!");
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) inline get(const SampleIndex& i) const
    {
        return feta::math::max(
            l_.template get<dim>(i), r_.template get<dim>(i));
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) inline get(const idx_t i) const
    {
        return feta::math::max(
            l_.template get<dim>(i), r_.template get<dim>(i));
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) inline get() const
    {
        return feta::math::max(l_.template get<dim>(), r_.template get<dim>());
    }
};

} // namespace detail
} // namespace expr
} // namespace vector
} // namespace feta
