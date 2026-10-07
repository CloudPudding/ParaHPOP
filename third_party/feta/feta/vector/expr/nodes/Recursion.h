#pragma once

/**
 * @file Recursion.h
 * @brief Generic compile-time recursion drivers for expression assignment
 *        (`RecursiveAssign`, `RecursiveAtomicAssign`, `RecursiveAtomicSum`).
 *
 * Part of the `OperationsDetail.h` facade. The per-node multi-load
 * specializations of `RecursiveAssign` live in `AssignMultiLoad.h`.
 */

#include <cmath>

#include "feta/math/math.h"
#include "feta/vector/expr/Expression.h"
#include "feta/vector/expr/Reduce.h"

namespace feta {
namespace vector {
namespace expr {
namespace detail {


/** @brief Atomic assignment */
template<typename T>
DEVICEHOST()
inline void atomicAssign(T* address, const T& val)
{
#ifdef __CUDA_ARCH__
    /* Reinterpret as 64-bit integer for atomic operation */
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long val_as_ull      = *((unsigned long long*)&val);
    atomicExch(address_as_ull, val_as_ull);
#else
    *address = val;
#endif
}

/**
 * @brief Recursive template which triggers expression evaluation.
 *
 * This unrolls the assignment operation `to = from` for each vector component.
 */
template<dims_t dim, typename ExprL, typename ExprR>
struct RecursiveAssign {
    DEVICEHOST()
    inline static void eval(const SampleIndex& i, ExprL& to, const ExprR& from)
    {
        static_assert(ExprL::VecDims == ExprR::VecDims,
            "Vectors must have same dimensions!");
        to.template get<dim>(i) = from.template get<dim>(i);
        RecursiveAssign<dim - 1, ExprL, ExprR>::eval(i, to, from);
    }

    DEVICEHOST()
    inline static void eval(const idx_t i, ExprL& to, const ExprR& from)
    {
        static_assert(ExprL::VecDims == ExprR::VecDims,
            "Vectors must have same dimensions!");
        to.template get<dim>(i) = from.template get<dim>(i);
        RecursiveAssign<dim - 1, ExprL, ExprR>::eval(i, to, from);
    }

    DEVICEHOST() inline static void eval(ExprL& to, const ExprR& from)
    {
        static_assert(ExprL::VecDims == ExprR::VecDims,
            "Vectors must have same dimensions!");
        to.template get<dim>() = from.template get<dim>();
        RecursiveAssign<dim - 1, ExprL, ExprR>::eval(to, from);
    }
};

template<typename ExprL, typename ExprR>
struct RecursiveAssign<0, ExprL, ExprR> {
    DEVICEHOST()
    inline static void eval(const SampleIndex& i, ExprL& to, const ExprR& from)
    {
        to.template get<0>(i) = from.template get<0>(i);
    }

    DEVICEHOST()
    inline static void eval(const idx_t i, ExprL& to, const ExprR& from)
    {
        to.template get<0>(i) = from.template get<0>(i);
    }

    DEVICEHOST() inline static void eval(ExprL& to, const ExprR& from)
    {
        to.template get<0>() = from.template get<0>();
    }
};

/** @brief Recursive atomic assign that triggers the expression evaluation
 *
 *  This unrolls the assignment operation `to = from` for each vector component.
 */
template<dims_t dim, typename ExprL, typename ExprR>
struct RecursiveAtomicAssign {
    DEVICEHOST()
    inline static void eval(const SampleIndex& i, ExprL& to, const ExprR& from)
    {
        static_assert(ExprL::VecDims == ExprR::VecDims,
            "Vectors must have same dimensions!");
        atomicAssign(&to.template get<dim>(i), from.template get<dim>(i));
        RecursiveAtomicAssign<dim - 1, ExprL, ExprR>::eval(i, to, from);
    }

    DEVICEHOST()
    inline static void eval(const idx_t i, ExprL& to, const ExprR& from)
    {
        static_assert(ExprL::VecDims == ExprR::VecDims,
            "Vectors must have same dimensions!");
        atomicAssign(&to.template get<dim>(i), from.template get<dim>(i));
        RecursiveAtomicAssign<dim - 1, ExprL, ExprR>::eval(i, to, from);
    }

    DEVICEHOST() inline static void eval(ExprL& to, const ExprR& from)
    {
        static_assert(ExprL::VecDims == ExprR::VecDims,
            "Vectors must have same dimensions!");
        atomicAssign(&to.template get<dim>(), from.template get<dim>());
        RecursiveAtomicAssign<dim - 1, ExprL, ExprR>::eval(to, from);
    }
};

template<typename ExprL, typename ExprR>
struct RecursiveAtomicAssign<0, ExprL, ExprR> {
    DEVICEHOST()
    inline static void eval(const SampleIndex& i, ExprL& to, const ExprR& from)
    {
        atomicAssign(&to.template get<0>(i), from.template get<0>(i));
    }

    DEVICEHOST()
    inline static void eval(const idx_t i, ExprL& to, const ExprR& from)
    {
        atomicAssign(&to.template get<0>(i), from.template get<0>(i));
    }

    DEVICEHOST() inline static void eval(ExprL& to, const ExprR& from)
    {
        atomicAssign(&to.template get<0>(), from.template get<0>());
    }
};


/**
 * @brief Recursive evaluator for AtomicSum expressions — invokes the
 *        expression's component get() which performs the per-component
 *        atomicAdd side-effect. This allows calling `view.atomicSum(expr)`
 *        as a statement (the View overload will use this helper).
 */
template<dims_t dim, typename Expr>
struct RecursiveAtomicSum {
    DEVICEHOST()
    inline static void eval(const SampleIndex& i, const Expr& expr)
    {
        expr.template get<dim>(i);
        RecursiveAtomicSum<dim - 1, Expr>::eval(i, expr);
    }
};

template<typename Expr>
struct RecursiveAtomicSum<0, Expr> {
    DEVICEHOST()
    inline static void eval(const SampleIndex& i, const Expr& expr)
    {
        expr.template get<0>(i);
    }
};

} // namespace detail
} // namespace expr
} // namespace vector
} // namespace feta
