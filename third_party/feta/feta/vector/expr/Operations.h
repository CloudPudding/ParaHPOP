/**
 * This header contains definitions for expression-templated vector operations.
 */
#pragma once

#include "feta/core/type_traits.h"
#include "feta/vector/expr/Expression.h"
#include "feta/vector/expr/OperationsDetail.h"
#include "feta/vector/expr/Reduce.h"

namespace feta {
namespace vector {
namespace expr {

/**
 * @brief Exposes `eval` methods which trigger component-wise expression
 * evaluation and assignment.
 */
template<dims_t dim, typename ExprL, typename ExprR>
using assign = detail::RecursiveAssign<dim - 1, ExprL, ExprR>;

/** @brief Exposes atomic eval method which triggers component-wise expression
 * evaluation and atomic assignment.
 */
template<dims_t dim, typename ExprL, typename ExprR>
using atomicAssign = detail::RecursiveAtomicAssign<dim - 1, ExprL, ExprR>;


/**
 * @brief Component-wise addition of two vector expressions.
 *
 * @tparam E1  Left-hand expression type.
 * @tparam E2  Right-hand expression type; must have the same `VecDims` as `E1`.
 * @param[in] l  Left-hand vector expression.
 * @param[in] r  Right-hand vector expression.
 * @return Lazy sum expression; evaluate by assigning with a `SampleIndex`.
 *
 * The DEVICEHOST overload is disabled when either operand is a packet
 * expression (carries SIMD-register fields). Packet operands route to
 * the HOST() overload below.
 */
template<typename E1, typename E2,
    typename = std::enable_if_t<
        !feta::core::expr::anyPacketExpression<E1, E2>::value>>
DEVICEHOST()
detail::Sum<E1, E2, false> operator+(
    const Expression<E1, typename E1::ComponentT>& l,
    const Expression<E2, typename E2::ComponentT>& r)
{
    return detail::Sum<E1, E2, false>(
        *static_cast<const E1*>(&l), *static_cast<const E2*>(&r));
}

/** @brief Packet overload of operator+ — host-only because PacketItem
 *  stores SIMD-register fields that nvcc rejects on device. */
template<typename E1, typename E2,
    typename = std::enable_if_t<
        feta::core::expr::anyPacketExpression<E1, E2>::value>,
    typename = void>
HOST()
detail::Sum<E1, E2, false> operator+(
    const Expression<E1, typename E1::ComponentT>& l,
    const Expression<E2, typename E2::ComponentT>& r)
{
    return detail::Sum<E1, E2, false>(
        *static_cast<const E1*>(&l), *static_cast<const E2*>(&r));
}

/**
 * @brief Multiply two operands (scalar * vector, vector * scalar, or
 *        component-wise vector * vector).
 *
 * Dispatch is resolved at compile time based on `std::is_arithmetic`
 * and `isExpression` traits.
 *
 * @tparam L  Left-hand type (scalar or vector expression).
 * @tparam R  Right-hand type (scalar or vector expression).
 * @param[in] l  Left operand.
 * @param[in] r  Right operand.
 * @return Lazy product expression.
 */
template<typename L, typename R>
DEVICEHOST()
decltype(auto) operator*(const L& l, const R& r)
{
    constexpr bool isScalarL = std::is_arithmetic<L>::value;
    constexpr bool isScalarR = std::is_arithmetic<R>::value;
    constexpr bool isVecL    = isExpression<L>::value;
    constexpr bool isVecR    = isExpression<R>::value;
    return detail::mult<L, R, isScalarL, isScalarR, isVecL, isVecR>::eval(l, r);
}

/**
 * @brief Divide two operands (vector / scalar, or component-wise
 *        vector / vector).
 *
 * @tparam L  Left-hand type (scalar or vector expression).
 * @tparam R  Right-hand type (scalar or vector expression).
 * @param[in] l  Left operand.
 * @param[in] r  Right operand.
 * @return Lazy quotient expression.
 */
template<typename L, typename R>
DEVICEHOST()
decltype(auto) operator/(const L& l, const R& r)
{
    constexpr bool isScalarL = std::is_arithmetic<L>::value;
    constexpr bool isScalarR = std::is_arithmetic<R>::value;
    constexpr bool isVecL    = isExpression<L>::value;
    constexpr bool isVecR    = isExpression<R>::value;
    return detail::div<L, R, isScalarL, isScalarR, isVecL, isVecR>::eval(l, r);
}

/**
 * @brief Negate a vector expression (unary minus).
 *
 * @tparam E  Vector expression type.
 * @param[in] expr  Expression to negate.
 * @return Lazy negation expression (scales all components by −1).
 */
template<typename E>
DEVICEHOST()
detail::CWiseScale<E, int> operator-(
    const Expression<E, typename E::ComponentT>& expr)
{
    return detail::CWiseScale<E, int>(*static_cast<const E*>(&expr), -1);
}

/**
 * @brief Component-wise subtraction of two vector expressions.
 *
 * Scalar overload (DEVICEHOST). Packet operands route to the HOST overload.
 */
template<typename E1, typename E2,
    typename = std::enable_if_t<
        !feta::core::expr::anyPacketExpression<E1, E2>::value>>
DEVICEHOST()
detail::Diff<E1, E2> operator-(const Expression<E1, typename E1::ComponentT>& l,
    const Expression<E2, typename E2::ComponentT>& r)
{
    return detail::Diff<E1, E2>(
        *static_cast<const E1*>(&l), *static_cast<const E2*>(&r));
}

/** @brief Packet overload of operator- — host-only. */
template<typename E1, typename E2,
    typename = std::enable_if_t<
        feta::core::expr::anyPacketExpression<E1, E2>::value>,
    typename = void>
HOST()
detail::Diff<E1, E2> operator-(const Expression<E1, typename E1::ComponentT>& l,
    const Expression<E2, typename E2::ComponentT>& r)
{
    return detail::Diff<E1, E2>(
        *static_cast<const E1*>(&l), *static_cast<const E2*>(&r));
}

} // namespace expr
} // namespace vector
} // namespace feta
