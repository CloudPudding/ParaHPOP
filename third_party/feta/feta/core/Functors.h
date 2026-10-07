#pragma once

#ifdef FETA_CPU_ONLY
#include <cmath>
#endif

#include "feta/math/math.h"

namespace feta {
namespace core {
namespace unaryOps {

namespace detail {
template<typename OP, typename I>
struct UnaryFunctor {
    DEVICEHOST() constexpr auto operator()(const I& x) const
    {
        return OP::eval(x);
    }
};
} // namespace detail

/** @brief Unary functor which returns itself */
template<typename T>
struct self : public detail::UnaryFunctor<self<T>, T> {
    DEVICEHOST() static constexpr T eval(const T& x) { return x; }
};

/** @brief Unary functor which squares its input. */
template<typename T>
struct square : public detail::UnaryFunctor<square<T>, T> {
    DEVICEHOST() static constexpr T eval(const T& x) { return x * x; }
};

/** @brief Unary functor which returns the absolute value of its input. */
template<typename T>
struct abs : public detail::UnaryFunctor<abs<T>, T> {
    DEVICEHOST() static constexpr T eval(const T& x) { return math::abs(x); }
};

/** @brief Unary functor which returns isfinite(x). */
template<typename T>
struct isFinite : public detail::UnaryFunctor<isFinite<T>, T> {
    DEVICEHOST() static constexpr bool eval(const T& x)
    {
        return math::isfinite(x);
    }
};

/** @brief Unary functor which implements the logical NOT operation. */
struct logicalNot : public detail::UnaryFunctor<logicalNot, bool> {
    DEVICEHOST() static constexpr bool eval(const bool& x) { return !x; }
};


} // namespace unaryOps

namespace binaryOps {

namespace detail {
template<typename OP, typename T>
struct BinaryFunctor {
    DEVICEHOST()
    constexpr decltype(auto) operator()(const T& x, const T& y) const
    {
        return OP::eval(x, y);
    }
};
} // namespace detail

/** @brief Binary functor which returns the sum of its inputs. */
template<typename T>
struct sum : public detail::BinaryFunctor<sum<T>, T> {
    DEVICEHOST() static constexpr T eval(const T& x, const T& y)
    {
        return x + y;
    }
};

/** @brief Binary functor which returns maximum of its inputs. */
template<typename T>
struct max : public detail::BinaryFunctor<max<T>, T> {
    DEVICEHOST() static constexpr T eval(const T& x, const T& y)
    {
        return math::max(x, y);
    }
};

/** @brief Binary function that returns the product of its inputs. */
template<typename T>
struct times : public detail::BinaryFunctor<times<T>, T> {
    DEVICEHOST() static constexpr T eval(const T& x, const T& y)
    {
        return x * y;
    }
};

/** @brief Binary functor which implements the logical AND operation. */
struct logicalAnd : public detail::BinaryFunctor<logicalAnd, bool> {
    DEVICEHOST() static constexpr bool eval(const bool& x, const bool& y)
    {
        return x && y;
    }
};

/** @brief Binary functor which implements the logical OR operation. */
struct logicalOr : public detail::BinaryFunctor<logicalOr, bool> {
    DEVICEHOST() static constexpr bool eval(const bool& x, const bool& y)
    {
        return x || y;
    }
};

} // namespace binaryOps
} // namespace core
} // namespace feta
