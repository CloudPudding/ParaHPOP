#pragma once

#include <experimental/type_traits> // std::experimental::is_detected
#include <type_traits>

namespace feta {
namespace core {

/** @brief Helper for static assertions that need to evaluate to false. */
template<typename T>
struct alwaysFalse {
    static constexpr bool value = false;
};

/** @brief Additional helper for enum-assertions that need to evaluate to false
 */
template<unsigned int i>
struct enumAlwaysFalse {
    static constexpr bool value = false;
};

namespace expr {

/**
 * @brief If `T` is a vector expression type, the member constant `value` is
 * `true`. Otherwise, it is `false`.
 */
template<typename T>
struct isExpression {
    template<typename U>
    using has_isVectorExpression_t = decltype(U::isVectorExpression);

    // True if T has a data member called isVectorExpression
    inline static constexpr bool value
        = std::experimental::is_detected<has_isVectorExpression_t, T>::value;
};

/**
 * @brief If `T` is directly accessible, the member constant `value` is `true`.
 *
 * A type is directly accessible if it has a method `get<dim>()`. This is in
 * contrast to, e.g., `VecNTArray`, which only has a `get<dim>(SampleIndex)`
 * because it requires specifying which vector in the array to access.
 * Examples of direct-accessible types: `VecView`, `VecNT`.
 */
template<typename T>
struct hasDirectAccess {
    /// @cond INTERNAL
    template<typename U>
    using has_get_t = decltype(std::declval<U&>().template get<0>());
    /// @endcond

    static constexpr bool value
        = std::experimental::is_detected<has_get_t, T>::value;
};

/**
 * @brief Detect packet-shaped expression types.
 *
 * A packet expression carries a static ``Width`` member (e.g. ``PacketItem``,
 * Sum/Diff over packets, etc.).  When this trait fires, the expression
 * stores or composes SIMD-register fields (``__m256d`` / ``__m512d``)
 * which nvcc rejects in device-side parses.  Code that touches such
 * types must therefore stay ``__host__``-only — never ``DEVICEHOST``.
 */
template<typename T>
struct isPacketExpression {
    template<typename U>
    using has_Width_t = decltype(U::Width);

    static constexpr bool value
        = std::experimental::is_detected<has_Width_t,
            std::remove_cv_t<std::remove_reference_t<T>>>::value;
};

template<typename L, typename R>
struct anyPacketExpression {
    static constexpr bool value
        = isPacketExpression<L>::value || isPacketExpression<R>::value;
};

} // namespace expr

/* Compatibility aliases: ``alwaysFalse`` + ``enumAlwaysFalse`` were
 * historically declared inside the ``expr`` sub-namespace.  They were
 * promoted to ``feta::core`` so the math.h static_asserts resolve via
 * the natural ``core::alwaysFalse`` lookup from inside ``feta::math``.
 * Old call sites that still qualify with ``core::expr::alwaysFalse``
 * are kept working via these aliases. */
namespace expr {
template<typename T>
using alwaysFalse = ::feta::core::alwaysFalse<T>;
template<unsigned int i>
using enumAlwaysFalse = ::feta::core::enumAlwaysFalse<i>;
} // namespace expr

} // namespace core
} // namespace feta
