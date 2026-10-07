#pragma once

#include <vector>

#include "feta/typedefs.h"

/**
 * @file IndexTraits.h
 * @brief Single source for the buffer index-type traits.
 *
 * Holds the primary templates (`IsIntIndex`/`IsBoolIndex`/`IsRange` and the
 * scalar `IsBool`/`IsInt`) plus the container specializations that need no
 * heavy type definitions (`std::vector<...>` and the forward-declared
 * `buffer::scalar::Array`). The RefArray/texture families specialize these
 * primaries next to their own type definitions (they need the full type), each
 * collapsed to a single `value = IsInt<T>::value` partial specialization.
 */

namespace feta {
namespace buffer {
namespace detail {

/* Valid indexes */
/** @brief Integer indexes */
template<typename T>
struct IsIntIndex {
    static constexpr bool value = false;
};
template<>
struct IsIntIndex<std::vector<int>> {
    static constexpr bool value = true;
};
template<>
struct IsIntIndex<std::vector<long int>> {
    static constexpr bool value = true;
};
template<>
struct IsIntIndex<std::vector<idx_t>> {
    static constexpr bool value = true;
};
template<>
struct IsIntIndex<std::vector<long unsigned int>> {
    static constexpr bool value = true;
};
/** @brief Boolean indexes */
template<typename T>
struct IsBoolIndex {
    static constexpr bool value = false;
};
template<>
struct IsBoolIndex<std::vector<bool>> {
    static constexpr bool value = true;
};

/** @brief Range type */
template<typename T>
struct IsRange {
    static constexpr bool value = false;
};

} // namespace detail


/** @brief Forward declare Scalar arrays */
namespace scalar {

/** @brief Distinguish boolean value */
template<typename T>
struct IsBool {
    static constexpr bool value = false;
};
template<>
struct IsBool<bool> {
    static constexpr bool value = true;
};

/** @brief Distinguish integer value */
template<typename T>
struct IsInt {
    static constexpr bool value = false;
};
template<>
struct IsInt<int> {
    static constexpr bool value = true;
};
template<>
struct IsInt<long int> {
    static constexpr bool value = true;
};
template<>
struct IsInt<idx_t> {
    static constexpr bool value = true;
};
template<>
struct IsInt<long unsigned int> {
    static constexpr bool value = true;
};

/** @brief Scalar array type */
template<typename DataT_>
class Array;
} // namespace scalar

namespace detail {

/** @brief Add index type traits for array */
template<typename DataT_>
struct IsIntIndex<scalar::Array<DataT_>> {
    static constexpr bool value = scalar::IsInt<DataT_>::value;
};

template<typename DataT_>
struct IsBoolIndex<scalar::Array<DataT_>> {
    static constexpr bool value = scalar::IsBool<DataT_>::value;
};

} // namespace detail

} // namespace buffer
} // namespace feta
