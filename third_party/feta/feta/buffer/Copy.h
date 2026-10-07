#pragma once

#include "feta/buffer/Info.h"

namespace feta {
namespace buffer {

namespace detail {
/**
 * @brief Select between memcpy-compatible and boolean-safe construction paths.
 *
 * Booleans cannot be safely copied with `memcpy`; this switcher specialises
 * that case.
 */
template<typename DataT>
struct Type {
    static constexpr bool isBool = false;
};
template<>
struct Type<bool> {
    static constexpr bool isBool = true;
};
} // namespace detail

/** @brief Handle the data copy across types */
template<typename DataT>
struct Copy {

    /** @brief Copy between pointers */
    static void copy(const DataT* from, DataT* to, const idx_t& size)
    {
        std::copy(from, from + size, to);
    }

    /** @brief Copy from std::vector to pointer */
    static void copy(const std::vector<DataT>& from, DataT* to)
    {
        if constexpr (detail::Type<DataT>::isBool) {
            for (idx_t i = 0; i < from.size(); i++)
                to[i] = from[i];
        } else {
            std::copy(from.data(), from.data() + from.size(), to);
        }
    }

    /** @brief Copy to std::vector from pointer */
    static void copy(const DataT* from, std::vector<DataT>& to)
    {
        if constexpr (detail::Type<DataT>::isBool) {
            for (idx_t i = 0; i < to.size(); i++)
                to[i] = from[i];
        } else {
            std::copy(from, from + to.size(), to.data());
        }
    }

    /** @brief Copy between info */
    static void copy(const Info<DataT>& from, Info<DataT>& to)
    {
        FETA_ASSERT(
            to.isValid(), "`To` buffer info points to non-allocated memory");
        FETA_ASSERT(from.isValid(),
            "`From` buffer info points to non-allocated memory");
        FETA_ASSERT(to.size() == from.size(), "Incompatible Sizes");
        copy(from.ptr, to.ptr, from.size());
    }

    /** @brief Copy from std::vector to Info */
    static void copy(const std::vector<DataT>& from, Info<DataT>& to)
    {
        FETA_ASSERT(
            to.isValid(), "`To` buffer info points to non-allocated memory");
        FETA_ASSERT(to.size() == from.size(), "Incompatible Sizes");
        copy(from, to.ptr);
    }

    /** @brief Copy from Info to std::vector */
    static void copy(const Info<DataT>& from, std::vector<DataT>& to)
    {
        FETA_ASSERT(
            from.isValid(), "`To` buffer info points to non-allocated memory");
        FETA_ASSERT(to.size() == from.size(), "Incompatible Sizes");
        copy(from.ptr, to);
    }
};

} // namespace buffer
} // namespace feta