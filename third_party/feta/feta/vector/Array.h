#pragma once

#include "feta/scalar/Array.h"
#include "feta/vector/texture/Array.h"

namespace feta {
namespace vector {
namespace detail {

/** @brief Non-owning reference to a non-texture Vector Array */
template<typename DataT, dims_t VectorDim, bool work, bool MaybeVolatile = false>
using RefArray =
    texture::detail::RefArray<DataT, VectorDim, work, false, MaybeVolatile>;
} // namespace detail

/**
 * @brief Owning host+device array of `VectorDim`-component vectors without texture support.
 *
 * Thin wrapper over `texture::Array<DataT, VectorDim, false>` that exposes
 * `GRef` and `WRef` aliases for use in kernel launches.
 *
 * @tparam DataT      Scalar element type (e.g. `double`, `float`).
 * @tparam VectorDim  Compile-time number of components per vector.
 */
template<typename DataT, dims_t VectorDim>
class Array : public texture::Array<DataT, VectorDim, false> {
    using ParentT = texture::Array<DataT, VectorDim, false>;

public:
    /** @brief Inherit the assignment operators */
    using ParentT::operator=;
    /* Expose the relevant types */
    using StreamT = typename ParentT::StreamT;
    using TexT    = typename ParentT::TexT;
    using BufT    = typename ParentT::BufT;

    /** @brief Explicit reference to RefArray.
     *
     * The 2nd parameter ``MaybeVolatile`` propagates the volatility
     * request down to the underlying RefArray.  Defaults to ``false``
     * for full backward-compatibility with all existing 1-arg call
     * sites.  Downstream wrappers (parm/brie/cudaj) thread this
     * through their own RefXXX template chains so a project-level
     * VolatileRef can fold all the way down to volatile pointers. */
    template<bool work, bool MaybeVolatile = false>
    using Ref = detail::RefArray<DataT, VectorDim, work, MaybeVolatile>;
    /** @brief Global reference type */
    using GRef = Ref<false>;
    /** @brief Work reference type */
    using WRef = Ref<true>;
    /** @brief Volatile work reference type.
     *
     * Use when the underlying storage is shared memory and the consumer
     * needs to defeat compiler caching / stack-backed mirrors.  Volatile
     * is only meaningful with ``work=true``. */
    using VolatileRef = Ref<true, true>;

    /** @brief Inherit constructors */
    Array(ParentT&& other)
        : ParentT{ std::move(other) }
    {
    }

    /**
     * @brief Construct a host-side array of `size` vectors, initialised to `initVal`.
     *
     * @param size      Number of vectors to allocate.
     * @param initVal   Initial scalar value for every component (default 0).
     * @param capacity  Optional pre-allocated capacity; uses `size` when 0.
     */
    Array(
        const idx_t& size, const DataT& initVal = 0, const idx_t& capacity = 0)
        : ParentT{ size, initVal, capacity }
    {
    }

    /** @brief Construct from std::vector of std::vector */
    Array(const BufT& vec)
        : ParentT{ vec }
    {
    }

    /** @brief Create an array clone, copying the data */
    Array clone() const { return Array(std::move(ParentT::clone())); }
};

} // namespace vector
} // namespace feta
