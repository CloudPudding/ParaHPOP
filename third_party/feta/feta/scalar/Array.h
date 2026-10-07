#pragma once

#include "feta/scalar/texture/Array.h"
#include "feta/buffer/detail/IndexTraits.h"

namespace feta {
namespace scalar {
namespace detail {

/** @brief Non-owning reference to scalar::Array */
template<typename DataT, bool work, bool MaybeVolatile = false>
using RefArray = texture::detail::RefArray<DataT, work, false, MaybeVolatile>;
} // namespace detail

/**
 * @brief Owning host+device array of scalar values without texture support.
 *
 * Thin wrapper over `texture::Array<DataT, false>` that exposes `GRef` and
 * `WRef` aliases for use in kernel launches.
 *
 * @tparam DataT  Scalar element type (e.g. `double`, `float`, `bool`, `int`).
 */
template<typename DataT>
class Array : public texture::Array<DataT, false> {
    using ParentT = texture::Array<DataT, false>;

public:
    /** @brief Inherit assignment operator */
    using ParentT::operator=;
    /* Expose relevant types */
    using StreamT = typename ParentT::StreamT;
    using TexT    = typename ParentT::TexT;
    using BufT    = typename ParentT::BufT;
    /** @brief Explicit reference to scalar::RefArray.
     *
     * The 2nd parameter ``MaybeVolatile`` propagates the volatility
     * request down to the underlying RefArray.  Defaults to ``false``
     * for full backward-compatibility with all existing 1-arg call
     * sites.  Downstream wrappers (parm/brie/cudaj) thread this
     * through their own RefXXX template chains so a project-level
     * VolatileRef can fold all the way down to volatile pointers. */
    template<bool work, bool MaybeVolatile = false>
    using Ref = detail::RefArray<DataT, work, MaybeVolatile>;
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

    /** @brief Construct copying the data from the given std::vector<DataT> */
    explicit Array(const BufT& vec)
        : ParentT{ vec }
    {
    }

    /**
     * @brief Host constructor
     *
     */
    Array(const idx_t& size, const DataT& initVal = 0)
        : ParentT{ size, initVal }
    {
    }

    /** @brief Create an array clone, copying the data */
    Array clone() const { return Array(std::move(ParentT::clone())); }
};

} // namespace scalar

/* Add the index type traits (value follows the element-type predicate, so one
   partial specialization replaces the per-int-type list). */
namespace buffer {
namespace detail {
template<typename T, bool work>
struct IsBoolIndex<feta::scalar::detail::RefArray<T, work>> {
    static constexpr bool value = scalar::IsBool<T>::value;
};
template<typename T, bool work>
struct IsIntIndex<feta::scalar::detail::RefArray<T, work>> {
    static constexpr bool value = scalar::IsInt<T>::value;
};
} // namespace detail
} // namespace buffer

} // namespace feta
