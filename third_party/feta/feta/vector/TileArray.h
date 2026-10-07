#pragma once

/**
 * @file feta/vector/TileArray.h
 * @brief Heap-allocated SoA-of-tiles: one ``Tile<T, D, K>`` per sample.
 *
 * A ``TileArray<T, D, K>`` is the heap counterpart of
 * ``feta::vector::Tile`` — it is just a thin specialisation of
 * ``feta::vector::Array<T, D * K>`` that exposes a ``slot<s>()``
 * accessor on its reference type.  The underlying storage and
 * access patterns are unchanged: the K slots occupy contiguous
 * component ranges, so per-slot access is a compile-time
 * ``segment<s * D, D>`` slice through the existing expression-
 * template machinery.
 *
 * Intended consumers: RK integrators that need an SoA scratch
 * holding K derivative stages of D components per sample (e.g.
 * ``parm::integrate::detail::DStates`` is a specialisation of
 * this type with extra recyclable scratch fields).
 */

#include "feta/vector/Array.h"

namespace feta {
namespace vector {

namespace detail {

/**
 * @brief Reference type for ``TileArray<T, D, K>`` — adds a per-slot
 *        accessor on top of the inherited ``feta::vector::Array``
 *        reference machinery.
 *
 * Inherits the parent ``RefArray<T, D*K, work>``, so all expression
 * template ops (``segment``, component access, ``operator[]``,
 * etc.) are available as on any feta vector array reference.
 *
 * @tparam ParentRefT  The parent ``feta::vector::Array<T, D *
 *                     K>::Ref<work>`` instantiation.
 * @tparam SlotDim     Per-slot vector dimension (D).
 * @tparam NumSlots    Number of slots (K).
 */
template<typename ParentRefT, dims_t SlotDim, idx_t NumSlots>
class RefTileArray : public ParentRefT {
    using Self = RefTileArray;

public:
    static constexpr dims_t SlotDims_ = SlotDim;
    static constexpr idx_t  NumSlots_ = NumSlots;

    /** @brief Default constructor — leaves the parent reference
     *  default-constructed (POD-like).  No ``DEVICEHOST()`` here:
     *  nvcc warns when ``__device__`` is applied to a function
     *  defaulted on first declaration; the parent's default ctor is
     *  itself ``DEVICEHOST``. */
    RefTileArray() = default;

    /** @brief Slice-construct from an instance of the parent
     *  reference type. */
    DEVICEHOST() RefTileArray(const ParentRefT& parent)
        : ParentRefT{ parent }
    {
    }

    /**
     * @brief Mutable view onto slot ``s`` for the bound sample range
     *        — delegates to ``segment<s * SlotDim, SlotDim>``.
     *
     * Useful when callers want named per-slot access matching the
     * stack ``Tile<T, D, K>::slot<s>()`` shape.
     */
    template<idx_t s>
    DEVICEHOST() inline decltype(auto) slot()
    {
        static_assert(
            s < NumSlots, "TileArray::Ref::slot<s>: slot index out of range");
        return this->template segment<s * SlotDim, SlotDim>();
    }

    /** @brief Read-only view onto slot ``s``. */
    template<idx_t s>
    DEVICEHOST() inline decltype(auto) slot() const
    {
        static_assert(
            s < NumSlots, "TileArray::Ref::slot<s>: slot index out of range");
        return this->template segment<s * SlotDim, SlotDim>();
    }
};

} // namespace detail

/**
 * @brief Heap-allocated SoA-of-tiles container.
 *
 * Inherits ``feta::vector::Array<DataT, SlotDim * NumSlots>``; only
 * adds the slot-aware ``Ref<work>`` alias and the matching
 * ``hostRef()`` / ``deviceRef()`` / ``ref()`` factories.
 *
 * @tparam DataT     Scalar component type.
 * @tparam SlotDim   Per-slot vector dimension (D).
 * @tparam NumSlots  Number of slots per sample (K).
 */
template<typename DataT, dims_t SlotDim, idx_t NumSlots>
class TileArray : public Array<DataT, SlotDim * NumSlots> {
    using ParentT = Array<DataT, SlotDim * NumSlots>;
    using Self    = TileArray;

public:
    /** @brief Per-slot vector dimension. */
    static constexpr dims_t SlotDims_ = SlotDim;
    /** @brief Number of slots per sample. */
    static constexpr idx_t  NumSlots_ = NumSlots;
    /** @brief Total component count per sample. */
    static constexpr dims_t TotalDims_ = SlotDim * NumSlots;

    /** @brief Reference type — augments the parent reference with
     *  the slot accessor.  ``MaybeVolatile`` is forwarded to the
     *  parent ``Array``'s ``Ref<work, MaybeVolatile>`` so the inner
     *  flat buffer carries the volatility flag. */
    template<bool work, bool MaybeVolatile = false>
    using Ref = detail::RefTileArray<
        typename ParentT::template Ref<work, MaybeVolatile>, SlotDim, NumSlots>;
    using GRef        = Ref<false>;
    using WRef        = Ref<true>;
    /** @brief Volatile work reference */
    using VolatileRef = Ref<true, true>;

    /** @brief Stream type, inherited. */
    using StreamT = typename ParentT::StreamT;
    using BufT    = typename ParentT::BufT;

    /** @brief Inherit Array's assignment operators. */
    using ParentT::operator=;

    /** @brief Construct from a parent rvalue (used by ``clone``). */
    TileArray(ParentT&& other)
        : ParentT{ std::move(other) }
    {
    }

    /**
     * @brief Construct a host-side tile array of ``size`` samples,
     *        initialised to ``initVal``.
     */
    TileArray(
        const idx_t& size, const DataT& initVal = 0, const idx_t& capacity = 0)
        : ParentT{ size, initVal, capacity }
    {
    }

    /** @brief Reference to the host array (slot-aware). */
    GRef hostRef() const { return GRef{ ParentT::hostRef() }; }
    GRef ref() const { return hostRef(); }

    /** @brief Reference to the device array (slot-aware). */
    GRef deviceRef() const { return GRef{ ParentT::deviceRef() }; }

    /** @brief Clone (deep copy). */
    TileArray clone() const { return TileArray(std::move(ParentT::clone())); }
};

} // namespace vector
} // namespace feta
