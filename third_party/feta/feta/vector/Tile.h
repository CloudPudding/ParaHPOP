#pragma once

/**
 * @file feta/vector/Tile.h
 * @brief Stack-allocated K-slot tile of D-dim vectors.
 *
 * A ``Tile<T, D, K>`` is the on-stack analogue of ``feta::vector::Array``
 * with a fixed compile-time number of slots: it holds ``K`` vectors of
 * dimension ``D``, contiguously, and exposes each slot as an
 * expression-template-friendly view via ``slot<s>()``.  The intended
 * use case is RK-style per-sample scratch storage (``K`` derivative
 * stages of ``D`` components each), where the whole compound stays in
 * registers / L1 and the K-stage chain is a compile-time fold.
 *
 * The implementation is just ``Item<T, D*K>`` with a thin slot
 * accessor that delegates to feta's existing ``segment<offset, len>``
 * machinery — so no new expression-template plumbing is required.
 *
 * The heap-allocated counterpart is ``feta::vector::TileArray<T, D,
 * K>`` (one tile per sample, SoA-layout); both share the same
 * conceptual surface so callers can write generic K-stage code that
 * targets either.
 */

#include "feta/vector/Item.h"

namespace feta {
namespace vector {

/**
 * @brief Compile-time-fixed K-slot stack tile of D-dim vectors.
 *
 * @tparam DataT      Scalar component type (e.g. ``double``, ``float``).
 * @tparam SlotDim    Number of components per slot (the "D" of "K-stage").
 * @tparam NumSlots   Number of slots (the "K" of "K-stage").
 *
 * Storage is a single ``Item<DataT, SlotDim * NumSlots>`` — slots
 * occupy contiguous component ranges.  Inheritance is public so the
 * full Item / Expression surface is available on the Tile (operators,
 * ``segment``, ``setZero``, etc.).  ``slot<s>()`` is a thin alias
 * over ``segment<s * SlotDim, SlotDim>()``.
 */
template<typename DataT, dims_t SlotDim, idx_t NumSlots>
class Tile : public Item<DataT, SlotDim * NumSlots> {
    using ParentT = Item<DataT, SlotDim * NumSlots>;
    using Self    = Tile;

public:
    /** @brief Per-slot vector dimension. */
    static constexpr dims_t SlotDims_  = SlotDim;
    /** @brief Number of slots. */
    static constexpr idx_t  NumSlots_  = NumSlots;
    /** @brief Total component count (for the inherited Item view). */
    static constexpr dims_t TotalDims_ = SlotDim * NumSlots;

    /** @brief Per-slot Item type — useful when callers need to copy a
     *  slot into a stand-alone vector. */
    using SlotItemT = Item<DataT, SlotDim>;

    /** @brief Inherit ``Item``'s constructors (default zero-init,
     *  initial-value, expression-eval, buffer assignment). */
    using ParentT::ParentT;
    using ParentT::operator=;

    /** @brief Default constructor — zero-initialises all slots. */
    DEVICEHOST() Tile() : ParentT{} {}

    /**
     * @brief Mutable view onto slot ``s`` as an expression-friendly
     *        D-dim view.
     *
     * The returned object is a ``ComponentView`` — it participates in
     * lazy expressions both as LHS (assignment) and RHS (read in a
     * larger expression).  Identical to ``segment<s * SlotDim,
     * SlotDim>()``.
     *
     * @tparam s  Compile-time slot index ``< NumSlots``.
     */
    template<idx_t s>
    DEVICEHOST() inline decltype(auto) slot()
    {
        static_assert(s < NumSlots, "Tile::slot<s>: slot index out of range");
        return this->template segment<s * SlotDim, SlotDim>();
    }

    /** @brief Read-only view onto slot ``s``. */
    template<idx_t s>
    DEVICEHOST() inline decltype(auto) slot() const
    {
        static_assert(s < NumSlots, "Tile::slot<s>: slot index out of range");
        return this->template segment<s * SlotDim, SlotDim>();
    }

    /**
     * @brief Raw pointer to the start of slot ``s``'s components in
     *        the underlying flat storage.
     *
     * Useful for tight inner loops where the expression-template
     * machinery would obscure the access pattern (e.g. RK substage
     * accumulation: ``ptr[d] += coef * kPtr[d]``).  All slots are
     * contiguous in the same buffer; ``slotPtr<s>()`` is just
     * ``data() + s * SlotDim``.
     */
    template<idx_t s>
    DEVICEHOST() inline DataT* slotPtr()
    {
        static_assert(s < NumSlots, "Tile::slotPtr<s>: slot index out of range");
        return this->data() + s * SlotDim;
    }

    template<idx_t s>
    DEVICEHOST() inline const DataT* slotPtr() const
    {
        static_assert(s < NumSlots, "Tile::slotPtr<s>: slot index out of range");
        return this->data() + s * SlotDim;
    }
};

} // namespace vector
} // namespace feta
