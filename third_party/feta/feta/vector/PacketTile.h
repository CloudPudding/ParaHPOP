#pragma once

/**
 * @file feta/vector/PacketTile.h
 * @brief Stack-allocated K-slot tile of W-lane SIMD vectors.
 *
 * Packet analogue of ``feta::vector::Tile<T, D, K>``.  Where Tile holds
 * ``K`` D-dim vectors of scalars (one per sample), PacketTile holds
 * ``K`` D-dim vectors of W-lane packets (W samples per slot).  The
 * intended use is RK K-stage scratch in a SIMD packet body — the
 * whole compound stays in registers + L1, and the K-stage chain is a
 * compile-time fold same as the scalar Tile.
 *
 * Implementation: thin wrapper over ``PacketItem<T, D*K, W>`` with a
 * ``slotPtr<s>()`` accessor that returns ``Packet<T, W>*`` pointing at
 * slot ``s``'s first packet.  Symmetric with ``Tile::slotPtr<s>()``
 * which returns ``T*`` — so the same RK accumulator templates can
 * operate on either via ``auto*`` deduction (scalar vs packet element
 * type).
 *
 * No new expression-template plumbing: PacketItem already participates
 * in the packet evaluation machinery (``packetGet`` / ``packetStore``),
 * and slot access here is raw-pointer for the inner-loop pattern that
 * the parm RK accumulators use.
 */

#include "feta/vector/PacketItem.h"

namespace feta {
namespace vector {

/**
 * @brief Compile-time-fixed K-slot stack tile of W-lane D-dim packets.
 *
 * @tparam DataT      Scalar component type.
 * @tparam SlotDim    Number of components per slot (D).
 * @tparam NumSlots   Number of slots (K).
 * @tparam W          SIMD width (lanes per packet).
 */
template<typename DataT, dims_t SlotDim, idx_t NumSlots, idx_t W>
class PacketTile : public PacketItem<DataT, SlotDim * NumSlots, W> {
    using ParentT = PacketItem<DataT, SlotDim * NumSlots, W>;
    using Self    = PacketTile;

public:
    using PacketT = typename ParentT::PacketT;

    static constexpr dims_t SlotDims_  = SlotDim;
    static constexpr idx_t  NumSlots_  = NumSlots;
    static constexpr dims_t TotalDims_ = SlotDim * NumSlots;
    static constexpr idx_t  Width_     = W;

    PacketTile() = default;

    /**
     * @brief Raw pointer to the first packet of slot ``s``.
     *
     * Symmetric with ``Tile::slotPtr<s>()`` (which returns scalar
     * ``T*``) — here we return ``Packet<T, W>*``, ``SlotDim`` packets
     * starting at ``data() + s * SlotDim``.  Tight inner loops use
     * ``auto* kD = tile.template slotPtr<col>(); accD[d] += coef *
     * kD[d];`` — the ``+=``/``*`` resolve to packet ops on
     * ``Packet<T, W>``.
     */
    template<idx_t s>
    inline PacketT* slotPtr()
    {
        static_assert(
            s < NumSlots, "PacketTile::slotPtr<s>: slot index out of range");
        return this->data() + s * SlotDim;
    }

    template<idx_t s>
    inline const PacketT* slotPtr() const
    {
        static_assert(
            s < NumSlots, "PacketTile::slotPtr<s>: slot index out of range");
        return this->data() + s * SlotDim;
    }
};

} // namespace vector
} // namespace feta
