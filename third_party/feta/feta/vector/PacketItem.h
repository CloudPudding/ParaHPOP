/**
 * @file PacketItem.h
 *
 * Register-resident SIMD vector: holds one Packet<DataT, W> per component.
 *
 * PacketItem is the SIMD analogue of vector::Item. Where Item stores one
 * scalar per component (register-resident for a single sample), PacketItem
 * stores one SIMD packet per component (register-resident for W samples).
 *
 * The key use case is **single-pass fused evaluation**: `packetCapture`
 * materialises an intermediate expression into a PacketItem that stays in
 * SIMD registers, avoiding the memory round-trip of a two-pass approach.
 *
 * @code
 *   feta::cpu::packetTiledFor<Real>(0, N, [&](auto pi) {
 *       auto vw   = feta::cpu::packetCapture<Real>(qR.quatRotate(vR), pi);
 *       auto expr = vw + omR.cross(vw);
 *       feta::cpu::packetAssign(outR, expr, pi);
 *   });
 * @endcode
 */
#pragma once

#include "feta/core/simd/simd.h"
#include "feta/vector/expr/Expression.h"

namespace feta {
namespace vector {

/**
 * @brief Register-resident SIMD vector holding one Packet per component.
 *
 * @tparam DataT    Scalar type (e.g. double, float).
 * @tparam VecDims  Number of vector components.
 * @tparam W        SIMD width (number of lanes per packet).
 */
template<typename DataT, dims_t VectorDim, idx_t W>
class PacketItem
    : public expr::Expression<PacketItem<DataT, VectorDim, W>, DataT> {
public:
    using ComponentT = DataT;
    using PacketT    = simd::Packet<DataT, W>;

    static constexpr dims_t VecDims  = VectorDim;
    static constexpr idx_t Width     = W;
    static constexpr bool isLeaf     = true;
    static constexpr bool isWritable = true;
    static constexpr bool work       = false;

    // Expression tree annotations
    static constexpr expr::ExprKind kind = expr::ExprKind::Leaf;
    static constexpr dims_t depth        = 0;
    static constexpr idx_t arraySize     = 1;

    PacketItem() = default;

    /** @brief Read component `dim`. */
    template<dims_t dim>
    const PacketT& packet() const
    {
        static_assert(dim < VectorDim, "Component index out of range");
        return data_[dim];
    }

    /** @brief Write component `dim`. */
    template<dims_t dim>
    PacketT& packet()
    {
        static_assert(dim < VectorDim, "Component index out of range");
        return data_[dim];
    }

    /**
     * @brief Scalar get — **deliberately disabled at compile time**.
     *
     * PacketItem is a SIMD register type.  Its correct read path is the
     * free-function `packetGet<dim>(item, pi)` (from PacketOps.h), which
     * returns `item.packet<dim>()` with zero overhead.  The scalar
     * `get<dim>(SampleIndex)` path inherited from `Expression` must never
     * be reached: calling it would produce zeros for every lane (the
     * original silent-zero stub) instead of the actual packet data.
     *
     * Any instantiation of this overload indicates that a `PacketItem`
     * has been used as the source of a *scalar* expression-template
     * evaluation (e.g. via `RecursiveAssign`).  That is a programming
     * error — use `packetGet<dim>(item, pi)` or `item.packet<dim>()`
     * instead, and reach subsets via `item.subset<Offset, Len>()`.
     */
    template<dims_t dim>
    inline DataT get(const SampleIndex&) const
    {
        static_assert(dependentFalse_<dim>,
            "PacketItem::get<dim>() was called in a scalar expression context."
            " Use packetGet<dim>(item, pi) or item.packet<dim>() instead."
            " For position/velocity subsets use item.subset<Offset, Len>().");
        return DataT{};
    }

    template<dims_t dim>
    inline DataT get(const idx_t) const
    {
        static_assert(dependentFalse_<dim>,
            "PacketItem::get<dim>() was called in a scalar expression context."
            " Use packetGet<dim>(item, pi) or item.packet<dim>() instead."
            " For position/velocity subsets use item.subset<Offset, Len>().");
        return DataT{};
    }

    /**
     * @brief Zero every component packet.
     *
     * ``Packet<DataT, W>::default`` is uninitialised (matches
     * SIMD-register default semantics on AVX-512), so callers
     * needing a zero-initialised accumulator must call ``setZero()``
     * explicitly.  Compile-time-recursive over components.
     */
    inline void setZero() { setZero_<VectorDim - 1>(); }

    /**
     * @brief Raw pointer to the underlying packet array.
     *
     * Symmetric with ``Item<DataT, VectorDim>::data()``: ``Item::data()``
     * returns ``DataT*`` (one scalar per component); here ``data()``
     * returns ``PacketT*`` (one SIMD packet per component).  Tight
     * inner loops can use ``auto* p = item.data(); p[d] += coef *
     * other[d]`` and the deduction picks up the right element type
     * for both scalar (Item) and packet (PacketItem) state vectors.
     */
    inline PacketT* data() { return data_; }
    inline const PacketT* data() const { return data_; }

    /**
     * @brief Compile-time slice — return a new ``PacketItem`` holding
     *        ``SubLen`` consecutive component packets starting at
     *        ``Offset``.
     *
     * Symmetric with ``Item::segment<O, L>()`` (which returns a
     * ``ComponentView``); since ``PacketItem`` doesn't participate
     * in the scalar expression-template machinery the same way, we
     * return a new ``PacketItem<DataT, SubLen, W>`` by value
     * (compile-time-unrolled copy of the L sub-packets).  Used by
     * RHS adapters that split a 6-dim state into 3-dim pos/vel
     * sub-packets.
     */
    template<dims_t Offset, dims_t SubLen>
    inline PacketItem<DataT, SubLen, W> subset() const
    {
        static_assert(Offset + SubLen <= VectorDim,
            "PacketItem::subset<O, L>: out of range");
        PacketItem<DataT, SubLen, W> out;
        copyPackets_<SubLen - 1, Offset>(out, *this);
        return out;
    }

    /**
     * @brief Compile-time write-slice — copy ``SubLen`` consecutive
     *        component packets from ``other`` into this PacketItem
     *        starting at ``Offset``.
     *
     * Mirror of ``subset`` for write-back: lets RHS adapters repack
     * a (vel, acc) pair into a 6-dim out PacketItem after the
     * acceleration evaluator returns the 3-dim acc packet.
     */
    template<dims_t Offset, dims_t SubLen>
    inline void setSubset(const PacketItem<DataT, SubLen, W>& other)
    {
        static_assert(Offset + SubLen <= VectorDim,
            "PacketItem::setSubset<O>: out of range");
        copyPacketsFrom_<SubLen - 1, Offset>(*this, other);
    }

private:
    /** @brief Dependent-false helper: ensures the get<dim>() static_assert
     *  fires only when the function is instantiated, not at class
     *  instantiation time. */
    template<dims_t>
    static constexpr bool dependentFalse_ = false;

    /** @brief Compile-time-unrolled component zeroing. */
    template<dims_t D>
    inline void setZero_()
    {
        if constexpr (D >= 1)
            setZero_<D - 1>();
        data_[D] = PacketT::zero();
    }

    /** @brief Compile-time-unrolled copy: ``out[I] = src[Offset + I]``
     *  for I = 0..N. */
    template<dims_t N, dims_t Offset, dims_t SubLen, idx_t Wrest>
    static inline void copyPackets_(
        PacketItem<DataT, SubLen, Wrest>& out, const PacketItem& src)
    {
        if constexpr (N >= 1) {
            copyPackets_<N - 1, Offset>(out, src);
        }
        out.template packet<N>() = src.template packet<Offset + N>();
    }

    /** @brief Compile-time-unrolled write: ``dst[Offset + I] =
     *  src[I]`` for I = 0..N. */
    template<dims_t N, dims_t Offset, dims_t SubLen, idx_t Wrest>
    static inline void copyPacketsFrom_(
        PacketItem& dst, const PacketItem<DataT, SubLen, Wrest>& src)
    {
        if constexpr (N >= 1) {
            copyPacketsFrom_<N - 1, Offset>(dst, src);
        }
        dst.template packet<Offset + N>() = src.template packet<N>();
    }

    PacketT data_[VectorDim];
};

/**
 * @brief Convenience alias using the platform's preferred SIMD width.
 */
template<typename DataT, dims_t VecDims>
using NativePacketItem
    = PacketItem<DataT, VecDims, simd::PreferredWidth<DataT>>;

} // namespace vector
} // namespace feta
