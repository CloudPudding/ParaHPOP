#pragma once

#include "brie/core/BodyInterpolator.h"
#include "brie/core/Traverser.h"

namespace brie {
namespace core {

/** @brief A collection of chain leaf accessors. The leaf type is generic:
 * it must expose `getValues`, `getDerivatives`, `getValuesAndDerivatives`
 * (scalar + array-with-sign variants). The walker logic is leaf-agnostic. */
template<class LeafT>
class RefChainInterpolator
    : public feta::scalar::Array<LeafT>::GRef {
    using ParentT = typename feta::scalar::Array<LeafT>::GRef;

    /* Array types */
    template<bool work>
    using Vec3RArrT =
        typename feta::vector::Array<Real, 3>::template Ref<work>::HandleT;
    template<bool work>
    using Vec6RArrT =
        typename feta::vector::Array<Real, 6>::template Ref<work>::HandleT;

public:
    /** @brief Factory method to construct from parent */
    DEVICEHOST()
    static RefChainInterpolator make(
        const ParentT& parent, const idx_t* const chain)
    {
        return RefChainInterpolator{ parent, chain };
    }

    /** @brief Run a position retrieval, starting from the given target
     * and running until the given center */
    DEVICEHOST()
    void getPosition(
        Vec3R& out, const Real& epoch, const ChainWalker& walker) const
    {
        idx_t t = walker.start_;
        for (idx_t i = 0; i < walker.steps_; i++) {
            (*this)[t].getValues(out, epoch);
            t = chain_[t];
        }
    }

    /** @brief Array version */
    template<bool work>
    DEVICEHOST()
    void getPosition(const SampleIndex& index, Vec3RArrT<work>& out,
        const Real& epoch, const ChainWalker& walker, const Real& sign) const
    {
        idx_t t = walker.start_;
        for (idx_t i = 0; i < walker.steps_; i++) {
            (*this)[t].template getValues<work>(index, out, epoch, sign);
            t = chain_[t];
        }
    }

    /** @brief Run a velocity retrieval, starting from the given target and
     * running until the given center */
    DEVICEHOST()
    void getVelocity(
        Vec3R& out, const Real& epoch, const ChainWalker& walker) const
    {
        idx_t t = walker.start_;
        for (idx_t i = 0; i < walker.steps_; i++) {
            (*this)[t].getDerivatives(out, epoch);
            t = chain_[t];
        }
    }

    /** @brief Array version */
    template<bool work>
    DEVICEHOST()
    void getVelocity(const SampleIndex& index, Vec3RArrT<work>& out,
        const Real& epoch, const ChainWalker& walker, const Real& sign) const
    {
        idx_t t = walker.start_;
        for (idx_t i = 0; i < walker.steps_; i++) {
            (*this)[t].template getDerivatives<work>(index, out, epoch, sign);
            t = chain_[t];
        }
    }

    /** @brief Run a full state retrieval, starting from the given target and
     * running until the given center */
    DEVICEHOST()
    void getPositionAndVelocity(
        Vec6R& out, const Real& epoch, const ChainWalker& walker) const
    {
        idx_t t = walker.start_;
        for (idx_t i = 0; i < walker.steps_; i++) {
            (*this)[t].getValuesAndDerivatives(out, epoch);
            t = chain_[t];
        }
    }

    /** @brief Array version */
    template<bool work>
    DEVICEHOST()
    void getPositionAndVelocity(const SampleIndex& index, Vec6RArrT<work>& out,
        const Real& epoch, const ChainWalker& walker, const Real& sign) const
    {
        idx_t t = walker.start_;
        for (idx_t i = 0; i < walker.steps_; i++) {
            (*this)[t].template getValuesAndDerivatives<work>(
                index, out, epoch, sign);
            t = chain_[t];
        }
    }

    /** @brief Per-thread value-returning chain walker — position only.
     *
     *  Accumulates ``sign * (*this)[t].getValues(idx, epoch)`` over all
     *  chain steps and returns the resulting ``Vec3R`` by value.  Mirrors
     *  the array-path overload's semantics (``out[index] += sign * cached``)
     *  but with a stack-local accumulator so the leaf can be a cache leaf
     *  (which requires a ``SampleIndex`` to pick a slot) and the result
     *  lives in registers — natural shape for per-thread consumers.
     *
     *  Only instantiable for leaf types with a value-returning
     *  ``getValues(SampleIndex, Real)`` overload (i.e. cache leaves). */
    DEVICEHOST() Vec3R getPosition(const SampleIndex& index, const Real& epoch,
        const ChainWalker& walker, const Real& sign) const
    {
        Vec3R out = Vec3R::Zeros();
        idx_t t = walker.start_;
        for (idx_t i = 0; i < walker.steps_; i++) {
            out += sign * (*this)[t].getValues(index, epoch);
            t = chain_[t];
        }
        return out;
    }

    /** @brief Packet (W-lane) chain-walker for position.
     *
     *  Mirrors ``getPosition(SampleIndex, ...)`` but operates on a
     *  W-lane packet of per-lane epochs.  COI is uniform across the
     *  packet (the caller — ``RefEphUnit::iGetPositionPacket`` — has
     *  already enforced uniform-COI via the predicate guard).  Each
     *  chain step calls the leaf's ``iGetValuesPacket(out, epoch)``
     *  which routes through the chebyshev packet evaluator (or per-
     *  lane scalar fallback when intervals diverge within the
     *  packet).  Sign baked into a packet broadcast for the
     *  multiplication.
     *
     *  PR-6δ-B2 packet ephemeris query.
     */
    template<idx_t W>
    inline void getPositionPacket(
        feta::vector::PacketItem<Real, 3, W>& out,
        const feta::simd::Packet<Real, W>& epochPkt,
        const ChainWalker& walker, const Real& sign) const
    {
        using PacketT = feta::simd::Packet<Real, W>;
        const PacketT signPkt = PacketT::broadcast(sign);
        idx_t t = walker.start_;
        for (idx_t i = 0; i < walker.steps_; i++) {
            /* Step accumulator (per-lane signed contribution). */
            feta::vector::PacketItem<Real, 3, W> step;
            step.setZero();
            (*this)[t].template iGetValuesPacket<W>(
                step, epochPkt);
            out.template packet<0>() = out.template packet<0>()
                + signPkt * step.template packet<0>();
            out.template packet<1>() = out.template packet<1>()
                + signPkt * step.template packet<1>();
            out.template packet<2>() = out.template packet<2>()
                + signPkt * step.template packet<2>();
            t = chain_[t];
        }
    }

    /** @brief Per-thread value-returning chain walker — velocity only. */
    DEVICEHOST() Vec3R getVelocity(const SampleIndex& index, const Real& epoch,
        const ChainWalker& walker, const Real& sign) const
    {
        Vec3R out = Vec3R::Zeros();
        idx_t t = walker.start_;
        for (idx_t i = 0; i < walker.steps_; i++) {
            out += sign * (*this)[t].getDerivatives(index, epoch);
            t = chain_[t];
        }
        return out;
    }

    /** @brief Per-thread value-returning chain walker — full state. */
    DEVICEHOST() Vec6R getPositionAndVelocity(const SampleIndex& index,
        const Real& epoch, const ChainWalker& walker, const Real& sign) const
    {
        Vec6R out = Vec6R::Zeros();
        idx_t t = walker.start_;
        for (idx_t i = 0; i < walker.steps_; i++) {
            out += sign * (*this)[t].getValuesAndDerivatives(index, epoch);
            t = chain_[t];
        }
        return out;
    }

    /* The chain item */
    const idx_t* chain_ = nullptr;
};

/** @brief The chain interpolator (generic on leaf accessor type). */
template<class LeafT>
class ChainInterpolator
    : public feta::scalar::Array<LeafT> {
    using ParentT = feta::scalar::Array<LeafT>;

public:
    /** @brief Expose the reference type.  ``MaybeVolatile`` is accepted
     * (ignored) for uniform 2-arg shape across containers —
     * RefChainInterpolator is work-invariant and not yet volatile-capable. */
    template<bool work, bool MaybeVolatile = false>
    using Ref = RefChainInterpolator<LeafT>;
    /** @brief Expose the global reference type */
    using GRef = Ref<false>;

    /** @brief Factory method to construct a flexible item */
    static ChainInterpolator flexible()
    {
        return ChainInterpolator(ParentT::flexible());
    }

    /** @brief Construct from size */
    ChainInterpolator(const idx_t& size)
        : ParentT{ size, LeafT{} }
    {
    }

    /** @brief Construct from parent type */
    ChainInterpolator(ParentT&& parent)
        : ParentT{ std::move(parent) }
    {
    }

    /** @brief Copy constructor is forbidden */
    ChainInterpolator(ChainInterpolator& other)       = delete;
    ChainInterpolator(const ChainInterpolator& other) = delete;

    /** @brief Move constructor is allowed */
    ChainInterpolator(ChainInterpolator&& other)
        : ParentT{ std::move(other) }
        , chain_{ std::exchange(other.chain_, nullptr) }
    {
    }

    /** @brief Copy assignment is forbidden */
    ChainInterpolator& operator=(ChainInterpolator& other)       = delete;
    ChainInterpolator& operator=(const ChainInterpolator& other) = delete;

    /** @brief Move assignment is allowed */
    ChainInterpolator& operator=(ChainInterpolator&& other)
    {
        ParentT::operator=(std::move(other));
        chain_ = std::exchange(other.chain_, nullptr);
        return *this;
    }

    /** @brief Bind to the given container */
    void bind(const idx_t* const ptr) const { chain_ = ptr; }

    /** @brief return a host reference */
    GRef hostRef() const { return GRef::make(ParentT::hostRef(), chain_); }
    GRef ref() const { return hostRef(); }

#ifndef BRIE_CPU_ONLY
    /** @brief return a device reference */
    GRef deviceRef() const { return GRef::make(ParentT::deviceRef(), chain_); }
#endif

protected:
    mutable const idx_t* chain_ = nullptr;
};

/** @brief Backwards-compat alias for the chebyshev ephemeris path.
 * Existing call sites that previously named `ChainInterpolator<UseTexture>`
 * should use this alias.
 *
 * The cache-flavour aliases `BodyCacheChain` / `RefBodyCacheChain` are
 * declared in `brie/core/BodyCache.h` (where the leaf type lives), to
 * avoid pulling cache-side includes into this file. */
template<bool UseTexture>
using EphChainInterpolator = ChainInterpolator<BodyInterpolator<UseTexture>>;
template<bool UseTexture>
using RefEphChainInterpolator = RefChainInterpolator<BodyInterpolator<UseTexture>>;

} // namespace core
} // namespace brie