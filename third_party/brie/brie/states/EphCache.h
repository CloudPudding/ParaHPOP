#pragma once

#ifndef BRIE_CPU_ONLY
#include <cuda.h>
#endif

#include <feta/feta.h>
#include <parm/util/Observer.h>
#include <vector>

#include "brie/core/BodyCache.h"
#include "brie/core/ChainInterpolator.h"
#include "brie/core/Traverser.h"
#include "brie/states/EphUnit.h"
#include "brie/states/detail/EphCacheFill.h"
#include "brie/typedefs.h"

namespace brie {
namespace states {

/* Forward declaration */
template<bool UseTexture, idx_t Dim,
    feta::core::memory::Device DeviceT
    = feta::core::memory::Device::CUDA_DEVICE>
class EphCache;

/**
 * @brief Non-owning device-/host-side reference to an `EphCache`.
 *
 * Mirrors `RefEphUnit`'s shape (just `traverser_` + `chain_`) so that
 * kernels capture the same per-kernel-parameter footprint as today's
 * EphUnit path.  The leaf type lives inside `chain_` and is a `BodyCache`
 * (POD pointer + counts), not a chebyshev `BodyInterpolator`.
 *
 * The query API mirrors `RefEphUnit::iGet*` array-path methods exactly.
 * Scalar (single-epoch) queries are intentionally absent — the cache has
 * no continuous-epoch semantics; epoch arguments are accepted only for
 * API parity with `RefEphUnit` and ignored at the leaf.
 */
template<bool work, idx_t Dim>
class RefEphCache {
    template<bool iwork>
    using Vec3RArrT =
        typename feta::vector::Array<Real, 3>::template Ref<iwork>::HandleT;
    template<bool iwork>
    using Vec6RArrT =
        typename feta::vector::Array<Real, 6>::template Ref<iwork>::HandleT;

public:
    /** @brief Type aliases for data members */
    using TraverserT = core::Traverser::Ref<work>;
    using ChainT =
        typename core::BodyCacheChain<Dim>::template Ref<work>;

    /** @brief Factory method to construct from data members */
    DEVICEHOST()
    static RefEphCache make(const TraverserT& traverser, const ChainT& chain)
    {
        return { traverser, chain };
    }

    /** @brief Number of bodies in the cache. */
    DEVICEHOST() vecdim_t nBodyUnits() const { return traverser_.size(); }

    /** @brief Position of `target` relative to `center` at the cache slot
     *  for sample `index`.  Mirrors `RefEphUnit::iGetPosition` exactly:
     *  the walker accumulates the target chain with `+1.0` and the center
     *  chain with `-1.0` into `x`.  Caller must zero-initialise `x[index]`
     *  before calling. */
    template<bool iwork>
    DEVICEHOST()
    void iGetPosition(const SampleIndex& index, Vec3RArrT<iwork>& x,
        const Real& epoch, const NaifId& target, const NaifId& center) const
    {
        core::WalkerPair walkers = traverser_.makeWalkers(target, center);
        chain_.template getPosition<iwork>(
            index, x, epoch, walkers.target_, 1.0);
        chain_.template getPosition<iwork>(
            index, x, epoch, walkers.center_, -1.0);
    }

    /** @brief Velocity of `target` relative to `center`.  Requires `Dim == 6`. */
    template<bool iwork>
    DEVICEHOST()
    void iGetVelocity(const SampleIndex& index, Vec3RArrT<iwork>& x,
        const Real& epoch, const NaifId& target, const NaifId& center) const
    {
        static_assert(Dim == 6,
            "iGetVelocity requires Dim == 6 (state cache); "
            "Dim == 3 caches store position only");
        core::WalkerPair walkers = traverser_.makeWalkers(target, center);
        chain_.template getVelocity<iwork>(
            index, x, epoch, walkers.target_, 1.0);
        chain_.template getVelocity<iwork>(
            index, x, epoch, walkers.center_, -1.0);
    }

    /** @brief Position+velocity of `target` relative to `center`.  Dim == 6 only. */
    template<bool iwork>
    DEVICEHOST()
    void iGetPositionAndVelocity(const SampleIndex& index, Vec6RArrT<iwork>& x,
        const Real& epoch, const NaifId& target, const NaifId& center) const
    {
        static_assert(Dim == 6,
            "iGetPositionAndVelocity requires Dim == 6 (state cache)");
        core::WalkerPair walkers = traverser_.makeWalkers(target, center);
        chain_.template getPositionAndVelocity<iwork>(
            index, x, epoch, walkers.target_, 1.0);
        chain_.template getPositionAndVelocity<iwork>(
            index, x, epoch, walkers.center_, -1.0);
    }

    /** @brief Value-returning ``iGetPosition`` overload (per-thread).
     *
     *  Lives alongside the array-path version: the array path is right
     *  for kernels that already maintain a per-sample-indexed
     *  shared-memory or device buffer (e.g. ``coiResolveKernel``,
     *  ``pointGravity``); the value-returning path is right for
     *  per-thread consumers (e.g. event Evaluables) that want the
     *  cached state in registers.  Numerical contract is identical —
     *  same striding, same chain-walker accumulation. */
    DEVICEHOST() Vec3R iGetPosition(const SampleIndex& index, const Real& epoch,
        const NaifId& target, const NaifId& center) const
    {
        core::WalkerPair walkers = traverser_.makeWalkers(target, center);
        Vec3R out = chain_.getPosition(index, epoch, walkers.target_, 1.0);
        out += chain_.getPosition(index, epoch, walkers.center_, -1.0);
        return out;
    }

    /** @brief Value-returning ``iGetVelocity`` overload (per-thread). */
    DEVICEHOST() Vec3R iGetVelocity(const SampleIndex& index, const Real& epoch,
        const NaifId& target, const NaifId& center) const
    {
        static_assert(Dim == 6,
            "iGetVelocity requires Dim == 6 (state cache); "
            "Dim == 3 caches store position only");
        core::WalkerPair walkers = traverser_.makeWalkers(target, center);
        Vec3R out = chain_.getVelocity(index, epoch, walkers.target_, 1.0);
        out += chain_.getVelocity(index, epoch, walkers.center_, -1.0);
        return out;
    }

    /** @brief Value-returning ``iGetPositionAndVelocity`` overload
     *  (per-thread).  Dim == 6 only. */
    DEVICEHOST() Vec6R iGetPositionAndVelocity(const SampleIndex& index,
        const Real& epoch, const NaifId& target, const NaifId& center) const
    {
        static_assert(Dim == 6,
            "iGetPositionAndVelocity requires Dim == 6 (state cache)");
        core::WalkerPair walkers = traverser_.makeWalkers(target, center);
        Vec6R out = chain_.getPositionAndVelocity(
            index, epoch, walkers.target_, 1.0);
        out += chain_.getPositionAndVelocity(
            index, epoch, walkers.center_, -1.0);
        return out;
    }

    /** @brief Clone for compatibility with FETA POD patterns. */
    DEVICEHOST() RefEphCache clone() const { return *this; }

    /** @brief Public POD members (mirrors RefEphUnit shape). */
    TraverserT traverser_;
    ChainT     chain_;
};

/**
 * @brief Owning per-body native-frame cache mirroring an `EphUnit`.
 *
 * `EphCache` allocates one `feta::core::memory::Container<Real, DeviceT, Dim>`
 * per body of its source `EphUnit`, sized for `nSamples` per-body slots.
 * After `fill*()` (Phase 3), each slot holds the body's native-frame
 * (parent-relative) position (Dim == 3) or full state (Dim == 6) at the
 * given per-sample epoch.  Queries use the brie chain walker — the same
 * one `RefEphUnit` uses — so `getPosition(target, center)` works for any
 * (target, center) pair without per-COI invalidation.
 *
 * Storage selection:
 *   - `DeviceT = Device::CUDA_DEVICE` — pure-device cache (typical GPU path).
 *   - `DeviceT = Device::CUDA_HOST`   — pure-host cache (CPU-only or pinned
 *     fall-back).
 * `UseTexture` only affects the *source* `EphUnit` (chebyshev coefficient
 * binding during fill); the cache itself is always pointer-loaded.
 *
 * Lifetime: `EphCache` holds a `parm::util::Observer<EphUnit>` to the source.
 * The source must outlive the cache, but moves of the source are tracked
 * automatically via the observer pattern — no explicit rebinding required.
 * Reads after a successful `fill*()` do not touch the source.
 */
template<bool UseTexture, idx_t Dim,
    feta::core::memory::Device DeviceT>
class EphCache {
    using Self = EphCache;

public:
    static_assert(Dim == 3 || Dim == 6,
        "EphCache supports Dim == 3 (position) or Dim == 6 (state)");

    /** @brief Source EphUnit type. */
    using SourceT = EphUnit<UseTexture>;

    /** @brief Per-body cache storage type. */
    using StoreT
        = feta::core::memory::Container<Real, DeviceT, Dim>;

    /** @brief Traverser type (mirrors EphUnit). */
    using TraverserT = core::Traverser;

    /** @brief Chain interpolator type (cache-leaf flavour). */
    using ChainT = core::BodyCacheChain<Dim>;

    /** @brief Reference type (POD captured by kernels). */
    template<bool work>
    using Ref = RefEphCache<work, Dim>;
    /** @brief Global reference. */
    using GRef = Ref<false>;

    /** @brief Construct an empty (sizeless) cache.  Useful as a placeholder. */
    EphCache()
        : source_{}
        , nSamples_{ 0 }
        , bodySlots_{}
        , traverser_{ TraverserT::flexible() }
        , chain_{ ChainT::flexible() }
    {
    }

    /** @brief Construct a cache mirroring `source`'s body set, sized for
     *  `nSamples` per-body slots.  Allocates `nBodies × nSamples × Dim`
     *  reals on `DeviceT`.  The cache registers itself as an observer of
     *  `source`, so subsequent moves of `source` are tracked automatically;
     *  the source must still outlive the cache. */
    EphCache(SourceT& source, const idx_t& nSamples)
        : source_{ source }
        , nSamples_{ nSamples }
        , bodySlots_{}
        , traverser_{ TraverserT::flexible() }
        , chain_{ ChainT::flexible() }
    {
        const idx_t nBodies = source.nBodyUnits();

        if (nBodies == 0)
            return;

        /* Allocate per-body cache slots. */
        bodySlots_.reserve(nBodies);
        for (idx_t b = 0; b < nBodies; b++) {
            bodySlots_.emplace_back(nSamples);
        }

        /* Copy traverser from source.  Going through `hostRef()` lazily
         * triggers source-side host chain construction the first time. */
        traverser_               = std::move(TraverserT(nBodies));
        auto srcTrav             = source.hostRef().traverser_;
        TraverserT::GRef dstTrav = traverser_.hostRef();
        for (idx_t b = 0; b < nBodies; b++) {
            dstTrav[b] = srcTrav[b];
        }

        /* Build chain with cache leaves bound to each body's slot.  The
         * BodyCache stores a feta vector handle directly — no raw
         * pointer leaks into the leaf type. */
        chain_                                   = std::move(ChainT(nBodies));
        typename ChainT::GRef cref               = chain_.hostRef();
        for (idx_t b = 0; b < nBodies; b++) {
            cref[b] = core::BodyCache<Dim>::make(
                slotRef_(b).handle(), nSamples);
        }

        /* Bind chain to the appropriate parent-position pointer.  We borrow
         * it from the source's metadata (which lives at least as long as
         * the cache by contract). */
#ifndef BRIE_CPU_ONLY
        if constexpr (DeviceT == feta::core::memory::Device::CUDA_DEVICE) {
            traverser_.upload();
            chain_.upload();
            chain_.bind(
                source.metadata().handleMembers().getDeviceData());
        } else
#endif
        {
            chain_.bind(
                source.metadata().handleMembers().getHostData());
        }
    }

    EphCache(EphCache&&) = default;
    EphCache(const EphCache&) = delete;
    EphCache& operator=(EphCache&&) = default;
    EphCache& operator=(const EphCache&) = delete;

    /** @brief Number of per-body slots (= nSamples). */
    idx_t nSamples() const { return nSamples_; }

    /** @brief Number of bodies in the cache. */
    idx_t nBodyUnits() const
    {
        return traverser_.size();
    }

    /** @brief Cache slot of body `bodyIdx` (raw underlying storage). */
    StoreT& slot(const idx_t& bodyIdx) { return bodySlots_[bodyIdx]; }
    const StoreT& slot(const idx_t& bodyIdx) const
    {
        return bodySlots_[bodyIdx];
    }

    /** @brief Typed `feta::vector::Array<Real, Dim>::GRef` view over body
     *  `bodyIdx`'s slot.  Same shape kernels see in `iGetPosition` /
     *  `iGetVelocity` walker reads, exposed for direct kernel parameter
     *  use (e.g. cudajectory's per-body conditional-copy kernels). */
    typename feta::vector::Array<Real, Dim>::GRef slotRef(
        const idx_t& bodyIdx) const
    {
        return slotRef_(bodyIdx);
    }

    /** @brief Return a non-owning reference to this cache, dispatching
     *  to host- or device-side based on `DeviceT`. */
    GRef ref() const
    {
#ifndef BRIE_CPU_ONLY
        if constexpr (DeviceT == feta::core::memory::Device::CUDA_DEVICE) {
            return GRef::make(traverser_.deviceRef(), chain_.deviceRef());
        } else
#endif
        {
            return GRef::make(traverser_.hostRef(), chain_.hostRef());
        }
    }

    /** @brief Fill all bodies' native-frame cache slots at the given
     *  per-sample epochs.  Internally fans out `nBodyUnits()` per-body
     *  fills.  For `DeviceT == CUDA_DEVICE` the source EphUnit must be
     *  uploaded; for `CUDA_HOST` no upload is required.
     *
     *  Const because the C++ state of the cache is unchanged — the
     *  per-sample slot writes go through the Container's mutable
     *  device pointer, matching FETA's `data() const → DataT*` idiom.
     *
     *  Two accepted signatures: a `feta::scalar::Array<Real>::GRef`
     *  (extracted via `.deviceRef()` / `.hostRef()` on a host-side
     *  Container) or its `HandleT` (the kernel-parameter shape, returned
     *  by parm-side accessors like `base.currentEpochs()`). */
    void fill(const typename feta::scalar::Array<Real>::GRef& epochs,
        typename SourceT::StreamT stream = 0) const
    {
        fill(epochs.handle(), stream);
    }

    void fill(
        const typename feta::scalar::Array<Real>::GRef::HandleT& epochs,
        typename SourceT::StreamT stream = 0) const
    {
        for (idx_t b = 0; b < nBodyUnits(); b++) {
            fillBody(b, epochs, stream);
        }
    }

    /** @brief Fill a single body's native-frame cache slot.  Exposed for
     *  per-body graph scheduling (cudajectory's pin/variable pattern).
     *  See `fill()` for the dual GRef/HandleT signatures. */
    void fillBody(const idx_t& bodyIdx,
        const typename feta::scalar::Array<Real>::GRef& epochs,
        typename SourceT::StreamT stream = 0) const
    {
        fillBody(bodyIdx, epochs.handle(), stream);
    }

    void fillBody(const idx_t& bodyIdx,
        const typename feta::scalar::Array<Real>::GRef::HandleT& epochs,
        typename SourceT::StreamT stream = 0) const
    {
#ifndef BRIE_CPU_ONLY
        if constexpr (DeviceT == feta::core::memory::Device::CUDA_DEVICE) {
            fillBodyDevice_(bodyIdx, epochs, stream);
            return;
        }
#endif
        fillBodyHost_(bodyIdx, epochs);
    }

private:
    /** @brief Build a `feta::vector::Array<Real, Dim>::GRef` view over
     *  body `bodyIdx`'s storage Container.  The returned GRef points at
     *  device memory for `DeviceT == CUDA_DEVICE` and host memory for
     *  `CUDA_HOST`. */
    typename feta::vector::Array<Real, Dim>::GRef slotRef_(
        const idx_t& bodyIdx) const
    {
        typename feta::vector::Array<Real, Dim>::GRef r;
        r.data_      = const_cast<Real*>(bodySlots_[bodyIdx].data());
        r.nVecs_     = nSamples_;
        r.dimOffset_ = nSamples_;
        r.tex_       = 0;
        r.texOffset_ = 0;
        return r;
    }

#ifndef BRIE_CPU_ONLY
    void fillBodyDevice_(const idx_t& bodyIdx,
        const typename feta::scalar::Array<Real>::GRef::HandleT& epochs,
        typename SourceT::StreamT stream) const
    {
        constexpr idx_t blockSize = 128;
        const idx_t nBlocks
            = (nSamples_ + blockSize - 1) / blockSize;
        kernel::fillBodyKernel<UseTexture, Dim><<<nBlocks, blockSize, 0, stream>>>(
            slotRef_(bodyIdx), source_->deviceRef(), bodyIdx, epochs);
    }
#endif

    void fillBodyHost_(const idx_t& bodyIdx,
        const typename feta::scalar::Array<Real>::GRef::HandleT& epochs) const
    {
        auto src    = source_->hostRef();
        auto leaf   = src.interpolators_[bodyIdx];
        auto slot   = slotRef_(bodyIdx);

        for (idx_t i = 0; i < nSamples_; i++) {
            const feta::SampleIndex idx = feta::SampleIndex::make(i);
            if constexpr (Dim == 3) {
                Vec3R val;
                val.setZero();
                leaf.getValues(val, epochs[idx]);
                slot[idx] = val;
            } else { /* Dim == 6 */
                Vec6R val;
                val.setZero();
                leaf.getValuesAndDerivatives(val, epochs[idx]);
                slot[idx] = val;
            }
        }
    }

private:
    parm::util::Observer<SourceT> source_;
    idx_t nSamples_;
    std::vector<StoreT> bodySlots_;
    TraverserT traverser_;
    ChainT chain_;
};

/** @brief Out-of-line definition of `EphUnit::makeCache`.  Lives here so
 *  that `EphCache` is fully defined at the point of return-by-value, while
 *  `EphUnit.h` carries only a forward declaration of `EphCache`. */
template<bool UseTexture>
template<idx_t Dim, feta::core::memory::Device DeviceT>
EphCache<UseTexture, Dim, DeviceT>
EphUnit<UseTexture>::makeCache(const idx_t& nSamples)
{
    return EphCache<UseTexture, Dim, DeviceT>(*this, nSamples);
}

/** @brief Position-only cache alias (Dim = 3). */
template<bool UseTexture,
    feta::core::memory::Device D = feta::core::memory::Device::CUDA_DEVICE>
using EphPosCache = EphCache<UseTexture, 3, D>;

/** @brief Full-state (position + velocity) cache alias (Dim = 6). */
template<bool UseTexture,
    feta::core::memory::Device D = feta::core::memory::Device::CUDA_DEVICE>
using EphStateCache = EphCache<UseTexture, 6, D>;

} // namespace states
} // namespace brie
