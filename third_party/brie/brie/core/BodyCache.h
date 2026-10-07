#pragma once

#include "brie/core/ChainInterpolator.h"
#include "brie/typedefs.h"
#include "brie/util/DeviceError.h"
#include "brie/util/throw.h"
#include <feta/feta.h>

namespace brie {
namespace core {

/** @brief A pre-cached body data leaf for the chain walker.
 *
 * BodyCache is a POD leaf compatible with `RefChainInterpolator`'s leaf
 * accessor contract.  Instead of evaluating chebyshev coefficients (as
 * `BodyInterpolator` does), it reads pre-cached values from a per-body
 * slot via a feta vector handle.
 *
 * Storage: the cache holds a
 * ``feta::vector::Array<Real, Dim>::GRef::HandleT`` directly.  Reads go
 * through the handle's expression-template ``operator[]`` (inherited from
 * ``feta::vector::expr::Expression``); component-range access uses
 * ``subset<start, HowMany>()`` so position / velocity halves of a Dim==6
 * cache are extracted with a single statement, not six per-component
 * scalar reads.  No raw ``Real*`` arithmetic surfaces at any call site.
 *
 * Frame: cache contents are *native-frame* (parent-relative — i.e. each
 * body's value is its position relative to its own native parent in the
 * SPK tree), NOT COI-relative.  The chain walker performs the
 * target-minus-center subtraction across slots, matching
 * `RefEphUnit::iGetPosition`.  Anyone applying COI subtraction to a
 * single slot's contents will get garbage.
 *
 * Templated on `Dim`:
 *   - `Dim == 3` — position-only cache.  `getValues` works;
 *      `getDerivatives` / `getValuesAndDerivatives` are static_assert-gated.
 *   - `Dim == 6` — full state cache.  All three accessors work.
 *      Position lives in components `[0, 3)`; velocity in `[3, 6)`.
 *
 * Scalar (single-epoch) accessors are intentionally unsupported (the
 * cache has no continuous-epoch semantics) and assert under
 * BRIE_DEBUG_MODE.  Only the array path is hot-callable.
 */
template<idx_t Dim>
class BodyCache {
    static_assert(Dim == 3 || Dim == 6,
        "BodyCache supports Dim == 3 (position) or Dim == 6 (state)");

    template<bool work>
    using Vec3RArrT =
        typename feta::vector::Array<Real, 3>::template Ref<work>::HandleT;
    template<bool work>
    using Vec6RArrT =
        typename feta::vector::Array<Real, 6>::template Ref<work>::HandleT;

public:
    /** @brief Slot handle type — feta vector handle bound to the body's
     *  per-sample storage block.  Carries pointer + SoA stride
     *  internally; no raw indexing required. */
    using SlotHandleT =
        typename feta::vector::Array<Real, Dim>::GRef::HandleT;

    /** @brief Factory: bind a body's per-sample slot.  ``slot`` is the
     *  feta vector handle returned by the cache's ``slotRef(...)`` (or
     *  any handle obtained from a vector RefArray of the same Dim). */
    DEVICEHOST()
    static BodyCache make(const SlotHandleT& slot, const idx_t& nSamples)
    {
        BodyCache c;
        c.data_     = slot;
        c.nSamples_ = nSamples;
        return c;
    }

    /** @brief Scalar path — unsupported by the cache. */
    DEVICEHOST() void getValues(Vec3R&, const Real&) const
    {
        assertScalarUnsupported_();
    }
    DEVICEHOST() void getDerivatives(Vec3R&, const Real&) const
    {
        static_assert(Dim == 6, "getDerivatives requires Dim == 6");
        assertScalarUnsupported_();
    }
    DEVICEHOST() void getValuesAndDerivatives(Vec6R&, const Real&) const
    {
        static_assert(Dim == 6, "getValuesAndDerivatives requires Dim == 6");
        assertScalarUnsupported_();
    }

    /** @brief Array path — accumulate `sign * cache[idx]` into `out[idx]`.
     *
     *  Reads components `[0, 3)` of the cache slot via a 3-component
     *  subset view of the handle (a no-op for Dim==3 caches; a stride
     *  re-base for Dim==6 caches) and adds `sign * cached` to
     *  `out[idx]` via expression-template assignment.  The `epoch`
     *  argument is ignored (epoch is implicit in the per-sample fill
     *  that produced the cache contents). */
    template<bool work>
    DEVICEHOST()
    void getValues(const SampleIndex& idx, Vec3RArrT<work>& out,
        const Real& /*epoch*/, const Real& sign) const
    {
        const Vec3R cached = data_.template subset<0, 3>()[idx];
        out[idx] += sign * cached;
    }

    /** @brief Array path velocity — components `[3, 6)`.  Dim == 6 only. */
    template<bool work>
    DEVICEHOST()
    void getDerivatives(const SampleIndex& idx, Vec3RArrT<work>& out,
        const Real& /*epoch*/, const Real& sign) const
    {
        static_assert(Dim == 6, "getDerivatives requires Dim == 6");
        const Vec3R cached = data_.template subset<3, 3>()[idx];
        out[idx] += sign * cached;
    }

    /** @brief Array path full state — components `[0, 6)`.  Dim == 6 only. */
    template<bool work>
    DEVICEHOST()
    void getValuesAndDerivatives(const SampleIndex& idx, Vec6RArrT<work>& out,
        const Real& /*epoch*/, const Real& sign) const
    {
        static_assert(Dim == 6, "getValuesAndDerivatives requires Dim == 6");
        const Vec6R cached = data_[idx];
        out[idx] += sign * cached;
    }

    /** @brief Per-thread value-returning leaf overload — components `[0, 3)`.
     *
     *  Returns the cache slot's first 3 components at sample ``idx`` via
     *  the handle's expression-template subset view.  The caller is
     *  responsible for applying any sign and accumulating across chain
     *  steps (the chain walker's value-returning overload does this).
     *
     *  Use case: per-thread consumers (e.g. event Evaluables) that want
     *  the cached state in registers rather than via a per-sample-indexed
     *  shared-memory or device buffer. */
    DEVICEHOST() Vec3R getValues(
        const SampleIndex& idx, const Real& /*epoch*/) const
    {
        return data_.template subset<0, 3>()[idx];
    }

    /** @brief Per-thread value-returning velocity leaf overload — components
     *  `[3, 6)`.  Dim == 6 only. */
    DEVICEHOST() Vec3R getDerivatives(
        const SampleIndex& idx, const Real& /*epoch*/) const
    {
        static_assert(Dim == 6, "getDerivatives requires Dim == 6");
        return data_.template subset<3, 3>()[idx];
    }

    /** @brief Per-thread value-returning full-state leaf overload —
     *  components `[0, 6)`.  Dim == 6 only. */
    DEVICEHOST() Vec6R getValuesAndDerivatives(
        const SampleIndex& idx, const Real& /*epoch*/) const
    {
        static_assert(Dim == 6, "getValuesAndDerivatives requires Dim == 6");
        return data_[idx];
    }

    /** @brief Operator != (matches the LeafT contract used elsewhere).
     *  Two BodyCaches are unequal if they wrap different slot
     *  storage (compared by underlying pointer) or different lengths. */
    DEVICEHOST() bool operator!=(const BodyCache& other) const
    {
        return data_.data() != other.data_.data() || nSamples_ != other.nSamples_;
    }

    /* Public members for PODification (mirrors BodyInterpolator). */
    SlotHandleT data_{};
    idx_t nSamples_ = 0;

private:
    DEVICEHOST()
    inline void assertScalarUnsupported_() const
    {
#ifdef BRIE_DEBUG_MODE
#ifdef __CUDA_ARCH__
        BRIE_GPU_THROW(err::INVALID_BODY_UNIT_TYPE);
#else
        BRIE_THROW(std::runtime_error,
            "BodyCache scalar path is unsupported - cache has no "
            "continuous-epoch semantics");
#endif
#endif
    }
};

/** @brief Cache-flavour chain alias.
 *
 *  `ChainInterpolator<BodyCache<Dim>>` — same walker, cache leaves. */
template<idx_t Dim>
using BodyCacheChain = ChainInterpolator<BodyCache<Dim>>;
template<idx_t Dim>
using RefBodyCacheChain = RefChainInterpolator<BodyCache<Dim>>;

} // namespace core
} // namespace brie
