#pragma once

#include "paraHPOP/model/environment.h"
#include "interface/config/model/Accelerations.h"

namespace paraHPOP {
namespace model {
namespace accelerations {

namespace src = interface::config::model::accelerations;

/**
 * @brief Non-owning reference to acceleration management class
 *
 */
template<bool work, bool MaybeVolatile = false>
class RefAccelerations {
    using FlagsT   = feta::vector::Array<bool, src::FEATSIZE>::GRef;
    using EnvT     = ::paraHPOP::model::environment::Env;
    using SourceT  = src::Features;
    using ScratchT =
        typename feta::vector::Array<Real, 3>::template Ref<work, MaybeVolatile>;

public:
    /** @brief Volatile is allowed only with work */
    static constexpr bool Volatile = MaybeVolatile && work;
    /** @brief expose acceleration flags */
    DEVICEHOST() FlagsT flags() const { return flags_; }

    /** @brief Check if the given acceleration source is active or not */
    template<SourceT SOURCE>
    DEVICEHOST()
    bool& active(const idx_t& i)
    {
        return flags_.get<SOURCE>(i);
    }
    template<SourceT SOURCE>
    DEVICEHOST()
    const bool& active(const idx_t& i) const
    {
        return flags_.get<SOURCE>(i);
    }

    /** @brief Activate the given acceleration source */
    template<SourceT SOURCE>
    DEVICEHOST()
    void activate(const idx_t& i)
    {
        active<SOURCE>(i) = true;
    }


    /** @brief Deactivate the given acceleration source */
    template<SourceT SOURCE>
    DEVICEHOST()
    void deactivate(const idx_t& i)
    {
        active<SOURCE>(i) = false;
    }

    /** @brief Evaluate total acceleration from per-sample raw fields.
     *
     *  Caller passes the four raw per-sample scalars (``mass``,
     *  ``area``, ``cr``, ``cd``) rather than pre-derived quantities.
     *  Internally:
     *    - SRP uses ``AMS = cr · area / mass`` (reflectivity-weighted
     *      area-to-mass).
     *    - Drag uses ``BC  = cd · area / mass`` (drag-coefficient-
     *      weighted ballistic coefficient).
     *  Both derived scalars are hoisted to the top of the body loop
     *  in ``eval_`` and stay constant across bodies for a given
     *  sample.  Raw signatures keep AMS and BC distinct — a cd↔cr
     *  swap at the call site is no longer possible. */
    DEVICEHOST()
    Vec3R eval(const SampleIndex& i, const Vec3R& pos, const Vec3R& vel,
        const mReal_t& epoch, const mReal_t& mass, const mReal_t& area,
        const mReal_t& cr, const mReal_t& cd, const brie::NaifId& COI,
        const typename EnvT::template Ref<work, MaybeVolatile>& env) const
    {
        return eval_(i, pos, vel, epoch, mass, area, cr, cd, COI, env);
    }

    /** @brief Packet (W-lane) total acceleration evaluation —
     *         uniform-COI path.
     *
     *  Caller (paraHPOP ``Propagator::monolithStep_``) has already
     *  enforced uniform-COI across the W lanes via the predicate guard
     *  in the packet RHS adapter.  Per-lane epochs and the four raw
     *  fields are packets.
     *
     *  - If any active body has SHAPE, OBLATENESS or ATMOSPHERE active,
     *    the method falls back to per-lane scalar ``eval``
     *    (correctness-equivalent; no SIMD on the math).
     *  - Otherwise (PointGravity + SRP only), runs the full packet path
     *    per body via ``RefEnvironment::getPositionPacket`` and
     *    computes ``AMS_packet = cr · area / mass`` inline. */
    template<idx_t W>
    feta::vector::PacketItem<Real, 3, W> evalPacket(
        const feta::PacketIndex<W>& pi,
        const feta::vector::PacketItem<Real, 3, W>& pos,
        const feta::vector::PacketItem<Real, 3, W>& vel,
        const feta::simd::Packet<Real, W>& epoch,
        const feta::simd::Packet<Real, W>& mass,
        const feta::simd::Packet<Real, W>& area,
        const feta::simd::Packet<Real, W>& cr,
        const feta::simd::Packet<Real, W>& cd,
        const brie::NaifId& COI, const typename EnvT::template Ref<work, MaybeVolatile>& env) const;

    /** @brief Create new global reference to the accelerations */
    DEVICEHOST() RefAccelerations clone() const { return *this; }

    /** @brief Internal flags representing active acceleration sources.
     *  Public for PODification. */
    FlagsT flags_;

private:
    DEVICEHOST()
    Vec3R eval_(const SampleIndex& i, const Vec3R& pos,
        [[maybe_unused]] const Vec3R& vel, const mReal_t& epoch,
        const mReal_t& mass, const mReal_t& area, const mReal_t& cr,
        const mReal_t& cd, const brie::NaifId& COI,
        const typename EnvT::template Ref<work, MaybeVolatile>& env) const;
};

} // namespace accelerations
} // namespace model
} // namespace paraHPOP