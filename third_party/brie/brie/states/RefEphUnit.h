#pragma once

#include "brie/core/ChainInterpolator.h"
#include "brie/core/Type2BodyUnit.h"
#include "brie/core/Type3BodyUnit.h"
#include "brie/states/metadata/EphUnit.h"
#include "brie/states/metadata/RefEphUnit.h"

namespace brie {
namespace states {

/**
 * @brief Non-owning reference class for a `brie::EphUnit`. Enables several
 * operations on data and metadata, including the actual core computation of
 * Position and Velocities.
 *
 */
template<bool work, bool UseTexture, bool MaybeVolatile = false>
class RefEphUnit {

    /* Array types */
    template<bool iwork>
    using Vec3RArrT =
        typename feta::vector::Array<Real, 3>::template Ref<iwork>::HandleT;
    template<bool iwork>
    using Vec6RArrT =
        typename feta::vector::Array<Real, 6>::template Ref<iwork>::HandleT;

public:
    /** @brief Volatile is allowed only with work */
    static constexpr bool Volatile = MaybeVolatile && work;
    /** @brief Type aliases for data members */
    using TraverserT     = typename core::Traverser::template Ref<work,
            MaybeVolatile>;
    using InterpolatorsT = typename core::EphChainInterpolator<
        UseTexture>::template Ref<work, MaybeVolatile>;

    /**
     * @brief Factory method to construct from data members
     *
     */
    DEVICEHOST()
    static RefEphUnit make(
        const TraverserT& traverser, const InterpolatorsT& interpolators)
    {
        return { traverser, interpolators };
    }

    /**
     * @brief Return the number of bodies in this `brie::EphUnit`
     *
     * @return Nbodies
     *
     */
    DEVICEHOST() vecdim_t nBodyUnits() const { return traverser_.size(); }

    /**
     * @brief Read-only access to the underlying body traverser.
     *
     * Exposes the chain structure (per-slot ``BodyLocator``: naif id ``id_``,
     * lookup-table slot ``pos_``, and center/parent slot ``centerPos_``; the
     * SSB root is denoted by ``size()``). Lets a consumer plan per-slot cache
     * work — e.g. classify which native cache slots are transited by the
     * active force model's chains versus those reachable only via
     * event-target point chains.
     */
    DEVICEHOST() const TraverserT& traverser() const { return traverser_; }

    /**
     * @brief Compute the position of the target NaifID, relative to the center
     * NaifID, at given epoch.
     *
     * @param epoch
     * @param targetNaifID
     * @param centerNaifID
     * @return positionVector
     *
     */
    DEVICEHOST()
    Vec3R getPosition(
        const Real& epoch, const NaifId& target, const NaifId& center) const
    {
#ifdef __CUDA_ARCH__
        return deviceGetPosition_(epoch, target, center);
#else
        return hostGetPosition_(epoch, target, center);
#endif
    }

    /**
     * @brief Compute the position of the target NaifID, relative to the center
     * NaifID, at given epoch.
     *
     * @param outPositionVector
     * @param epoch
     * @param targetNaifID
     * @param centerNaifID
     *
     */
    DEVICEHOST()
    void iGetPosition(Vec3R& out, const Real& epoch, const NaifId& target,
        const NaifId& center) const
    {
#ifdef __CUDA_ARCH__
        deviceiGetPosition_(out, epoch, target, center);
#else
        hostiGetPosition_(out, epoch, target, center);
#endif
    }

    /** @brief Array version */
    template<bool iwork>
    DEVICEHOST()
    void iGetPosition(const SampleIndex& index, Vec3RArrT<iwork>& x,
        const Real& epoch, const NaifId& target, const NaifId& center) const
    {
        /* find common center */
        core::WalkerPair walkers = traverser_.makeWalkers(target, center);

        /* read ephemeris */
        interpolators_.template getPosition<iwork>(
            index, x, epoch, walkers.target_, 1.0);
        interpolators_.template getPosition<iwork>(
            index, x, epoch, walkers.center_, -1.0);
    }

    /** @brief Packet (W-lane) position retrieval — uniform-COI path.
     *
     *  Caller (cudajectory ``RefEnvironment::getPositionPacket``)
     *  has already verified that all W lanes share the same
     *  ``(target, center)`` chain (typically via the uniform-COI
     *  predicate at packet RHS entry).  Per-lane epochs may
     *  differ; the underlying chebyshev packet evaluator handles
     *  uniform-interval predicate + per-lane scalar fallback.
     *
     *  PR-6δ-B2 packet ephemeris query.
     */
    template<idx_t W>
    inline void iGetPositionPacket(
        feta::vector::PacketItem<Real, 3, W>& out,
        const feta::simd::Packet<Real, W>& epochPkt,
        const NaifId& target, const NaifId& center) const
    {
        /* Walker pair construction is scalar (NaifId is scalar; COI
         * uniform across packet by caller's predicate). */
        core::WalkerPair walkers = traverser_.makeWalkers(target, center);

        /* Walk both chains, accumulating into the packet output. */
        interpolators_.template getPositionPacket<W>(
            out, epochPkt, walkers.target_, 1.0);
        interpolators_.template getPositionPacket<W>(
            out, epochPkt, walkers.center_, -1.0);
    }

    /**
     * @brief Compute the velocity of the target NaifID, relative to the center
     * NaifID, at given epoch.
     *
     * @param epoch
     * @param targetNaifID
     * @param centerNaifID
     * @return velocityVector
     *
     */
    DEVICEHOST()
    Vec3R getVelocity(
        const Real& epoch, const NaifId& target, const NaifId& center) const
    {
#ifdef __CUDA_ARCH__
        return deviceGetVelocity_(epoch, target, center);
#else
        return hostGetVelocity_(epoch, target, center);
#endif
    }

    /**
     * @brief Compute the velocity of the target NaifID, relative to the center
     * NaifID, at given epoch.
     *
     * @param outVelocityVector
     * @param epoch
     * @param targetNaifID
     * @param centerNaifID
     *
     */
    DEVICEHOST()
    void iGetVelocity(Vec3R& out, const Real& epoch, const NaifId& target,
        const NaifId& center) const
    {
#ifdef __CUDA_ARCH__
        deviceiGetVelocity_(out, epoch, target, center);
#else
        hostiGetVelocity_(out, epoch, target, center);
#endif
    }

    /** @brief Array version */
    template<bool iwork>
    DEVICEHOST()
    void iGetVelocity(const SampleIndex& index, Vec3RArrT<iwork>& x,
        const Real& epoch, const NaifId& target, const NaifId& center) const
    {
        /* find common center */
        core::WalkerPair walkers = traverser_.makeWalkers(target, center);

        /* read ephemeris */
        interpolators_.template getVelocity<iwork>(
            index, x, epoch, walkers.target_, 1.0);
        interpolators_.template getVelocity<iwork>(
            index, x, epoch, walkers.center_, -1.0);
    }

    /**
     * @brief Compute the Cartesian State of the target NaifID, relative to the
     * center NaifID, at given epoch.
     *
     * @param epoch
     * @param targetNaifID
     * @param centerNaifID
     * @return cartesianStateVector
     *
     */
    DEVICEHOST()
    Vec6R getPositionAndVelocity(
        const Real& epoch, const NaifId& target, const NaifId& center) const
    {
#ifdef __CUDA_ARCH__
        return deviceGetPositionAndVelocity_(epoch, target, center);
#else
        return hostGetPositionAndVelocity_(epoch, target, center);
#endif
    }

    /**
     * @brief Compute the Cartesian state of the target NaifID, relative to the
     * center NaifID, at given epoch.
     *
     * @param outCartesianState
     * @param epoch
     * @param targetNaifID
     * @param centerNaifID
     *
     */
    DEVICEHOST()
    void iGetPositionAndVelocity(Vec6R& out, const Real& epoch,
        const NaifId& target, const NaifId& center) const
    {
#ifdef __CUDA_ARCH__
        deviceiGetPositionAndVelocity_(out, epoch, target, center);
#else
        hostiGetPositionAndVelocity_(out, epoch, target, center);
#endif
    }

    /** @brief Array version */
    template<bool iwork>
    DEVICEHOST()
    void iGetPositionAndVelocity(const SampleIndex& index, Vec6RArrT<iwork>& x,
        const Real& epoch, const NaifId& target, const NaifId& center) const
    {
        /* find common center */
        core::WalkerPair walkers = traverser_.makeWalkers(target, center);

        /* read ephemeris */
        interpolators_.template getPositionAndVelocity<iwork>(
            index, x, epoch, walkers.target_, 1.0);
        interpolators_.template getPositionAndVelocity<iwork>(
            index, x, epoch, walkers.center_, -1.0);
    }

    /** @brief Create a new global reference to this ephemeris unit */
    DEVICEHOST() RefEphUnit<work, UseTexture> clone() const { return *this; }

    /** @brief Data members made public for PODification */

    /* The traverser */
    TraverserT traverser_;

    /* The interpolators */
    InterpolatorsT interpolators_;

    /** @brief Host position dispatcher */
    Vec3R hostGetPosition_(
        const Real& epoch, const NaifId& target, const NaifId& center) const;
    void hostiGetPosition_(Vec3R& out, const Real& epoch, const NaifId& target,
        const NaifId& center) const;

#ifndef BRIE_CPU_ONLY
    /** @brief Device position dispatcher */
    DEVICE()
    Vec3R deviceGetPosition_(
        const Real& epoch, const NaifId& target, const NaifId& center) const;
    DEVICE()
    void deviceiGetPosition_(Vec3R& out, const Real& epoch,
        const NaifId& target, const NaifId& center) const;
#endif

    /** @brief Device-Host core for get position */
    DEVICEHOST()
    inline Vec3R getPosition_(
        const Real& epoch, const NaifId& target, const NaifId& center) const
    {
        Vec3R x;
        iGetPosition_(x, epoch, target, center);
        return x;
    }
    DEVICEHOST()
    inline void iGetPosition_(Vec3R& x, const Real& epoch, const NaifId& target,
        const NaifId& center) const
    {
        /* find common center */
        core::WalkerPair walkers = traverser_.makeWalkers(target, center);

        /* read ephemeris */
        interpolators_.getPosition(x, epoch, walkers.center_);
        x = -x;
        interpolators_.getPosition(x, epoch, walkers.target_);
    }

    /** @brief Host Velocity dispatcher */
    Vec3R hostGetVelocity_(
        const Real& epoch, const NaifId& target, const NaifId& center) const;
    void hostiGetVelocity_(Vec3R& out, const Real& epoch, const NaifId& target,
        const NaifId& center) const;

#ifndef BRIE_CPU_ONLY
    /** @brief Device Velocity dispatcher */
    DEVICE()
    Vec3R deviceGetVelocity_(
        const Real& epoch, const NaifId& target, const NaifId& center) const;
    DEVICE()
    void deviceiGetVelocity_(Vec3R& out, const Real& epoch,
        const NaifId& target, const NaifId& center) const;
#endif

    /** @brief Device-host core for get velocity */
    DEVICEHOST()
    inline Vec3R getVelocity_(
        const Real& epoch, const NaifId& target, const NaifId& center) const
    {
        Vec3R x;
        iGetVelocity_(x, epoch, target, center);
        return x;
    }
    DEVICEHOST()
    inline void iGetVelocity_(Vec3R& x, const Real& epoch, const NaifId& target,
        const NaifId& center) const
    {
        /* find common center */
        core::WalkerPair walkers = traverser_.makeWalkers(target, center);

        /* read ephemeris */
        interpolators_.getVelocity(x, epoch, walkers.center_);
        x = -x;
        interpolators_.getVelocity(x, epoch, walkers.target_);
    }

    /** @brief Host Position and Velocity dispatcher */
    Vec6R hostGetPositionAndVelocity_(
        const Real& epoch, const NaifId& target, const NaifId& center) const;
    void hostiGetPositionAndVelocity_(Vec6R& out, const Real& epoch,
        const NaifId& target, const NaifId& center) const;

#ifndef BRIE_CPU_ONLY
    /** @brief Device Position and Velocity dispatcher */
    DEVICE()
    Vec6R deviceGetPositionAndVelocity_(
        const Real& epoch, const NaifId& target, const NaifId& center) const;
    DEVICE()
    void deviceiGetPositionAndVelocity_(Vec6R& out, const Real& epoch,
        const NaifId& target, const NaifId& center) const;
#endif

    /** @brief Device-host core for get position and velocity */
    DEVICEHOST()
    inline Vec6R getPositionAndVelocity_(
        const Real& epoch, const NaifId& target, const NaifId& center) const
    {
        Vec6R x;
        iGetPositionAndVelocity_(x, epoch, target, center);
        return x;
    }
    DEVICEHOST()
    inline void iGetPositionAndVelocity_(Vec6R& x, const Real& epoch,
        const NaifId& target, const NaifId& center) const
    {
        /* find common center */
        core::WalkerPair walkers = traverser_.makeWalkers(target, center);

        /* read ephemeris */
        interpolators_.getPositionAndVelocity(x, epoch, walkers.center_);
        x = -x;
        interpolators_.getPositionAndVelocity(x, epoch, walkers.target_);
    }
};

/* Explicit instantiations */
extern template class RefEphUnit<true, true>;
extern template class RefEphUnit<true, false>;
extern template class RefEphUnit<false, true>;
extern template class RefEphUnit<false, false>;

} // namespace states
} // namespace brie