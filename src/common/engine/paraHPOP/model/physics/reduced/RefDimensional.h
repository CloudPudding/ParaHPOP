/* Reduced version of the physical model, that does not include events and
 * interactions */
#pragma once

#include "paraHPOP/model/accelerations.h"
#include "paraHPOP/model/environment.h"
#include "paraHPOP/model/samples/states/cartesian/States.h"

namespace paraHPOP {
namespace model {
namespace physics {
namespace reduced {

/**
 * @brief Non-owning reference to The Reduced Physics
 *
 */
template<bool work, bool MaybeVolatile = false>
class RefDimensional {
public:
    /** @brief Volatile is allowed only with work */
    static constexpr bool Volatile = MaybeVolatile && work;
    /* Get data types */
    using StatesT = typename samples::states::cartesian::States::template Ref<
        work, MaybeVolatile>;
    using StateT  = typename StatesT::StateT;
    using EnvT
        = typename environment::Env::template Ref<work, MaybeVolatile>;
    using AccT = typename accelerations::Accelerations::template Ref<work,
        MaybeVolatile>;

    /** @brief Constexpr to mark if this physical model is non dimensional */
    static constexpr bool IsNondimensional = false;

    /** @brief Factory method to construct from data members */
    DEVICEHOST() static RefDimensional make(EnvT&& env, AccT&& accs)
    {
        return RefDimensional(std::move(env), std::move(accs));
    }

    /** @brief Expose environment */
    DEVICEHOST() const EnvT& env() const { return env_; }

    /** @brief Expose accelerations */
    DEVICEHOST() const AccT& accs() const { return accs_; }

    /** @brief Evaluate physical model from raw per-sample fields.
     *  See ``RefAccelerations::eval`` for the AMS/BC derivation. */
    DEVICEHOST()
    Vec3R eval(const SampleIndex& idx, const Vec3R& pos, const Vec3R& vel,
        const Real& epoch, const Real& mass, const Real& area,
        const Real& cr, const Real& cd, const brie::NaifId& COI) const
    {
#ifdef __CUDA_ARCH__
        return deviceEval_(idx, pos, vel, epoch, mass, area, cr, cd, COI);
#else
        return hostEval_(idx, pos, vel, epoch, mass, area, cr, cd, COI);
#endif
    }

    /** @brief Packet (W-lane) physical model evaluation —
     *         uniform-COI path.
     *
     *  Thin wrapper over ``RefAccelerations::evalPacket`` mirroring
     *  scalar ``eval`` 's shape.  Caller has enforced uniform-COI.
     *  Pure-packet path runs PointGravity + SRP; per-lane scalar
     *  fallback covers J2/SH/drag cases. */
    template<idx_t W>
    inline feta::vector::PacketItem<Real, 3, W> evalPacket(
        const feta::PacketIndex<W>& pi,
        const feta::vector::PacketItem<Real, 3, W>& pos,
        const feta::vector::PacketItem<Real, 3, W>& vel,
        const feta::simd::Packet<Real, W>& epoch,
        const feta::simd::Packet<Real, W>& mass,
        const feta::simd::Packet<Real, W>& area,
        const feta::simd::Packet<Real, W>& cr,
        const feta::simd::Packet<Real, W>& cd,
        const brie::NaifId& COI) const
    {
        return accs_.template evalPacket<W>(
            pi, pos, vel, epoch, mass, area, cr, cd, COI, this->env());
    }

    /** @brief Get a new global reference to this physical model */
    DEVICEHOST()
    RefDimensional<work> clone() const { return *this; }

    /** @brief Data members made public for PODification */
    EnvT env_  = EnvT{};
    AccT accs_ = AccT{};

protected:
    DEVICEHOST()
    inline Vec3R eval_(const SampleIndex& idx, const Vec3R& pos,
        const Vec3R& vel, const Real& epoch, const Real& mass,
        const Real& area, const Real& cr, const Real& cd,
        const brie::NaifId& COI) const
    {
        return accs_.eval(
            idx, pos, vel, epoch, mass, area, cr, cd, COI, this->env());
    }

    /** @brief Host version */
    Vec3R hostEval_(const SampleIndex& idx, const Vec3R& pos, const Vec3R& vel,
        const Real& epoch, const Real& mass, const Real& area, const Real& cr,
        const Real& cd, const brie::NaifId& COI) const;

    /** @brief Device version */
    DEVICE()
    Vec3R deviceEval_(const SampleIndex& idx, const Vec3R& pos,
        const Vec3R& vel, const Real& epoch, const Real& mass,
        const Real& area, const Real& cr, const Real& cd,
        const brie::NaifId& COI) const;
};

/* Explicit instantiation */
extern template class RefDimensional<true>;
extern template class RefDimensional<false>;

} // namespace reduced
} // namespace physics
} // namespace model
} // namespace paraHPOP