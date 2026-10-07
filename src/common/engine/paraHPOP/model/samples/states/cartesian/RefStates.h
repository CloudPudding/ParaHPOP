#pragma once

#include "paraHPOP/model/samples/states/cartesian/State.h"

namespace paraHPOP {
namespace model {
namespace samples {
namespace states {
namespace cartesian {

/** @brief Reference cartesian states */
template<bool work, bool MaybeVolatile = false>
class RefStates
    : public feta::vector::Array<Real, 6>::template Ref<work, MaybeVolatile> {
    using Self    = RefStates<work, MaybeVolatile>;
    using ParentT =
        typename feta::vector::Array<Real, 6>::template Ref<work, MaybeVolatile>;
    using HeadT      = feta::vector::expr::ComponentView<Self, 0, 3>;
    using ConstHeadT = feta::vector::expr::ConstComponentView<Self, 0, 3>;
    using TailT      = feta::vector::expr::ComponentView<Self, 3, 3>;
    using ConstTailT = feta::vector::expr::ConstComponentView<Self, 3, 3>;

public:
    /** @brief Volatile is allowed only with work */
    static constexpr bool Volatile = MaybeVolatile && work;
    /** @brief single state type for this state */
    using StateT    = State;
    using PosT      = HeadT;
    using VelT      = TailT;
    using ConstPosT = ConstHeadT;
    using ConstVelT = ConstTailT;

    /** @brief Expose View types */
    using ParentT::ConstViewT;
    using ParentT::ViewT;
    using ParentT::operator[];

    /** @brief Factory method to construct from Parent Type */
    DEVICEHOST() static RefStates make(const ParentT& v)
    {
        RefStates out;
        static_cast<ParentT&>(out) = v;
        return out;
    }

    /** @brief Obtain a view onto the 3D position vector */
    DEVICEHOST() PosT pos() { return PosT(*this); }

    /** @brief Obtain a view onto the 3D position vector */
    DEVICEHOST() ConstPosT pos() const { return ConstPosT(*this); }

    /** @brief Obtain a view onto the 3D velocity vector */
    DEVICEHOST() VelT vel() { return VelT(*this); }

    /** @brief Obtain a view onto the 3D velocity vector */
    DEVICEHOST() ConstVelT vel() const { return ConstVelT(*this); }

    /** @brief ODE cast helper */
    template<idx_t ORDER>
    DEVICEHOST()
    static void ODEcast(
        const SampleIndex& idx, Self& dStates, const Self& states)
    {
        static_assert(ORDER == 1, "Only first order equations are implemented");
        if constexpr (ORDER == 1) {
            /* Move the velocity */
            dStates.pos()[idx] = states.vel();
        }
    }

    /** @brief ODE cast helper */
    template<idx_t ORDER>
    DEVICEHOST()
    static void ODEcast(const SampleIndex& idx, Self& dStates,
        const Self& states, const Vec3R& acc)
    {
        static_assert(ORDER == 1, "Only first order equations are implemented");
        if constexpr (ORDER == 1) {
            /* Move the velocity */
            dStates.pos()[idx] = states.vel();
            /* Move the acceleration */
            dStates.vel()[idx] = acc;
        }
    }
};

} // namespace cartesian
} // namespace states
} // namespace samples
} // namespace model
} // namespace paraHPOP
