#pragma once

#include "paraHPOP/typedefs.h"

namespace paraHPOP {
namespace model {
namespace samples {
namespace states {
namespace cartesian {

/**
 * @brief Single cartesian state
 *
 */
class State : public Vec6R {
    using ParentT = Vec6R;
    using HalfT   = Vec3R;
    using Self    = State;

public:
    /** @brief Inherit constructors from Vec6R so State can be constructed from
     * expressions and buffers just like Vec6R (enables `State s = view;`) */
    using ParentT::ParentT;

    /** @brief Static method to construct from position and velocity components
     */
    DEVICEHOST() static State make(const Vec3R& pos, const Vec3R& vel)
    {
        State out;
        out.template head<3>() = pos;
        out.template tail<3>() = vel;
        return out;
    }

    /** @brief Obtain a view onto the 3D position vector */
    DEVICEHOST() decltype(auto) pos() { return ParentT::template head<3>(); }

    /** @brief Obtain a view onto the 3D position vector */
    DEVICEHOST()
    decltype(auto) pos() const { return ParentT::template head<3>(); }

    /** @brief Obtain a view onto the 3D velocity vector */
    DEVICEHOST() decltype(auto) vel() { return ParentT::template tail<3>(); }

    /** @brief Obtain a view onto the 3D velocity vector */
    DEVICEHOST()
    decltype(auto) vel() const { return ParentT::template tail<3>(); }

    /**
     * @brief STATIC METHODS
     *
     */

    /** @brief Convert the given state into physical representation */
    DEVICEHOST()
    static inline Vec6R toPhysicalState(const Vec6R& state) { return state; }

    /** @brief Expression template-related forwarder */
    template<typename Expr>
    DEVICEHOST()
    static inline decltype(auto) toPhysicalState(const Expr& expr)
    {
        return expr;
    }

    /** @brief Given the state vector and the independent integration variable,
     * retrieve the physical time */
    DEVICEHOST()
    static inline Real getPhysicalTime(
        [[maybe_unused]] const Self& state, const Real& t)
    {
        return t;
    }

    /** @brief sample index, expression templated, and reference array based
     * extraction of the physical time */
    template<typename Expr>
    DEVICEHOST()
    static inline Real getPhysicalTime([[maybe_unused]] const SampleIndex& i,
        [[maybe_unused]] const Expr& expr, const Real& t)
    {
        return t;
    }

    /** @brief Cast the given acceleration to either first or second order ODE
     * representation */
    template<idx_t ORDER>
    DEVICEHOST()
    static inline decltype(auto) ODEcast(Self state, const Vec3R& physicalAcc)
    {
        static_assert(ORDER == 1 || ORDER == 2,
            "Only first or second order equations are implemented");
        if constexpr (ORDER == 1) {
            Vec6R out;
            out.head<3>() = state.template tail<3>();
            out.tail<3>() = physicalAcc;
            return out;
        } else
            return physicalAcc;
    }
};

} // namespace cartesian
} // namespace states
} // namespace samples
} // namespace model
} // namespace paraHPOP
