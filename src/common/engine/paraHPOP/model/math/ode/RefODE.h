#pragma once

#include <parm/util.h>

#include "paraHPOP/model/samples/metadata/Orbit.h"
#include "paraHPOP/util.h"

namespace paraHPOP {
namespace model {

/* Short hand for vector array */
using feta::vector::Array;

/**
 * @brief Adapter which allows to solve ODEs, storing the physical states in the
 * given reference (Work or Global representation)
 *
 */
template<typename ModelT, idx_t ORDER_, bool work, bool MaybeVolatile = false>
class RefODE {
public:
    /** @brief Volatile is allowed only with work */
    static constexpr bool Volatile  = MaybeVolatile && work;
    static constexpr idx_t ORDER    = ORDER_;
    using RefModelT                 = typename ModelT::GRef;
    using SamplesT                  = typename RefModelT::SamplesT;
    using MetadataT                 = typename SamplesT::MetadataT;
    using FullModelPhysicsT         = typename RefModelT::PhysicsT;
    using PhysicsT                  = typename FullModelPhysicsT::ReducedT;
    using StatesT                   = typename SamplesT::StatesT;
    using StateT                    = typename StatesT::StateT;
    static constexpr idx_t StateDim = StatesT::VecDims / 2;

    using Self = RefODE;

    /* This type except work boolean */
    template<bool iWork>
    using SelfT = RefODE<ModelT, ORDER, iWork>;

    /* Additional data buffer */

    // /* Additional Real parameters */

    // /* Additional integer parameters */


    /** @brief Factory method to construct from model reference */
    DEVICEHOST() static RefODE make(const RefModelT& model)
    {
        return make(model.samples().metadata(), model.physics().reduced());
    }

    /** @brief Factory method to construct from data members */
    DEVICEHOST()
    static RefODE make(const MetadataT& metadata, const PhysicsT& physics)
    {
        static_assert(ORDER == 1 || ORDER == 2,
            "Only first or second order equations are implemented");
        return { metadata, physics };
    }

    /** @brief Extract the ODE result */

    /** @brief Evaluate the ODE at the intermediate step */
    DEVICEHOST()
    StateT eval(const SampleIndex& idx,
        const feta::scalar::Array<typename StatesT::ComponentT>::GRef& t,
        const StatesT& states) const
    {
        /* Prepare the intermediate states */
        return StateT::template ODEcast<ORDER>(
            states[idx], physeval_(idx, states, t[idx]));
    }

    /** @brief Evaluate the ode for the given already extracted individual state
     * and time */
    DEVICEHOST()
    StateT eval(
        const SampleIndex& idx, const StateT& state, const Real& t) const
    {
        return StateT::template ODEcast<ORDER>(state, physeval_(idx, state, t));
    }

    /** @brief Overloads - Let operator() inline call eval */
    DEVICEHOST()
    inline StateT operator()(const SampleIndex& idx,
        const feta::scalar::Array<typename StatesT::ComponentT>::GRef& t,
        const StatesT& states) const
    {
        return eval(idx, t, states);
    }
    DEVICEHOST()
    inline StateT operator()(
        const SampleIndex& idx, const StateT& state, const Real& t) const
    {
        return eval(idx, state, t);
    }

    /** @brief Required buffer size (in bytes) when constructing in shared
     * memory. Uses feta work references for type-safe array management */
    static size_t bufBytes(const size_t& blockSize)
    {
        return StatesT::VecDims * blockSize
            * sizeof(typename StatesT::ComponentT);
    }

    /** @brief Return a newly created global reference reference for this
       ODE */
    DEVICEHOST() Self clone() const { return *this; }

    /** @brief Data members made public for PODification */
    MetadataT metadata_;
    PhysicsT physics_;

private:
    /** @brief Internally evaluate physical model for the given intermediate
       step. Optimized to cache metadata access outside the hot path */
    DEVICEHOST()
    Vec3R physeval_(
        const SampleIndex& idx, const StatesT& states, const Real& t) const
    {
        /* Extract physical state components */
        feta::vector::Item<Real, StateDim> pos, vel;
        {
            feta::vector::Item<Real, StatesT::VecDims> physicalState
                = StateT::toPhysicalState(states[idx]);
            pos = physicalState.template head<StateDim>();
            vel = physicalState.template tail<StateDim>();
        }
        /* Directly return the acceleration*/
        const Real physt = StateT::template getPhysicalTime(idx, states, t);
        return minimalPhysEval_(idx, physt, pos, vel);
    }

    /** @brief Internally evaluate physical model for the given intermediate
   step, for the given individual state*/
    DEVICEHOST()
    Vec3R physeval_(
        const SampleIndex& idx, const StateT& state, const Real& t) const
    {
        /* Extract physical state components */
        feta::vector::Item<Real, StateDim> pos, vel;
        {
            feta::vector::Item<Real, StatesT::VecDims> physicalState
                = StateT::toPhysicalState(state);
            pos = physicalState.template head<StateDim>();
            vel = physicalState.template tail<StateDim>();
        }
        const Real physt = StateT::template getPhysicalTime(idx, state, t);
        return minimalPhysEval_(idx, physt, pos, vel);
    }

    /** @brief direct physics evaluation - minimal.
     *
     *  Reads the four raw per-sample fields (``mass``, ``area``,
     *  ``cr``, ``cd``) and passes them through to ``physics_.eval``.
     *  The derived quantities AMS (= cr·area/mass, SRP-specific) and
     *  BC (= cd·area/mass, drag-specific) are computed inside
     *  ``RefAccelerations::eval_()`` once per sample, hoisted above
     *  the per-body loop — same arithmetic cost as pre-computing at
     *  this scope, but keeps the call site free of mismatch risk. */
    DEVICEHOST()
    Vec3R minimalPhysEval_(const SampleIndex& idx, const Real& phystime,
        const feta::vector::Item<Real, StateDim>& pos,
        const feta::vector::Item<Real, StateDim>& vel) const
    {
        const Real mass        = metadata_.own().mass()[idx];
        const Real area        = metadata_.own().area()[idx];
        const Real cr          = metadata_.own().cr()[idx];
        const Real cd          = metadata_.own().cd()[idx];
        const brie::NaifId coi = metadata_.own().cois()[idx];
        return physics_.eval(idx, pos, vel, phystime, mass, area, cr, cd, coi);
    }
};

} // namespace model
} // namespace paraHPOP