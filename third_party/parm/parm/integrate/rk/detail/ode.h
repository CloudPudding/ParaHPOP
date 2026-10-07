/*
    This header file contains detail functions for the computation of the
    ODE during a Runge-Kutta step
*/
#pragma once

#include "parm/typedefs.h"
#include "parm/util.h"

namespace parm {
namespace integrate {
namespace rk {
namespace detail {
namespace ode {

/** @brief Perform the intermediate summation of the DStates for the ODE
 * evaluation */
template<idx_t row, idx_t col, typename DStatesT, typename Tableau>
struct RecurseSumIntermediateDStates {

    /** @brief Short hands */
    using StateT     = typename DStatesT::StateT;
    using ComponentT = typename DStatesT::ComponentT;

    /** @brief Host/device function to accumulate into an output state */
    DEVICEHOST()
    static FORCEINLINE() void sumInto(const SampleIndex& idx,
        const typename DStatesT::GRef& dStates, StateT& acc,
        const ComponentT& dt)
    {
        /* Recurse to previous column */
        if constexpr (col >= 1) {
            RecurseSumIntermediateDStates<row, col - 1, DStatesT,
                Tableau>::sumInto(idx, dStates, acc, dt);
        }
        /* accumulate */
        if constexpr (Tableau::template a<row, col>() != 0)
            acc += dt * Tableau::template a<row, col>()
                * dStates.template stage<col>()[idx];
    }

    /** @brief Packet-aware accumulation into a PacketItem. */
    template<feta::idx_t W>
    static inline void sumInto(const feta::PacketIndex<W>& pi,
        const typename DStatesT::GRef& dStates,
        feta::vector::PacketItem<ComponentT, DStatesT::StatesT::VecDims, W>&
            acc,
        const feta::simd::Packet<ComponentT, W>& dt)
    {
        using PacketT             = feta::simd::Packet<ComponentT, W>;
        constexpr feta::dims_t VD = DStatesT::StatesT::VecDims;

        if constexpr (col >= 1) {
            RecurseSumIntermediateDStates<row, col - 1, DStatesT,
                Tableau>::sumInto(pi, dStates, acc, dt);
        }
        if constexpr (Tableau::template a<row, col>() != 0) {
            auto stage = feta::cpu::packetCapture<ComponentT>(
                dStates.template stage<col>(), pi);
            auto coeff
                = dt * PacketT::broadcast(Tableau::template a<row, col>());
            [&]<feta::dims_t... D>(std::integer_sequence<feta::dims_t, D...>) {
                ((acc.template packet<D>() = fmadd(coeff,
                      stage.template packet<D>(), acc.template packet<D>())),
                    ...);
            }(std::make_integer_sequence<feta::dims_t, VD>{});
        }
    }
};

/**
 * @brief Compile-time unrolled evaluation of \f$\sum_j a_ij \cdot k_j\f$
 */
template<idx_t stage, typename DStatesT, typename Tableau>
struct SumIntermediateDStates {
    using StateT     = typename DStatesT::StateT;
    using ComponentT = typename DStatesT::ComponentT;

    /** @brief In-place accumulation helper */
    DEVICEHOST()
    static FORCEINLINE() void sumInto(const SampleIndex& idx,
        const typename DStatesT::GRef& dStates, StateT& acc,
        const ComponentT& dt)
    {
        RecurseSumIntermediateDStates<stage - 1, stage - 1, DStatesT,
            Tableau>::sumInto(idx, dStates, acc, dt);
    }

    /** @brief Packet-aware accumulation helper */
    template<feta::idx_t W>
    static inline void sumInto(const feta::PacketIndex<W>& pi,
        const typename DStatesT::GRef& dStates,
        feta::vector::PacketItem<ComponentT, DStatesT::StatesT::VecDims, W>&
            acc,
        const feta::simd::Packet<ComponentT, W>& dt)
    {
        RecurseSumIntermediateDStates<stage - 1, stage - 1, DStatesT,
            Tableau>::sumInto(pi, dStates, acc, dt);
    }
};

/** @brief ODE evaluation helpers - preparation of intermediate states and
 * result retrieval */
template<idx_t stage, typename DStatesT, typename Tableau>
struct Eval {

    /** @brief Short hands */
    using ComponentT    = typename DStatesT::ComponentT;
    using ComponentArrT = GRefArrT<ComponentT>;
    using StatesT       = typename DStatesT::StatesT;
    using StateT        = typename DStatesT::StateT;

    /** @brief Prepare the intermediate state */
    DEVICEHOST()
    static FORCEINLINE() void prepareIntermediateState(const SampleIndex& i,
        typename DStatesT::IntermediateStatesT::GRef& intermediateStates,
        const typename ComponentArrT::HandleT& currentEpochs,
        const typename StatesT::GRef& states,
        const typename DStatesT::GRef& dStates,
        const typename ComponentArrT::HandleT& dt)
    {
        /* reworked simpler version for stage = 0*/
        if constexpr (stage == 0) {
            intermediateStates.states_[i] = states[i];
            intermediateStates.epochs_[i] = currentEpochs[i];
            return;
        } else {
            /* Compute intermediate state and epoch */
            ComponentT t         = currentEpochs[i];
            StateT state         = states[i];
            const ComponentT dti = dt[i];
            t += Tableau::template c<stage - 1>() * dti;
            /* Compute the intermediate state */
            SumIntermediateDStates<stage, DStatesT, Tableau>::sumInto(
                i, dStates, state, dti);
            /* Do the final write */
            intermediateStates.states_[i] = state;
            intermediateStates.epochs_[i] = t;
        }
    }

    /** @brief Packet-aware intermediate state preparation. */
    template<feta::idx_t W>
    static inline void prepareIntermediateState(const feta::PacketIndex<W>& pi,
        typename DStatesT::IntermediateStatesT::GRef& intermediateStates,
        const typename ComponentArrT::HandleT& currentEpochs,
        const typename StatesT::GRef& states,
        const typename DStatesT::GRef& dStates,
        const typename ComponentArrT::HandleT& dt)
    {
        using PacketT             = feta::simd::Packet<ComponentT, W>;
        constexpr feta::dims_t VD = StatesT::VecDims;

        if constexpr (stage == 0) {
            // Copy states and epochs directly
            auto st = feta::cpu::packetCapture<ComponentT>(states, pi);
            feta::cpu::packetAssign(intermediateStates.states_, st, pi);
            PacketT epochs = util::Host::packetLoad<ComponentT, W>(
                currentEpochs.data(), pi);
            util::Host::packetStore<ComponentT, W>(
                intermediateStates.epochs_.data(), pi, epochs);
        } else {
            PacketT epochs = util::Host::packetLoad<ComponentT, W>(
                currentEpochs.data(), pi);
            PacketT dti = util::Host::packetLoad<ComponentT, W>(dt.data(), pi);

            // t = currentEpoch + c_{stage-1} * dt
            PacketT cCoeff
                = PacketT::broadcast(Tableau::template c<stage - 1>());
            PacketT t = fmadd(cCoeff, dti, epochs);

            // state = states[pi] + sum_j a_{stage,j} * dt * k_j
            auto st = feta::cpu::packetCapture<ComponentT>(states, pi);
            feta::vector::PacketItem<ComponentT, VD, W> state;
            [&]<feta::dims_t... D>(std::integer_sequence<feta::dims_t, D...>) {
                ((state.template packet<D>() = st.template packet<D>()), ...);
            }(std::make_integer_sequence<feta::dims_t, VD>{});

            SumIntermediateDStates<stage, DStatesT, Tableau>::sumInto(
                pi, dStates, state, dti);

            feta::cpu::packetAssign(intermediateStates.states_, state, pi);
            util::Host::packetStore<ComponentT, W>(
                intermediateStates.epochs_.data(), pi, t);
        }
    }
};

} // namespace ode
} // namespace detail
} // namespace rk
} // namespace integrate
} // namespace parm
