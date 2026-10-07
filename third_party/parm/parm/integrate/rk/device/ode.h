/* ODE related device code */
#pragma once

#include "parm/integrate/base/Simulation.h"
#include "parm/integrate/rk/coefs/tableau.h"
#include "parm/integrate/rk/detail/ode.h"

namespace parm {
namespace integrate {
namespace rk {
namespace device {
namespace ode {
namespace kernel {

#define PIBLOCKSIZE 256
#define PIACTIVEBLOCKS 4

/** @brief ODE preparation at the given stage: compute and store the given
 * intermediate state and epoch */
template<idx_t stage, typename DStatesT, typename Tableau>
KERNEL()
__launch_bounds__(PIBLOCKSIZE, PIACTIVEBLOCKS) void prepareIntermediateState(
    typename DStatesT::IntermediateStatesT::GRef intermediateStates,
    GRID_CONSTANT()
        typename GRefArrT<typename DStatesT::ComponentT>::HandleT currentEpochs,
    GRID_CONSTANT() typename DStatesT::StatesT::GRef states,
    GRID_CONSTANT() typename DStatesT::GRef dStates,
    GRID_CONSTANT()
        typename GRefArrT<typename DStatesT::ComponentT>::HandleT dt,
    GRID_CONSTANT() typename GRefArrT<bool>::HandleT terminated)
{
    const SampleIndex i
        = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);

    if (i.global() >= states.size())
        return;

    using RefIStatesT = typename DStatesT::IntermediateStatesT::GRef;
    using RefDStatesT = typename DStatesT::GRef;

    /* Do nothing if terminated */
    if (terminated[i])
        return;

    /* Write set to zero the dstate for this stage */
    using ZerosT = typename feta::vector::ZerosNT<typename DStatesT::ComponentT,
        DStatesT::StatesT::VecDims>;
    typename DStatesT::StatesT::GRef dStateStage
        = dStates.template extractStage<stage>();
    dStateStage[i] = ZerosT();

    /* Compute intermediate state and epoch */
    detail::ode::Eval<stage, DStatesT, Tableau>::prepareIntermediateState(
        i, intermediateStates, currentEpochs, states, dStates, dt);
}

} // namespace kernel

/** @brief Compile-time unrolled recursive evaluation of the ODE */
template<idx_t stage, typename DStatesT, typename Tableau, typename ODE>
struct RecurseEval {
    using Self  = RecurseEval;
    using BaseT = parm::integrate::base::Simulation;

    /** @brief Stream capturing evaluation */
    static inline void eval(const ODE& ode,
        const typename DStatesT::StatesT::GRef& states,
        typename DStatesT::GRef& dStates, const BaseT::GRef& base,
        const cudaStream_t& stream, const idx_t& nBlocks,
        const idx_t& blockSize)
    {
        if constexpr (stage >= 1) {
            RecurseEval<stage - 1, DStatesT, Tableau, ODE>::eval(
                ode, states, dStates, base, stream, nBlocks, blockSize);
        }
        /* Prepare the intermediate state */
        kernel::prepareIntermediateState<stage, DStatesT, Tableau>
            <<<nBlocks, blockSize, 0, stream>>>(dStates.intermediateStates(),
                base.currentEpochs(), states, dStates, base.nextDts(),
                base.terminated());
        /* Run the ODE evaluation pipeline (defined externally) */
        ode.eval(dStates.template extractStage<stage>(),
            dStates.intermediateStates_.states(),
            dStates.intermediateStates_.epochs(), base.terminated(), stream);
    }

    /** @brief Parm-Graph filling evaluator */
    static inline void eval(util::graph::Graph& graph, const ODE& ode,
        const typename DStatesT::StatesT::GRef& states,
        typename DStatesT::GRef& dStates, const BaseT::GRef& base,
        const cudaStream_t& stream, const idx_t& nBlocks,
        const idx_t& blockSize)
    {
        if constexpr (stage >= 1) {
            RecurseEval<stage - 1, DStatesT, Tableau, ODE>::eval(
                graph, ode, states, dStates, base, stream, nBlocks, blockSize);
        }
        util::graph::StreamCapturer capturer(stream);
        capturer.begin();
        kernel::prepareIntermediateState<stage, DStatesT, Tableau>
            <<<nBlocks, blockSize, 0, stream>>>(dStates.intermediateStates(),
                base.currentEpochs(), states, dStates, base.nextDts(),
                base.terminated());
        /* Pass ``PIBLOCKSIZE`` as the idealBlockSize so a future
         * ``Launcher::setLogicalSize`` re-tunes both gridDim and
         * blockDim through ``computeBlocks``. */
        graph.addNode(capturer.end(), {}, PIBLOCKSIZE);
        /* The ``<stage>`` template arg lets consumers gate per-stage
         * work (e.g. stage-0 ephemeris cache short-circuit). */
        ode.template eval<stage>(graph,
            dStates.template extractStage<stage>(),
            dStates.intermediateStates_.states(),
            dStates.intermediateStates_.epochs(), base.terminated(), stream);
    }
};

/** @brief Recursive ODE evaluation wrapper.
 *
 * Launches one kernel per RK stage, captured into a CUDA graph for
 * replay; supports per-stage callbacks and external integration. */
template<typename DStatesT, typename Tableau, typename ODE>
struct Eval {
    /** @brief CUDA-Graph-returning evaluator */
    static cudaGraph_t eval(const ODE& ode,
        const typename DStatesT::StatesT::GRef& states,
        typename DStatesT::GRef& dStates, const base::Simulation::GRef& base,
        const cudaStream_t& stream)
    {
        util::graph::StreamCapturer capturer(stream);
        capturer.begin();
        constexpr idx_t blockSize = PIBLOCKSIZE;
        const idx_t nBlocks       = (states.size() + blockSize - 1) / blockSize;
        RecurseEval<Tableau::nStages() - 1, DStatesT, Tableau, ODE>::eval(
            ode, states, dStates, base, stream, nBlocks, blockSize);
        return capturer.end();
    }

    /** @brief Parm-Graph filling evaluator */
    static void eval(util::graph::Graph& graph, const ODE& ode,
        const typename DStatesT::StatesT::GRef& states,
        typename DStatesT::GRef& dStates, const base::Simulation::GRef& base,
        const cudaStream_t& stream)
    {
        constexpr idx_t blockSize = PIBLOCKSIZE;
        const idx_t nBlocks       = (states.size() + blockSize - 1) / blockSize;
        RecurseEval<Tableau::nStages() - 1, DStatesT, Tableau, ODE>::eval(
            graph, ode, states, dStates, base, stream, nBlocks, blockSize);
    }
};

} // namespace ode
} // namespace device
} // namespace rk
} // namespace integrate
} // namespace parm
