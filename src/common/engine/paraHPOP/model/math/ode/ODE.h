#pragma once

#include "paraHPOP/model/math/ode/RefODE.h"

namespace paraHPOP {
namespace model {

/* ODE finalization kernels */
namespace ode {
namespace device {

/** @brief ODE finalization */
template<typename ODET>
__global__ void finalize(typename ODET::StatesT::GRef dStates,
    GRID_CONSTANT() typename ODET::StatesT::GRef states,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated)
{
    const SampleIndex i
        = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);

    if (i.global() >= dStates.size())
        return;

    if (terminated[i])
        return;

    /* Finalize the ode cast */
    using StatesT = typename ODET::StatesT::GRef;
    StatesT::template ODEcast<ODET::ORDER>(i, dStates, states);
}


/** @brief ODE Full evaluation kernel */
template<typename ODET>
__global__ void eval(GRID_CONSTANT() typename ODET::GRef ode,
    GRID_CONSTANT() typename ODET::StatesT::GRef dStates,
    GRID_CONSTANT() typename ODET::StatesT::GRef states,
    GRID_CONSTANT()
        typename feta::scalar::Array<typename ODET::StatesT::ComponentT>::GRef
            epochs,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated)
{
    const SampleIndex i(threadIdx.x + blockIdx.x * blockDim.x);

    if (i.global() >= dStates.size())
        return;

    using ComponentT = typename ODET::StatesT::ComponentT;
    using RefStatesT = typename ODET::StatesT::GRef;

    /* Leverage shared memory */

#if __CUDA_ARCH__ < 700
    using RefODET = typename ODET::GRef;
    __shared__ RefODET sharedODE;

    using RefEpochsT = typename feta::scalar::Array<ComponentT>::GRef;
    __shared__ RefEpochsT sharedEpochs;
    __shared__ RefStatesT sharedStates;
#endif

    __shared__ RefStatesT sharedDStates;

    /* Copy-in the non-const items */
    if (i.work() == parm::util::Threads::blockLeader()) {
#if __CUDA_ARCH__ < 700
        sharedEpochs = epochs;
        sharedStates = states;
        sharedODE    = ode;
#endif
        sharedDStates = dStates;
    }
    __syncthreads();

    /* Do nothing if terminated */
    if (terminated[i])
        return;

#if __CUDA_ARCH__ < 700
    sharedDStates[i] = sharedODE.eval(i, sharedEpochs, sharedStates);
#else
    sharedDStates[i] = ode.eval(i, epochs, states);
#endif
}
} // namespace device

namespace host {
/** @brief ODE finalization launcher */
template<typename ODET>
void eval(const typename ODET::GRef& ode, typename ODET::StatesT::GRef& dStates,
    const typename ODET::StatesT::GRef& states,
    const typename feta::scalar::Array<
        typename ODET::StatesT::ComponentT>::GRef& epochs,
    const feta::scalar::Array<bool>::GRef::HandleT& terminated)
{
    parm::util::Host::launchIfNot(
        dStates.size(), terminated, [&](const SampleIndex& i) {
            dStates[i] = ode.eval(i, epochs, states);
        });
}

} // namespace host
} // namespace ode


/** @brief Reference holder-owning Type manager */
template<typename ModelT, idx_t ORDER_, bool needsStateRecast = false>
class ODE {
    using HolderT = parm::util::ReferenceHolder<ModelT>;
    using Self    = ODE;

public:
    static constexpr idx_t ORDER = ORDER_;
    /** @brief Expose the states type */
    using StatesT = typename ModelT::SamplesT::StatesT;
    /** @brief Expose the component type */
    using ComponentT = typename StatesT::ComponentT;
    /** @brief and the component array type (epochs) */
    using EpochsT = feta::scalar::Array<ComponentT>;
    /** @brief Short-hand for the boolean array type */
    using BoolArrayT = feta::scalar::Array<bool>;

    /** @brief Reference Types */
    template<bool work, bool MaybeVolatile = false>
    using Ref  = RefODE<ModelT, ORDER, work, MaybeVolatile>;
    using GRef = Ref<false>;
    using WRef = Ref<true>;
    /** @brief Volatile work reference */
    using VolatileRef = Ref<true, true>;

    /** @brief Default constructor is forbidden */
    ODE() = delete;

    /** @brief Copy constructor is forbidden */
    ODE(ODE& other)       = delete;
    ODE(const ODE& other) = delete;

    /** @brief Construction from model reference */
    ODE(const ModelT& model)
        : holder_{ model }
    {
        static_assert(ORDER == 1 || ORDER == 2,
            "Only first or second order equations are implemented");
    }

    /** @brief Move construction from data member */
    ODE(HolderT&& holder)
        : holder_{ std::move(holder) }
    {
        static_assert(ORDER == 1 || ORDER == 2,
            "Only first or second order equations are implemented");
    }

    /** @brief Move constructor */
    ODE(ODE&& other)
        : holder_{ std::move(other.holder_) }
    {
        static_assert(ORDER == 1 || ORDER == 2,
            "Only first or second order equations are implemented");
    }

    /** @brief Copy assignmeny is forbidden */
    ODE& operator=(ODE& other)       = delete;
    ODE& operator=(const ODE& other) = delete;

    /** @brief Move assignment is allowed */
    ODE& operator=(ODE&& other)
    {
        holder_ = std::move(other.holder_);
        return *this;
    }

    /** @brief Run the ODE evaluation pipeline */
    void eval(StatesT::GRef dStates, const StatesT::GRef& states,
        const EpochsT::GRef& epochs,
        const BoolArrayT::GRef::HandleT& terminated,
        const cudaStream_t& stream = 0) const
    {

        /* Extract the adequate reference */
        ensureRef(terminated.isHostRef());

        if (terminated.isHostRef()) {
            ode::host::eval<Self>(ref_, dStates, states, epochs, terminated);
        } else {
            /* Old monolithic kernel logic */
            // const idx_t nBlocks
            //     = (dStates.size() + PARAHPOP_BLOCKSIZE - 1) / PARAHPOP_BLOCKSIZE;
            // ode::device::eval<Self><<<nBlocks, PARAHPOP_BLOCKSIZE, 0, stream>>>(
            //     ref_, dStates, states, epochs, terminated);

            /* Forward the graph construction downstream to the model */
            holder_->eval(dStates, states, epochs, terminated, stream);

            /* and finalize */
            const idx_t nBlocks
                = (dStates.size() + PARAHPOP_BLOCKSIZE - 1) / PARAHPOP_BLOCKSIZE;
            ode::device::finalize<Self>
                <<<nBlocks, PARAHPOP_BLOCKSIZE, 0, stream>>>(
                    dStates, states, terminated);
        }
    }

    /** @brief Call the finalize kernel */
    cudaGraph_t finalize(StatesT::GRef dStates, const StatesT::GRef& states,
        const BoolArrayT::GRef::HandleT& terminated,
        const cudaStream_t& stream = 0) const
    {

        PARAHPOP_ASSERT(!terminated.isHostRef(),
            "Finalize should not be called on the host - this indicates a "
            "problem in the ODE evaluation pipeline");

        /* Extract the adequate reference */
        ensureRef(terminated.isHostRef());

        parm::util::graph::StreamCapturer capturer(stream);
        capturer.begin();
        const idx_t nBlocks
            = (dStates.size() + PARAHPOP_BLOCKSIZE - 1) / PARAHPOP_BLOCKSIZE;
        ode::device::finalize<Self><<<nBlocks, PARAHPOP_BLOCKSIZE, 0, stream>>>(
            dStates, states, terminated);
        return capturer.end();
    }


    /** @brief Graph-based dispatcher.
     *
     *  The `stage` template parameter carries the RK substage index from
     *  parm's `RecurseEval<stage,…>` down to `Dimensional::eval(graph,
     *  …)`, which uses it to short-circuit the per-body native fill +
     *  `coiResolveKernel` / `cacheRotation` launches at stage 0 (the
     *  step-0 epoch — pinned cache holds the values already). */
    template<idx_t stage>
    void eval(parm::util::graph::Graph& graph, StatesT::GRef dStates,
        const StatesT::GRef& states, const EpochsT::GRef& epochs,
        const BoolArrayT::GRef::HandleT& terminated,
        const cudaStream_t& stream = 0) const
    {
        /* Extract the adequate reference */
        ensureRef(terminated.isHostRef());

        /* last node */
        idx_t lastNode = graph.lastNode();

        /* Forward the graph construction downstream to the model */
        holder_->template eval<stage>(
            graph, dStates, states, epochs, terminated, stream);
        idx_t odeNode = graph.lastNode();

        /* and finalize (works here because we are only placing dpos where it
         * belongs )*/
        // graph.addNode(
        //     finalize(dStates, states, terminated, stream), { lastNode });
        /* Note: for the cartesian ODE we only need to run a memcpy */
        parm::util::graph::StreamCapturer capturer(stream);
        capturer.begin();
        dStates.template subset<0, 3>().fetch(
            states.template subset<3, 3>(), stream);
        graph.addNode(capturer.end(), { lastNode });
        lastNode = graph.lastNode();

        /* add empty node to join */
        graph.addEmptyNode({ lastNode, odeNode });
    }

    /** @brief Get host reference */
    GRef hostRef() const { return GRef::make(holder_.hostRef()); }

    /** @brief Get device reference */
    GRef deviceRef() const { return GRef::make(holder_.deviceRef()); }

private:
    /** @brief Ensure that the reference is extracted */
    void ensureRef(const bool& useHostRef = true) const
    {
        if (!extractedRef_) {
            ref_          = useHostRef ? hostRef() : deviceRef();
            extractedRef_ = true;
        }
    }

    mutable GRef ref_;
    mutable bool extractedRef_ = false;
    HolderT holder_;
};

} // namespace model
} // namespace paraHPOP