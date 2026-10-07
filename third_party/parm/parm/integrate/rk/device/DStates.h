#pragma once

#ifndef PARM_CPU_ONLY

#include "parm/integrate/detail/DStates.h"
#include "parm/integrate/rk/device/ode.h"

namespace parm {
namespace integrate {
namespace rk {
namespace device {

/** @brief Device Dstates - include the evaluation of the ODE */
template<typename StatesT_, typename Tableau>
class DStates
    : public parm::integrate::detail::DStates<StatesT_, Tableau::nStages()> {
    using ParentT =
        typename parm::integrate::detail::DStates<StatesT_, Tableau::nStages()>;

    /* This type */
    using Self = DStates;

public:
    /** @brief Expose Stream type */
    using StreamT = typename ParentT::StreamT;
    /** @brief Expose states type */
    using StatesT = typename ParentT::StatesT;
    /** @brief Expose single state type */
    using StateT = typename ParentT::StateT;

    /** @brief Expose parent reference types */
    using GRef        = typename ParentT::GRef;
    using WRef        = typename ParentT::WRef;
    using VolatileRef = typename ParentT::VolatileRef;

    /** @brief Inherit constructors */
    using ParentT::ParentT;

    /** @brief Inherit assignment operators */
    using ParentT::operator=;

    /** @brief Return a const reference to the block size associated to
     * the computation of these dstates */
    const idx_t& blockSize() const { return blockSize_; }

    /** @brief Return a const reference to the CUDA stream associated to the
     * computation of these dstates */
    const StreamT& stream() const { return stream_; }

    /** @brief Set the cuda stream */
    void stream(const StreamT& stream) { stream_ = stream; }

    /** @brief Set the block size (and internally update the number of blocks)
     */
    void blockSize(const idx_t& blockSize)
    {
        PARM_ASSERT(blockSize != 0, "Block size must be different than 0!");
        blockSize_ = blockSize;
        nBlocks_   = (this->size() + blockSize_ - 1) / blockSize_;
    }

    /**
     * @brief Ensure the the integrator's internal data structures are
     * initialised on the GPU. Must be called before time-stepping.
     */
    void ensureInitDevice() { ParentT::ensureInitDevice(stream_); }

    /** @brief Expose the cuda graph */
    const cudaGraph_t& graph() const { return graph_; }

    /** @brief Setup the graph computation */
    template<typename ODE>
    inline cudaGraph_t graph(const ODE& myode,
        const typename StatesT::GRef& states,
        const base::Simulation::GRef& base)
    {
        PARM_ASSERT(this->initialised_, "DStates must be initialised!");
        PARM_ASSERT(blockSize_ != 0, "Block size must be set!");

        /* Allocate shared memory size */
        GRef selfref = this->deviceRef();

        return ode::Eval<Self, Tableau, ODE>::eval(
            myode, states, selfref, base, stream_);
    }

    /** @brief Capture into the given parm graph */
    template<typename ODE>
    inline void graph(util::graph::Graph& graph, const ODE& myode,
        const typename StatesT::GRef& states,
        const base::Simulation::GRef& base)
    {
        PARM_ASSERT(this->initialised_, "DStates must be initialised!");
        PARM_ASSERT(blockSize_ != 0, "Block size must be set!");

        /* Allocate shared memory size */
        GRef selfref = this->deviceRef();

        ode::Eval<Self, Tableau, ODE>::eval(
            graph, myode, states, selfref, base, stream_);
    }

    /** @brief Compute the DStates by evaluating the given ODE */
    template<typename ODE>
    inline void compute(const ODE& myode, const typename StatesT::GRef& states,
        const base::Simulation::GRef& base)
    {
        /* Setup this graph */
        if (!graphCreated_)
            graph_ = graph<ODE>(myode, states, base);
        /* Launch the execution */
        launch_();
    }

protected:
    /** @brief Launch the graph */
    inline void launch_()
    {
        if (!graphInstantiated_) {
            cudaGraphInstantiate(&instance_, graph_, NULL, NULL, 0);
            graphInstantiated_ = true;
        }
        PARM_KERNEL_PRE();
        cudaGraphLaunch(instance_, stream_);
        PARM_KERNEL_POST();
    }

    /* Running features */
    cudaStream_t stream_ = 0;
    idx_t blockSize_     = 0;
    idx_t nBlocks_;
    bool graphCreated_      = false;
    bool graphInstantiated_ = false;
    cudaGraph_t graph_;
    cudaGraphExec_t instance_;
};

} // namespace device
} // namespace rk
} // namespace integrate
} // namespace parm

#endif
