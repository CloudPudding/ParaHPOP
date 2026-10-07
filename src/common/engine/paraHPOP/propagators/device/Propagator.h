#pragma once
#include "paraHPOP/propagators/FixedStep.h"
#include "paraHPOP/IntegrationTableau.h"
#include <parm/integrate/rk/device/DStates.h>
#include <parm/util/graph/Stream.h>
namespace paraHPOP::propagators::device {
using States = model::CartesianDim::StatesT;
using DStates = parm::integrate::rk::device::DStates<States,IntegrationTableau>;
// All threads reach the barrier, including tail lanes of a partial block.
static __global__ void advance(DStates::GRef derivatives, States::GRef states, Base::GRef base, Real step) {
    __shared__ Real scratch[256*6];
    __shared__ feta::vector::Array<Real,6>::WRef::HandleT accum;
    if(threadIdx.x==0) { accum.ptr=scratch; accum.dOffset=blockDim.x; }
    __syncthreads();
    auto i=SampleIndex::make(threadIdx.x,blockIdx.x,blockDim.x);
    if(i.global()>=states.size() || base.terminated()[i]) return;
    accum[i]=Vec6R(0);
    [&]<idx_t... s>(std::integer_sequence<idx_t,s...>) {
        ((accum[i] += IntegrationTableau::template b<s>()*derivatives.template stage<s>()[i]),...);
    }(std::make_integer_sequence<idx_t,IntegrationTableau::nStages()>{});
    Vec6R sum=accum[i];
    states[i]=states[i]+base.nextDts()[i]*sum;
    finishStep(base,i,step);
}
class Propagator {
    using BaseT=Base;
    model::CartesianDim& model_;
    parm::util::graph::Stream stream_;
    cudaStream_t stream() const { return stream_.cuda(); }
public:
    explicit Propagator(model::CartesianDim& model) : model_(model) {}
    ~Propagator() { model_.physics().env().releaseEphCache(model::environment::Environment::CacheTarget::DEVICE); }
    void run(Real step, Real duration, idx_t maxSteps) {
        // Independent corrections use geocentric states, as required by
        // the public driver. Reject off-centre internal calls explicitly.
        if (model_.physics().env().earthCorrections().any()) {
            auto cois = model_.samples().metadata().hostRef().own().cois();
            for (idx_t j = 0; j < model_.samples().size(); ++j)
                PARAHPOP_ASSERT(cois[SampleIndex::make(j)] == NaifId{399},
                    "GPU Earth corrections require geocentric target states.");
        }
        auto count=prepare(model_,step,duration);
        auto& env=model_.physics().env();
        env.allocateEphCache(model_.samples().size(),model_.numRotations(),model::environment::Environment::CacheTarget::DEVICE,{brie::NaifId(399)});
        model_.upload(stream());
        bootstrapPinnedCache_();
        DStates derivatives(model_.samples().size());
        derivatives.stream(stream()); derivatives.blockSize(PARAHPOP_BLOCKSIZE);
        derivatives.upload(stream());
        stream_.synchronize();
        auto states=model_.samples().states().deviceRef();
        auto base=model_.samples().metadata().base().deviceRef();
        auto ode=model_.ode<1>();
        parm::util::graph::Graph graph; graph.stream(stream());
        derivatives.graph(graph,ode,states,base);
        parm::util::graph::StreamCapturer capturer(stream());
        capturer.begin();
        advance<<<(states.size()+255)/256,256,0,stream()>>>(derivatives.deviceRef(),states,base,step);
        graph.addNode(capturer.end(),{},256);
        setupPinCacheCopy_(graph,base);
        auto launcher=graph.launcher();
        for(idx_t n=0;n<std::min(count,maxSteps);++n) launcher.launch();
        model_.download(stream());
        stream_.synchronize();
        postProcess(model_);
    }
private:
    template<typename LaunchFn>
    inline idx_t captureKernelNode_(parm::util::graph::Graph& graph,
        parm::util::graph::StreamCapturer& capturer, LaunchFn&& launch,
        std::initializer_list<idx_t> deps, idx_t idealBlockSize)
    {
        capturer.begin();
        launch();
        graph.addNode(capturer.end(), deps, idealBlockSize);
        return graph.lastNode();
    }
    template<typename AccsT>
    static bool needsBodyFrame_(const AccsT& accs, idx_t i)
    {
        namespace src = interface::config::model::accelerations;
        return accs.template active<src::SHAPE>(i)
            || accs.template active<src::OBLATENESS>(i)
            || accs.template active<src::ATMOSPHERE>(i);
    }
    inline void bootstrapPinnedCache_()
    {
        namespace src      = interface::config::model::accelerations;
        namespace envkern  = paraHPOP::model::environment::kernel;

        auto& env  = this->model_.physics().env();
        auto& accs = this->model_.physics().reduced().accs();
        auto bodies = env.bodies().hostRef();
        auto eref   = env.deviceRef();

        auto base       = this->model_.samples().metadata().base().deviceRef();
        auto epochs     = base.currentEpochs();
        auto terminated = base.terminated();
        auto cois       = this->model_.samples().metadata().deviceRef()
                              .own().cois();

        idx_t nBlocks, blockSize;
        paraHPOP::model::physics::reduced::computeBlocks(
            this->model_.samples().size(), nBlocks, blockSize,
            envkern::EnvKernelLaunch::maxBlockSize);

        for (idx_t b = 0; b < env.ephNativePinned().nBodyUnits(); b++) {
            env.ephNativePinned().fillBody(b, epochs, this->stream());
        }

        for (idx_t i = 0; i < bodies.size(); i++) {
            if (!accs.anyActive(i) && !env.needsTidePosition(i))
                continue;
            if (accs.template active<src::ATMOSPHERE>(i)) {
                envkern::coiResolveStateKernel<<<nBlocks, blockSize, 0,
                    this->stream()>>>(env.ephPinned(i), env.velPinned(i),
                    env.ephNativePinned().ref(), bodies[i], cois, epochs,
                    terminated);
                
                envkern::omegaResolveKernel<<<nBlocks, blockSize, 0,
                    this->stream()>>>(env.omegaPinned(i),
                    env.deviceRef().orientations(), env.orientationTarget(i),
                    epochs,
                    terminated);
            } else {
                envkern::coiResolveKernel<<<nBlocks, blockSize, 0,
                    this->stream()>>>(env.ephPinned(i),
                    env.ephNativePinned().ref(), bodies[i], cois, epochs,
                    terminated);
            }
        }

        for (idx_t i = 0; i < bodies.size(); i++) {
            if (!needsBodyFrame_(accs, i)
                && !(bodies[i] == NaifId{399} && env.earthCorrections().tidesActive()))
                continue;
            envkern::cacheRotation<<<nBlocks, blockSize, 0,
                this->stream()>>>(env.quatPinned(i), epochs, terminated,
                eref.orientations(), env.orientationTarget(i));
        }

    }
    inline void setupPinCacheCopy_(parm::util::graph::Graph& graph,
        const typename BaseT::GRef& base)
    {
        namespace src      = interface::config::model::accelerations;
        namespace envkern  = paraHPOP::model::environment::kernel;

        auto& env  = this->model_.physics().env();
        auto& accs = this->model_.physics().reduced().accs();
        auto bodies = env.bodies().hostRef();
        auto eref   = env.deviceRef();

        auto epochs     = base.currentEpochs();
        auto terminated = base.terminated();

        auto cois       = this->model_.samples().metadata().deviceRef()
                              .own().cois();

        idx_t nBlocks, blockSize;
        paraHPOP::model::physics::reduced::computeBlocks(
            this->model_.samples().size(), nBlocks, blockSize,
            envkern::EnvKernelLaunch::maxBlockSize);

        const idx_t anchor = graph.lastNode();
        std::vector<idx_t> tier1Terminals;
        std::vector<idx_t> terminals;
        const idx_t nNative = env.ephNativeVariable().nBodyUnits();
        tier1Terminals.reserve(nNative);
        terminals.reserve(bodies.size() * 2);

        parm::util::graph::StreamCapturer capturer(this->stream());

        const idx_t idealBlockSize
            = static_cast<idx_t>(envkern::EnvKernelLaunch::maxBlockSize);

        for (idx_t b = 0; b < nNative; b++) {
            tier1Terminals.push_back(captureKernelNode_(graph, capturer, [&] {
                envkern::copyNativeCacheKernel<6><<<nBlocks, blockSize, 0,
                    this->stream()>>>(env.ephNativePinned().slotRef(b),
                    env.ephNativeVariable().slotRef(b), terminated);
            }, { anchor }, idealBlockSize));
        }

        if (tier1Terminals.empty())
            tier1Terminals.push_back(anchor);
        graph.addEmptyNode(tier1Terminals);
        const idx_t tier1Barrier = graph.lastNode();

        for (idx_t i = 0; i < bodies.size(); i++) {
            if (!accs.anyActive(i) && !env.needsTidePosition(i))
                continue;

            terminals.push_back(captureKernelNode_(graph, capturer, [&] {
                if (accs.template active<src::ATMOSPHERE>(i)) {
                    envkern::coiResolveStateKernel<<<nBlocks, blockSize, 0,
                        this->stream()>>>(env.ephPinned(i), env.velPinned(i),
                        env.ephNativePinned().ref(), bodies[i], cois, epochs,
                        terminated);
                } else {
                    envkern::coiResolveKernel<<<nBlocks, blockSize, 0,
                        this->stream()>>>(env.ephPinned(i),
                        env.ephNativePinned().ref(), bodies[i], cois, epochs,
                        terminated);
                }
            }, { tier1Barrier }, idealBlockSize));

            if (accs.template active<src::ATMOSPHERE>(i)) {
                terminals.push_back(captureKernelNode_(graph, capturer, [&] {
                    envkern::omegaResolveKernel<<<nBlocks, blockSize, 0,
                        this->stream()>>>(env.omegaPinned(i),
                        env.deviceRef().orientations(),
                        env.orientationTarget(i), epochs, terminated);
                }, { tier1Barrier }, idealBlockSize));
            }
        }

        for (idx_t i = 0; i < bodies.size(); i++) {
            if (!needsBodyFrame_(accs, i)
                && !(bodies[i] == NaifId{399} && env.earthCorrections().tidesActive()))
                continue;

            terminals.push_back(captureKernelNode_(graph, capturer, [&] {
                envkern::cacheRotation<<<nBlocks, blockSize, 0,
                    this->stream()>>>(env.quatPinned(i), epochs, terminated,
                    eref.orientations(), env.orientationTarget(i));
            }, { tier1Barrier }, idealBlockSize));
        }

        if (terminals.empty())
            terminals.push_back(tier1Barrier);
        graph.addEmptyNode(terminals);
    }
};
}
