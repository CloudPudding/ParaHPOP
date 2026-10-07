#pragma once
#include "paraHPOP/propagators/FixedStep.h"
#include "paraHPOP/IntegrationTableau.h"
#include "paraHPOP/model/accelerations/host/kernels.h"
#include <parm/integrate/rk/host/Step.h>
namespace paraHPOP::propagators::host {
class Propagator {
    using BaseT = Base;
    model::CartesianDim& model_;
public:
    explicit Propagator(model::CartesianDim& model) : model_(model) {}
    ~Propagator() { model_.physics().env().releaseEphCache(model::environment::Environment::CacheTarget::HOST); }
    void run(Real step, Real duration, idx_t maxSteps, idx_t threads) {
        auto count = prepare(model_,step,duration);
        auto& env=model_.physics().env();
        env.allocateEphCache(model_.samples().size(),model_.numRotations(),model::environment::Environment::CacheTarget::HOST);
        bootstrapPinnedCache_();
        auto base=model_.samples().metadata().base().hostRef();
        auto states=model_.samples().states().hostRef();
        auto ode=model_.ode<1>(); auto rhs=ode.hostRef();
        for(idx_t n=0;n<std::min(count,maxSteps);++n) {
            #pragma omp parallel for num_threads(threads) schedule(static)
            for(idx_t j=0;j<states.size();++j) {
                auto i=SampleIndex::make(j);
                if(base.terminated()[i]) continue;
                Vec6R state=states[i];
                parm::integrate::rk::host::stepFixed<IntegrationTableau>(i,state,base.nextDts()[i],base.currentEpochs()[i],rhs);
                states[i]=state;
                finishStep(base,i,step);
            }
            pinCacheCopy_(base);
        }
        postProcess(model_);
    }
private:
    inline void bootstrapPinnedCache_()
    {
        namespace src   = interface::config::model::accelerations;
        auto& env       = this->model_.physics().env();
        auto& accs      = this->model_.physics().reduced().accs();
        auto bodies     = env.bodies().hostRef();
        auto eref       = env.hostRef();
        auto base       = this->model_.samples().metadata().base().hostRef();
        auto epochs     = base.currentEpochs();
        auto terminated = base.terminated();

        env.ephNativePinnedHost().fill(epochs);

        namespace hk = paraHPOP::model::accelerations::host::kernel;
        for (idx_t i = 0; i < bodies.size(); i++) {
            if (!(accs.template active<src::SHAPE>(i)
                    || accs.template active<src::OBLATENESS>(i)))
                continue;
            hk::cacheRotation(env.quatPinnedHost(i), epochs, terminated,
                eref.orientations(), env.orientationTarget(i));
        }

    }
    inline void pinCacheCopy_(BaseT::GRef& base)
    {
        namespace src              = interface::config::model::accelerations;
        auto& env                  = this->model_.physics().env();
        auto& accs                 = this->model_.physics().reduced().accs();
        auto bodies                = env.bodies().hostRef();
        auto terminated            = base.terminated();

        const idx_t N              = this->model_.samples().size();
        const idx_t bytesPerSample = model_.physics().reduced().hostBytesPerSample();

        const idx_t nNative = env.ephNativeVariableHost().nBodyUnits();
        for (idx_t b = 0; b < nNative; b++) {
            auto pinnedSlot   = env.ephNativePinnedHost().slotRef(b);
            auto variableSlot = env.ephNativeVariableHost().slotRef(b);
            feta::cpu::packetBatchedFor<Real>(
                idx_t{ 0 }, N,
                [&](const auto& pi) {
                    constexpr idx_t W
                        = std::remove_cvref_t<decltype(pi)>::width;
                    auto inactive
                        = parm::util::Host::loadMask<Real, W>(terminated, pi);
                    auto active = parm::util::Host::applyTail<W>(~inactive, pi);
                    if (!active.anyTrue())
                        return;
                    parm::util::Host::dispatchMasked(
                        pi, active, [&](const SampleIndex& i) {
                            Vec6R val     = variableSlot[i];
                            pinnedSlot[i] = val;
                        });
                },
                bytesPerSample,
                false);
        }

        for (idx_t i = 0; i < bodies.size(); i++) {
            if (!(accs.template active<src::SHAPE>(i)
                    || accs.template active<src::OBLATENESS>(i)))
                continue;
            auto pinnedSlot   = env.quatPinnedHost(i);
            auto variableSlot = env.quatVariableHost(i);
            feta::cpu::packetBatchedFor<Real>(
                idx_t{ 0 }, N,
                [&](const auto& pi) {
                    constexpr idx_t W
                        = std::remove_cvref_t<decltype(pi)>::width;
                    auto inactive
                        = parm::util::Host::loadMask<Real, W>(terminated, pi);
                    auto active = parm::util::Host::applyTail<W>(~inactive, pi);
                    if (!active.anyTrue())
                        return;
                    parm::util::Host::dispatchMasked(
                        pi, active, [&](const SampleIndex& j) {
                            feta::vector::Item<Real, 4> val = variableSlot[j];
                            pinnedSlot[j]                   = val;
                        });
                },
                bytesPerSample,
                false);
        }

    }
};
}
