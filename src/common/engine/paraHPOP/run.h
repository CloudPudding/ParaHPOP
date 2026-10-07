#pragma once
#include "paraHPOP/propagators/host/Propagator.h"
#include "paraHPOP/propagators/device/Propagator.h"
namespace paraHPOP {
inline interface::samples::Collection run(const interface::config::Model& cfg,
    const interface::samples::Collection& samples, Real duration, Real step,
    idx_t maxSteps, idx_t threads, bool gpu) {
    model::CartesianDim model(cfg);
    model.loadSamples(model::CartesianDim::SamplesT(samples));
    if(gpu) { propagators::device::Propagator runner(model); runner.run(step,duration,maxSteps); }
    else { propagators::host::Propagator runner(model); runner.run(step,duration,maxSteps,threads); }
    return model.interfaceSamples();
}
}
