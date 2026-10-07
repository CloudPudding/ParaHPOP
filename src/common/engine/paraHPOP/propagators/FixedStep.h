#pragma once
#include "paraHPOP/model.h"
#include "interface/util/TimeConversion.h"
#include <limits>
namespace paraHPOP::propagators {
using Base = parm::integrate::base::Simulation;
inline idx_t prepare(model::CartesianDim& model, Real step, Real duration) {
    auto base = model.samples().metadata().base().hostRef();
    idx_t launches = 0;
    for (idx_t j=0; j<model.samples().size(); ++j) {
        auto i = SampleIndex::make(j);
        Real start = base.currentEpochs()[i];
        using Conv = interface::util::TimeConversion;
        base.startEpochs()[i] = Conv::mjdToSpice(start);
        base.currentEpochs()[i] = base.startEpochs()[i];
        base.endEpochs()[i] = Conv::mjdToSpice(start + duration/86400.0);
        model.physics().env().assertEarthCorrectionsEpochCoverageEt(
            base.startEpochs()[i], base.endEpochs()[i]);
        base.resetEndOfSimulation(i);
        base.setInitialStep(i,step);
        Real span = base.endEpochs()[i] - base.currentEpochs()[i];
        if (!(span > 0) || !std::isfinite(span) ||
            std::ceil(span/step) >= static_cast<Real>(std::numeric_limits<idx_t>::max()))
            throw std::runtime_error("Duration/step cannot be represented by the integration time/index types.");
        base.nextDts()[i] = std::min(step,span);
        launches = std::max(launches,static_cast<idx_t>(std::ceil(span/step))+1);
    }
    return launches;
}
DEVICEHOST() inline void finishStep(Base::GRef base, SampleIndex i, Real step) {
    base.currentEpochs()[i] += base.nextDts()[i];
    Real remaining = base.endEpochs()[i] - base.currentEpochs()[i];
    bool finished = remaining <= 0;
    base.terminated()[i] = finished;
    base.endOfSimulation()[i] = finished;
    base.nextDts()[i] = finished ? 0 : feta::math::min(step,remaining);
}
inline void postProcess(model::CartesianDim& model) {
    auto b=model.samples().metadata().base().hostRef();
    using Conv=interface::util::TimeConversion;
    for(idx_t j=0;j<model.samples().size();++j) {
        auto i=SampleIndex::make(j);
        b.currentEpochs()[i]=Conv::spiceToMjd(b.currentEpochs()[i]);
        b.startEpochs()[i]=Conv::spiceToMjd(b.startEpochs()[i]);
        b.endEpochs()[i]=Conv::spiceToMjd(b.endEpochs()[i]);
    }
}
}
