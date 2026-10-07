#pragma once
#include "paraHPOP/model/math/detail/RefModel.h"
#include "interface/config/Model.h"
#include "interface/samples/Samples.h"
namespace paraHPOP::model::math::detail {
template<typename SamplesT_, typename PhysicsT_>
class Model {
    using Self = Model;
public:
    static constexpr idx_t ORDER = PhysicsT_::ORDER;
    static constexpr idx_t PhysicsDim = PhysicsT_::Dim;
    using SamplesT = SamplesT_;
    using StatesT = typename SamplesT::StatesT;
    using PhysicsT = PhysicsT_;
    using EpochsT = feta::scalar::Array<typename StatesT::ComponentT>;
    using BoolArrayT = feta::scalar::Array<bool>;
    template<bool W, bool V = false> using Ref = RefModel<SamplesT,PhysicsT,W,V>;
    using GRef = Ref<false>;
    using WRef = Ref<true>;
    using VolatileRef = Ref<true,true>;
    template<idx_t Order> using ODE = model::ODE<Self,Order>;
    explicit Model(const interface::config::Model& cfg) : physics_(cfg) {}
    void loadSamples(SamplesT&& s) { samples_ = std::move(s); }
    SamplesT& samples() { return samples_; }
    const SamplesT& samples() const { return samples_; }
    PhysicsT& physics() { return physics_; }
    const PhysicsT& physics() const { return physics_; }
    idx_t numRotations() const {
        namespace src = interface::config::model::accelerations;
        return physics_.accs().template anyActive<src::SHAPE>() ||
               physics_.accs().template anyActive<src::OBLATENESS>() ||
               physics_.accs().template anyActive<src::ATMOSPHERE>() ||
               physics_.env().earthCorrections().tidesActive() ? 1 : 0;
    }
    void upload(cudaStream_t s = 0) { samples_.upload(s); physics_.upload(s); }
    void download(cudaStream_t s = 0) { samples_.download(s); }
    GRef hostRef() const { return GRef::make(samples_.hostRef(),physics_.hostRef()); }
    GRef deviceRef() const { return GRef::make(samples_.deviceRef(),physics_.deviceRef()); }
    template<idx_t Order> ODE<Order> ode() const { return ODE<Order>(*this); }
    interface::samples::Collection interfaceSamples() const { return samples_.interface(); }
    void eval(StatesT::GRef& dStates, const StatesT::GRef& states,
        const EpochsT::GRef& epochs,
        const BoolArrayT::GRef::HandleT& terminated,
        const cudaStream_t& stream = 0) const
    {
        /* Forward the evaluation downstream to the reduced physics part */
        physics_.reduced().eval(dStates, states, epochs, terminated,
            samples_.metadata().deviceRef(), stream);
    }
    template<idx_t stage>
    void eval(parm::util::graph::Graph& graph, StatesT::GRef& dStates,
        const StatesT::GRef& states, const EpochsT::GRef& epochs,
        const BoolArrayT::GRef::HandleT& terminated,
        const cudaStream_t& stream = 0) const
    {
        /* Forward the evaluation downstream to the reduced physics part */
        physics_.reduced().template eval<stage>(graph, dStates, states,
            epochs, terminated, samples_.metadata().deviceRef(), stream);
    }

private:
    SamplesT samples_ = SamplesT(0);
    PhysicsT physics_;
};
}
