#pragma once
#include "paraHPOP/model/samples/metadata/Orbit.h"
#include "paraHPOP/model/samples/states/cartesian/States.h"
#include "interface/samples/Samples.h"
namespace paraHPOP::model::samples::cartesian {
template<bool W,bool V=false> struct RefSamples {
    static constexpr bool Volatile=W&&V;
    using MetadataT=metadata::Orbit::Ref<W,V>;
    using StatesT=states::cartesian::States::Ref<W,V>;
    MetadataT metadata_;
    StatesT states_;
    DEVICEHOST() static RefSamples make(MetadataT m,StatesT s){return {m,s};}
    DEVICEHOST() const MetadataT& metadata() const{return metadata_;}
    DEVICEHOST() const StatesT& states() const{return states_;}
    DEVICEHOST() idx_t size() const{return states_.size();}
    DEVICEHOST() RefSamples clone() const{return *this;}
};
class Samples {
public:
    using StatesT=states::cartesian::States;
    using StateT=StatesT::StateT;
    using MetadataT=metadata::Orbit;
    template<bool W,bool V=false> using Ref=RefSamples<W,V>;
    using GRef=Ref<false>; using WRef=Ref<true>; using VolatileRef=Ref<true,true>;
    explicit Samples(idx_t n):metadata_(n),states_(n){}
    explicit Samples(const interface::samples::Collection& input):metadata_(input),states_(input.states()){}
    Samples(Samples&&)=default;
    Samples& operator=(Samples&&)=default;
    idx_t size() const{return states_.size();}
    MetadataT& metadata(){return metadata_;}
    const MetadataT& metadata() const{return metadata_;}
    StatesT& states(){return states_;}
    const StatesT& states() const{return states_;}
    GRef hostRef() const{return GRef::make(metadata_.hostRef(),states_.hostRef());}
    GRef deviceRef() const{return GRef::make(metadata_.deviceRef(),states_.deviceRef());}
    void upload(cudaStream_t stream=0){metadata_.upload(stream);states_.upload(stream);}
    void download(cudaStream_t stream=0){metadata_.download(stream);states_.download(stream);}
    interface::samples::Collection interface() const {
        interface::samples::Collection out(size());
        auto m=metadata_.hostRef();
        states_.hostRef().toBuffer(out.states());
        m.base().ids().toBuffer(out.ids());
        m.base().currentEpochs().toBuffer(out.epochs());
        m.base().endOfSimulation().toBuffer(out.status().endOfSimulation());
        m.own().cois().toBuffer(out.centres());
        m.own().mass().toBuffer(out.mass());
        m.own().area().toBuffer(out.area());
        m.own().cd().toBuffer(out.cd());
        m.own().cr().toBuffer(out.cr());
        return out;
    }
private:
    MetadataT metadata_;
    StatesT states_;
};
}
