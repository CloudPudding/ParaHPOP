#pragma once
#include "interface/typedefs.h"
namespace interface::samples {
class Status {
    feta::buffer::scalar::Array<bool> finished_;
public:
    explicit Status(idx_t n=0):finished_(n){}
    auto& endOfSimulation(){return finished_;}
    const auto& endOfSimulation() const{return finished_;}
};
class Collection {
    using RealBuffer=feta::buffer::scalar::Array<Real>;
    using IdBuffer=feta::buffer::scalar::Array<NaifId>;
    feta::buffer::vector::Array<Real,6> states_;
    RealBuffer epochs_,mass_,area_,cd_,cr_;
    IdBuffer ids_,centres_;
    Status status_;
public:
    explicit Collection(idx_t n=0):states_(n),epochs_(n),mass_(n),area_(n),cd_(n),cr_(n),ids_(n),centres_(n),status_(n){}
    idx_t size() const{return states_.size();}
#define PARAHPOP_FIELD(name) auto& name(){return name##_;} const auto& name() const{return name##_;}
    PARAHPOP_FIELD(states)
    PARAHPOP_FIELD(epochs)
    PARAHPOP_FIELD(mass)
    PARAHPOP_FIELD(area)
    PARAHPOP_FIELD(cd)
    PARAHPOP_FIELD(cr)
    PARAHPOP_FIELD(ids)
    PARAHPOP_FIELD(centres)
    PARAHPOP_FIELD(status)
#undef PARAHPOP_FIELD
};
}
