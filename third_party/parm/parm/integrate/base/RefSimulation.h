#pragma once

#include "parm/typedefs.h"
#include "parm/util.h"
#include <omp.h>

namespace parm {
namespace integrate {
namespace base {

enum SimulationBoolMembers { TERMINATED, ENDOFSIM, SIMBOOLSIZE };

enum SimulationIntMembers { SAMPLEID, SIMINTSIZE };

enum SimulationRealMembers { DTNEXT, CURRENTEPOCH, STARTEPOCH, ENDEPOCH, SIMREALSIZE };

template<bool work, bool MaybeVolatile = false>
class RefSimulation;

template<bool work, bool MaybeVolatile = false>
class View;
template<bool work, bool MaybeVolatile = false>
class ConstView;

template<bool work, bool MaybeVolatile>
class ConstView {
    
    using RefT = RefSimulation<work, MaybeVolatile>;
    using Self = ConstView<work, MaybeVolatile>;

public:
    
    DEVICEHOST() bool terminated() const;

    DEVICEHOST() bool endOfSimulation() const;

    DEVICEHOST() mInt_t id() const;

    DEVICEHOST() Real epoch() const;

    const SampleIndex idx_;
    const RefT& ref_;
};

template<bool work, bool MaybeVolatile>
class View {
    
    using RefT = RefSimulation<work, MaybeVolatile>;

public:
    
    DEVICEHOST() bool& terminated();
    DEVICEHOST() bool terminated() const;

    DEVICEHOST() bool& endOfSimulation();
    DEVICEHOST() bool endOfSimulation() const;

    DEVICEHOST() mInt_t& id();
    DEVICEHOST() mInt_t id() const;

    DEVICEHOST() Real& epoch();
    DEVICEHOST() Real epoch() const;

    const SampleIndex idx_;
    RefT& ref_;
};

template<bool work, bool MaybeVolatile>
class RefSimulation
    : public util::MultiContainer<SIMINTSIZE, SIMBOOLSIZE,
          SIMREALSIZE>::template Ref<work, MaybeVolatile> {
public:
    
    static constexpr bool Volatile = MaybeVolatile && work;

private:
    
    using ParentT = typename util::MultiContainer<SIMINTSIZE, SIMBOOLSIZE,
        SIMREALSIZE>::template Ref<work, MaybeVolatile>;
    
    using typename ParentT::RefVecBoolArrayT;
    using typename ParentT::RefVecIntArrayT;
    using typename ParentT::RefVecRealArrayT;
    
    using RefRealArrayT =
        typename feta::scalar::Array<Real>::template Ref<work, MaybeVolatile>;
    using RefIntArrayT  =
        typename feta::scalar::Array<mInt_t>::template Ref<work, MaybeVolatile>;
    using RefBoolArrayT =
        typename feta::scalar::Array<bool>::template Ref<work, MaybeVolatile>;
    
    using RealHandleT = typename RefRealArrayT::HandleT;
    using IntHandleT  = typename RefIntArrayT::HandleT;
    using BoolHandleT = typename RefBoolArrayT::HandleT;

    friend class RefSimulation<!work, MaybeVolatile>;

public:
    
    using ViewT      = View<work, MaybeVolatile>;
    using ConstViewT = ConstView<work, MaybeVolatile>;

    DEVICEHOST() static RefSimulation make(const ParentT& other)
    {
        return RefSimulation{ other };
    }

    template<SimulationBoolMembers item>
    DEVICEHOST()
    BoolHandleT boolComponent() const
    {
        return this->boolMembers_.template component<item>();
    }

    template<SimulationIntMembers item>
    DEVICEHOST()
    IntHandleT intComponent() const
    {
        return this->intMembers_.template component<item>();
    }

    template<SimulationRealMembers item>
    DEVICEHOST()
    RealHandleT realComponent() const
    {
        return this->realMembers_.template component<item>();
    }

    DEVICEHOST()
    BoolHandleT terminated() const { return this->boolComponent<TERMINATED>(); }

    DEVICEHOST()
    BoolHandleT endOfSimulation() const
    {
        return this->boolComponent<ENDOFSIM>();
    }

    DEVICEHOST() IntHandleT ids() const
    {
        return this->intComponent<SAMPLEID>();
    }

    DEVICEHOST()
    RealHandleT nextDts() const { return this->realComponent<DTNEXT>(); }

    DEVICEHOST()
    RealHandleT currentEpochs() const
    {
        return this->realComponent<CURRENTEPOCH>();
    }

    DEVICEHOST()
    RealHandleT startEpochs() const
    {
        return this->realComponent<STARTEPOCH>();
    }

    DEVICEHOST()
    RealHandleT endEpochs() const { return this->realComponent<ENDEPOCH>(); }

    DEVICEHOST()
    void resetSampleID(const SampleIndex& idx)
    {
        this->ids()[idx] = idx.global();
    }

    DEVICEHOST()
    void resetTerminal(const SampleIndex& idx)
    {
        this->boolMembers_.template get<TERMINATED>(idx) = false;
    }

    DEVICEHOST()
    void resetEndOfSimulation(const SampleIndex& idx)
    {
        if (this->boolMembers_.template get<ENDOFSIM>(idx)) {
            this->boolMembers_.template get<TERMINATED>(idx) = false;
            this->boolMembers_.template get<ENDOFSIM>(idx)   = false;
        }
    }

    DEVICEHOST() void setInitialStep(const SampleIndex& idx, Real dt) {
        nextDts()[idx] = dt;
    }

    DEVICEHOST()
    ViewT operator[](const SampleIndex& idx)
    {
        ParentT::assertInBounds(idx);
        return ViewT(idx, *this);
    }
    DEVICEHOST()
    ConstViewT operator[](const SampleIndex& idx) const
    {
        ParentT::assertInBounds(idx);
        return ConstViewT(idx, *this);
    }

    DEVICEHOST()
    RefSimulation clone() const { return *this; }
};

#define PARM_SIM_CONSTVIEW_GET(NAME, TYPE, MEMBER, FIELD)                      \
    template<bool work, bool MaybeVolatile>                                    \
    DEVICEHOST() TYPE ConstView<work, MaybeVolatile>::NAME() const             \
    {                                                                          \
        return ref_.MEMBER().template get<FIELD>(idx_);                        \
    }

#define PARM_SIM_VIEW_GET(NAME, TYPE, MEMBER, FIELD)                           \
    template<bool work, bool MaybeVolatile>                                    \
    DEVICEHOST() TYPE& View<work, MaybeVolatile>::NAME()                       \
    {                                                                          \
        return ref_.MEMBER().template get<FIELD>(idx_);                        \
    }                                                                          \
    template<bool work, bool MaybeVolatile>                                    \
    DEVICEHOST() TYPE View<work, MaybeVolatile>::NAME() const                  \
    {                                                                          \
        return ref_.MEMBER().template get<FIELD>(idx_);                        \
    }

PARM_SIM_CONSTVIEW_GET(terminated, bool, boolMembers, TERMINATED)
PARM_SIM_CONSTVIEW_GET(endOfSimulation, bool, boolMembers, ENDOFSIM)
PARM_SIM_CONSTVIEW_GET(id, mInt_t, intMembers, SAMPLEID)
PARM_SIM_CONSTVIEW_GET(epoch, Real, realMembers, CURRENTEPOCH)

PARM_SIM_VIEW_GET(terminated, bool, boolMembers, TERMINATED)
PARM_SIM_VIEW_GET(endOfSimulation, bool, boolMembers, ENDOFSIM)
PARM_SIM_VIEW_GET(id, mInt_t, intMembers, SAMPLEID)
PARM_SIM_VIEW_GET(epoch, Real, realMembers, CURRENTEPOCH)

#undef PARM_SIM_CONSTVIEW_GET
#undef PARM_SIM_VIEW_GET

} // namespace base
} // namespace integrate
} // namespace parm