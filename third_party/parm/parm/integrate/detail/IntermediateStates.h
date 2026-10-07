#pragma once

#include "parm/typedefs.h"


namespace parm {
namespace integrate {
namespace detail {

/** @brief Forward declare Ref intermediate states */
template<typename IStatesT, bool work, bool MaybeVolatile = false>
class RefIntermediateStates;

/* Forward declarations */
template<typename IStatesT, bool work, bool MaybeVolatile = false>
class ConstIntermediateStateView;
template<typename IStatesT, bool work, bool MaybeVolatile = false>
class IntermediateStateView;

/** @brief POD struct - intermediate state */
template<typename StatesT>
struct IntermediateState {
    using ComponentT = typename StatesT::ComponentT;
    using StateT     = typename StatesT::ItemT;
    ComponentT epoch;
    StateT state;
};

/** @brief Const view onto an intermediate state */
template<typename IStatesT, bool work, bool MaybeVolatile>
class ConstIntermediateStateView {
    /* The reference data type */
    using RefT = RefIntermediateStates<IStatesT, work, MaybeVolatile>;
    /* this view type */
    using Self = ConstIntermediateStateView<IStatesT, work, MaybeVolatile>;
    /* other view type */
    using Other = ConstIntermediateStateView<IStatesT, !work, MaybeVolatile>;

    /* Friend other item view */
    friend class ConstIntermediateStateView<IStatesT, !work, MaybeVolatile>;

    /* Friend item view */
    friend class IntermediateStateView<IStatesT, work, MaybeVolatile>;
    friend class IntermediateStateView<IStatesT, !work, MaybeVolatile>;

public:
    /** @brief Expose the epoch */
    DEVICEHOST() typename IStatesT::ComponentT epoch() const;

    /** @brief Expose the state */
    DEVICEHOST() decltype(auto) state() const;

    /** @brief Public data members (POD-like layout for device compatibility). */
    const SampleIndex idx_;
    const RefT& ref_;
};

/** @brief View onto an intermediate state */
template<typename IStatesT, bool work, bool MaybeVolatile>
class IntermediateStateView {
    /* The reference data type */
    using RefT = RefIntermediateStates<IStatesT, work, MaybeVolatile>;

public:
    /** @brief Expose the epoch */
    DEVICEHOST() typename IStatesT::ComponentT& epoch();
    DEVICEHOST() typename IStatesT::ComponentT epoch() const;

    /** @brief Expose the state */
    DEVICEHOST() decltype(auto) state();
    DEVICEHOST() decltype(auto) state() const;

    /** @brief Public data members (POD-like layout for device compatibility). */
    const SampleIndex idx_;
    const RefT& ref_;
};

/**
 * @brief Provide non-owning access to intermediate epoch-state pairs used during
 *        Runge-Kutta integration.
 *
 * @tparam IStatesT  Owning ``IntermediateStates`` type.
 * @tparam work      ``false`` for global memory; ``true`` for shared memory.
 */
template<typename IStatesT, bool work, bool MaybeVolatile>
class RefIntermediateStates {

    static constexpr idx_t StateDims = IStatesT::StatesT::VecDims;
    using Self
        = RefIntermediateStates<IStatesT, work, MaybeVolatile>;
    using ParentT = typename VecNTArray<typename IStatesT::ComponentT,
        StateDims + IStatesT::StatesT::VecDims>::template Ref<work,
        MaybeVolatile>;

    /* Friend other reference */
    friend class RefIntermediateStates<IStatesT, !work, MaybeVolatile>;

    /** @brief Extract data types */
    using EpochsT =
        typename feta::scalar::Array<typename IStatesT::ComponentT>::
            template Ref<work, MaybeVolatile>;
    using StatesT
        = typename IStatesT::StatesT::template Ref<work, MaybeVolatile>;

public:
    /** @brief Volatile is allowed only with work */
    static constexpr bool Volatile = MaybeVolatile && work;
    /* Item view types */
    using ViewT
        = IntermediateStateView<IStatesT, work, MaybeVolatile>;
    using ConstViewT
        = ConstIntermediateStateView<IStatesT, work, MaybeVolatile>;

    /** @brief Factory method to construct from the data elements */
    DEVICEHOST()
    static RefIntermediateStates make(
        const EpochsT epochs, const StatesT states)
    {
        RefIntermediateStates ref;
        ref.epochs_ = epochs;
        ref.states_ = states;
        return ref;
    }

    /** @brief Expose the epochs array */
    DEVICEHOST() EpochsT epochs() const { return epochs_; }

    /** @brief Expose the states array */
    DEVICEHOST() StatesT states() const { return states_; }

    /** @brief Get a view onto the i-th intermediate state */
    DEVICEHOST()
    ViewT operator[](const SampleIndex& idx) { return ViewT(idx, *this); }
    DEVICEHOST()
    ConstViewT operator[](const SampleIndex& idx) const
    {
        return ConstViewT(idx, *this);
    }

    /** @brief Clone this reference */
    DEVICEHOST()
    Self clone() const { return *this; }

    /** @brief Public data members (POD-like layout for device compatibility). */
    EpochsT epochs_;
    StatesT states_;
};

/* Finalize const view method implementations */
template<typename IStatesT, bool work, bool MaybeVolatile>
DEVICEHOST()
typename IStatesT::ComponentT
ConstIntermediateStateView<IStatesT, work, MaybeVolatile>::epoch() const
{
    return ref_.epochs()[idx_];
}
template<typename IStatesT, bool work, bool MaybeVolatile>
DEVICEHOST()
decltype(auto)
ConstIntermediateStateView<IStatesT, work, MaybeVolatile>::state() const
{
    return ref_.states()[idx_];
}

/* Finalize view method implementations */
template<typename IStatesT, bool work, bool MaybeVolatile>
DEVICEHOST()
typename IStatesT::ComponentT&
IntermediateStateView<IStatesT, work, MaybeVolatile>::epoch()
{
    return ref_.epochs()[idx_];
}
template<typename IStatesT, bool work, bool MaybeVolatile>
DEVICEHOST()
typename IStatesT::ComponentT
IntermediateStateView<IStatesT, work, MaybeVolatile>::epoch() const
{
    return ref_.epochs()[idx_];
}
template<typename IStatesT, bool work, bool MaybeVolatile>
DEVICEHOST()
decltype(auto) IntermediateStateView<IStatesT, work, MaybeVolatile>::state()
{
    return ref_.states()[idx_];
}
template<typename IStatesT, bool work, bool MaybeVolatile>
DEVICEHOST()
decltype(auto)
IntermediateStateView<IStatesT, work, MaybeVolatile>::state() const
{
    return ref_.states()[idx_];
}

/**
 * @brief Own and manage the intermediate epoch-state pair arrays for RK integration.
 *
 * @tparam StatesT_  FETA ``vector::Array`` type holding the ODE state vectors.
 */
template<typename StatesT_>
class IntermediateStates {

    static constexpr idx_t StateDims = StatesT_::VecDims;
    using Self                       = IntermediateStates<StatesT_>;

public:
    using StatesT                  = StatesT_;
    using ComponentT               = typename StatesT_::ComponentT;
    using EpochsT                  = feta::scalar::Array<ComponentT>;
    static constexpr idx_t VecDims = StatesT_::VecDims;

    /* Stream types */
    using StreamT = typename StatesT::StreamT;

    /** @brief Generic work/global non-owning reference type.  The
     * ``MaybeVolatile`` parameter propagates volatility to the
     * underlying RefIntermediateStates and to the inner EpochsT /
     * StatesT feta refs. */
    template<bool work, bool MaybeVolatile = false>
    using Ref = detail::RefIntermediateStates<Self, work, MaybeVolatile>;
    /** @brief Global reference type */
    using GRef = Ref<false>;
    /** @brief Work reference type */
    using WRef = Ref<true>;
    /** @brief Volatile work reference */
    using VolatileRef = Ref<true, true>;

    /** @brief Default constructor is forbidden */
    IntermediateStates() = delete;

    /** @brief Copy constructor is forbidden */
    IntermediateStates(IntermediateStates& other) = delete;
    IntermediateStates(const IntermediateStates&) = delete;

    /** @brief Move constructor from data elements */
    IntermediateStates(EpochsT&& epochs, StatesT&& states)
        : epochs_{ std::move(epochs) }
        , states_{ std::move(states) }
    {
    }

    /** @brief Move constructor */
    IntermediateStates(IntermediateStates&& other)
        : epochs_{ std::move(other.epochs_) }
        , states_{ std::move(other.states_) }
        , initialised_{ std::exchange(other.initialised_, false) }
    {
    }

    /** @brief Generic host constructor with memory allocation */
    IntermediateStates(const idx_t& nSamples)
        : epochs_{ nSamples }
        , states_{ nSamples }
    {
    }

    /** @brief Copy-assignment is forbidden */
    IntermediateStates& operator=(IntermediateStates& other) = delete;
    IntermediateStates& operator=(const IntermediateStates&) = delete;

    /** @brief Move assignment operator */
    IntermediateStates& operator=(IntermediateStates&& other)
    {
        epochs_      = std::move(other.epochs_);
        states_      = std::move(other.states_);
        initialised_ = std::exchange(other.initialised_, false);
        return *this;
    }

    /** @brief return a RefIntermediateStates object pointing to the host array
     */
    GRef hostRef() const
    {
        return GRef::make(epochs_.hostRef(), states_.hostRef());
    }

#ifndef PARM_CPU_ONLY
    /** @brief return a RefIntermediateStates object pointing to the device
     * array
     */
    GRef deviceRef() const
    {
        return GRef::make(epochs_.deviceRef(), states_.deviceRef());
    }

    /** @brief Upload the data to the device */
    void upload(const StreamT& stream = 0)
    {
        epochs_.upload(stream);
        states_.upload(stream);
        initialised_ = true;
    }

    /** @brief Download the data from the device */
    void download(const StreamT& stream = 0)
    {
        epochs_.download(stream);
        states_.download(stream);
    }

    /**
     * @brief Ensure the IntermediateStates are initialised on the GPU. Must be
     * called before time-stepping.
     */
    void ensureInitDevice(const StreamT& stream = 0)
    {
        if (!initialised_) {
            this->upload(stream);
            initialised_ = true;
        }
    }

    /** @brief Clear data from the GPU */
    void clearDevice()
    {
        epochs_.clearDevice();
        states_.clearDevice();
        initialised_ = false;
    }

#endif

protected:
    EpochsT epochs_;
    StatesT states_;

    /* Flag to ensure the device initialization */
    bool initialised_ = false;
};

} // namespace detail
} // namespace integrate
} // namespace parm