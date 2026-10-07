#pragma once

#include "parm/integrate/detail/IntermediateStates.h"

#include <feta/vector/TileArray.h>

namespace parm {
namespace integrate {
namespace detail {

/**
 * @brief Provide non-owning access to the DStates scratch storage.
 *
 * The ``DStates`` structure is used as a working area by Runge-Kutta
 * integrators.  It stores one derivative vector ("dState") per RK substep,
 * where "dState" is the right-hand-side evaluation (first derivative of the
 * state vector).
 *
 * Inherits from ``feta::vector::TileArray<T, D, K>::Ref<work>`` (= a
 * ``RefTileArray``) — that gives us the SoA-of-tiles parent
 * (per-sample storage of K derivative stages × D-dim each) plus the
 * ``slot<s>()`` accessor; this class adds the recyclable
 * ``intermediateStates_`` scratch and the higher-level
 * ``stage<s>()`` / ``extractStage<s>()`` / ``nextStates()`` /
 * ``dtNow()`` / ``dtNext()`` accessors that the RK integrator uses.
 *
 * @tparam DStatesT  Owning ``DStates`` type.
 * @tparam work      ``false`` for global memory access; ``true`` for shared memory.
 */
template<typename DStatesT, bool work, bool MaybeVolatile = false>
class RefDStates
    : public DStatesT::ParentT::template Ref<work, MaybeVolatile> {
    using Self = RefDStates<DStatesT, work, MaybeVolatile>;

public:
    /** @brief Volatile is allowed only with work */
    static constexpr bool Volatile = MaybeVolatile && work;
    /* Short hands */
    static constexpr idx_t StateDims = DStatesT::VecDims;
    static constexpr idx_t nStages   = DStatesT::nStages;
    /** @brief Parent reference type — ``feta::vector::TileArray::Ref<work,
     *  MaybeVolatile>`` (= ``RefTileArray``).  Provides the underlying
     *  flat buffer plus the ``slot<s>()`` accessor that ``stage<s>()``
     *  aliases below. */
    using ParentT
        = typename DStatesT::ParentT::template Ref<work, MaybeVolatile>;
    using StatesT
        = typename DStatesT::StatesT::template Ref<work, MaybeVolatile>;
    using StateT          = typename DStatesT::StateT;
    using ComponentArrayT = typename feta::scalar::Array<typename DStatesT::
            ComponentT>::template Ref<work, MaybeVolatile>;
    using IntermediateStatesT = typename DStatesT::IntermediateStatesT::
        template Ref<work, MaybeVolatile>;

    /** @brief Factory method to construct from Parent and IntermediateStates */
    DEVICEHOST()
    static RefDStates make(
        const ParentT other, const IntermediateStatesT intermediateStates)
    {
        RefDStates ref;
        static_cast<ParentT&>(ref) = other;
        ref.intermediateStates_    = intermediateStates;
        return ref;
    }

    /**
     * @brief Returns a view onto a particular dState vector.
     *
     * Thin alias over ``RefTileArray::slot<stageIdx>()`` (inherited
     * from ``ParentT``) — the slot/stage indexing is identical;
     * we keep the ``stage<>`` name for backwards compatibility.
     *
     * @tparam stageIdx Index of the dState to expose
     */
    template<idx_t stageIdx>
    DEVICEHOST()
    decltype(auto) stage()
    {
        static_assert(stageIdx < nStages,
            "Requested stage index exceeds number of stored stages");
        return ParentT::template slot<stageIdx>();
    }

    template<idx_t stageIdx>
    DEVICEHOST()
    decltype(auto) stage() const
    {
        static_assert(stageIdx < nStages,
            "Requested stage index exceeds number of stored stages");
        return ParentT::template slot<stageIdx>();
    }

    /** @brief Extract the current stage DStates into a StatesT object */
    template<idx_t stageIdx>
    DEVICEHOST()
    StatesT extractStage() const
    {
        StatesT out;
        out.data_      = this->data() + stageIdx * StateDims * this->size();
        out.nVecs_     = this->size();
        out.dimOffset_ = this->dimOffset();
        return out;
    }

    /** @brief Expose the intermediate states */
    DEVICEHOST()
    IntermediateStatesT intermediateStates() const
    {
        return intermediateStates_;
    }

    /** @brief Next states - we can recycle intermediateStates_ */
    DEVICEHOST() StatesT nextStates() const
    {
        return intermediateStates_.states();
    }

    /** @brief DtNow - we can recycle stage 0, dimension 0 */
    DEVICEHOST()
    ComponentArrayT dtNow() const { return this->template component<0>(); }

    /** @brief DtNext - we can recycle stage 0, dimension 1 */
    DEVICEHOST()
    ComponentArrayT dtNext() const { return this->template component<1>(); }

    /** @brief Clone this reference object */
    DEVICEHOST()
    Self clone() const { return *this; }

    /** @brief Public data members (POD-like layout for device compatibility). */
    /* Intermediate states scratch */
    IntermediateStatesT intermediateStates_;
};

/**
 * @brief Structure used to store intermediate step results.
 *
 * The ``DStates`` structure is used as a "working area" by Runge-Kutta
 * integrators. It stores one "dState" per RK substep. By "dState" we mean
 * what is returned by the right-hand side (RHS), i.e. the first derivative
 * of the state vector. It includes an extra "stage" to store the cumulative
 * weighted sum of the evaluated dStates.
 *
 * Specialisation of ``feta::vector::TileArray<ComponentT, StateDims,
 * nStages>`` — the heap-allocated SoA-of-tiles container.  TileArray
 * provides the underlying flat buffer (``nSamples × StateDims ×
 * nStages``) plus per-stage slot access; ``DStates`` adds the
 * recyclable ``intermediateStates_`` scratch (used by the RK
 * pipeline as ``nextStates`` / ``dtNow`` / ``dtNext``) and the
 * ``stage<>`` / ``extractStage<>`` accessors that consumers expect.
 */
template<typename StatesT_, idx_t nStages_>
class DStates
    : public feta::vector::TileArray<typename StatesT_::ComponentT,
          StatesT_::VecDims, nStages_> {

    static constexpr idx_t StateDims = StatesT_::VecDims;
    using Self                       = DStates<StatesT_, nStages_>;

public:
    /* Short hands */
    static constexpr idx_t nStages = nStages_;
    using ComponentT               = typename StatesT_::ComponentT;
    using ParentT
        = feta::vector::TileArray<ComponentT, StateDims, nStages>;
    using StreamT             = typename ParentT::StreamT;
    using StatesT             = StatesT_;
    using StateT              = typename StatesT::GRef::ItemT;
    using IntermediateStatesT = IntermediateStates<StatesT_>;
    static constexpr idx_t VecDims = StatesT_::VecDims;

    /** @brief Reference types.  ``MaybeVolatile`` flows down through
     * the underlying TileArray + IntermediateStates refs. */
    template<bool work, bool MaybeVolatile = false>
    using Ref = RefDStates<Self, work, MaybeVolatile>;
    /** @brief Global reference type */
    using GRef = Ref<false>;
    /** @brief Work reference type */
    using WRef = Ref<true>;
    /** @brief Volatile work reference */
    using VolatileRef = Ref<true, true>;

    /** @brief Default constructor is forbidden */
    DStates() = delete;

    /** @brief Construct with size */
    DStates(const idx_t& nSamples)
        : ParentT{ nSamples }
        , intermediateStates_{ nSamples }
    {
    }

    /** @brief Copy constructor is forbidden */
    DStates(DStates& other)                                    = delete;
    DStates(const DStates& other)                              = delete;
    DStates(ParentT& other)                                    = delete;
    DStates(ParentT& other, IntermediateStatesT& intermStates) = delete;
    DStates(const ParentT& other)                              = delete;
    DStates(const ParentT& other, const IntermediateStatesT& intermStates)
        = delete;

    /** @brief Move constructor from the data elements */
    DStates(ParentT&& other, IntermediateStatesT&& intermStates)
        : ParentT{ std::move(other) }
        , intermediateStates_{ std::move(intermStates) }
    {
    }

    /** @brief Move constructor from parent */
    DStates(ParentT&& other)
        : ParentT{ std::move(other) }
        , intermediateStates_{ this->size() }
    {
    }

    /** @brief Move constructor */
    DStates(DStates&& other)
        : ParentT{ std::move(other) }
        , intermediateStates_{ std::move(other.intermediateStates_) }
        , initialised_{ std::exchange(other.initialised_, false) }
    {
    }

    /** @brief Copy assignment is forbidden */
    DStates& operator=(DStates& other)       = delete;
    DStates& operator=(const DStates& other) = delete;

    /** @brief Move assignment operator */
    DStates& operator=(DStates&& other)
    {
        ParentT::operator=(std::move(other));
        intermediateStates_ = std::move(other.intermediateStates_);
        initialised_        = std::exchange(other.initialised_, false);
        return *this;
    }

    /** @brief Expose the total data size */
    static idx_t totalSize(const idx_t& nSamples)
    {
        return nSamples * StateDims * nStages_;
    }

    /** @brief Return a RefDStates pointing to the host array. */
    GRef hostRef() const
    {
        return GRef::make(ParentT::hostRef(), intermediateStates_.hostRef());
    }
    GRef ref() const { return hostRef(); }

#ifndef PARM_CPU_ONLY
    /** @brief Return a RefDStates pointing to the device array. */
    GRef deviceRef() const
    {
        return GRef::make(
            ParentT::deviceRef(), intermediateStates_.deviceRef());
    }

    /** @brief Upload (both parent and intermediate states) */
    void upload(const StreamT& stream = 0)
    {
        ParentT::upload(stream);
        intermediateStates_.upload(stream);
        initialised_ = true;
    }

    /** @brief Download (both parent and intermediate states) */
    void download(const StreamT& stream = 0)
    {
        ParentT::download(stream);
        intermediateStates_.download(stream);
    }

    /**
     * @brief Ensure the DStates are initialised on the GPU. Must be called
     * before time-stepping.
     */
    void ensureInitDevice(const StreamT& stream = 0)
    {
        if (!initialised_) {
            ParentT::upload(stream);
            intermediateStates_.ensureInitDevice(stream);
            initialised_ = true;
        }
    }

    /** @brief Clear data from the GPU */
    void clearDevice()
    {
        ParentT::clearDevice();
        intermediateStates_.clearDevice();
        initialised_ = false;
    }
#endif

    /** @brief Expose the intermediate states */
    IntermediateStatesT& intermediateStates() { return intermediateStates_; }
    const IntermediateStatesT& intermediateStates() const
    {
        return intermediateStates_;
    }

protected:
    /* Intermediate states scratch */
    IntermediateStatesT intermediateStates_;

    /** @brief Add a flag to ensure the device initialization */
    bool initialised_ = false;
};

} // namespace detail
} // namespace integrate
} // namespace parm