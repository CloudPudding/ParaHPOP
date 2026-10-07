#pragma once

#include "paraHPOP/model/samples/metadata/own/Orbit.h"

namespace paraHPOP {
namespace model {
namespace samples {
namespace metadata {

/** @brief Forward declarations */
template<bool work>
class View;
template<bool work>
class ConstView;

/** @brief Individual const item view */
template<bool work>
class ConstView {
    using Self  = ConstView;
    using Other = ConstView<!work>;
    /* Friend other item view */
    friend class ConstView<!work>;

    /* Friend const item view */
    friend class View<work>;
    friend class View<!work>;

public:
    /* base type */
    using BaseT =
        typename parm::integrate::base::RefSimulation<work>::ConstViewT;
    using OwnT = typename own::RefOrbit<work>::ConstViewT;


    /** @brief Factory method to construct by members */
    DEVICEHOST() static ConstView make(BaseT&& base, OwnT&& own)
    {
        return ConstView(std::move(base), std::move(own));
    }

    /** @brief Expose the base part */
    DEVICEHOST() BaseT base() const { return base_; }

    /** @brief Expose the own part */
    DEVICEHOST() OwnT own() const { return own_; }

    /** @brief Data members made public for PODification */
    BaseT base_;
    OwnT own_;
};

/** @brief Individual Item View */
template<bool work>
class View {
    using Self  = View;
    using Other = View<!work>;

    /* Friend other item view */
    friend class View<!work>;

    /* Friend const item view */
    friend class ConstView<work>;
    friend class ConstView<!work>;
    using ConstSelf  = ConstView<work>;
    using ConstOther = ConstView<!work>;

public:
    /* base type */
    using BaseT = typename parm::integrate::base::RefSimulation<work>::ViewT;
    using OwnT  = typename own::RefOrbit<work>::ViewT;

    /** @brief Factory method to construct by members */
    DEVICEHOST() static View make(BaseT&& base, OwnT&& own)
    {
        return View(std::move(base), std::move(own));
    }

    /** @brief Expose the base part */
    DEVICEHOST() BaseT base() const { return base_; }

    /** @brief Expose the own part */
    DEVICEHOST() OwnT own() const { return own_; }

    /** @brief Data members made public for PODification */
    BaseT base_;
    OwnT own_;
};

/** @brief Reference Orbit metadata - allows short-hand access to metadata
 * elements */
template<bool work, bool MaybeVolatile = false>
class RefOrbit {
    using Self = RefOrbit<work, MaybeVolatile>;

    /* Friend other reference */
    friend class RefOrbit<!work, MaybeVolatile>;

public:
    /** @brief Volatile is allowed only with work */
    static constexpr bool Volatile = MaybeVolatile && work;
    /* Simulation base type */
    using BaseT =
        typename parm::integrate::base::Simulation::Ref<work, MaybeVolatile>;
    /* Own orbit type */
    using OwnT = typename own::Orbit::template Ref<work, MaybeVolatile>;

    /** @brief Stream type, inherited from BaseT (FETA mode-aware). */
    using StreamT = typename BaseT::StreamT;

    /** @brief Expose the item view type */
    using ViewT      = View<work>;
    using ConstViewT = ConstView<work>;

    /**
     * @brief Factory method to construct from data members
     *
     */
    DEVICEHOST() static RefOrbit make(BaseT&& base, OwnT&& own)
    {
        return RefOrbit{ std::move(base), std::move(own) };
    }
    /**
     * @brief Return the number of samples in the corresponding
     * `paraHPOP::model::samples::metadata::Orbit`
     *
     */
    DEVICEHOST() mSize_t size() const { return base_.size(); }

    /** @brief Get the item view */
    DEVICEHOST()
    ViewT operator[](const SampleIndex& idx)
    {
        return ViewT(base_[idx], own_[idx]);
    }
    DEVICEHOST()
    ConstViewT operator[](const SampleIndex& idx) const
    {
        return ConstViewT(base_[idx], own_[idx]);
    }

    /**
     * @brief Get reference to base members
     *
     */
    DEVICEHOST() BaseT& base() { return base_; }
    DEVICEHOST() const BaseT& base() const { return base_; }

    /**
     * @brief Get reference to own members
     *
     */
    DEVICEHOST() OwnT& own() { return own_; }
    DEVICEHOST() const OwnT& own() const { return own_; }

    /** @brief Async copy from another ``RefOrbit`` (composite ref-to-ref
     *  fetch). Recurses through ``base`` + ``own``, both inherited
     *  ``parm::util::RefMultiContainer::fetch`` implementations.
     */
    void fetch(const RefOrbit& other, const StreamT& stream = 0)
    {
        base_.fetch(other.base_, stream);
        own_.fetch(other.own_, stream);
    }

    /** @brief Per-slot composite copy. Chains ``base.copyFrom`` +
     *  ``own.copyFrom`` (both inherited from
     *  ``parm::util::RefMultiContainer``). */
    DEVICEHOST() void copyFrom(const SampleIndex& dst_i, const RefOrbit& src,
        const SampleIndex& src_i)
    {
        base_.copyFrom(dst_i, src.base_, src_i);
        own_.copyFrom(dst_i, src.own_, src_i);
    }

    /**
     * @brief Create new global reference for this object
     *
     */
    DEVICEHOST() RefOrbit clone() const { return *this; }

    /** @brief Data members made public for PODification */
    BaseT base_;
    OwnT own_;
};

} // namespace metadata
} // namespace samples
} // namespace model
} // namespace paraHPOP