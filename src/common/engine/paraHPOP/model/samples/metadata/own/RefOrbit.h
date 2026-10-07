#pragma once

#include "paraHPOP/typedefs.h"

namespace paraHPOP {
namespace model {
namespace samples {
namespace metadata {
namespace own {

/**
 * @brief Integer elements for Orbital simulations
 *
 */
enum OrbitInt {

    COI, /* Center of integration */

    /* Leave the following item as last -- it only acts as enum size */
    ORBITINTSIZE

};

/** @brief Boolean elements for orbital simulations */
enum OrbitBool {

    DUMBOOL, /* dummy. TODO: manage 0 dimensions */

    /* Leave the following item as last -- it only acts as enum size */
    ORBITBOOLSIZE

};

/**
 * @brief Real elements for Orbital simulations
 *
 * The packed Area-To-Mass times Cr scalar (``AMS = Cr * Area / Mass``) has
 * been split into four independent per-sample fields.  ``MASS`` is kg,
 * ``AREA`` is km^2, ``CR`` and ``CD`` are dimensionless reflectivity /
 * drag coefficients.  SRP reads (Cr, Area, Mass); drag reads (Cd, Area,
 * Mass).  Storing the components instead of the packed scalar lets future
 * effects (stochastic tumbling on Area/Cr, thrust-induced mass changes)
 * mutate the source fields without recomputing a cache.
 */
enum OrbitReal {

    /** Spacecraft mass, [kg]. */
    MASS,

    /** Cross-sectional area exposed to SRP / drag, [km^2]. */
    AREA,

    /** Solar Radiation Pressure reflectivity coefficient, dimensionless. */
    CR,

    /** Drag coefficient, dimensionless. */
    CD,

    /* Leave the following item as last -- it only acts as enum size */
    ORBITREALSIZE

};

/** @brief Expand view type with direct accessors */
template<bool work>
class View : public parm::util::View<work, ORBITINTSIZE, ORBITBOOLSIZE,
                 ORBITREALSIZE> {
    using ParentT
        = parm::util::View<work, ORBITINTSIZE, ORBITBOOLSIZE, ORBITREALSIZE>;
    using RefT = typename ParentT::RefT;

public:
    /** @brief Factory method to construct from index and reference */
    DEVICEHOST() static View make(const SampleIndex& idx, RefT& ref)
    {
        return View(ParentT(idx, ref));
    }

    /** @brief access the coi */
    DEVICEHOST() mInt_t& coi()
    {
        return this->ref_.intMembers().template get<COI>(this->idx_);
    }
    DEVICEHOST() const mInt_t& coi() const
    {
        return this->ref_.intMembers().template get<COI>(this->idx_);
    }

    /** @brief access the spacecraft mass */
    DEVICEHOST() mReal_t& mass()
    {
        return this->ref_.realMembers().template get<MASS>(this->idx_);
    }
    DEVICEHOST() const mReal_t& mass() const
    {
        return this->ref_.realMembers().template get<MASS>(this->idx_);
    }

    /** @brief access the cross-sectional area */
    DEVICEHOST() mReal_t& area()
    {
        return this->ref_.realMembers().template get<AREA>(this->idx_);
    }
    DEVICEHOST() const mReal_t& area() const
    {
        return this->ref_.realMembers().template get<AREA>(this->idx_);
    }

    /** @brief access the SRP reflectivity coefficient */
    DEVICEHOST() mReal_t& cr()
    {
        return this->ref_.realMembers().template get<CR>(this->idx_);
    }
    DEVICEHOST() const mReal_t& cr() const
    {
        return this->ref_.realMembers().template get<CR>(this->idx_);
    }

    /** @brief access the drag coefficient */
    DEVICEHOST() mReal_t& cd()
    {
        return this->ref_.realMembers().template get<CD>(this->idx_);
    }
    DEVICEHOST() const mReal_t& cd() const
    {
        return this->ref_.realMembers().template get<CD>(this->idx_);
    }
};

/** @brief Expand the const view type with direct accessors */
template<bool work>
class ConstView : public parm::util::ConstView<work, ORBITINTSIZE,
                      ORBITBOOLSIZE, ORBITREALSIZE> {
    using ParentT = parm::util::ConstView<work, ORBITINTSIZE, ORBITBOOLSIZE,
        ORBITREALSIZE>;
    using RefT    = typename ParentT::RefT;

public:
    /** @brief Factory method to construct from index and reference */
    DEVICEHOST() static ConstView make(const SampleIndex& idx, const RefT& ref)
    {
        return ConstView(ParentT(idx, ref));
    }

    /** @brief access the coi */
    DEVICEHOST() const mInt_t& coi() const
    {
        return this->ref_.intMembers().template get<COI>(this->idx_);
    }

    /** @brief access the spacecraft mass */
    DEVICEHOST() const mReal_t& mass() const
    {
        return this->ref_.realMembers().template get<MASS>(this->idx_);
    }

    /** @brief access the cross-sectional area */
    DEVICEHOST() const mReal_t& area() const
    {
        return this->ref_.realMembers().template get<AREA>(this->idx_);
    }

    /** @brief access the SRP reflectivity coefficient */
    DEVICEHOST() const mReal_t& cr() const
    {
        return this->ref_.realMembers().template get<CR>(this->idx_);
    }

    /** @brief access the drag coefficient */
    DEVICEHOST() const mReal_t& cd() const
    {
        return this->ref_.realMembers().template get<CD>(this->idx_);
    }
};

/** @brief Reference to the Unique orbital metadata */
template<bool work, bool MaybeVolatile = false>
class RefOrbit
    : public parm::util::MultiContainer<ORBITINTSIZE, ORBITBOOLSIZE,
          ORBITREALSIZE>::template Ref<work, MaybeVolatile> {
    using ParentT
        = typename parm::util::MultiContainer<ORBITINTSIZE, ORBITBOOLSIZE,
            ORBITREALSIZE>::template Ref<work, MaybeVolatile>;
    using RefIntArrayT =
        typename feta::scalar::Array<mInt_t>::template Ref<work, MaybeVolatile>;
    using RefRealArrayT =
        typename feta::scalar::Array<mReal_t>::template Ref<work, MaybeVolatile>;
    using IntHandleT    = typename RefIntArrayT::HandleT;
    using RealHandleT   = typename RefRealArrayT::HandleT;
    /* Friend other reference */
    friend class RefOrbit<!work, MaybeVolatile>;

public:
    /** @brief Volatile is allowed only with work */
    static constexpr bool Volatile = MaybeVolatile && work;
    /** @brief Item view type */
    using ViewT      = View<work>;
    using ConstViewT = ConstView<work>;

    /** @brief Factory method to construct from the parent type */
    DEVICEHOST()
    static RefOrbit make(const ParentT& other) { return RefOrbit{ other }; }

    /** @brief Short-hand access to the COIs */
    DEVICEHOST()
    IntHandleT cois() const
    {
        return this->intMembers_.template component<COI>();
    }

    /** @brief Short-hand access to the spacecraft masses */
    DEVICEHOST()
    RealHandleT mass() const
    {
        return this->realMembers_.template component<MASS>();
    }

    /** @brief Short-hand access to the cross-sectional areas */
    DEVICEHOST()
    RealHandleT area() const
    {
        return this->realMembers_.template component<AREA>();
    }

    /** @brief Short-hand access to the SRP reflectivity coefficients */
    DEVICEHOST()
    RealHandleT cr() const
    {
        return this->realMembers_.template component<CR>();
    }

    /** @brief Short-hand access to the drag coefficients */
    DEVICEHOST()
    RealHandleT cd() const
    {
        return this->realMembers_.template component<CD>();
    }

    /** @brief Re-expose square bracket operators */
    DEVICEHOST()
    ViewT operator[](const SampleIndex& idx) { return ViewT::make(idx, *this); }
    DEVICEHOST()
    ConstViewT operator[](const SampleIndex& idx) const
    {
        return ConstViewT::make(idx, *this);
    }
};

} // namespace own
} // namespace metadata
} // namespace samples
} // namespace model
} // namespace paraHPOP
