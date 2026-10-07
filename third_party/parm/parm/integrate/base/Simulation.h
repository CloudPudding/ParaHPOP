#pragma once

#include "parm/integrate/base/RefSimulation.h"

namespace parm {
namespace integrate {
namespace base {



/**
 * @brief Owning per-sample metadata container for Runge-Kutta simulations.
 *
 * Stores, for each sample, termination flags,
 * integer fields (sample ID), and real fields (current epoch,
 * step sizes, start/end epochs).  Backed by three FETA ``vector::Array``
 * members (SoA layout) that support host↔device transfer.
 *
 * Obtain a non-owning reference via ``hostRef()`` or ``deviceRef()``.
 * Pass the reference (``GRef``) to integrators and kernels.
 *
 * @see RefSimulation  for the full accessor API.
 */
class Simulation
    : public util::MultiContainer<SIMINTSIZE, SIMBOOLSIZE, SIMREALSIZE> {
    using ParentT = util::MultiContainer<SIMINTSIZE, SIMBOOLSIZE, SIMREALSIZE>;
    /* Implicit types inheritance */
    using typename ParentT::VecBoolArrayT;
    using typename ParentT::VecIntArrayT;
    using typename ParentT::VecRealArrayT;
    using Self = Simulation;

public:
    /** @brief Generic work/global non-owning reference type. */
    template<bool work, bool MaybeVolatile = false>
    using Ref = RefSimulation<work, MaybeVolatile>;
    /** @brief Global (device or host) reference type. */
    using GRef = Ref<false>;
    /** @brief Shared-memory (work) reference type. */
    using WRef = Ref<true>;
    /** @brief Volatile work reference */
    using VolatileRef = Ref<true, true>;

    /** @brief Inherit constructors */
    using ParentT::ParentT;

    /** @brief Inherit assignment operators */
    using ParentT::operator=;

    /** @brief Move-construct from parent */
    Simulation(ParentT&& other)
        : ParentT{ std::move(other) }
    {
    }

    /** @brief Move-assign from parent */
    Simulation& operator=(ParentT&& other)
    {
        ParentT::operator=(std::move(other));
        return *this;
    }

    /** @brief Initialize sample IDs on the host */
    void initializeIDs()
    {
        GRef ref = this->hostRef();
        idx_t s  = this->size();

#pragma omp parallel for simd num_threads(8)
        for (idx_t i = 0; i < s; i++) {
            SampleIndex ii = SampleIndex::make(i);
            ref.resetSampleID(ii);
        }
    }

    /** @brief Return a Simulation reference pointing to the host
     * arrays. */
    GRef hostRef() const { return GRef::make(ParentT::hostRef()); }
    GRef ref() const { return hostRef(); }

#ifndef PARM_CPU_ONLY
    /** @brief Return a Simulation reference pointing to the device
     * arrays.
     */
    GRef deviceRef() const { return GRef::make(ParentT::deviceRef()); }

#endif

    /** @brief Clone these metadata */
    Self clone() const { return Self(std::move(ParentT::clone())); }
};

} // namespace base
} // namespace integrate
} // namespace parm