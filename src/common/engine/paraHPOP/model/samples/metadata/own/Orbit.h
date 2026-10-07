#pragma once

#include "paraHPOP/model/samples/metadata/own/RefOrbit.h"

namespace paraHPOP {
namespace model {
namespace samples {
namespace metadata {
namespace own {

/** @brief Reference to the Unique orbital metadata */
class Orbit : public parm::util::MultiContainer<ORBITINTSIZE, ORBITBOOLSIZE,
                  ORBITREALSIZE> {
    using ParentT = parm::util::MultiContainer<ORBITINTSIZE, ORBITBOOLSIZE,
        ORBITREALSIZE>;

public:
    /** @brief Reference type */
    template<bool work, bool MaybeVolatile = false>
    using Ref = RefOrbit<work, MaybeVolatile>;
    /** @brief Global reference type */
    using GRef = Ref<false>;
    /** @brief Work reference type */
    using WRef = Ref<true>;
    /** @brief Volatile work reference */
    using VolatileRef = Ref<true, true>;

    using typename ParentT::EndSharedMemDataT;
    using typename ParentT::SharedMemDataT;

    /** @brief Inherit constructors */
    Orbit(ParentT&& other)
        : ParentT{ std::move(other) }
    {
    }

    /** @brief Clone these metadata */
    Self clone() const { return Self(std::move(ParentT::clone())); }
};

} // namespace own
} // namespace metadata
} // namespace samples
} // namespace model
} // namespace paraHPOP