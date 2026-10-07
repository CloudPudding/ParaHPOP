#pragma once

#include "paraHPOP/model/samples/states/cartesian/RefStates.h"


namespace paraHPOP {
namespace model {
namespace samples {
namespace states {
namespace cartesian {

/**
 * @brief Cartesian state representation. Maps to a 6-dimensional vector.
 */
class States : public feta::vector::Array<Real, 6> {
    using ParentT = feta::vector::Array<Real, 6>;
    using Self    = States;

public:
    /** @brief Generic work/global non-owning reference type */
    template<bool work, bool MaybeVolatile = false>
    using Ref = RefStates<work, MaybeVolatile>;
    /** @brief Global reference type */
    using GRef = Ref<false>;
    /** @brief Work reference type */
    using WRef = Ref<true>;
    /** @brief Volatile work reference */
    using VolatileRef = Ref<true, true>;
    /** @brief Single state type */
    using StateT = State;
    /** @brief Interface type */
    using InterfaceT = feta::buffer::vector::Array<Real,6>;

    /** @brief inherit the constructors */
    using ParentT::ParentT;

    /** @brief inherit the assignment operators */
    using ParentT::operator=;

    /** @brief Create from Interface */
    States(const InterfaceT& iStates)
        : ParentT{ iStates }
    {
    }

    /** @brief Construct from parent type */
    States(ParentT&& v)
        : ParentT{ std::move(v) }
    {
    }

    /** @brief Return a host reference */
    GRef hostRef() const { return GRef::make(ParentT::hostRef()); }
    /** @brief Return a device reference */
    GRef deviceRef() const { return GRef::make(ParentT::deviceRef()); }

    /** @brief Write to the given interface samples */
    void interface(InterfaceT& out) const
    {
        PARAHPOP_ASSERT(this->size() == out.size(),
            "Incompatible sizes for interface writing");
        hostRef().toBuffer(out);
    }

    /** @brief Return the interface states */
    InterfaceT interface() const { return this->buffer(); }

    /** @brief Clone the current states */
    Self clone() const { return Self(std::move(ParentT::clone())); }
};

} // namespace cartesian
} // namespace states
} // namespace samples
} // namespace model
} // namespace paraHPOP
