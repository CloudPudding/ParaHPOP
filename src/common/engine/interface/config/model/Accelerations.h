#pragma once

#include "interface/config/model/environment/bodies/BodySource.h"

namespace interface {
namespace config {
namespace model {

/* TEMPORARY */
namespace environment {
class Bodies;
}
namespace accelerations = environment::body;

class Accelerations {
    // using FlagsT = feta::scalar::Array<bool>;
    using FlagsT = feta::vector::Array<bool, accelerations::FEATSIZE>;
    using Self   = Accelerations;

public:
    /** @brief Default constructor makes all flags inactive */
    Accelerations() = default;

    /** @brief Construct from the Given size - activates point gravity by
     * default */
    Accelerations(const idx_t& size)
        : flags_{ size }
    {
        if (size == 0)
            return;
        /* Activate point gravity by default */
        FlagsT::GRef flagsref = flags_.ref();
        for (idx_t i = 0; i < size; i++) {
            flagsref.template get<accelerations::POINTGRAVITY>(i) = true;
        }
    }

    /** @brief Accelerations are constructed from body features, not from JSON.
     *  Use environment().bodies()["Earth"].gravity().oblateness().activate()
     *  to set features, then model.refresh() (or let simulation auto-refresh)
     *  to sync flags. */
    Accelerations(const json& j) = delete;

    /** @brief Construct from the environment's active bodies - forwarded to
     * after active bodies definition */
    inline Accelerations(const model::environment::Bodies& abodies);

    /** @brief Copy constructor */
    Accelerations(const Accelerations& other)
        : flags_{ std::move(other.flags_.clone()) }
    {
    }

    /** @brief Construct from given flags*/
    Accelerations(const FlagsT& flags)
        : flags_{ std::move(flags.clone()) }
    {
    }

    /** @brief Move constructor from another Accelerations object*/
    Accelerations(Accelerations&& other)
        : flags_{ std::move(other.flags_) }
    {
    }

    /** @brief Move constructor from flags*/
    Accelerations(FlagsT&& flags)
        : flags_{ std::move(flags) }
    {
    }

    /** @brief Copy assignment operators */
    Accelerations& operator=(const Accelerations& other)
    {
        this->flags_ = std::move(other.flags_.clone());
        return *this;
    }

    /** @brief Move assignment operator */
    Accelerations& operator=(Accelerations&& other)
    {
        this->flags_ = std::move(other.flags_);
        return *this;
    }

    /** @brief expose acceleration flags */
    FlagsT& flags() { return flags_; }
    const FlagsT& flags() const { return flags_; }

    /** @brief Access the i-th body */
    accelerations::Body operator[](const idx_t& i)
    {
        return accelerations::Body(flags_.ref(), i);
    }
    accelerations::ConstBody operator[](const idx_t& i) const
    {
        return accelerations::ConstBody(flags_.ref(), i);
    }

protected:
    /* flexi type is needed for default construction with feta arrays */
    FlagsT flags_ = FlagsT::flexible();
};

} // namespace model
} // namespace config
} // namespace interface