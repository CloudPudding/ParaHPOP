#pragma once

#include "paraHPOP/model/accelerations/RefAccelerations.h"
#include "paraHPOP/typedefs.h"
#include "paraHPOP/util.h"
#include "interface/config/model/Accelerations.h"

namespace paraHPOP {
namespace model {
namespace accelerations {

/**
 * @brief Acceleration management class
 *
 */
class Accelerations {
    using SourcesT = src::Features;
    using ScratchT = feta::vector::Array<Real, 6>;
    using EnvT     = ::paraHPOP::model::environment::Env;
    using FlagsT   = feta::vector::Array<bool, src::FEATSIZE>;

public:
    using InterfaceT = interface::config::model::Accelerations;
    /** @brief reference types */
    template<bool work, bool MaybeVolatile = false>
    using Ref = RefAccelerations<work, MaybeVolatile>;
    /** @brief global reference type */
    using GRef = Ref<false>;
    /** @brief work reference type */
    using WRef = Ref<true>;
    /** @brief Volatile work reference */
    using VolatileRef = Ref<true, true>;

    /** @brief Static method to turn a vec  */

    /** @brief Default constructor makes all flags inactive */
    Accelerations() = default;

    /** @brief Construct from interface config type */
    Accelerations(const InterfaceT& iaccs)
        : Accelerations{ std::move(iaccs.flags().clone()) }
    {
    }

    /** @brief Copy constructor is forbidden */
    Accelerations(Accelerations& other)       = delete;
    Accelerations(const Accelerations& other) = delete;

    /** @brief Construct from given flags - move only*/
    Accelerations(FlagsT& flags)       = delete;
    Accelerations(const FlagsT& flags) = delete;

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

    /** @brief Move assignment operator */
    Accelerations& operator=(Accelerations&& other)
    {
        this->flags_ = std::move(other.flags_);
        return *this;
    }

    /** @brief expose acceleration flags */
    FlagsT& flags() { return flags_; }
    const FlagsT& flags() const { return flags_; }

    /** @brief Check if the given acceleration source is active or not */
    template<SourcesT SOURCE>
    bool& active(const idx_t& i)
    {
        return flags_.data()[SOURCE * flags_.size() + i];
    }
    template<SourcesT SOURCE>
    const bool& active(const idx_t& i) const
    {
        return flags_.data()[SOURCE * flags_.size() + i];
    }

    /** @brief Return whether any acceleration source is active for the given
     * index */
    bool anyActive(const idx_t& i) const
    {
        bool any = false;
        anyActive_<static_cast<SourcesT>(0)>(any, i);
        return any;
    }

    /** @brief Return whether any force-emitting source is active for the
     *  given index.  OCCULTING is excluded: it is a passive capability
     *  (the body casts a shadow, consumed by the occultation precompute
     *  kernel) and never emits a per-body force chain.  Use ::anyActive
     *  when gating per-sample cache resolution — occulters need their
     *  resolved ephemeris slots filled like any force body. */
    bool anyForceActive(const idx_t& i) const
    {
        bool any = false;
        anyForceActive_<static_cast<SourcesT>(0)>(any, i);
        return any;
    }

    /** @brief Return whether the given acceleration source is active for any
     * index */
    template<SourcesT SOURCE>
    bool anyActive() const
    {
        bool any = false;
        for (idx_t i = 0; i < flags_.size(); i++) {
            any |= active<SOURCE>(i);
            if (any)
                break;
        }
        return any;
    }

    /** @brief Activate the given acceleration source */
    template<SourcesT SOURCE>
    void activate(const idx_t& i)
    {
        active<SOURCE>(i) = true;
    }

    /** @brief Deactivate the given acceleration source */
    template<SourcesT SOURCE>
    void deactivate(const idx_t& i)
    {
        active<SOURCE>(i) = false;
    }

    /** @brief Upload the flags */
    void upload(const cudaStream_t& stream = 0) { flags_.upload(stream); }

    /** @brief Download the flags */
    void download(const cudaStream_t& stream = 0) { flags_.download(stream); }

    /** @brief Clear the device data */
    void clearDevice() { flags_.clearDevice(); }

    /** @brief return host reference */
    GRef hostRef() const { return GRef(flags_.hostRef()); }

    /** @brief return device reference */
    GRef deviceRef() const { return GRef(flags_.deviceRef()); }

private:
    /** @brief recursion to check whether any source is active for the given
     * index.  The flag read is guarded by the same bound as the recursion
     * so the terminal instantiation (SOURCE == FEATSIZE) never indexes
     * one row past the flags array. */
    template<SourcesT SOURCE>
    void anyActive_(bool& any, const idx_t& i) const
    {
        if constexpr (SOURCE < SourcesT::FEATSIZE) {
            any |= active<SOURCE>(i);
            anyActive_<static_cast<SourcesT>(static_cast<idx_t>(SOURCE) + 1)>(
                any, i);
        }
    }

    /** @brief ::anyActive_ variant skipping OCCULTING (see
     * ::anyForceActive). */
    template<SourcesT SOURCE>
    void anyForceActive_(bool& any, const idx_t& i) const
    {
        if constexpr (SOURCE < SourcesT::FEATSIZE) {
            if constexpr (SOURCE != SourcesT::OCCULTING)
                any |= active<SOURCE>(i);
            anyForceActive_<
                static_cast<SourcesT>(static_cast<idx_t>(SOURCE) + 1)>(any, i);
        }
    }

    FlagsT flags_ = FlagsT::flexible();
};

} // namespace accelerations
} // namespace model
} // namespace paraHPOP