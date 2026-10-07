#pragma once

#include "paraHPOP/model/samples/metadata/RefOrbit.h"
#include "interface/samples/Samples.h"

namespace paraHPOP {
namespace model {
namespace samples {
namespace metadata {

class Orbit {
    /* Interface samples */
    using ISamples = interface::samples::Collection;
    using Self     = Orbit;

public:
    /** @brief Reference types */
    template<bool work, bool MaybeVolatile = false>
    using Ref = RefOrbit<work, MaybeVolatile>;
    /** @brief global reference type **/
    using GRef = Ref<false>;
    /** @brief work reference type **/
    using WRef = Ref<true>;
    /** @brief Volatile work reference */
    using VolatileRef = Ref<true, true>;
    /** @brief Data type aliases*/
    using BaseT = parm::integrate::base::Simulation;
    using OwnT  = own::Orbit;

    /**
     * @brief Copy constructor is forbidden
     *
     */
    Orbit(Orbit& other) = delete;

    /**
     * @brief Move constructor from the data elements
     *
     */
    Orbit(BaseT&& base, OwnT&& own)
        : base_{ std::move(base) }
        , own_{ std::move(own) }
    {
    }

    /**
     * @brief Move constructor
     *
     */
    Orbit(Orbit&& other)
        : base_{ std::move(other.base_) }
        , own_{ std::move(other.own_) }
    {
    }

    /**
     * @brief Basic host constructor
     *
     */
    Orbit(const mSize_t& NSamples)
        : base_{ NSamples }
        , own_{ NSamples }
    {
    }

    /** @brief Construct from Interface samples */
    Orbit(const ISamples& iSamples)
        : Orbit{ iSamples.size() }
    {
        GRef ref = this->hostRef();

        /* Copy in the metadata items */
        /* Epochs */
        ref.base().currentEpochs().fromBuffer(iSamples.epochs());
        ref.base().startEpochs().fetch(ref.base().currentEpochs(), size());
        /* COIs */
        ref.own().cois().fromBuffer(iSamples.centres());
        /* IDs */
        ref.base().ids().fromBuffer(iSamples.ids());
        /* Mass / Area / Cr / Cd */
        ref.own().mass().fromBuffer(iSamples.mass());
        ref.own().area().fromBuffer(iSamples.area());
        ref.own().cr().fromBuffer(iSamples.cr());
        ref.own().cd().fromBuffer(iSamples.cd());
    }

    /**
     * @brief Copy assignment is forbidden
     *
     */
    Orbit& operator=(const Orbit& other) = delete;

    /**
     * @brief Move assignment operator
     *
     */
    Orbit& operator=(Orbit&& other)
    {
        base_ = std::move(other.base_);
        own_  = std::move(other.own_);

        return *this;
    }

    /**
     * @brief Return the number of samples in the metadata
     *
     * @return NSamples
     *
     */
    mSize_t size() const { return base_.size(); }

    /**
     * @brief Copy the metadata from host to device
     *
     * @param stream (optional) CUDA stream for asynchronous copy
     *
     */
    void upload(const cudaStream_t& stream = 0)
    {
        base_.upload(stream);
        own_.upload(stream);
    }

    /**
     * @brief Copy the metadata from device to host
     *
     * @param stream (optional) CUDA stream for asynchronous copy
     *
     */
    void download(const cudaStream_t& stream = 0)
    {
        base_.download(stream);
        own_.download(stream);
    }

    /** @brief Clear data from the GPU */
    void clearDevice()
    {
        base_.clearDevice();
        own_.clearDevice();
    }

    /** @brief Expose base members */
    BaseT& base() { return base_; }
    const BaseT& base() const { return base_; }

    /** @brief Expose own members */
    OwnT& own() { return own_; }
    const OwnT& own() const { return own_; }

    /**
     * @brief Return a non-owning `paraHPOP::model::samples::RefOrbit` that
     * allows for operations with the host data
     *
     * @return hostRefOrbitMetaData
     *
     */
    GRef hostRef() const
    {
        return GRef::make(base_.hostRef(), OwnT::GRef::make(own_.hostRef()));
    }

    /**
     * @brief Return a non-owning `paraHPOP::model::samples::RefOrbit` that
     * allows for operations with the device data
     *
     * @return deviceRefOrbitMetaData
     *
     */
    GRef deviceRef() const
    {
        return GRef::make(
            base_.deviceRef(), OwnT::GRef::make(own_.deviceRef()));
    }

    /** @brief Clone these metadata */
    Self clone() const
    {
        return Self(std::move(base_.clone()), std::move(own_.clone()));
    }

protected:
    BaseT base_;
    OwnT own_;
};

} // namespace metadata
} // namespace samples
} // namespace model
} // namespace paraHPOP