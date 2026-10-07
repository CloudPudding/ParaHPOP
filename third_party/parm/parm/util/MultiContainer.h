#pragma once

#include "parm/util/RefMultiContainer.h"

namespace parm {
namespace util {

/**
 * @brief Heterogeneous per-sample metadata container (integers, booleans, reals).
 *
 * Stores three FETA ``vector::Array`` members in Structure-of-Arrays layout,
 * one for each scalar type.  Supports asynchronous host↔device transfers and
 * exposes non-owning ``Ref`` wrappers for use in kernels.
 *
 * Copy construction and copy assignment are disabled; use move semantics or
 * ``clone()``.
 *
 * @tparam IntSize   Number of integer fields per sample.
 * @tparam BoolSize  Number of boolean fields per sample.
 * @tparam RealSize  Number of real (``double``) fields per sample.
 */
template<idx_t IntSize, idx_t BoolSize, idx_t RealSize>
class MultiContainer {
protected:
    /* Data types aliases */
    using VecIntArrayT  = feta::vector::Array<mInt_t, IntSize>;
    using VecBoolArrayT = feta::vector::Array<bool, BoolSize>;
    using VecRealArrayT = feta::vector::Array<Real, RealSize>;
    using Self          = MultiContainer;

public:
    /** @brief Generic work/global non-owning reference type.  The 2nd
     * ``MaybeVolatile`` parameter propagates the volatility request
     * down to the contained feta refs. */
    template<bool work, bool MaybeVolatile = false>
    using Ref
        = RefMultiContainer<work, IntSize, BoolSize, RealSize, MaybeVolatile>;
    /** @brief Global (device or host) reference type. */
    using GRef = Ref<false>;
    /** @brief Shared-memory (work) reference type. */
    using WRef = Ref<true>;
    /** @brief Volatile shared-memory work reference type. */
    using VolatileRef = Ref<true, true>;
    /** @brief Shared memory data type */
    using SharedMemDataT    = typename GRef::SharedMemDataT;
    using EndSharedMemDataT = typename GRef::EndSharedMemDataT;
    /** @brief generic data types */
    using IntT  = typename GRef::IntT;
    using RealT = typename GRef::RealT;

    /** @brief Default constructor is forbidden */
    MultiContainer() = delete;

    /** @brief Copy constructor is forbidden. */
    MultiContainer(MultiContainer& other)       = delete;
    MultiContainer(const MultiContainer& other) = delete;
    MultiContainer(VecIntArrayT& intMembers, VecBoolArrayT& boolMembers,
        VecRealArrayT& realMembers)
        = delete;
    MultiContainer(const VecIntArrayT& intMembers,
        const VecBoolArrayT& boolMembers, const VecRealArrayT& realMembers)
        = delete;

    /** @brief Move-construct from the data elements. */
    MultiContainer(VecIntArrayT&& intMembers, VecBoolArrayT&& boolMembers,
        VecRealArrayT&& realMembers)
        : intMembers_{ std::move(intMembers) }
        , boolMembers_{ std::move(boolMembers) }
        , realMembers_{ std::move(realMembers) }
    {
    }

    /** @brief Move constructor. */
    MultiContainer(MultiContainer&& other)
        : intMembers_{ std::move(other.intMembers_) }
        , boolMembers_{ std::move(other.boolMembers_) }
        , realMembers_{ std::move(other.realMembers_) }
    {
    }

    /**
     * @brief Construct with the given number of samples; allocate host memory.
     *
     * @param NSamples  Number of samples (rows in each SoA array).
     */
    MultiContainer(const idx_t& NSamples)
        : intMembers_{ NSamples }
        , boolMembers_{ NSamples }
        , realMembers_{ NSamples }
    {
    }

    /** @brief Copy-assignment is forbidden. */
    MultiContainer& operator=(MultiContainer& other)       = delete;
    MultiContainer& operator=(const MultiContainer& other) = delete;

    /** @brief Move assignment operator. */
    MultiContainer& operator=(MultiContainer&& other)
    {
        intMembers_  = std::move(other.intMembers_);
        boolMembers_ = std::move(other.boolMembers_);
        realMembers_ = std::move(other.realMembers_);

        return *this;
    }

    /** @brief Return the number of samples. */
    idx_t size() const { return intMembers_.size(); }

#ifndef PARM_CPU_ONLY

    /**
     * @brief Upload all metadata arrays from host to device.
     *
     * @param stream  CUDA stream for asynchronous copy (default 0).
     */
    void upload(const cudaStream_t& stream = 0)
    {
        intMembers_.upload(stream);
        boolMembers_.upload(stream);
        realMembers_.upload(stream);
    }

    /**
     * @brief Download all metadata arrays from device to host.
     *
     * @param stream  CUDA stream for asynchronous copy (default 0).
     */
    void download(const cudaStream_t& stream = 0)
    {
        intMembers_.download(stream);
        boolMembers_.download(stream);
        realMembers_.download(stream);
    }

    /** @brief Clear data from the GPU */
    void clearDevice()
    {
        intMembers_.clearDevice();
        boolMembers_.clearDevice();
        realMembers_.clearDevice();
    }

#endif

    /** @brief Expose the int members */
    VecIntArrayT& intMembers() { return this->intMembers_; }
    const VecIntArrayT& intMembers() const { return this->intMembers_; }

    /** @brief Expose the bool members */
    VecBoolArrayT& boolMembers() { return this->boolMembers_; }
    const VecBoolArrayT& boolMembers() const { return this->boolMembers_; }

    /** @brief Expose the real members */
    VecRealArrayT& realMembers() { return this->realMembers_; }
    const VecRealArrayT& realMembers() const { return this->realMembers_; }

    /** @brief Return a non-owning reference to the host data. */
    GRef hostRef() const
    {
        return GRef::make(intMembers_.hostRef(), boolMembers_.hostRef(),
            realMembers_.hostRef());
    }
    GRef ref() const { return hostRef(); }

#ifndef PARM_CPU_ONLY

    /** @brief Return a non-owning reference to the device data. */
    GRef deviceRef() const
    {
        return GRef::make(intMembers_.deviceRef(), boolMembers_.deviceRef(),
            realMembers_.deviceRef());
    }

#endif

    /** @brief Clone these metadata */
    Self clone() const
    {
        return Self(std::move(intMembers_.clone()),
            std::move(boolMembers_.clone()), std::move(realMembers_.clone()));
    }

protected:
    VecIntArrayT intMembers_;
    VecBoolArrayT boolMembers_;
    VecRealArrayT realMembers_;
};

} // namespace util
} // namespace parm