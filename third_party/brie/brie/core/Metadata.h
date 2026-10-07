#pragma once

#include "brie/core/RefMetadata.h"

namespace brie {
namespace core {

/** @brief the Metadata */
template<idx_t INTSIZE, idx_t HANDLESIZE, idx_t REALSIZE>
class Metadata {
    using Self = Metadata<INTSIZE, HANDLESIZE, REALSIZE>;

public:
    /** @brief Expose the data types*/
    using IntArrayT    = feta::vector::texture::Array<NaifId, INTSIZE, false>;
    using HandleArrayT = feta::vector::texture::Array<idx_t, HANDLESIZE, false>;
    using RealArrayT   = feta::vector::texture::Array<Real, REALSIZE, false>;

    /** @brief Expose the reference types.  ``MaybeVolatile`` flows
     * down to the inner FETA vector refs (Int/Handle/Real). */
    template<bool work, bool MaybeVolatile = false>
    using Ref = RefMetadata<INTSIZE, HANDLESIZE, REALSIZE, work, MaybeVolatile>;
    /** @brief Global reference type */
    using GRef = Ref<false>;
    using WRef = Ref<true>;
    /** @brief Volatile work reference */
    using VolatileRef = Ref<true, true>;

    /** @brief Expose the stream and texture types */
    using StreamT = typename IntArrayT::StreamT;
    using TexT    = typename IntArrayT::StreamT;

    /** @brief Expose the integer data size */
    static constexpr idx_t IntSize = INTSIZE;
    /** @brief Expose the handle data size */
    static constexpr idx_t HandleSize = HANDLESIZE;
    /** @brief Expose the real data size */
    static constexpr idx_t RealSize = REALSIZE;

    /** @brief Flexible factory method */
    static Metadata flexible()
    {
        return Metadata(std::move(IntArrayT::flexible()),
            std::move(HandleArrayT::flexible()),
            std::move(RealArrayT::flexible()));
    }

    /**
     * @brief Basic host constructor
     *
     */
    Metadata(const idx_t& Nbodies)
        : intMembers_{ Nbodies }
        , handleMembers_{ Nbodies }
        , realMembers_{ Nbodies }
    {
    }

    /**
     * @brief Copy constructor is forbidden
     *
     */
    Metadata(Metadata& other)       = delete;
    Metadata(const Metadata& other) = delete;

    /**
     * @brief Move constructor
     *
     */
    Metadata(Metadata&& other)
        : intMembers_{ std::move(other.intMembers_) }
        , handleMembers_{ std::move(other.handleMembers_) }
        , realMembers_{ std::move(other.realMembers_) }
    {
    }

    /** @brief Move constructor from all data members */
    Metadata(IntArrayT&& intMembers, HandleArrayT&& handleMembers,
        RealArrayT&& realMembers)
        : intMembers_{ std::move(intMembers) }
        , handleMembers_{ std::move(handleMembers) }
        , realMembers_{ std::move(realMembers) }
    {
    }

    Metadata& operator=(const Metadata& other) = delete;

    /**
     * @brief Move assignment operator
     *
     */
    Metadata& operator=(Metadata&& other)
    {
        this->intMembers_    = std::move(other.intMembers_);
        this->handleMembers_ = std::move(other.handleMembers_);
        this->realMembers_   = std::move(other.realMembers_);

        return *this;
    }

    /**
     * @brief Return the number of bodies in these metadata
     *
     * @return Nbodies
     *
     */
    idx_t size() const { return this->intMembers_.size(); }

    /**
     * @brief Copy the metadata from host to device
     *
     * @param stream (Optional) CUDA stream for asynchronous copy
     *
     */
    void upload(const StreamT& stream = 0)
    {
        this->intMembers_.upload(stream);
        this->handleMembers_.upload(stream);
        this->realMembers_.upload(stream);
    }

#ifndef BRIE_CPU_ONLY
    /**
     * @brief Copy the metadata from device to host
     *
     * @param stream (Optional) CUDA stream for asynchronous copy
     *
     */
    void download(const StreamT& stream = 0)
    {
        this->intMembers_.download(stream);
        this->handleMembers_.download(stream);
        this->realMembers_.download(stream);
    }

    /** @brief Clear data from the GPU */
    void clearDevice()
    {
        this->intMembers_.clearDevice();
        this->handleMembers_.clearDevice();
        this->realMembers_.clearDevice();
    }
#endif
    /**
     * @brief Return a non-owning `brie::states::metadata::RefMetadata` that
     * allows for operations with the host data
     *
     * @return hostRefEphUnit
     *
     */
    GRef hostRef() const
    {
        return GRef::make(this->intMembers_.hostRef(),
            this->handleMembers_.hostRef(), this->realMembers_.hostRef());
    }
    GRef ref() const { return hostRef(); }

#ifndef BRIE_CPU_ONLY
    /**
     * @brief Return a non-owning `brie::states::metadata::RefMetadata` that
     * allows for operations with the device data
     *
     * @return deviceRefEphUnit
     *
     */
    GRef deviceRef() const
    {
        return GRef::make(this->intMembers_.deviceRef(),
            this->handleMembers_.deviceRef(), this->realMembers_.deviceRef());
    }
#endif

    /** @brief Clone this object. */
    Metadata clone() const
    {
        return Metadata(std::move(intMembers_.clone()),
            std::move(handleMembers_.clone()), std::move(realMembers_.clone()));
    }

    /** @brief Expose the int members */
    IntArrayT& intMembers() { return intMembers_; }
    const IntArrayT& intMembers() const { return intMembers_; }

    /** @brief Expose the handle members */
    HandleArrayT& handleMembers() { return handleMembers_; }
    const HandleArrayT& handleMembers() const { return handleMembers_; }

    /** @brief Expose the real members */
    RealArrayT& realMembers() { return realMembers_; }
    const RealArrayT& realMembers() const { return realMembers_; }

protected:
    /** @brief Integer metadata members **/
    IntArrayT intMembers_;
    /** @brief Handle metadata members **/
    HandleArrayT handleMembers_;
    /** @brief Double metadata members **/
    RealArrayT realMembers_;
};
} // namespace core
} // namespace brie