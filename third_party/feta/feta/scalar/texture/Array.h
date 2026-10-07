#pragma once

#include "feta/core/memory.h"
#include "feta/scalar/texture/RefArray.h"

namespace feta {
namespace scalar {
namespace texture {

/* Declare the container type */
template<typename DataT>
struct Container {
#ifndef FETA_CPU_ONLY
    using T = core::memory::EnhancedContainer<DataT,
        core::memory::Device::CUDA_HOST, core::memory::Device::CUDA_DEVICE>;
#else
    using T = core::memory::EnhancedContainer<DataT, core::memory::Device::CPU,
        core::memory::Device::NONE>;
#endif
};

/**
 * @brief Owning array of scalar values which has a corresponding copy on the
 * GPU.
 *
 * This class manages two arrays: one on the host and one on the GPU.
 *
 * The host memory is managed RAII-style: allocation happens in the
 * constructor, and deallocation in the destructor.
 *
 * Device memory is allocated upon calling memcpyHostToDevice from the host, and
 * deallocated in the destructor.
 *
 * This class is only meant for use on the host. When passing a `ScalarArray`
 * into a `__global__` kernel, use `ScalarArray::deviceRef()` to obtain a
 * non-owning reference to the device-allocated array.
 */
template<typename DataT, bool UseTexture>
class Array : public Container<DataT>::T {
    using Self    = Array;
    using ParentT = typename Container<DataT>::T;

public:
/** @brief Expose the texture and stream types */
#ifndef FETA_CPU_ONLY
    using TexT    = cudaTextureObject_t;
    using StreamT = cudaStream_t;
#else
    using TexT    = unsigned int;
    using StreamT = unsigned int;
#endif

    /** @brief Expose the associated standard type */
    using BufT     = buffer::scalar::Array<DataT>;
    using BufInfoT = buffer::Info<DataT>;

    /* Generic Reference type */
    template<bool work, bool MaybeVolatile = false>
    using Ref = detail::RefArray<DataT, work, UseTexture, MaybeVolatile>;
    /* Global reference type */
    using GRef = Ref<false>;
    /* Work reference type */
    using WRef = Ref<true>;

    /** @brief Factory method to construct a flexible array */
    static Array flexible() { return Array(true); }

    /** @brief Factory method to borrow from a buffer type */
    static Array borrow(const BufT& buf)
    {
        return Array(std::move(ParentT::template borrow(buf.container())));
    }

    /**
     * @brief Host constructor that distinguishes size and capacity. Capacity is
     * used with the container, size is what governs the RefArray behaviour
     *
     */
    Array(const idx_t& size, const DataT& initVal = 0,
        const idx_t& capacity = 0, const bool& flexible = false)
        : ParentT{ capacity == 0 ? size : capacity, initVal, flexible }
        , size_{ size }
    {
    }

    /** @brief Construct from the given buffer info */
    Array(const BufInfoT& i)
        : Array{ i.size() }
    {
        this->hostRef().fromBuffer(i);
    }

    /** @brief Construct copying the data from the given
     * feta::buffer::scalar::Array<DataT> */
    explicit Array(const BufT& buf)
        : Array{ buf.size() }
    {
        this->hostRef().fromBuffer(buf);
    }

    /** @brief Construct copying the data from the given
     * feta::buffer::scalar::Array<DataT>::SliceT*/
    Array(const typename BufT::SliceT& buf)
        : Array{ buf.size() }
    {
        this->hostRef().fromBuffer(buf);
    }

    /**
     * @brief Move construction is allowed
     *
     */
    Array(Array&& other)
        : ParentT{ std::move(other) }
        , size_{ std::exchange(other.size_, 0) }
    {
    }

    /** @brief Move construct from the parent container */
    Array(ParentT&& other)
        : ParentT{ std::move(other) }
        , size_{ this->interface_.size() }
    {
    }

    /**
     * @brief Move assignment
     *
     */
    Array& operator=(Array&& other)
    {
        ParentT::operator=(std::move(other));
        size_ = std::exchange(other.size_, 0);
        return *this;
    }

    /**
     * @brief Destructor
     *
     */
    ~Array() { }

    /** @brief Return pointer to host data. Is never null. */
    inline DataT* getHostData() const { return this->data(); }

    /** @brief Return the array capacity (i.e. the container size) */
    inline idx_t capacity() const { return this->interface_.size(); }

    /** @brief Return the number of elements in this array. */
    inline idx_t size() const { return size_; }

    /** @brief Upload and bind to texture if required */
    void upload(const StreamT& stream = 0)
    {
        ParentT::upload(stream);
        if constexpr (UseTexture) {
            this->workable_.bindToTexture();
        }
    }
    void blockingUpload()
    {
        ParentT::blockingUpload();
        if constexpr (UseTexture) {
            this->workable_.bindToTexture();
        }
    }

    /**
     * @brief Get host reference. If unlocked is true, returns a reference with
     * size that equals the capacity, allowing the manipulation of off-size
     * elements
     *
     */
    GRef hostRef(const bool& unlocked = false) const
    {
        idx_t sz = unlocked ? this->capacity() : size_;
        return { this->getHostData(), sz, 0, 0 };
    }
    GRef ref() const { return hostRef(); }

#ifndef FETA_CPU_ONLY

    /**
     * @brief Return pointer to device data. May be null if called from the host
     * and memcpyHostToDevice was not called.
     */
    inline DataT* getDeviceData() const
    {
        static_assert(!ParentT::WorkableT::IsNone,
            "Cannot extract Device Data from None");
        return this->workable_.data();
    }

    /**
     * @brief Get device reference
     *
     */
    GRef deviceRef(const bool& unlocked = false) const
    {
        idx_t sz = unlocked ? this->capacity() : size_;
        if constexpr (UseTexture)
            return { this->getDeviceData(), sz, this->workable_.tex().native(),
                0 };
        else
            return { this->getDeviceData(), sz, 0, 0 };
    }

#endif

    /** @brief Return a buffer array */
    BufT buffer() const { return this->hostRef().toBuffer(); }

    /** @brief Copy to the given buffer */
    void toBuffer(BufT& buf) const { this->hostRef().toBuffer(buf); }

    /** @brief Copy to the given buffer (slice) */
    void toBuffer(typename BufT::SliceT& buf) const
    {
        this->hostRef().toBuffer(buf);
    }

    /** @brief Return the buffer info for this array */
    BufInfoT info() const
    {
        BufInfoT i;
        i.ptr  = this->data();
        i.ndim = 1;
        i.shape.push_back(size());
        i.strides.push_back(1);
        return i;
    }

    /** @brief Clear the device data */
    void clearDevice() { this->clearWorkable(); }

    /** @brief Create an array clone, copying the data */
    Array clone() const { return Array(*this); }

protected:
    /**
     * @brief Default constructor - internal use only
     *
     */
    Array(const bool& flexible = false)
        : ParentT{ flexible }
    {
    }

    /** @brief Copy constructor - internal use only */
    Array(const Array& other)
        : ParentT{ other }
        , size_{ other.size_ }
    {
    }

    /** @brief Copy assignment - internal use only */
    Array& operator=(const Array& other)
    {
        ParentT::operator=(other);
        size_ = other.size_;
        return *this;
    }

    /** @brief Actual size */
    idx_t size_ = 0;
};

} // namespace texture
} // namespace scalar
} // namespace feta