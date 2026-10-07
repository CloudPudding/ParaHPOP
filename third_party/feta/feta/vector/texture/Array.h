#pragma once

#include "feta/vector/texture/RefArray.h"
#include <iostream>

namespace feta {
namespace vector {
namespace texture {

/* Declare the container type */
template<typename DataT, idx_t VectorDim>
struct Container {
#ifndef FETA_CPU_ONLY
    using T = core::memory::EnhancedContainer<DataT,
        core::memory::Device::CUDA_HOST, core::memory::Device::CUDA_DEVICE,
        VectorDim>;
#else
    using T = core::memory::EnhancedContainer<DataT, core::memory::Device::CPU,
        core::memory::Device::NONE, VectorDim>;
#endif
};

/**
 * @brief Owning array of n-dimensional vectors which has a corresponding
 * copy on the GPU. Uses `VecNTArray<DataType>` internally.
 *
 * This class is only meant for use on the host. When passing a `VecNTArray`
 * into a `__global__` kernel, use either `VecNTArray::deviceRef()`, to obtain
 * a non-owning reference to the device-allocated array, or
 * `VecNTArray::textureRef()`, to obtain a non-owning reference to the texture
 * binding of the device-allocated array
 */
template<typename DataT, idx_t VectorDim, bool UseTexture>
class Array : public Container<DataT, VectorDim>::T {
    using Self    = Array;
    using ParentT = typename Container<DataT, VectorDim>::T;

public:
/** @brief Expose the texture and stream types */
#ifndef FETA_CPU_ONLY
    using TexT    = cudaTextureObject_t;
    using StreamT = cudaStream_t;
#else
    using TexT    = int;
    using StreamT = int;
#endif
    /** @brief Buffer type */
    using BufT     = buffer::vector::Array<DataT, VectorDim>;
    using BufInfoT = buffer::Info<DataT>;

    /** @brief Generic work/global non-owning reference type */
    template<bool work, bool MaybeVolatile = false>
    using Ref = detail::RefArray<DataT, VectorDim, work, UseTexture,
        MaybeVolatile>;
    /** @brief Global reference type */
    using GRef = Ref<false>;
    /** @brief Work reference type */
    using WRef = Ref<true>;

    /** @brief Datatype of the vector elements */
    using ComponentT = DataT;

    /** @brief Number of components in the vectors */
    static constexpr idx_t VecDims = VectorDim;

    /** @brief Factory method to construct a flexible array */
    static Array flexible() { return Array(true); }

    /** @brief Factory method to borrow from a buffer type */
    static Array borrow(const BufT& buf)
    {
        return Array(std::move(ParentT::template borrow(buf.container())));
    }

    /**
     * @brief Host constructor. Host memory is allocated and will be
     * deallocated in destructor. Values are initialized to 0.
     */
    Array(const idx_t size, const DataT& initVal = 0, const idx_t& capacity = 0,
        const bool& flexible = false)
        : ParentT{ capacity == 0 ? size : capacity, initVal, flexible }
        , nVecs_{ size }
    {
    }

    /** @brief Move construction is allowed. */
    Array(Array&& other)
        : ParentT{ std::move(other) }
        , nVecs_{ std::exchange(other.nVecs_, 0) }
    {
    }

    /** @brief Move construct from the parent container */
    Array(ParentT&& other)
        : ParentT{ std::move(other) }
        , nVecs_{ this->interface_.size() }
    {
    }

    /** @brief Construct from Buffer */
    explicit Array(const BufT& buf)
        : Array{ buf.size() }
    {
        this->hostRef().fromBuffer(buf);
    }


    /** @brief Construct copying the data from the given buffer slice*/
    Array(const typename BufT::SliceT& buf)
        : Array{ buf.size() }
    {
        this->hostRef().fromBuffer(buf);
    }

    /** @brief Move-assign from another `Array`, transferring ownership of all buffers. */
    Array& operator=(Array&& other)
    {
        ParentT::operator=(std::move(other));
        nVecs_ = std::exchange(other.nVecs_, 0);

        return *this;
    }

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

    /** @brief Clear the device data */
    void clearDevice() { this->clearWorkable(); }

    /** @brief Return the array capacity (i.e. the container size) */
    inline idx_t capacity() const { return this->interface_.size(); }

    /** @brief Returns the number of vectors in this array. */
    inline idx_t size() const { return nVecs_; }

    /** @brief Return pointer to host data. Is never null. */
    inline DataT* getHostData() const { return this->data(); }

    /** @brief Return a RefVecNTArray pointing to the host array. */
    GRef hostRef(const bool& unlocked = false) const
    {
        idx_t sz = unlocked ? this->capacity() : nVecs_;
        GRef r;
        r.data_      = getHostData();
        r.nVecs_     = sz;
        r.dimOffset_ = this->capacity();
        r.tex_       = 0;
        r.texOffset_ = 0;
        return r;
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

    /** @brief Return a RefVecNTArray pointing to the device array. */
    GRef deviceRef(const bool& unlocked = false) const
    {
        idx_t sz = unlocked ? this->capacity() : nVecs_;
        GRef r;
        r.data_      = getDeviceData();
        r.nVecs_     = sz;
        r.dimOffset_ = this->capacity();
        if constexpr (UseTexture) {
            r.tex_       = this->workable_.tex().native();
            r.texOffset_ = 0;
        } else {
            r.tex_       = 0;
            r.texOffset_ = 0;
        }
        return r;
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
        i.ndim = 2;
        i.shape.push_back(VectorDim);
        i.shape.push_back(size());
        i.strides.push_back(size());
        i.strides.push_back(1);
        return i;
    }

    /** @brief Create an array clone, copying the data */
    Array clone() const { return Array(*this); }

protected:
    /** @brief Default constructor for internal use; allocates nothing. */
    Array(const bool& flexible = false)
        : ParentT{ flexible }
    {
    }

    /** @brief Copy constructor for internal use only */
    Array(const Array& other)
        : ParentT{ other }
        , nVecs_{ other.nVecs_ }
    {
    }

    /** @brief Copy assignment for internal use only */
    Array& operator=(const Array& other)
    {
        ParentT::operator=(other);
        nVecs_ = other.nVecs_;
        return *this;
    }

    idx_t nVecs_ = 0;
};

} // namespace texture
} // namespace vector
} // namespace feta