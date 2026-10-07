#pragma once

#include "feta/buffer/scalar/Array.h"
#include "feta/buffer/detail/IndexTraits.h"
#include "feta/core/SampleIndex.h"
#include "feta/core/TextureHelpers.h"
#include <type_traits>

#ifndef FETA_CPU_ONLY
#include <cuda.h>
#endif

namespace feta {
namespace scalar {
namespace texture {
namespace detail {

/** @brief Switcher for std::vector-based construction (booleans cannot use
 * memcpy)*/
template<typename DataT>
struct Type {
    static constexpr bool isBool = false;
};
template<>
struct Type<bool> {
    static constexpr bool isBool = true;
};

/* Texture fetcher */
#ifndef FETA_CPU_ONLY
template<typename DataT>
DEVICE()
inline DataT texFetch(const idx_t& idx, const cudaTextureObject_t& tex_)
{
    if constexpr (helper::Fetch<DataT>::needsConversion) {
        /* Float and double have different conversions*/
        typename helper::Fetch<DataT>::T result
            = tex1Dfetch<helper::Fetch<DataT>::T>(tex_, idx);
        if constexpr (helper::DoublePrecision<DataT>::T) {
            /* This is a double */
            DataT out = __hiloint2double(result.y, result.x);
            return out;
        } else { /* This is a float */
            DataT out = __int_as_float(result);
            return out;
        }
    } else {
        /* Simple type casting is enough */
        DataT out = tex1Dfetch<helper::Fetch<DataT>::T>(tex_, idx);
        return out;
    }
}
#endif

/** @brief Struct to compile-time gate volatile pointers */
template<typename DataT, bool Volatile = false>
struct MaybeVolatile {
    using T = DataT*;
};
template<typename DataT>
struct MaybeVolatile<DataT, true> {
    using T = volatile DataT*;
};

/** @brief Pointer and texture switcher */
template<typename DataT, bool UseTexture, bool Volatile = false>
struct Pointer {
    using T                 = typename MaybeVolatile<DataT, Volatile>::T;
    static constexpr T Null = nullptr;
};
#ifndef FETA_CPU_ONLY
template<typename DataT>
struct Pointer<DataT, true, false> {
#ifdef __CUDA_ARCH__
    using T                 = cudaTextureObject_t;
    static constexpr T Null = 0;
#else
    using T                 = typename MaybeVolatile<DataT, false>::T;
    static constexpr T Null = nullptr;
#endif
};
template<typename DataT>
struct Pointer<DataT, true, true> {
#ifdef __CUDA_ARCH__
    using T                 = cudaTextureObject_t;
    static constexpr T Null = 0;
#else
    using T                 = typename MaybeVolatile<DataT, true>::T;
    static constexpr T Null = nullptr;
#endif
};
#endif

/** @brief Forward declaration of RefArray.  Default for ``MaybeVolatile``
 * lives on the actual class declaration below — declaring it here too
 * would be an illegal redefinition of the default. */
template<typename DataT, bool work, bool UseTexture, bool MaybeVolatile>
class RefArray;

/** @brief POD data handle */
template<typename DataT, bool work, bool MaybeVolatile = false>
struct Handle {
    /** @brief Volatile is allowed only with work */
    static constexpr bool Volatile = MaybeVolatile && work;
    using RDataT   = typename helper::DataReturn<DataT, false, Volatile>::T;
    using PointerT = typename Pointer<DataT, false, Volatile>::T;
    using Self     = Handle<DataT, work, Volatile>;
    /** @brief Expose the associated standard type */
    using BufT = buffer::scalar::Array<DataT>;
/** @brief Expose the stream type */
#ifndef FETA_CPU_ONLY
    using StreamT = cudaStream_t;
#else
    using StreamT = int;
#endif

    /** @brief The inner pointer (or tex reference) */
    PointerT ptr = Pointer<DataT, false, Volatile>::Null;

    /** @brief Access the given data */
    DEVICEHOST() inline decltype(auto) operator[](const idx_t& idx) const
    {
#ifdef __CUDA_ARCH__
        /* Shared memory - return by reference to avoid register pressure */
        if constexpr (!work && helper::HasLdg<DataT>::value) {
            return __ldg(ptr + idx);
        } else {
            return ptr[idx];
        }
#else
        return ptr[idx];
#endif
    }
    DEVICEHOST() inline decltype(auto) operator[](const SampleIndex& i) const
    {
        if constexpr (work) {
            return (*this)[i.work()];
        } else {
            return (*this)[i.global()];
        }
    }
    DEVICEHOST() inline RDataT operator[](const idx_t& idx) { return ptr[idx]; }
    DEVICEHOST() inline RDataT operator[](const SampleIndex& i)
    {
        if constexpr (work) {
            return ptr[i.work()];
        } else {
            return ptr[i.global()];
        }
    }

    /** @brief Return the data pointed to */
    DEVICEHOST() PointerT data() const { return ptr; }

    /** @brief Fetch the data from the given RefArray */
    void fetch(const RefArray<DataT, work, false, Volatile>& other,
        const StreamT& stream = 0);

    /** @brief Fetch the data from the given other handle and size */
    void fetch(const Handle<DataT, work, Volatile>& other, const idx_t& size,
        const StreamT& stream = 0);

    /** @brief Copy the data from the given buffer */
    void fromBuffer(const BufT& buf);

    /** @brief Copy the data to the given buffer */
    void toBuffer(BufT& buf) const;

    /** @brief Return whether this is a host reference */
    bool isHostRef() const;
};

/** @brief Handle Switcher */
template<typename DataT, bool work, bool UseTexture, bool MaybeVolatile = false>
struct HandleSwitcher {
    using T = Handle<DataT, work, MaybeVolatile>;
};

#ifndef FETA_CPU_ONLY
/** @brief Texture POD data handle */
template<typename DataT>
struct TextureHandle {
    using PointerT = Pointer<DataT, true>::T;
    using Self     = TextureHandle<DataT>;

    /** @brief The inner pointer (or tex reference) */
    PointerT ptr = Pointer<DataT, true>::Null;
    /** @brief The offset within the data */
    idx_t texOffset = 0;

    /** @brief Access the given data */
    DEVICEHOST() inline DataT operator[](const idx_t& idx) const
    {
#ifdef __CUDA_ARCH__
        return texFetch<DataT>(idx + texOffset, ptr);
#else
        return ptr[idx + texOffset];
#endif
    }
    DEVICEHOST() inline DataT operator[](const SampleIndex& i) const
    {
#ifdef __CUDA_ARCH__
        return texFetch<DataT>(i.global() + texOffset, ptr);
#else
        return ptr[i.global() + texOffset];
#endif
    }
};

template<typename DataT, bool work>
struct HandleSwitcher<DataT, work, true> {
    using T = TextureHandle<DataT>;
};
#endif

/**
 * @brief Non-owning reference to scalar::texture::Array
 *
 */
template<typename DataT, bool work, bool UseTexture, bool MaybeVolatile = false>
class RefArray {
public:
    static constexpr bool Volatile = MaybeVolatile && work;

private:
    using RDataT
        = typename helper::DataReturn<DataT, UseTexture, Volatile>::T;
    using TypeT  = detail::Type<DataT>;
    using Self   = RefArray<DataT, work, UseTexture, MaybeVolatile>;

public:
/** @brief Expose the texture and stream types */
#ifndef FETA_CPU_ONLY
    using TexT    = cudaTextureObject_t;
    using StreamT = cudaStream_t;
#else
    using TexT    = int;
    using StreamT = int;
#endif

    /** @brief Expose the pointer type
     *
     * ``data_`` is the regular memory pointer regardless of UseTexture (the
     * texture object lives in the separate ``tex_`` field).  Use the
     * non-texture, volatility-aware ``Pointer`` so ``data_[idx]`` indexing
     * stays valid for work-refs. */
    using PointerT = typename Pointer<DataT, false, Volatile>::T;

    /** @brief Expose the associated standard type */
    using BufT = buffer::scalar::Array<DataT>;

    /** @brief Expose array item type */
    using ItemT = DataT;
    /** @brief Expose the Component type */
    using ComponentT = DataT;

    /** @brief Expose the buffer info type */
    using BufInfoT = typename BufT::InfoT;

    /** @brief Expose view and const view types */
    using ViewT      = DataT&;
    using ConstViewT = const DataT&;

    /** @brief Expose the handle type */
    using HandleT =
        typename HandleSwitcher<DataT, work, UseTexture, MaybeVolatile>::T;

    /**
     * @brief Square brackets operator - idx_t
     *
     */
    DEVICEHOST() decltype(auto) operator[](const idx_t& idx) const
    {
#ifdef __CUDA_ARCH__
        FETA_GPU_ASSERT(idx < this->size(), ::feta::err::OUT_OF_RANGE_SCALAR);
        /* Shared memory - return by reference to avoid register pressure */
        /* Texture */
        if constexpr (UseTexture && !work) {
            return texFetch<DataT>(idx + texOffset_, tex_);
        }
        /* Global memory - use __ldg which returns by value if available */
        else if constexpr (helper::HasLdg<DataT>::value && !work) {
            return __ldg(data_ + idx);
        } else {
            return data_[idx];
        }
#else
        FETA_ASSERT(idx < this->size(),
            "attempt to access elements out of range in Scalar Array");
        return data_[idx];
#endif
    }
    /**
     * @brief Square brackets operator - SampleIndex
     *
     */
    DEVICEHOST() DataT operator[](const SampleIndex& idx) const
    {
        /* Host code uses the normal [] operator */
        if constexpr (work) {
            return (*this)[idx.work()];
        } else {
            return (*this)[idx.global()];
        }
    }

    /**
     * @brief Square brackets operator - idx_t
     *
     */
    DEVICEHOST() RDataT operator[](const idx_t& idx)
    {
#ifdef __CUDA_ARCH__
        FETA_GPU_ASSERT(idx < this->size(), ::feta::err::OUT_OF_RANGE_SCALAR);
        /* texture */
        static_assert(!UseTexture,
            "Non-const access not supported for texture references");
#else
        FETA_ASSERT(idx < this->size(),
            "attempt to access elements out of range in "
            "Scalar Array");
#endif
        return data_[idx];
    }
    /**
     * @brief Square brackets operator - SampleIndex
     *
     */
    DEVICEHOST() RDataT operator[](const SampleIndex& idx)
    {
        /* Host code uses the normal [] operator */
        if constexpr (work) {
            return (*this)[idx.work()];
        } else {
            return (*this)[idx.global()];
        }
    }

    /** @brief Returns the number of elements in this array. */
    DEVICEHOST() inline idx_t size() const { return size_; }

    /** @brief Return a data handler */
    DEVICEHOST() HandleT handle() const
    {
        HandleT h;
#ifdef __CUDA_ARCH__
        if constexpr (UseTexture && !work) {
            h.ptr       = tex_;
            h.texOffset = texOffset_;
        } else {
            h.ptr = data_;
        }
#else
        h.ptr = data_;
#endif
        return h;
    }

    /** @brief Return a pointer to the underlying scalar data. */
    DEVICEHOST() PointerT data() const { return data_; }

    /**
     * @brief Returns a pointer to the element following the last element in
     * the scalar array.
     */
    DEVICEHOST() PointerT end() const { return data_ + size_; }

    /**
     * @brief Minimum size (in elements of type `DataType`) required for
     * allocation on an arbitrary buffer, if the object were to be
     * constructed for `arrayLen` elements.
     */
    DEVICEHOST() static size_t bufSize(const size_t& arrayLen)
    {
        return arrayLen;
    }

    /**
     * @brief Minimum size (in bytes) required for allocation on an
     * arbitrary buffer, if the object were to be constructed for `arrayLen`
     * elements.
     */
    DEVICEHOST() static size_t bufBytes(const size_t& arrayLen)
    {
        return bufSize(arrayLen) * sizeof(DataT);
    }

    /**
     * @brief Get work reference
     *
     */
    DEVICEHOST()
    static RefArray<DataT, true, false> workRef(
        DataT* const buf, const idx_t& size)
    {
        return { buf, size, 0, 0 };
    }

    /**
     * @brief Get a volatile work reference.
     *
     * Same as ``workRef`` except the returned array's inner pointer is
     * ``volatile DataT*``.  Use this in shared-memory shmem-scratch
     * patterns where the compiler would otherwise route reads through
     * a stack-backed mirror (see ``[[sh-phasor-shmem-failed]]``).
     */
    DEVICEHOST()
    static RefArray<DataT, true, false, true> volatileWorkRef(
        DataT* const buf, const idx_t& size)
    {
        return { buf, size, 0, 0 };
    }

    /**
     * @brief Return texture
     *
     */
    DEVICEHOST() TexT tex() const { return tex_; }

    /** @brief Create a new global reference to this scalar array */
    DEVICEHOST() RefArray clone() const { return *this; }

#ifndef FETA_CPU_ONLY

    /** @brief Async memcpy from host to the given device array */
    void upload(RefArray& other, const StreamT& stream = 0) const
    {
        assertHost_();
        other.assertDevice_();
        assertSize_(other.size());
        FETA_CHECK(cudaMemcpyAsync(other.data(), data(), size() * sizeof(DataT),
            cudaMemcpyHostToDevice, stream));
    }

    /** @brief Async memcpy from device to this data */
    void download(const RefArray& other, const StreamT& stream = 0)
    {
        assertHost_();
        other.assertDevice_();
        assertSize_(other.size());
        FETA_CHECK(cudaMemcpyAsync(data(), other.data(), size() * sizeof(DataT),
            cudaMemcpyDeviceToHost, stream));
    }

#endif

    /** @brief Generic fetch from the other ref array */
    void fetch(
        const RefArray& other, [[maybe_unused]] const StreamT& stream = 0)
    {
        const DataT* src = other.data();
#ifndef FETA_CPU_ONLY
        if (this->isHostRef()) {
            if (other.isHostRef()) {
                FETA_CHECK(cudaMemcpyAsync(data(), src, size() * sizeof(DataT),
                    cudaMemcpyHostToHost, stream));
            } else {
                FETA_CHECK(cudaMemcpyAsync(data(), src, size() * sizeof(DataT),
                    cudaMemcpyDeviceToHost, stream));
            }
        } else {
            if (isHostRef(src)) {
                FETA_CHECK(cudaMemcpyAsync(data(), src, size() * sizeof(DataT),
                    cudaMemcpyHostToDevice, stream));
            } else {
                FETA_CHECK(cudaMemcpyAsync(data(), src, size() * sizeof(DataT),
                    cudaMemcpyDeviceToDevice, stream));
            }
        }
#else
        std::copy(src, src + size_, data_);
#endif
    }

    /** @brief Generic fetch from the other handle array */
    void fetch(const HandleT& other, [[maybe_unused]] const StreamT& stream = 0)
    {
        const DataT* src = other.data();
#ifndef FETA_CPU_ONLY
        if (this->isHostRef()) {
            if (isHostRef(src)) {
                FETA_CHECK(cudaMemcpyAsync(data(), src, size() * sizeof(DataT),
                    cudaMemcpyHostToHost, stream));
            } else {
                FETA_CHECK(cudaMemcpyAsync(data(), src, size() * sizeof(DataT),
                    cudaMemcpyDeviceToHost, stream));
            }
        } else {
            if (isHostRef(src)) {
                FETA_CHECK(cudaMemcpyAsync(data(), src, size() * sizeof(DataT),
                    cudaMemcpyHostToDevice, stream));
            } else {
                FETA_CHECK(cudaMemcpyAsync(data(), src, size() * sizeof(DataT),
                    cudaMemcpyDeviceToDevice, stream));
            }
        }
#else
        std::copy(src, src + size_, data_);
#endif
    }

    /** @brief Read the data from buffer */
    void fromBuffer(const BufT& buf)
    {
        /* Assert the size compatibility */
        FETA_ASSERT(size_ == buf.size(),
            "Incompatible std::vector and feta::scalar::Array sizes");
        std::copy(buf.data(), buf.data() + size_, data_);
    }

    /** @brief Read the data from the given buffer info */
    void fromBuffer(const BufInfoT& i)
    {
        /* Assert the size compatibility */
        FETA_ASSERT(size_ == i.size(),
            "Incompatible feta::buffer::Info and feta::scalar::Array "
            "sizes");
        std::copy(i.ptr, i.ptr + size_, data_);
    }

    /** @brief Read the data from buffer (slice) */
    void fromBuffer(const typename BufT::SliceT& buf)
    {
        /* Assert the size compatibility */
        FETA_ASSERT(size_ == buf.size(),
            "Incompatible std::vector and feta::scalar::Array sizes");
        for (idx_t i = 0; i < size_; i++)
            (*this)[i] = buf[i];
    }

    /** @brief Write the data to buffer (void) */
    void toBuffer(BufT& buf) const
    {
        /* Assert the size compatibility */
        FETA_ASSERT(size_ == buf.size(),
            "Incompatible std::vector and feta::scalar::Array sizes");
        std::copy(data_, data_ + size_, buf.data());
    }

    /** @brief Write the data to buffer (slice) */
    void toBuffer(typename BufT::SliceT& buf) const
    {
        /* Assert the size compatibility */
        FETA_ASSERT(size_ == buf.size(),
            "Incompatible std::vector and feta::scalar::Array sizes");
        for (idx_t i = 0; i < size_; i++)
            buf[i] = (*this)[i];
    }

    /** @brief Write the data to buffer (return) */
    BufT toBuffer() const
    {
        BufT buf(this->size());
        toBuffer(buf);
        return buf;
    }

    /** @brief Return `true` if this reference points to host (non-device)
     * memory. */
    bool isHostRef() const { return isHostRef(data_); }

    /** @brief Return whether the given input pointer is a host reference */
    static bool isHostRef(const DataT* const ptr)
    {
#ifdef FETA_CPU_ONLY
        return true;
#else
        cudaPointerAttributes attributes;
        cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);
        if (err != cudaSuccess) {
            /* assume host pointer if error */
            cudaGetLastError(); // clear the error
            return true;
        }
        if (attributes.type == cudaMemoryTypeDevice)
            return false;
        else
            return true;
#endif
    }

    /** @brief Return the first element of this array */
    DEVICEHOST() decltype(auto) front() const { return (*this)[0]; }

    /** @brief Return the last element of this array */
    DEVICEHOST() decltype(auto) back() const { return (*this)[size_ - 1]; }

    /** @brief Expose the texture offset */
    DEVICEHOST() idx_t texOffset() const { return texOffset_; }

    /** @brief Data members - made public for PODification.  ``data_`` is
     * volatility-aware via ``PointerT`` (``DataT*`` or ``volatile DataT*``
     * depending on ``Volatile``); ``nullptr`` implicit-converts to both. */
    PointerT data_   = nullptr;
    idx_t size_      = 0;
    TexT tex_        = 0;
    idx_t texOffset_ = 0;

private:
    /** @brief Assert that this is a host reference */
    void assertHost_() const
    {
        FETA_ASSERT(isHostRef(), "This is not a host reference");
    }

    /** @brief Assert that this is a device reference */
    void assertDevice_() const
    {
        FETA_ASSERT(!isHostRef(), "This is not a device reference");
    }

    /** @brief Assert that sizes are compatible */
    void assertSize_(const idx_t& sz) const
    {
        FETA_ASSERT(
            size_ == sz, "Incompatible sizes for reference-based memcpy");
    }
};

/* Finalize the implementations of fetch and fromBuffer for Handle.  The
 * 3rd template parameter (``MaybeVolatile``) is threaded through so the
 * out-of-class definition signatures match the in-class declarations. */
template<typename DataT, bool work, bool MaybeVolatile>
void Handle<DataT, work, MaybeVolatile>::fetch(
    const RefArray<DataT, work, false,
        Handle<DataT, work, MaybeVolatile>::Volatile>& other,
    const StreamT& stream)
{
    /* Create a ref array from this */
    RefArray<DataT, work, false,
        Handle<DataT, work, MaybeVolatile>::Volatile>
        ref = { ptr, other.size(), 0, 0 };
    /* call fetch */
    ref.fetch(other, stream);
}

template<typename DataT, bool work, bool MaybeVolatile>
void Handle<DataT, work, MaybeVolatile>::fetch(
    const Handle<DataT, work,
        Handle<DataT, work, MaybeVolatile>::Volatile>& other,
    const idx_t& size, const StreamT& stream)
{
    /* Create a ref array from this */
    RefArray<DataT, work, false,
        Handle<DataT, work, MaybeVolatile>::Volatile>
        ref = { ptr, size, 0, 0 };
    /* call fetch */
    ref.fetch(other, stream);
}

template<typename DataT, bool work, bool MaybeVolatile>
void Handle<DataT, work, MaybeVolatile>::fromBuffer(const BufT& buf)
{
    /* Create a ref array from this */
    RefArray<DataT, work, false,
        Handle<DataT, work, MaybeVolatile>::Volatile>
        ref = { ptr, buf.size(), 0, 0 };
    /* call fromBuffer */
    ref.fromBuffer(buf);
}

template<typename DataT, bool work, bool MaybeVolatile>
void Handle<DataT, work, MaybeVolatile>::toBuffer(BufT& buf) const
{
    /* Create a ref array from this */
    RefArray<DataT, work, false,
        Handle<DataT, work, MaybeVolatile>::Volatile>
        ref = { ptr, buf.size(), 0, 0 };
    /* call fromBuffer */
    ref.toBuffer(buf);
}

template<typename DataT, bool work, bool MaybeVolatile>
bool Handle<DataT, work, MaybeVolatile>::isHostRef() const
{
    return RefArray<DataT, work, false,
        Handle<DataT, work, MaybeVolatile>::Volatile>::isHostRef(ptr);
}

} // namespace detail
} // namespace texture
} // namespace scalar

/* Add the index type traits (value follows the element-type predicate, so one
   partial specialization replaces the per-int-type list). */
namespace buffer {
namespace detail {
template<typename T, bool work, bool UseTexture>
struct IsBoolIndex<feta::scalar::texture::detail::RefArray<T, work, UseTexture>> {
    static constexpr bool value = scalar::IsBool<T>::value;
};
template<typename T, bool work, bool UseTexture>
struct IsIntIndex<feta::scalar::texture::detail::RefArray<T, work, UseTexture>> {
    static constexpr bool value = scalar::IsInt<T>::value;
};
} // namespace detail
} // namespace buffer
} // namespace feta