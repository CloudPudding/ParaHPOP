/* The core memory management module */
#pragma once

#include "feta/core/memory/Devices.h"

namespace feta {
namespace core {
namespace memory {

/** @brief Boolean concept */
template<typename T>
concept IsBoolean = std::is_same_v<T, bool>;

/** @brief Arithmetic concept */
template<typename T>
concept IsArithmetic = std::is_arithmetic_v<T>;

/** @brief Trivial concept */
template<typename T>
concept IsTrivial = std::is_trivially_default_constructible_v<T>
    && std::is_trivially_copyable_v<T>;

#ifndef FETA_CPU_ONLY
/** @brief Default data */
template<typename T>
DEVICE()
constexpr T DeviceDefault()
{
    if constexpr (IsBoolean<T>)
        return false;
    else if constexpr (IsArithmetic<T>)
        return 0;
    else
        return T();
}
#endif
/** @brief Default data */
template<typename T>
constexpr T Default()
{
    if constexpr (IsBoolean<T>)
        return false;
    else if constexpr (IsArithmetic<T>)
        return 0;
    else
        return T();
}

/** @brief Memory allocator class */
template<Device Target>
struct Allocator {

    /** @brief Allocate the memory */
    template<typename DataT>
    static inline void malloc(DataT** data, const idx_t& size,
        const bool& isBorrowed = false, const bool& manageRegistration = false)
    {
        static_assert(
            feta::core::expr::enumAlwaysFalse<static_cast<unsigned int>(
                Target)>::value,
            "Not implemented for this Device type");
    }

    /** @brief Free the memory previously allocated by `malloc`. */
    template<typename DataT>
    static inline void free(DataT* data, const idx_t& size,
        const bool& isBorrowed = false, const bool& manageRegistration = false)
    {
        static_assert(
            feta::core::expr::enumAlwaysFalse<static_cast<unsigned int>(
                Target)>::value,
            "Not implemented for this Device type");
    }

    /** @brief Bind to the given texture object */
    template<typename DataT, typename TexT>
    static inline void bindToTexture(DataT* data, TexT& tex, const idx_t& size)
    {
        static_assert(
            feta::core::expr::enumAlwaysFalse<static_cast<unsigned int>(
                Target)>::value,
            "Not implemented for this Device type");
    }

    /** @brief Destroy the given texture object */
    template<typename TexT>
    static inline void destroyTexture(TexT& tex)
    {
        static_assert(
            feta::core::expr::enumAlwaysFalse<static_cast<unsigned int>(
                Target)>::value,
            "Not implemented for this Device type");
    }
};

/* Specializations*/

/* None - reduces to no-ops */
template<>
template<typename DataT>
inline void Allocator<Device::NONE>::malloc([[maybe_unused]] DataT** data,
    [[maybe_unused]] const idx_t& size, [[maybe_unused]] const bool& isBorrowed,
    [[maybe_unused]] const bool& manageRegistration)
{
    /* do nothing */
}

template<>
template<typename DataT>
inline void Allocator<Device::NONE>::free([[maybe_unused]] DataT* data,
    [[maybe_unused]] const idx_t& size, [[maybe_unused]] const bool& isBorrowed,
    [[maybe_unused]] const bool& manageRegistration)
{
    /* do nothing */
}

/* CPU */

template<>
template<typename DataT>
inline void Allocator<Device::CPU>::malloc(DataT** data, const idx_t& size,
    [[maybe_unused]] const bool& isBorrowed,
    [[maybe_unused]] const bool& manageRegistration)
{
    if (isBorrowed)
        return;
    if constexpr (std::is_arithmetic_v<DataT>) {
        /* SIMD-friendly over-aligned allocation, paired with the matching
         * aligned `::operator delete[]` in `free`. Applies to arithmetic
         * element types only; complex types take the plain new[]/delete[]
         * path below to avoid aligned-delete / array-cookie mismatches. */
        *data = static_cast<DataT*>(
            ::operator new[](size * sizeof(DataT), std::align_val_t{ 64 }));
        std::fill(*data, *data + size, Default<DataT>());
    } else {
        /* Plain new[] default-constructs each element in place. */
        *data = new DataT[size];
    }
}

template<>
template<typename DataT>
inline void Allocator<Device::CPU>::free(DataT* data,
    [[maybe_unused]] const idx_t& size,
    [[maybe_unused]] const bool& isBorrowed,
    [[maybe_unused]] const bool& manageRegistration)
{
    if (isBorrowed)
        return;

    /* defensive early return for null pointers */
    if (data == nullptr)
        return;

    if constexpr (std::is_arithmetic_v<DataT>) {
        ::operator delete[](data, std::align_val_t{ 64 });
    } else {
        delete[] data;
    }
}

#ifndef FETA_CPU_ONLY

namespace detail {

/* Kernel to set all data in device memory to 0 */
template<typename DataT>
KERNEL()
void zeroKernel(DataT* const __restrict__ data, const size_t size)
{
    const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size)
        return;
    data[idx] = DeviceDefault<DataT>();
}

} // namespace detail

/* CUDA Host */

template<>
template<typename DataT>
inline void Allocator<Device::CUDA_HOST>::malloc(DataT** data,
    const idx_t& size, const bool& isBorrowed, const bool& manageRegistration)
{
    if (isBorrowed) {
        /* Simply register the already allocated memory to make it pageable and
         * enable async transfers, if it is not already registered */
        if (manageRegistration) {
            /* Not registered, so we register it */
            FETA_CHECK(cudaHostRegister(
                *data, size * sizeof(DataT), cudaHostRegisterDefault));
        }
    } else {
        /* Allocate anew with cudaMallocHost */
        FETA_CHECK(cudaMallocHost(data, size * sizeof(DataT)));
        if constexpr (IsTrivial<DataT>) {
            std::fill(*data, *data + size, Default<DataT>());
        } else {
            for (idx_t i = 0; i < size; i++)
                new ((*data) + i) DataT(Default<DataT>());
        }
    }
}

template<>
template<typename DataT>
inline void Allocator<Device::CUDA_HOST>::free(DataT* data, const idx_t& size,
    const bool& isBorrowed, const bool& manageRegistration)
{
    /* defensive early return for null pointers */
    if (data == nullptr)
        return;

    if (isBorrowed) {
        /* Unregister the memory, if we are managing the registration */
        if (manageRegistration) { /* Accepting errors path: */
            cudaError_t err = cudaHostUnregister(data);
            /* If the pointer no longer corresponds to a registered region,
             * treat it as already cleaned up to avoid spurious crashes.
             * This can happen inadvertently if the borrowed memory is freed
             * before the borrowing container is destroyed (e.g., both are
             * declared in the same scope). It is safe to ignore these errors
             * because they do not lead to a memory leak.
             */
            if (err == cudaSuccess) {
                return;
            } else if (err == cudaErrorInvalidValue
                || err == cudaErrorHostMemoryNotRegistered) {
                /* one of the errors to bypass - release it to avoid
                 cudaGetLastError() return something we would not need to be
                 checking against */
                cudaGetLastError();
            } else {
                // FETA_CHECK(err);
                /* make it throw directly otherwise */
                feta::err::detail::checkCudaApiError(
                    err, __FILE__, __LINE__, __func__);
            }
        }
        return;
    } else {
        /* Free the memory */
        if constexpr (!IsTrivial<DataT>) {
            for (idx_t i = 0; i < size; i++)
                data[i].~DataT();
        }
        FETA_CHECK(cudaFreeHost(data));
    }
}

/* CUDA Device */

template<>
template<typename DataT>
inline void Allocator<Device::CUDA_DEVICE>::malloc(DataT** data,
    const idx_t& size, const bool& isBorrowed,
    [[maybe_unused]] const bool& manageRegistration)
{
    if (isBorrowed)
        return;

    FETA_CHECK(cudaMalloc(data, size * sizeof(DataT)));

    // constexpr int blockSize = 256;
    // const int numBlocks = (size + blockSize - 1) / blockSize;
    // detail::zeroKernel<<<numBlocks, blockSize>>>(*data, size);

    // /* Initialize to zero - cudaMemset sets all bytes to 0, which is the
    //  * correct bit pattern for all arithmetic types and bool */
    // FETA_CHECK(cudaMemset(*data, 0, size * sizeof(DataT)));
    // FETA_CHECK(cudaDeviceSynchronize());
}

template<>
template<typename DataT>
inline void Allocator<Device::CUDA_DEVICE>::free(DataT* data,
    [[maybe_unused]] const idx_t& size, const bool& isBorrowed,
    [[maybe_unused]] const bool& manageRegistration)
{
    if (isBorrowed)
        return;

    /* defensive early return for null pointers */
    if (data == nullptr)
        return;

    FETA_CHECK(cudaFree(data));
}

template<>
template<typename DataT, typename TexT>
inline void Allocator<Device::CUDA_DEVICE>::bindToTexture(
    DataT* data, TexT& tex, const idx_t& size)
{
    /* Create resource descriptor */
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType           = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = data;
    resDesc.res.linear.desc
        = cudaCreateChannelDesc<typename helper::Fetch<DataT>::T>();
    resDesc.res.linear.sizeInBytes = size * sizeof(DataT);

    /* Crete texture descriptor */
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    /* create texture object */
    FETA_CHECK(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));
}

template<>
template<typename TexT>
inline void Allocator<Device::CUDA_DEVICE>::destroyTexture(TexT& tex)
{
    FETA_CHECK(cudaDestroyTextureObject(tex));
}

#endif

} // namespace memory
} // namespace core
} // namespace feta