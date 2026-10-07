#pragma once

#include "feta/core/memory/Devices.h"

namespace feta {
namespace core {
namespace memory {

/**
 * @brief Dispatch memory copies between two devices using the appropriate API.
 *
 * Specialisations cover all valid `areCompatible<From, To>` pairs (CPU, CUDA_HOST,
 * CUDA_DEVICE). Instantiating an unsupported pair produces a `static_assert` failure.
 *
 * @tparam From  Source device.
 * @tparam To    Destination device.
 */
template<Device From, Device To>
struct Copy {

    /**
     * @brief Copy `NumElements` scalars from `from` to `to` synchronously.
     *
     * @param from         Source pointer.
     * @param to           Destination pointer.
     * @param NumElements  Number of scalar elements to copy.
     */
    template<typename DataT>
    static inline void blocking(
        const DataT* from, DataT* to, const idx_t& NumElements)
    {
        static_assert(
            feta::core::expr::enumAlwaysFalse<static_cast<unsigned int>(
                From)>::value,
            "Not implemented for this Device type");
    }

    /**
     * @brief Copy `NumElements` scalars from `from` to `to` asynchronously on `stream`.
     *
     * @param from         Source pointer.
     * @param to           Destination pointer.
     * @param NumElements  Number of scalar elements to copy.
     * @param stream       CUDA stream to enqueue the transfer on.
     */
    template<typename DataT,
        typename StreamT = typename StreamSwitcher<From>::T>
    static inline void async(const DataT* from, DataT* to,
        const idx_t& NumElements, const StreamT& stream)
    {
        static_assert(
            feta::core::expr::enumAlwaysFalse<static_cast<unsigned int>(
                From)>::value,
            "Not implemented for this Device type");
    }
};

/* Specialisations */

/* Host Only */

/* CPU - CPU */
template<>
template<typename DataT>
inline void Copy<Device::CPU, Device::CPU>::blocking(
    const DataT* from, DataT* to, const idx_t& NumElements)
{
    std::copy(from, from + NumElements, to);
}

#ifndef FETA_CPU_ONLY

/* CPU - CUDA host */
template<>
template<typename DataT>
inline void Copy<Device::CPU, Device::CUDA_HOST>::blocking(
    const DataT* from, DataT* to, const idx_t& NumElements)
{
    std::copy(from, from + NumElements, to);
}

/* CUDA host - CPU */
template<>
template<typename DataT>
inline void Copy<Device::CUDA_HOST, Device::CPU>::blocking(
    const DataT* from, DataT* to, const idx_t& NumElements)
{
    std::copy(from, from + NumElements, to);
}

/* CUDA host - CUDA host */
template<>
template<typename DataT>
inline void Copy<Device::CUDA_HOST, Device::CUDA_HOST>::blocking(
    const DataT* from, DataT* to, const idx_t& NumElements)
{
    // Avoid CUDA runtime for host-to-host copies; use std::copy for speed
    std::copy(from, from + NumElements, to);
}

template<>
template<typename DataT, typename StreamT>
inline void Copy<Device::CUDA_HOST, Device::CUDA_HOST>::async(const DataT* from,
    DataT* to, const idx_t& NumElements, const StreamT& stream)
{
    FETA_CHECK(cudaMemcpyAsync(
        to, from, NumElements * sizeof(DataT), cudaMemcpyHostToHost, stream));
}

/* Host - device cases */

/* CPU - CUDA Device */
template<>
template<typename DataT>
inline void Copy<Device::CPU, Device::CUDA_DEVICE>::blocking(
    const DataT* from, DataT* to, const idx_t& NumElements)
{
    FETA_CHECK(cudaMemcpy(
        to, from, NumElements * sizeof(DataT), cudaMemcpyHostToDevice));
}

/* CUDA host - CUDA Device */

template<>
template<typename DataT>
inline void Copy<Device::CUDA_HOST, Device::CUDA_DEVICE>::blocking(
    const DataT* from, DataT* to, const idx_t& NumElements)
{
    FETA_CHECK(cudaMemcpy(
        to, from, NumElements * sizeof(DataT), cudaMemcpyHostToDevice));
}

template<>
template<typename DataT, typename StreamT>
inline void Copy<Device::CUDA_HOST, Device::CUDA_DEVICE>::async(
    const DataT* from, DataT* to, const idx_t& NumElements,
    const StreamT& stream)
{
    FETA_CHECK(cudaMemcpyAsync(
        to, from, NumElements * sizeof(DataT), cudaMemcpyHostToDevice, stream));
}

/* Device - host cases */

/* CUDA Device - CPU */
template<>
template<typename DataT>
inline void Copy<Device::CUDA_DEVICE, Device::CPU>::blocking(
    const DataT* from, DataT* to, const idx_t& NumElements)
{
    FETA_CHECK(cudaMemcpy(
        to, from, NumElements * sizeof(DataT), cudaMemcpyDeviceToHost));
}

/* CUDA Device - CUDA host */
template<>
template<typename DataT>
inline void Copy<Device::CUDA_DEVICE, Device::CUDA_HOST>::blocking(
    const DataT* from, DataT* to, const idx_t& NumElements)
{
    FETA_CHECK(cudaMemcpy(
        to, from, NumElements * sizeof(DataT), cudaMemcpyDeviceToHost));
}

template<>
template<typename DataT, typename StreamT>
inline void Copy<Device::CUDA_DEVICE, Device::CUDA_HOST>::async(
    const DataT* from, DataT* to, const idx_t& NumElements,
    const StreamT& stream)
{
    FETA_CHECK(cudaMemcpyAsync(
        to, from, NumElements * sizeof(DataT), cudaMemcpyDeviceToHost, stream));
}

/* CUDA Device - CUDA device */
template<>
template<typename DataT>
inline void Copy<Device::CUDA_DEVICE, Device::CUDA_DEVICE>::blocking(
    const DataT* from, DataT* to, const idx_t& NumElements)
{
    FETA_CHECK(cudaMemcpy(
        to, from, NumElements * sizeof(DataT), cudaMemcpyDeviceToDevice));
}

template<>
template<typename DataT, typename StreamT>
inline void Copy<Device::CUDA_DEVICE, Device::CUDA_DEVICE>::async(
    const DataT* from, DataT* to, const idx_t& NumElements,
    const StreamT& stream)
{
    FETA_CHECK(cudaMemcpyAsync(to, from, NumElements * sizeof(DataT),
        cudaMemcpyDeviceToDevice, stream));
}

#endif

} // namespace memory
} // namespace core
} // namespace feta