/* The core memory management module */
#pragma once

#include "feta/core/TextureHelpers.h"
#include "feta/core/type_traits.h"
#include "feta/err/DeviceError.h"
#include "feta/err/throw.h"
#include "feta/typedefs.h"

namespace feta {
namespace core {
namespace memory {

/*  Follow DLPACK memory cases - inspired from

    https://github.com/dmlc/dlpack/tree/main

    (as accessed on 16/10/2025)

*/
enum class Device : unsigned int {
    /** @brief None - handle only to support unified/single/non-paired
       containers */
    NONE,
    /** @brief CPU device */
    CPU,
    /** @brief CUDA GPU device */
    CUDA_DEVICE,
    /**
     * @brief Pinned CUDA CPU memory by cudaMallocHost
     */
    CUDA_HOST,
    /** @brief OpenCL devices. */
    OPENCL,
    /** @brief Vulkan buffer for next generation graphics. */
    VULKAN,
    /** @brief Metal for Apple GPU. */
    METAL,
    /** @brief Verilog simulator buffer */
    VERILOG,
    /** @brief ROCm GPUs for AMD GPUs */
    ROCM_HOST,
    /**
     * @brief Pinned ROCm CPU memory allocated by hipMallocHost
     */
    ROCM_DEVICE,
    /**
     * @brief Reserved extension device type,
     * used for quickly test extension device
     * The semantics can differ depending on the implementation.
     */
    EXTDEV,
    /**
     * @brief CUDA managed/unified memory allocated by cudaMallocManaged
     */
    CUDA_MANAGED,
    /**
     * @brief Unified shared memory allocated on a oneAPI non-partititioned
     * device. Call to oneAPI runtime is required to determine the device
     * type, the USM allocation type and the sycl context it is bound to.
     *
     */
    ONEAPI,
    /** @brief GPU support for next generation WebGPU standard. */
    WEBGPU,
    /** @brief Qualcomm Hexagon DSP */
    HEXAGON,
    /** @brief Microsoft MAIA devices */
    MAIA,
    /** @brief AWS Trainium */
    TRAINIUM,
    /* Leave the following item as last - it only acts as enum size */
    SIZE
};

/** @brief Identify devices that use unified (shared) host/device addressing. */
template<Device Target>
concept UnifiedArchitecture = Target == Device::CUDA_MANAGED
    || Target == Device::ONEAPI || Target == Device::MAIA
    || Target == Device::TRAINIUM || Target == Device::WEBGPU
    || Target == Device::VULKAN || Target == Device::METAL;

/** @brief Identify devices whose memory can be filled via `std::fill`. */
template<Device Target>
concept isFillable = Target == Device::CPU || Target == Device::CUDA_HOST
    || Target == Device::ROCM_HOST || UnifiedArchitecture<Target>;

/** @brief Identify devices that support CUDA texture memory. */
template<Device Target>
concept hasTextureSupport = Target == Device::CUDA_DEVICE;

/** @brief Select the appropriate stream type for a given device target. */
template<Device Target>
struct StreamSwitcher {
    using T = unsigned int;
};

#ifndef FETA_CPU_ONLY
/* Cuda specialization */
template<Device Target>
    requires(Target == Device::CUDA_HOST || Target == Device::CUDA_DEVICE
        || Target == Device::CUDA_MANAGED)
struct StreamSwitcher<Target> {
    using T = cudaStream_t;
};
#endif


/** @brief Identify devices that support asynchronous (stream-based) memory copies. */
template<Device Target>
concept hasAsynchronousCopySupport = Target == Device::CUDA_DEVICE
    || Target == Device::CUDA_HOST || Target == Device::CUDA_MANAGED;

/** @brief Verify that a memory copy from `From` to `To` is a supported transfer path. */
template<Device From, Device To>
concept areCompatible = (From == Device::CPU && To == Device::CPU)
    || (From == Device::CPU && To == Device::CUDA_HOST)
    || (From == Device::CUDA_HOST && To == Device::CPU)
    || (From == Device::CUDA_HOST && To == Device::CUDA_HOST)
    || (From == Device::CPU && To == Device::CUDA_DEVICE)
    || (From == Device::CUDA_DEVICE && To == Device::CPU)
    || (From == Device::CUDA_HOST && To == Device::CUDA_DEVICE)
    || (From == Device::CUDA_DEVICE && To == Device::CUDA_HOST)
    || (From == Device::CUDA_DEVICE && To == Device::CUDA_DEVICE);

/** @brief Verify that a `From` container can lend its pointer to a `To` container without a copy. */
template<Device From, Device To>
concept areBorrowable = (From == Device::CPU && To == Device::CPU)
    || (From == Device::CUDA_HOST && To == Device::CUDA_HOST)
    || (From == Device::CUDA_DEVICE && To == Device::CUDA_DEVICE)
    || (From == Device::CUDA_MANAGED && To == Device::CUDA_MANAGED)
    || (From == Device::CPU && To == Device::CUDA_HOST)
    || (From == Device::CUDA_HOST && To == Device::CPU);

/** @brief Identify the `NONE` device, used as a no-op placeholder. */
template<Device Target>
concept isNone = Target == Device::NONE;

/** @brief Identify devices that are self-standing (directly accessible without a paired workable). */
template<Device Target>
concept isSelfStanding = Target == Device::CPU || Target == Device::CUDA_HOST
    || Target == Device::CUDA_MANAGED;

} // namespace memory
} // namespace core
} // namespace feta