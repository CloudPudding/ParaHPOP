#pragma once

/**
 * @brief Mark a function as device-only (`__device__`).
 *
 * In `FETA_CPU_ONLY` builds this expands to nothing.
 */
/**
 * @brief Mark a function as host-only (`__host__`).
 *
 * In `FETA_CPU_ONLY` builds this expands to nothing.
 */
/**
 * @brief Mark a function as callable from both device and host
 *        (`__device__ __host__`).
 *
 * In `FETA_CPU_ONLY` builds this expands to nothing.
 */
/**
 * @brief Prevent inlining of a function (`__noinline__` on CUDA,
 *        compiler-specific attribute on CPU).
 */
/**
 * @brief Mark a function as a CUDA kernel entry point (`__global__`).
 *
 * In `FETA_CPU_ONLY` builds this is not defined (kernels are
 * replaced by plain functions via the build system).
 */
/**
 * @brief Qualify a kernel parameter as grid-constant read-only
 *        (`const __grid_constant__` on Volta+, plain `const` otherwise).
 */
/**
 * @brief Context-aware device/host qualifier.
 *
 * Expands to `__device__` when compiling device code
 * (`__CUDA_ARCH__` defined) and to `__host__` otherwise.
 * In `FETA_CPU_ONLY` builds this expands to nothing.
 */
/**
 * @brief Declare a `__shared__` variable in a CUDA kernel.
 *
 * Expands to nothing when `__CUDA_ARCH__` is not defined.
 */
/**
 * @brief Force-inline a function (`__forceinline__` on device,
 *        `inline` on host).
 */
#ifndef FETA_CPU_ONLY
#define DEVICE(...) __device__
#define HOST(...) __host__
#define DEVICEANDHOST(...) __device__ __host__
#define NOINLINE(...) __noinline__
#define KERNEL(...) __global__
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
#define GRID_CONSTANT(...) const __grid_constant__
#else
#define GRID_CONSTANT(...) const
#endif
#if defined(__CUDA_ARCH__)
#define DEVICEHOST(...) DEVICE()
#define SHARED(...) __shared__
#else
#define DEVICEHOST(...) HOST()
#define SHARED(...)
#endif
#else
#define DEVICE(...)
#define HOST(...)
#define DEVICEANDHOST(...)
#define DEVICEHOST(...)
#ifdef _WIN32
#define NOINLINE(...) __declspec(noinline)
#else
#define NOINLINE(...) __attribute__((noinline))
#endif
#endif
#if defined(__CUDA_ARCH__)
#define FORCEINLINE(...) __forceinline__
#else
#define FORCEINLINE(...) inline
#endif