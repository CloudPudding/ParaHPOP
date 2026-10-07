/**
 * @file Prefetch.h
 *
 * SIMD software prefetch utilities for hiding memory latency in the
 * packet evaluation loop.
 *
 * On CPUs with SoA layout, consecutive packets for the same dimension
 * are contiguous, but switching dimensions causes a stride of
 * `dimOffset * sizeof(DataT)`.  Issuing a prefetch for the *next*
 * packet's address while computing the *current* packet hides L1/L2
 * miss latency.
 */
#pragma once

#include "feta/typedefs.h"

#if defined(__SSE2__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
#include <immintrin.h>
#define FETA_HAS_PREFETCH 1
#else
#define FETA_HAS_PREFETCH 0
#endif

namespace feta {
namespace simd {

/**
 * @brief Prefetch a cache line containing address `ptr` into L1 cache.
 *
 * Compiles to `_mm_prefetch(ptr, _MM_HINT_T0)` when SIMD headers are
 * available, otherwise a no-op.
 */
inline void prefetchL1(const void* ptr)
{
#if FETA_HAS_PREFETCH
    _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0);
#else
    (void)ptr;
#endif
}

/**
 * @brief Prefetch into L2 cache (non-temporal hint).
 */
inline void prefetchL2(const void* ptr)
{
#if FETA_HAS_PREFETCH
    _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T1);
#else
    (void)ptr;
#endif
}

/**
 * @brief Prefetch for write — brings cache line into L1 in exclusive state.
 *
 * Uses _MM_HINT_ET0 where supported (write-intent prefetch), which avoids
 * the read-for-ownership penalty when the line is later written.
 * Falls back to _MM_HINT_T0 on older compilers/platforms.
 */
inline void prefetchL1W(const void* ptr)
{
#if FETA_HAS_PREFETCH
#if defined(__AVX512F__) || defined(__AVX2__)
    // _MM_HINT_ET0 = exclusive to L1 (write intent) — supported since Broadwell
    _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_ET0);
#else
    _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0);
#endif
#else
    (void)ptr;
#endif
}

/**
 * @brief Prefetch for write into L2 cache.
 */
inline void prefetchL2W(const void* ptr)
{
#if FETA_HAS_PREFETCH
#if defined(__AVX512F__) || defined(__AVX2__)
    _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_ET1);
#else
    _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T1);
#endif
#else
    (void)ptr;
#endif
}

/// Default prefetch distances (in packets, not bytes).
/// These can be overridden by users via template parameters.
inline constexpr idx_t DEFAULT_PREFETCH_L1_DISTANCE = 2;   ///< packets ahead for L1
inline constexpr idx_t DEFAULT_PREFETCH_L2_DISTANCE = 8;   ///< packets ahead for L2

} // namespace simd
} // namespace feta
