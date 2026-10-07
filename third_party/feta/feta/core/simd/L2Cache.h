/**
 * @file L2Cache.h
 *
 * Runtime detection of the L2 cache size for tile-size sizing.
 *
 * Used by ``feta::cpu::optimalTileSize`` and indirectly by
 * ``feta::cpu::packetBatchedFor`` to pick a tile size that keeps each
 * thread's working set L2-resident across substages.  Detected once
 * per process via ``std::call_once`` and cached.
 *
 * Detection uses POSIX ``sysconf(_SC_LEVEL2_CACHE_SIZE)`` on Linux.
 * Falls back to a 1 MiB default when sysconf returns 0 (e.g. some
 * containers, virtualised environments, or non-Linux platforms).
 */
#pragma once

#include "feta/typedefs.h"

#include <mutex>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace feta {
namespace cpu {
namespace detail {

/** @brief Conservative fallback when sysconf returns 0 (1 MiB).
 *
 *  Modern x86-64 cores typically have 256 KiB – 2 MiB private L2;
 *  1 MiB is a safe middle ground that doesn't over-tile on small
 *  L2 cores nor under-tile on larger ones. */
inline constexpr idx_t L2_CACHE_SIZE_FALLBACK = idx_t{ 1 } << 20;

/** @brief Return the per-core L2 cache size in bytes.
 *
 *  Detected once per process and cached.  On non-Linux or when sysconf
 *  fails, returns ``L2_CACHE_SIZE_FALLBACK``.
 *
 *  Note: ``_SC_LEVEL2_CACHE_SIZE`` reports the per-core L2 on most
 *  modern x86-64 systems where L2 is core-private; on older
 *  shared-L2 designs it would over-report.  The downstream tile sizer
 *  derates by a factor (default 0.5) to leave headroom for state +
 *  per-step scratch, so a small over-report is harmless. */
inline idx_t l2CacheSize()
{
    static idx_t cached = 0;
    static std::once_flag flag;
    std::call_once(flag, []() {
#if defined(_SC_LEVEL2_CACHE_SIZE)
        const long sz = ::sysconf(_SC_LEVEL2_CACHE_SIZE);
        cached        = (sz > 0) ? static_cast<idx_t>(sz)
                                 : L2_CACHE_SIZE_FALLBACK;
#else
        cached = L2_CACHE_SIZE_FALLBACK;
#endif
    });
    return cached;
}

} // namespace detail
} // namespace cpu
} // namespace feta
