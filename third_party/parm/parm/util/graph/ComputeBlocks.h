#pragma once

#include "parm/typedefs.h"

#ifndef PARM_CPU_ONLY

#include <algorithm>

#include <cuda_runtime.h>

namespace parm {
namespace util {
namespace graph {

/** @brief Cached single-device SM count. */
inline int currentSMCount()
{
    static const int nSMs = []() {
        int dev = 0;
        int n   = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
        return n;
    }();
    return nSMs;
}

/** @brief Block-size policy for an N-sample kernel.
 *
 *  Computes ``(blockSize, nBlocks)`` over ``numStates`` work items
 *  given the kernel's ``__launch_bounds__``-derived
 *  ``idealBlockSize`` cap and a target ``blocksPerSM``. Targets
 *  ``nSMs * blocksPerSM`` blocks, rounds up to WARP=32, clamps to
 *  ``[WARP, idealBlockSize]``.
 *
 *  The caller supplies ``blocksPerSM`` rather than calling
 *  ``cudaOccupancyMaxActiveBlocksPerMultiprocessor`` because that
 *  fails with ``cudaErrorInvalidResourceHandle`` when the consumer
 *  DSO links a statically-linked cudart distinct from the kernel
 *  stub's. */
inline void computeBlocks(idx_t numStates, idx_t& nBlocks, idx_t& blockSize,
    idx_t idealBlockSize = 256, idx_t blocksPerSM = 4)
{
    constexpr idx_t WARP = 32;

    /* Handle the 0 numStates case. Defaults to warp-sized blocksize but 0 num
     * blocks */
    if (numStates == 0) {
        nBlocks   = 0;
        blockSize = WARP;
        return;
    }

    const idx_t gridSize = static_cast<idx_t>(currentSMCount()) * blocksPerSM;

    idx_t bs  = (numStates + gridSize - 1) / gridSize;
    bs        = ((bs + WARP - 1) / WARP) * WARP;
    bs        = std::max<idx_t>(WARP, std::min<idx_t>(idealBlockSize, bs));
    blockSize = bs;
    nBlocks   = (numStates + bs - 1) / bs;
}

} // namespace graph
} // namespace util
} // namespace parm

#endif // PARM_CPU_ONLY
