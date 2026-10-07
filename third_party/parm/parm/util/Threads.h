#pragma once

#include "parm/typedefs.h"
#include "parm/util/DeviceError.h"

#ifndef PARM_CPU_ONLY

namespace parm {
namespace util {

/** @brief Collection of thread/warp utilities */
struct Threads {

    /** @brief Return the minimum index active thread in this block (i.e. the
     * lowest thread ID that hasn't returned somewhere before the invokation of
     * this function) */
    DEVICE() static inline idx_t blockLeader()
    {
        __shared__ idx_t minThreadId;
        /* note: this race condition is harmless as we are computing the min
         * later on */
        minThreadId = blockDim.x;
        __syncthreads();
        // Each active thread votes for its thread ID
        atomicMin(&minThreadId, threadIdx.x);
        __syncthreads();

        return minThreadId;
    }

    /** @brief Return the warp leader - i.e. the minimum active thread ID in the
     * current warp that meets the given condition. By default, the lowest Lane
     * ID in the warp of the active thread is returned (i.e. if return has not
     * been invoked before this function)*/
    DEVICE() static inline idx_t warpLeader(const bool& condition = true)
    {
        // get the mask of active threads in the warp that meet the condition
        const idx_t mask = __ballot_sync(0xFFFFFFFF, condition);
        // return the index of the least significant bit in the mask (i.e. the
        // minimum active thread ID in the warp that meets the condition)
        PARM_GPU_ASSERT(mask != 0, err::FAILED_WARP_LEADER);
        return __ffs(mask) - 1;
    }
};

} // namespace util
} // namespace parm

#endif