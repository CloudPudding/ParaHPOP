#pragma once

#include <parm/util/DeviceError.h>

#if defined(BRIE_DEBUG_MODE) && !defined(BRIE_CPU_ONLY)

#define BRIE_CHECK(expr)                                                       \
    {                                                                          \
        ::feta::err::detail::checkCudaApiError(                                \
            (expr), __FILE__, __LINE__, __func__);                             \
    }

#define BRIE_KERNEL_PRE()                                                      \
    {                                                                          \
        /* Reset via brie wrapper so host->device symbol ops happen within     \
         * the same shared object that defines the device flag. */             \
        ::brie::err::resetDeviceErrors();                                      \
        PARM_KERNEL_PRE();                                                     \
    }

#define BRIE_KERNEL_POST()                                                     \
    {                                                                          \
        PARM_KERNEL_POST();                                                    \
        /* Check via brie wrapper in the same DSO to avoid invalid device      \
         * symbol due to cross-module cudaMemcpyFromSymbol. */                 \
        ::brie::err::checkDeviceErrors(                                        \
            __FILE__, __LINE__, static_cast<const char*>(__func__));           \
    }

#define BRIE_GPU_THROW(errCode)                                                \
    {                                                                          \
        ::feta::err::detail::reportDeviceError(                                \
            errCode, ::brie::err::deviceErrorFlag);                            \
    }

#define BRIE_GPU_ASSERT(condition, errCode)                                    \
    {                                                                          \
        if (!(condition))                                                      \
            ::feta::err::detail::reportDeviceError(                            \
                errCode, ::brie::err::deviceErrorFlag);                        \
    }

#else // BRIE_DEBUG_MODE && !defined(BRIE_CPU_ONLY)

#define BRIE_CHECK(expr)                                                       \
    {                                                                          \
        expr;                                                                  \
    }

#define BRIE_KERNEL_PRE()                                                      \
    {                                                                          \
    }

#define BRIE_KERNEL_POST()                                                     \
    {                                                                          \
    }

#define BRIE_GPU_THROW(errCode)                                                \
    {                                                                          \
    }

#define BRIE_GPU_ASSERT(condition, errCode)                                    \
    {                                                                          \
    }

#endif // BRIE_DEBUG_MODE && !defined(BRIE_CPU_ONLY)

#ifndef BRIE_CPU_ONLY

namespace brie {
namespace err {

using DeviceFlag = feta::err::DeviceFlag;
/* Leverage macros to detect if we are compiling into a shared module. If so,
 * make the variable extern (will be defined later) to keep a single instance */
#ifndef BRIE_INTO_SHARED_LIBRARY
[[maybe_unused]] static DEVICE() DeviceFlag deviceErrorFlag;
#else
[[maybe_unused]] extern DEVICE() DeviceFlag deviceErrorFlag;
#endif

enum BrieErrorTypes : DeviceFlag {
    NO_ERROR                          = (DeviceFlag)0,
    INVALID_NAIF                      = (DeviceFlag)1 << 0,
    TARGET_BODY_NOT_IN_UNIT           = (DeviceFlag)1 << 1,
    CENTER_BODY_NOT_IN_UNIT           = (DeviceFlag)1 << 2,
    OUT_OF_RANGE_TARGET_EPOCH         = (DeviceFlag)1 << 3,
    OUT_OF_RANGE_CENTER_EPOCH         = (DeviceFlag)1 << 4,
    CACHE_NOT_AVAILABLE               = (DeviceFlag)1 << 5,
    OUT_OF_CACHE_RANGE                = (DeviceFlag)1 << 6,
    OUT_OF_CACHE_BODY_UNITS           = (DeviceFlag)1 << 7,
    CACHE_REQUESTED_BUT_NOT_AVAILABLE = (DeviceFlag)1 << 8,
    INVALID_BODY_UNIT_TYPE            = (DeviceFlag)1 << 9,
    NO_SHARED_CENTER                  = (DeviceFlag)1 << 10,
    BODY_NOT_IN_TRAVERSER             = (DeviceFlag)1 << 11,
    BODY_NOT_IN_TREE                  = (DeviceFlag)1 << 12,
    ERR_MAX                           = (DeviceFlag)1 << 63
};

const static feta::err::ErrorMessageMap errorMessages{ //
    { INVALID_NAIF, "Unrecognised NAIF ID" },
    { BODY_NOT_IN_TRAVERSER, "Body not found in traverser" },
    { TARGET_BODY_NOT_IN_UNIT, "TARGET body not in the current unit" },
    { CENTER_BODY_NOT_IN_UNIT, "CENTER body not in the current unit" },
    { OUT_OF_RANGE_TARGET_EPOCH,
        "Epoch for TARGET not in the available range" },
    { OUT_OF_RANGE_CENTER_EPOCH,
        "Epoch for CENTER not in the available range" },
    { CACHE_NOT_AVAILABLE, "Cache is not available" },
    { OUT_OF_CACHE_RANGE,
        "Requested level exceeds number of levels in the cache" },
    { OUT_OF_CACHE_BODY_UNITS,
        "Requested body units exceed number of body units in the cache" },
    { CACHE_REQUESTED_BUT_NOT_AVAILABLE,
        "Cache was requested but is not enabled" },
    { INVALID_BODY_UNIT_TYPE, "Invalid body unit type" },
    { NO_SHARED_CENTER,
        "No shared center found between target and center bodies" },
    { BODY_NOT_IN_TREE, "Requested body not found in the gravity tree" }
};

// Host-side wrappers to manipulate/check the device error flag from code
// that may live in a different shared library (e.g., Python extension modules).
// Implemented in cudajectory/cudaj/util/DeviceError.cu where the device flag
// is defined, ensuring cudaMemcpyToSymbol/cudaMemcpyFromSymbol target a symbol
// in the same CUDA module.

void resetDeviceErrors();
void checkDeviceErrors(const char* file, int line, const char* func);

} // namespace err
} // namespace brie

#endif