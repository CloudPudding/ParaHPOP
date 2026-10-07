#pragma once

#include <feta/feta.h>

#if defined(PARM_DEBUG_MODE) && !defined(PARM_CPU_ONLY)

#define PARM_CHECK(expr)                                                       \
    {                                                                          \
        ::feta::err::detail::checkCudaApiError(                                \
            (expr), __FILE__, __LINE__, __func__);                             \
    }

#define PARM_KERNEL_PRE()                                                      \
    {                                                                          \
        ::feta::err::detail::resetDeviceErrors(::parm::err::deviceErrorFlag);  \
        FETA_KERNEL_PRE();                                                     \
    }

#define PARM_KERNEL_POST()                                                     \
    {                                                                          \
        FETA_KERNEL_POST();                                                    \
        ::feta::err::detail::checkDeviceErrors(::parm::err::deviceErrorFlag,   \
            ::parm::err::errorMessages, __FILE__, __LINE__,                    \
            static_cast<const char*>(__func__));                               \
    }

#define PARM_GPU_THROW(errCode)                                                \
    {                                                                          \
        ::feta::err::detail::reportDeviceError(                                \
            errCode, ::parm::err::deviceErrorFlag);                            \
    }

#define PARM_GPU_ASSERT(condition, errCode)                                    \
    {                                                                          \
        if (!(condition))                                                      \
            ::feta::err::detail::reportDeviceError(                            \
                errCode, ::parm::err::deviceErrorFlag);                        \
    }

#else // PARM_DEBUG_MODE && !PARM_CPU_ONLY

#define PARM_CHECK(expr)                                                       \
    {                                                                          \
        expr;                                                                  \
    }

#define PARM_KERNEL_PRE()                                                      \
    {                                                                          \
    }

#define PARM_KERNEL_POST()                                                     \
    {                                                                          \
    }

#define PARM_GPU_THROW(errCode)                                                \
    {                                                                          \
    }

#define PARM_GPU_ASSERT(condition, errCode)                                    \
    {                                                                          \
    }

#endif // PARM_DEBUG_MODE && !PARM_CPU_ONLY

/* Define the required routines calling the CUDA API if not in CPU only mode*/
#ifndef PARM_CPU_ONLY

namespace parm {
namespace err {

using DeviceFlag = feta::err::DeviceFlag;

/* Leverage macros to detect if we are compiling into a shared module. If so,
 * make the variable extern (will be defined later) to keep a single instance */
#ifndef PARM_INTO_SHARED_LIBRARY
[[maybe_unused]] static __device__ DeviceFlag deviceErrorFlag;
#else
[[maybe_unused]] extern __device__ DeviceFlag deviceErrorFlag;
#endif

/** @brief Bit-flag error codes reported from device kernels via ``PARM_GPU_THROW``. */
enum ParmErrorTypes : DeviceFlag {
    NO_ERROR              = (DeviceFlag)0,       ///< No error.
    RK_OVERSTEP           = (DeviceFlag)1 << 0,  ///< A sample overstepped its end time.
    INTERP_OUTOFBOUNDS    = (DeviceFlag)1 << 1,  ///< Interpolation query is out of bounds.
    INTERP_INVALID_DEGREE = (DeviceFlag)1 << 2,  ///< Invalid polynomial degree for interpolation.
    FAILED_WARP_LEADER    = (DeviceFlag)1 << 3,  ///< Warp leader failed a required condition.
    ERR_MAX = (DeviceFlag)1 << 63                ///< Sentinel — maximum error flag value.
};

const static feta::err::ErrorMessageMap errorMessages = {
    { RK_OVERSTEP,
        "RK integration error: a sample overstepped its end time, which "
        "shouldn't happen" },
    { INTERP_OUTOFBOUNDS,
        "Interpolation error: requested value of interpolation variable is out "
        "of bounds." },
    { INTERP_INVALID_DEGREE,
        "Interpolation error: invalid polynomial degree." },
    { FAILED_WARP_LEADER,
        "Warp leader error: the warp leader should always meet the condition, "
        "so this should never happen." }
    // Error messages corresponding to error variants
    // { MY_ERROR, "you did a thing wrong" },
};

} // namespace err
} // namespace parm

#endif