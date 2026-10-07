#pragma once

#include "interface/typedefs.h"

#ifdef PARAHPOP_DEBUG_MODE

#define PARAHPOP_CHECK(expr)                                                      \
    {                                                                          \
        ::feta::err::detail::checkCudaApiError(                                \
            (expr), __FILE__, __LINE__, __func__);                             \
    }

#define PARAHPOP_KERNEL_PRE()                                                     \
    {                                                                          \
        ::feta::err::detail::resetDeviceErrors(::paraHPOP::err::deviceErrorFlag); \
        BRIE_KERNEL_PRE();                                                     \
    }

#define PARAHPOP_KERNEL_POST()                                                    \
    {                                                                          \
        BRIE_KERNEL_POST();                                                    \
        ::feta::err::detail::checkDeviceErrors(::paraHPOP::err::deviceErrorFlag,  \
            ::paraHPOP::err::errorMessages, __FILE__, __LINE__,                   \
            static_cast<const char*>(__func__));                               \
    }

#define PARAHPOP_GPU_THROW(errCode)                                               \
    {                                                                          \
        ::feta::err::detail::reportDeviceError(                                \
            errCode, ::paraHPOP::err::deviceErrorFlag);                           \
    }

#define PARAHPOP_GPU_ASSERT(condition, errCode)                                   \
    {                                                                          \
        if (!(condition))                                                      \
            ::feta::err::detail::reportDeviceError(                            \
                errCode, ::paraHPOP::err::deviceErrorFlag);                       \
    }

#else // PARAHPOP_DEBUG_MODE

#define PARAHPOP_CHECK(expr)                                                      \
    {                                                                          \
        expr;                                                                  \
    }

#define PARAHPOP_KERNEL_PRE()                                                     \
    {                                                                          \
    }

#define PARAHPOP_KERNEL_POST()                                                    \
    {                                                                          \
    }

#define PARAHPOP_GPU_THROW(errCode)                                               \
    {                                                                          \
    }

#define PARAHPOP_GPU_ASSERT(condition, errCode)                                   \
    {                                                                          \
    }

#endif // PARAHPOP_DEBUG_MODE


namespace paraHPOP {
namespace err {

using DeviceFlag = feta::err::DeviceFlag;
[[maybe_unused]] static DEVICE() DeviceFlag deviceErrorFlag;

enum CudajErrorTypes : DeviceFlag {
    NO_ERROR             = (DeviceFlag)0,
    RK_OVERSTEP          = (DeviceFlag)1 << 0,
    INCOMPATIBLESIZES    = (DeviceFlag)1 << 1,
    INCOMPATIBLEVECSIZES = (DeviceFlag)1 << 2,
    INVALID_BODY         = (DeviceFlag)1 << 3,
    // Example error variant
    // MY_ERROR = (DeviceFlag)1 << 1,
    ERR_MAX = (DeviceFlag)1 << 63
};

const static feta::err::ErrorMessageMap errorMessages = {
    { RK_OVERSTEP,
        "RK integration error: a sample overstepped its end time, which "
        "shouldn't happen" },
    { INCOMPATIBLESIZES,
        "Incompatible Scalar Array and Compile-time Vector sizes." },
    { INCOMPATIBLEVECSIZES,
        "Incompatible VectorArray elements and Row Dimension." },
    { INVALID_BODY, "This Naif ID is not allowed in this context." },
    // Error messages corresponding to error variants
    // { MY_ERROR, "you did a thing wrong" },
};

} // namespace err
} // namespace paraHPOP
