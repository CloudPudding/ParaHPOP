#pragma once

#include <exception>
#include <map>
#include <vector>

#ifndef FETA_CPU_ONLY
#include <cuda.h>
#endif

#include "throw.h"
#include <feta/typedefs.h>

#if defined(FETA_DEBUG_MODE) && !defined(FETA_CPU_ONLY)

#define FETA_CHECK(expr)                                                       \
    {                                                                          \
        ::feta::err::detail::checkCudaApiError(                                \
            (expr), __FILE__, __LINE__, __func__);                             \
    }

#define FETA_KERNEL_PRE()                                                      \
    {                                                                          \
        ::feta::err::detail::resetDeviceErrors(::feta::err::deviceErrorFlag);  \
    }

#define FETA_KERNEL_POST()                                                     \
    {                                                                          \
        FETA_CHECK(cudaDeviceSynchronize());                                   \
        FETA_CHECK(cudaGetLastError());                                        \
        ::feta::err::detail::checkDeviceErrors(::feta::err::deviceErrorFlag,   \
            ::feta::err::errorMessages, __FILE__, __LINE__,                    \
            static_cast<const char*>(__func__));                               \
    }

#define FETA_GPU_THROW(errCode)                                                \
    {                                                                          \
        ::feta::err::detail::reportDeviceError(                                \
            errCode, ::feta::err::deviceErrorFlag);                            \
    }

#define FETA_GPU_ASSERT(condition, errCode)                                    \
    {                                                                          \
        if (!(condition))                                                      \
            ::feta::err::detail::reportDeviceError(                            \
                errCode, ::feta::err::deviceErrorFlag);                        \
    }

#else // FETA_DEBUG_MODE && !FETA_CPU_ONLY

#define FETA_CHECK(expr)                                                       \
    {                                                                          \
        expr;                                                                  \
    }

#define FETA_KERNEL_PRE()                                                      \
    {                                                                          \
    }

#define FETA_KERNEL_POST()                                                     \
    {                                                                          \
    }

#define FETA_GPU_THROW(errCode)                                                \
    {                                                                          \
    }

#define FETA_GPU_ASSERT(condition, errCode)                                    \
    {                                                                          \
    }

#endif // FETA_DEBUG_MODE && !FETA_CPU_ONLY

/* Define the required routines calling the CUDA API if not in CPU only mode*/
#ifndef FETA_CPU_ONLY

namespace feta {
namespace err {

/** @brief Unsigned integer type used as a bitmask for device-side error flags. */
using DeviceFlag      = unsigned long long int;
/** @brief Map from individual error flag bits to human-readable messages. */
using ErrorMessageMap = std::map<DeviceFlag, const char*>;

/* Leverage macros to detect if we are compiling into a shared module. If so,
 * make the variable extern (will be defined later) to keep a single instance */
#ifndef FETA_INTO_SHARED_LIBRARY
[[maybe_unused]] static __device__ DeviceFlag deviceErrorFlag;
#else
[[maybe_unused]] extern __device__ DeviceFlag deviceErrorFlag;
#endif

/** @brief Bitmask error codes raised from device kernels in debug mode. */
enum FetaErrorTypes : DeviceFlag {
    NO_ERROR               = (DeviceFlag)0,            ///< No error.
    OUT_OF_RANGE_SCALAR    = (DeviceFlag)1 << 0,       ///< Scalar array index out of range.
    OUT_OF_RANGE_VECTOR    = (DeviceFlag)1 << 1,       ///< Vector array index out of range.
    INVALID_DIMENSION      = (DeviceFlag)1 << 3,       ///< Invalid vector dimension access.
    SAMPLE_INDEX_INIT      = (DeviceFlag)1 << 4,       ///< Invalid SampleIndex initialisation.
    OUT_OF_RANGE_COMPONENT = (DeviceFlag)1 << 5,       ///< Vector component index out of range.
    ERR_MAX                = (DeviceFlag)1 << 63       ///< Sentinel (maximum flag value).
};

const static ErrorMessageMap errorMessages = { //
    { OUT_OF_RANGE_SCALAR,
        "attempt to access elements out of range in Scalar Array" },
    { OUT_OF_RANGE_VECTOR,
        "attempt to access elements out of range in Vector Array" },
    { INVALID_DIMENSION, "attempt to access invalid vector dimension" },
    { SAMPLE_INDEX_INIT, "invalid initialization of SampleIndex" },
    { OUT_OF_RANGE_COMPONENT, "VectorView: out of bounds component access" }
};

/** @brief Exception thrown when a device kernel reports an error via the error flag. */
class DeviceError : public std::runtime_error {
public:
    /** @brief Construct with an error message.
     *  @param[in] message  Human-readable error description.
     */
    DeviceError(const std::string& message)
        : std::runtime_error(message)
    {
    }

    /**
     * @brief Decode a bitmask device flag into a human-readable message string.
     *
     * @param[in] deviceFlag  Bitmask of `FetaErrorTypes` values.
     * @param[in] messageMap  Map from individual flag bits to message strings.
     * @return Concatenated error description.
     */
    static std::string flagToMessage(
        const DeviceFlag& deviceFlag, const ErrorMessageMap& messageMap)
    {
        DeviceFlag errorKind = 1;
        std::vector<const char*> messages(0);
        for (idx_t i = 0; i < 8 * sizeof(DeviceFlag); i++, errorKind <<= 1) {
            if (deviceFlag & errorKind) {
                if (messageMap.find(errorKind) != messageMap.end()) {
                    messages.push_back(messageMap.at(errorKind));
                } else {
                    messages.push_back("[unknown error]");
                }
            }
        }

        std::string message = "";
        if (messages.size() == 1) {
            message = "GPU error: " + std::string(messages[0]);
        } else if (messages.size() > 1) {
            message = "Multiple GPU errors: ";
            for (auto it : messages) {
                message += std::string(it) + "; ";
            }
        } else {
            message = "GPU error (empty flag)";
        }
        return message;
    }
};

/** @brief Exception thrown when a CUDA API call returns an error code. */
class CUDAError : public std::runtime_error {
public:
    /** @brief Construct with an error message.
     *  @param[in] message  CUDA error description (prefixed with "CUDA Error: ").
     */
    CUDAError(const std::string& message)
        : std::runtime_error("CUDA Error: " + message)
    {
    }
};

namespace detail {

/**
 * @brief Check the error code of a CUDA API call, throwing on an error.
 *
 * @param[in] errorCode  Return value of a CUDA API function.
 * @param[in] file       Source file name (usually `__FILE__`).
 * @param[in] line       Source line number (usually `__LINE__`).
 * @param[in] func       Enclosing function name (usually `__func__`).
 * @throws CUDAError  If `errorCode` is not `cudaSuccess`.
 */
static void checkCudaApiError(
    cudaError_t errorCode, const char* file, idx_t line, const char* func)
{
    if (errorCode != cudaSuccess) {
        Throw<CUDAError>(file, line, func)(
            std::string(cudaGetErrorString(errorCode)));
    }
}

/** @brief Reset the device flag used to report errors in kernels.
 *  @param[in,out] flag  Device-side error flag to zero out.
 */
[[maybe_unused]] static void resetDeviceErrors(DeviceFlag& flag)
{
    DeviceFlag zero = 0;
    FETA_CHECK(cudaMemcpyToSymbol(flag, &zero, sizeof(DeviceFlag)));
}

/**
 * @brief Check for error flags raised during kernel execution.
 *
 * @param[in] flag        Device-side error flag to read.
 * @param[in] messageMap  Map from flag bits to error descriptions.
 * @param[in] file        Source file name.
 * @param[in] line        Source line number.
 * @param[in] func        Enclosing function name.
 * @throws DeviceError  If any error bits are set in `flag`.
 */
[[maybe_unused]] static void checkDeviceErrors(DeviceFlag& flag,
    const ErrorMessageMap& messageMap, const std::string& file, int line,
    const std::string& func)
{
    // check for kernel launch errors
    checkCudaApiError(cudaGetLastError(), file.c_str(), line, func.c_str());
    // check for device function errors
    DeviceFlag error;
    // check for device function errors
    checkCudaApiError(cudaMemcpyFromSymbol(&error, flag, sizeof(DeviceFlag)),
        file.c_str(), line, func.c_str());

    if (error != 0) {
        Throw<DeviceError>(file, line, func)(
            DeviceError::flagToMessage(error, messageMap));
    }
}

/** @brief Raise an error flag from device code.
 *  @param[in] errorCode  Error bit(s) to set (from `FetaErrorTypes`).
 *  @param[in,out] flag   Device-side flag; bits are atomically OR-ed in.
 */
[[maybe_unused]] __device__ static void reportDeviceError(
    DeviceFlag errorCode, DeviceFlag& flag)
{
    atomicOr(&flag, errorCode);
}

} // namespace detail
} // namespace err
} // namespace feta
#endif