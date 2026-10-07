#pragma once

#include "parm/typedefs.h"
#include "parm/util/DeviceError.h"
#include "parm/util/log.h"
#include "parm/util/throw.h"

#ifndef PARM_CPU_ONLY

namespace parm {
namespace util {
namespace graph {

/** @brief Stream manager for non-default streams */
class Stream {
public:
    /** @brief Default constructor */
    Stream() { create_(); }

    /** @brief Copy constructor */
    Stream(Stream& other)       = delete;
    Stream(const Stream& other) = delete;

    /** @brief Move constructor */
    Stream(Stream&& other)
        : created_{ std::exchange(other.created_, false) }
        , cuda_{ std::exchange(other.cuda_, nullptr) }
    {
    }

    /** @brief Copy constructor is forbidden */
    Stream& operator=(Stream& other)       = delete;
    Stream& operator=(const Stream& other) = delete;

    /** @brief Move assignment operator*/
    Stream& operator=(Stream&& other) noexcept
    {
        destroy_();
        created_ = std::exchange(other.created_, false);
        cuda_    = std::exchange(other.cuda_, nullptr);
        return *this;
    }

    /** @brief Synchronize the host wrt this stream */
    void synchronize() const { PARM_CHECK(cudaStreamSynchronize(cuda_)); }

    /** @brief Get the low-level stream */
    const cudaStream_t& cuda() const { return cuda_; }

    /** @brief Default constructor */
    ~Stream() { destroy_(); }

protected:
    void create_()
    {
        if (!created_) {
            PARM_CHECK(cudaStreamCreate(&cuda_));
            created_ = true;
        }
    }

    void destroy_()
    {
        if (created_ && cuda_ != nullptr) {
            PARM_CHECK(cudaStreamDestroy(cuda_));
            created_ = false;
            cuda_    = nullptr;
        }
    }

    bool created_      = false;
    cudaStream_t cuda_ = nullptr;
};

} // namespace graph
} // namespace util
} // namespace parm

#endif
