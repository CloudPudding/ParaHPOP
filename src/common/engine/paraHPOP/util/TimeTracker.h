#pragma once

#include "paraHPOP/typedefs.h"
#include "interface/util/err.h"

namespace paraHPOP {
namespace util {

namespace device {

/** @brief Execution time tracking structure */
struct TimeTracker {

    /** @brief Default constructor craetes the events */
    TimeTracker() { create_(); }

    /** @brief Construct with a stream assignment */
    TimeTracker(const cudaStream_t& s) { this->stream(s); }

    /** @brief Copy is forbidden: cudaEvent_t handles own device-side
     *  resources and must not be shared between instances (sharing leads to
     *  double cudaEventDestroy on destruction, which surfaces as
     *  non-deterministic CUDA errors on later, unrelated launches). */
    TimeTracker(const TimeTracker&)            = delete;
    TimeTracker& operator=(const TimeTracker&) = delete;

    /** @brief Move constructor: transfer handle ownership and leave the
     *  source in a "not created" state so its destructor is a no-op. */
    TimeTracker(TimeTracker&& other) noexcept
        : started_{ other.started_ }
        , finished_{ other.finished_ }
        , created_{ other.created_ }
        , stream_{ other.stream_ }
    {
        other.created_ = false;
    }

    /** @brief Move assignment: destroy the current handles first, then
     *  take ownership of the source's handles. */
    TimeTracker& operator=(TimeTracker&& other) noexcept
    {
        if (this != &other) {
            destroy_();
            started_       = other.started_;
            finished_      = other.finished_;
            created_       = other.created_;
            stream_        = other.stream_;
            other.created_ = false;
        }
        return *this;
    }

    /** @brief Set the stream to be tracked */
    void stream(const cudaStream_t& s) { this->stream_ = s; }
    const cudaStream_t& stream() const { return stream_; }

    /** @brief Record the start */
    void start()
    {
        if (!created_)
            create_();
        cudaEventRecord(started_, stream_);
    }

    /** @brief Record the end */
    void finish() { cudaEventRecord(finished_, stream_); }

    /** @brief Return the elapsed time from start to finish in milliseconds */
    Real getMilliseconds()
    {
        float dt;
        cudaEventSynchronize(finished_);
        cudaEventElapsedTime(&dt, started_, finished_);
        return static_cast<Real>(dt);
    }

    /** @brief Reset everything */
    void reset()
    {
        destroy_();
        create_();
    }

    /** @brief Hard reset leaving the cuda Events as not created */
    void hardReset() { destroy_(); }

    /** @brief Destructor destroys the events */
    ~TimeTracker() { destroy_(); }

protected:
    /** @brief Create the events */
    void create_()
    {
        if (!created_) {
            PARAHPOP_CHECK(cudaEventCreate(&started_));
            PARAHPOP_CHECK(
                cudaEventCreateWithFlags(&finished_, cudaEventBlockingSync));
            created_ = true;
        }
    }

    /** @brief Destroy the events */
    void destroy_()
    {
        if (created_) {
            PARAHPOP_CHECK(cudaEventDestroy(started_));
            PARAHPOP_CHECK(cudaEventDestroy(finished_));
            created_ = false;
        }
    }

    /** @brief data members */
    cudaEvent_t started_;
    cudaEvent_t finished_;
    bool created_        = false;
    cudaStream_t stream_ = 0; /* Default stream tracked by default */
};

} // namespace device

namespace host {

using clock_t = std::chrono::high_resolution_clock;
using time_t  = std::chrono::time_point<clock_t>;

/** @brief Execution time tracking structure */
struct TimeTracker {

    /** @brief Default constructor only */
    TimeTracker() { }

    /** @brief Record the start */
    void start() { started_ = clock_t::now(); }

    /** @brief Record the end */
    void finish() { finished_ = clock_t::now(); }

    /** @brief Return the elapsed time from start to finish in milliseconds */
    Real getMilliseconds()
    {
        using ms = std::chrono::milliseconds;
        return std::chrono::duration_cast<ms>(finished_ - started_).count();
    }

    /** @brief Reset everything */
    void reset()
    {
        started_  = time_t();
        finished_ = time_t();
    }

    /** @brief Hard reset (just to comply with the interface) */
    void hardReset() { reset(); }

protected:
    /** @brief data members */
    time_t started_;
    time_t finished_;
};

} // namespace host

/** @brief Time tracker switcher structure */
template<bool Device>
struct TimeTrackerSwitcher {
    using T = device::TimeTracker;
};

template<>
struct TimeTrackerSwitcher<false> {
    using T = host::TimeTracker;
};

} // namespace util
} // namespace paraHPOP