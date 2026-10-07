#pragma once

#include "parm/util/graph/Stream.h"

#ifndef PARM_CPU_ONLY

namespace parm {
namespace util {
namespace graph {

/**
 * @brief RAII helper for recording kernel launches into a CUDA graph via stream capture.
 *
 * Call ``begin()`` before launching kernels on the associated stream, and
 * ``end()`` afterwards to obtain a ``cudaGraph_t`` that can be added to a
 * ``Graph`` with ``Graph::addNode()``.
 *
 * @note Available only when ``PARM_CPU_ONLY`` is not defined.
 * @see Graph::addNode() — consumes the captured graph.
 */
class StreamCapturer {
    using Self = StreamCapturer;

public:
    /** @brief Construct by assigning the stream */
    StreamCapturer(const cudaStream_t& stream = 0)
        : stream_{ stream }
    {
    }

    /** @brief Copy constructor */
    StreamCapturer(const StreamCapturer& other)
        : stream_{ other.stream_ }
    {
    }

    /** @brief Move constructor */
    StreamCapturer(StreamCapturer&& other)
        : stream_{ other.stream_ }
    {
        other.stream_ = 0;
    }

    /** @brief Assignment from stream */
    Self& operator=(const cudaStream_t& str)
    {
        stream_ = str;
        return *this;
    }

    /** @brief Copy assignment */
    Self& operator=(const StreamCapturer& other)
    {
        stream_ = other.stream_;
        return *this;
    }

    /** @brief Move assignment */
    Self& operator=(StreamCapturer&& other)
    {
        stream_       = other.stream_;
        other.stream_ = 0;
        return *this;
    }

    /** @brief Update the stream */
    Self& stream(const cudaStream_t& stream)
    {
        stream_ = stream;
        return *this;
    }

    /** @brief Expose the stream */
    const cudaStream_t& stream() const { return stream_; }

    /** @brief Begin the capture */
    void begin()
    {
        PARM_CHECK(
            cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
    }

    /** @brief End the capture and return the created graph */
    cudaGraph_t end()
    {
        cudaGraph_t graph;
        PARM_CHECK(cudaStreamEndCapture(stream_, &graph));
        return graph;
    }

protected:
    cudaStream_t stream_;
};

} // namespace graph
} // namespace util
} // namespace parm

#endif