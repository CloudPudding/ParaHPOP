#pragma once

#include "feta/err/throw.h"
#include "feta/typedefs.h"

#include <array>
#include <iostream>
#include <sstream>
#include <utility>

namespace feta {
namespace buffer {

/** @brief Default data */
template<typename T>
constexpr T Default()
{
    if constexpr (std::is_same<T, bool>::value)
        return false;
    else if constexpr (std::is_arithmetic<T>::value)
        return 0;
    else
        return T();
}

/** @brief Buffer info structure */
template<typename DataT>
struct Info {

    /** @brief Default constructor */
    Info() = default;

    /** @brief Copy constructor */
    Info(const Info& other) { *this = other; }

    /** @brief Move constructor */
    Info(Info&& other) { *this = std::move(other); }

    /** @brief Copy assignment operator */
    Info& operator=(const Info& other)
    {
        this->ptr     = other.ptr;
        this->format  = other.format;
        this->ndim    = other.ndim;
        this->shape   = other.shape;
        this->strides = other.strides;

        return *this;
    }

    /** @brief Move assignment operator */
    Info& operator=(Info&& other)
    {
        this->ptr     = std::exchange(other.ptr, nullptr);
        this->format  = std::exchange(other.format, "");
        this->ndim    = std::exchange(other.ndim, 0);
        this->shape   = std::exchange(other.shape, std::vector<idx_t>());
        this->strides = std::exchange(other.strides, std::vector<idx_t>());

        return *this;
    }

    /** @brief Return the overall array size (as number of elements) */
    static idx_t size(const std::vector<idx_t>& shp)
    {
        if (shp.empty())
            return 0;
        else {
            idx_t out = 1;
            for (const idx_t& dim : shp)
                out *= dim;
            return out;
        }
    }
    idx_t size() const { return Info::size(shape); }

    /** @brief Return the overall array size in bytes */
    idx_t byteSize() const { return itemSize * size(); }

    /** @brief Return a typed buffer pointer */
    void* voidptr() const { return static_cast<void*>(ptr); }

    /** @brief Return the bytes strides */
    std::vector<ssize_t> outShape() const
    {
        std::vector<ssize_t> out;
        for (const idx_t& dim : shape)
            out.push_back(static_cast<ssize_t>(dim));
        return out;
    }

    /** @brief Return the strides cast to `ssize_t` for interop with external buffer APIs. */
    std::vector<ssize_t> outStrides() const
    {
        std::vector<ssize_t> out;
        for (const idx_t& stride : strides)
            out.push_back(static_cast<ssize_t>(stride));
        return out;
    }

    /** @brief Return the bytes strides */
    std::vector<ssize_t> byteStrides() const
    {
        std::vector<ssize_t> out;
        for (const idx_t& stride : strides)
            out.push_back(static_cast<ssize_t>(stride * itemSize));
        return out;
    }

    /** @brief Return whether this buffer info points to something valid */
    bool isValid() const { return ptr != nullptr; }

    /** @brief Pointer to the buffer*/
    DataT* ptr = nullptr;
    /** @brief Size of one scalar in bytes */
    static constexpr idx_t itemSize = sizeof(DataT);
    /** @brief Format description */
    std::string format = "";
    /** @brief Number of dimensions */
    idx_t ndim = 0;
    /** @brief Multi-dimensional array shape */
    std::vector<idx_t> shape = std::vector<idx_t>();
    /** @brief Array strides */
    std::vector<idx_t> strides = std::vector<idx_t>();
};

} // namespace buffer
} // namespace feta