#pragma once

#include "feta/buffer/Slice.h"
#include "feta/core/Functors.h"
#include "feta/core/memory.h"

namespace feta {
namespace buffer {
namespace scalar {

/* Declare the container type */
template<typename DataT>
using Container = core::memory::Container<DataT, core::memory::Device::CPU>;

/** @brief Buffer array that owns the data */
template<typename DataT_>
class Array {
    using Self       = Array;
    using CopyT      = Copy<DataT_>;
    using ContainerT = Container<DataT_>;

public:
    /** @brief Expose the info type */
    using InfoT = Info<DataT_>;

    /** @brief Basic properties and data types */
    static constexpr bool IsVector = false;
    using DataT                    = DataT_;
    using ItemT                    = DataT;
    using ViewT                    = DataT&;
    using ConstViewT               = const DataT&;

    /** @brief Iterators */
    using iterator       = detail::iterator<Self>;
    using const_iterator = detail::const_iterator<Self>;
    using value_iterator = detail::value_iterator<Self, IsVector>;

    /** @brief Slices */
    using SliceT      = detail::Slice<Self>;
    using ConstSliceT = detail::ConstSlice<Self>;

    /** @brief Factory method to construct from an std::vector */
    static Array make(const std::vector<DataT>& vec)
    {
        Array out(static_cast<idx_t>(vec.size()));
        CopyT::copy(vec, out.data());
        return out;
    }

    /** @brief Factory method to construct by copying the data from the
     * given pointer and size */
    static Array make(const DataT* const src, const idx_t& sz)
    {
        Array out(sz);
        CopyT::copy(src, out.data(), sz);
        return out;
    }

    /** @brief Default constructor */
    Array() = default;

    /** @brief Construct by allocating the memory for the given 1D overall
     * size
     */
    Array(const idx_t& sz, const DataT& initVal = Default<DataT>())
        : container_{ sz, initVal }
        , size_{ sz }
    {
        if (sz > 0) {
            updateShape_();
        }
    }
    Array(const int& sz, const DataT& initVal = Default<DataT>())
    {

        FETA_ASSERT(sz >= 0,
            "Negative size cannot be used to construct a buffer array");
        if (sz > 0) {
            container_ = std::move(ContainerT(static_cast<idx_t>(sz), initVal));
            size_      = static_cast<idx_t>(sz);
            updateShape_();
        }
    }

    /** @brief Construct by copying data from the given buffer info */
    Array(const InfoT& i)
    {
        if (i.size() > 0) {
            /* Assert that info is valid */
            if (!i.isValid()) {
                std::stringstream msg;
                msg << "This info points to a non-allocated buffer";
                if constexpr (std::is_same<DataT, bool>::value)
                    msg << ", for bools ";
                else if constexpr (std::is_arithmetic<DataT>::value)
                    msg << ", for arithmetic type";
                else
                    msg << ", for complex type";
                msg << ", with size " << size();
                FETA_THROW(std::runtime_error, msg.str().c_str());
            }
            /* Allocate and copy the data */
            container_ = std::move(ContainerT(i.size()));
            size_      = i.size();
            updateShape_();
            CopyT::copy(i.ptr, data(), i.size());
        }
    }

    /** @brief Copy constructor physically copies the memory */
    Array(const Array& other)
        : Array{ std::move(Array(other.info())) }
    {
    }

    /** @brief Generic copy-constructor from size and type-compatible type
     */
    Array(const SliceT& other)
        : Array{ other.size() }
    {
        *this = other;
    }
    Array(const ConstSliceT& other)
        : Array{ other.size() }
    {
        *this = other;
    }

    /** @brief Move Constructor */
    Array(Array&& other) { *this = std::move(other); }

    /** @brief Fill with the given value */
    void fill(const DataT& value = Default<DataT>())
    {
        FETA_ASSERT(size_ > 0, "Size of this array is set to 0");
        container_.fill(value);
    }

    /** @brief Copy assignment copies the memory and the attributes */
    Array& operator=(const Array& other)
    {
        *this = std::move(Array(other.info()));
        return *this;
    }

    /** @brief Move assignment is allowed */
    Array& operator=(Array&& other)
    {
        reset();

        /* Take over other */
        container_ = std::move(other.container_);
        format_    = std::exchange(other.format_, "");
        shape_     = std::exchange(other.shape_, { 0 });
        size_      = std::exchange(other.size_, 0);

        return *this;
    }

    /** @brief Generic assignment from size and type-compatible type */
    Array& operator=(const SliceT& other)
    {
        FETA_ASSERT(
            size() == other.size(), "Incompatible sizes for Slice Assignment!");

        for (idx_t i = 0; i < size(); i++)
            (*this)[i] = other[i];

        return *this;
    }
    Array& operator=(const ConstSliceT& other)
    {
        FETA_ASSERT(
            size() == other.size(), "Incompatible sizes for Slice Assignment!");

        for (idx_t i = 0; i < size(); i++)
            (*this)[i] = other[i];

        return *this;
    }

    /** @brief Return the buffer info corresponding to this array */
    InfoT info() const
    {
        InfoT out;
        if (!container_.isValid())
            return out;
        out.ptr    = const_cast<DataT*>(data());
        out.format = format_;
        out.ndim   = ndim_;
        out.shape.push_back(shape_[0]);
        out.strides.push_back(strides_[0]);
        return out;
    }

    /** @brief Expose the container */
    inline ContainerT& container() { return container_; }
    inline const ContainerT& container() const { return container_; }

    /** @brief Destruct by deallocating the memory */
    ~Array() { reset(); }

    /** @brief Create and copy the data to the given std::vector */
    std::vector<DataT> stdVector() const
    {
        /* An unallocated (size-0) array has no data pointer to read; return an
         * empty vector rather than dereferencing it. */
        if (size() == 0)
            return {};
        std::vector<DataT> out = std::vector<DataT>(size());
        CopyT::copy(data(), out);
        return out;
    }

    /** @brief Return the format description */
    const std::string& format() const { return format_; }
    /** @brief Update the format description */
    Array& format(const std::string& newf)
    {
        format_ = newf;
        return *this;
    }

    /** @brief Expose the data */
    inline DataT* data() { return container_.data(); }
    inline const DataT* data() const { return container_.data(); }

    /** @brief Access the data elements with bound and allocation checks */
    DataT& at(const int& i)
    {
        FETA_ASSERT(std::abs(i) < size_, "Attempted out of bound access.");
        return (*this)[i];
    }
    const DataT& at(const int& i) const
    {
        FETA_ASSERT(std::abs(i) < size_, "Attempted out of bound access.");
        return (*this)[i];
    }
    DataT& at(const idx_t& i)
    {
        FETA_ASSERT(i < size_, "Attempted out of bound access.");
        return (*this)[i];
    }
    const DataT& at(const idx_t& i) const
    {
        FETA_ASSERT(i < size_, "Attempted out of bound access.");
        return (*this)[i];
    }

    /** @brief Access the data elements */
    DataT& operator[](const int& i)
    {
        idx_t idx;
        if (i < 0) {
            idx = static_cast<idx_t>(size_ + i);
        } else
            idx = static_cast<idx_t>(i);
        return data()[idx];
    }
    const DataT& operator[](const int& i) const
    {
        idx_t idx;
        if (i < 0) {
            idx = static_cast<idx_t>(size_ + i);
        } else
            idx = static_cast<idx_t>(i);
        return data()[idx];
    }
    DataT& operator[](const idx_t& i) { return data()[i]; }
    const DataT& operator[](const idx_t& i) const { return data()[i]; }

    /** @brief Create a slice, according to the given input indexes */
    template<typename T>
    SliceT operator[](const T& ir)
    {
        return SliceT(*this, detail::Indexes<false>::make<T, Self>(ir, *this));
    }
    template<typename T>
    ConstSliceT operator[](const T& ir) const
    {
        return ConstSliceT(
            *this, detail::Indexes<false>::make<T, Self>(ir, *this));
    }

    /** @brief Create a slice, according to the given input indexes, with
     * bounds checking */
    template<typename T>
    SliceT at(const T& ir)
    {
        return SliceT(*this, detail::Indexes<true>::make<T, Self>(ir, *this));
    }
    template<typename T>
    ConstSliceT at(const T& ir) const
    {
        return ConstSliceT(
            *this, detail::Indexes<true>::make<T, Self>(ir, *this));
    }

    /** @brief Add element at the end of the array */
    void push_back(const DataT& value)
    {
        if (size_ == capacity())
            resize_();
        data()[size_++] = value;
        updateShape_();
    }

    /** @brief Extend the array, adding elements at the end of the array */
    template<typename T>
    void extend(const T& other)
    {
        idx_t oldsize = size();
        expand(other.size());
#pragma omp simd
        for (idx_t i = 0; i < other.size(); i++) {
            (*this)[oldsize + i] = other[i];
        }
    }

    /** @brief Iterators */
    iterator begin() { return iterator(*this, 0); }
    iterator end() { return iterator(*this, size_); }
    const_iterator begin() const { return const_iterator(*this, 0); }
    const_iterator end() const { return const_iterator(*this, size_); }
    const_iterator cbegin() const { return const_iterator(*this, 0); }
    const_iterator cend() const { return const_iterator(*this, size_); }
    value_iterator vbegin() const { return value_iterator(*this, 0); }
    value_iterator vend() const { return value_iterator(*this, size_); }

    /** @brief Reverse iterators */
    iterator rbegin() { return iterator(*this, size_ - 1); }
    iterator rend() { return iterator(*this, -1); }
    const_iterator rbegin() const { return const_iterator(*this, size_ - 1); }
    const_iterator rend() const { return const_iterator(*this, -1); }
    const_iterator crbegin() const { return const_iterator(*this, size_ - 1); }
    const_iterator crend() const { return const_iterator(*this, -1); }
    value_iterator vrbegin() const { return value_iterator(*this, size_ - 1); }
    value_iterator vrend() const { return value_iterator(*this, -1); }

    /** @brief Return the array size */
    const idx_t& size() const { return size_; }
    /** @brief Return the array capacity */
    idx_t capacity() const { return container_.size(); }

    /** @brief Clear the data and the size, without reducing the capacity */
    void clear()
    {
        fill();
        size_ = 0;
    }

    /** @brief Whether the container is allocated or not */
    bool allocated() const { return container_.isValid(); }

    /** @brief Return whether the array is empty */
    bool empty() const { return size_ == 0; }

    /** @brief Reset to default, also deallocating the memory and setting
     * the capacity to 0 */
    void reset()
    {
        if (capacity() > 0) {
            container_ = std::move(ContainerT());
            size_      = 0;
            format_    = "";
            updateShape_();
        }
    }

    /** @brief Reserve the given memory, possibly updating the container's
     * capacity */
    void reserve(const idx_t& addedCapacity)
    {
        if (capacity() == 0) {
            container_ = std::move(ContainerT(addedCapacity));
        } else {
            /* We may need to increase the capacity */
            idx_t freespace = capacity() - size_;
            if (addedCapacity > freespace) {
                idx_t newcapacity = capacity() + (addedCapacity - freespace);
                ContainerT newc(newcapacity);
                CopyT::copy(data(), newc.data(), size_);
                container_ = std::move(newc);
            }
        }
    }

    /** @brief Expand the container of the given number of elements */
    void expand(const idx_t& addedCapacity)
    {
        reserve(addedCapacity);
        fillCapacity_(addedCapacity);
    }

protected:
    /** @brief Resize the array increasing the capacity */
    void resize_()
    {
        ContainerT newc(capacity() + 1000);
        if (size_ > 0) {
            CopyT::copy(data(), newc.data(), size_);
        }
        container_ = std::move(newc);
    }

    /** @brief Update shape according to the size */
    void updateShape_() { shape_[0] = size_; }

    /** @brief Increase the size to match the capacity */
    void fillCapacity_(const idx_t& addedCapacity)
    {
        size_ += addedCapacity;
        updateShape_();
    }

    /** @brief The container */
    ContainerT container_;
    /** @brief Size of one scalar in bytes */
    static constexpr idx_t itemSize_ = sizeof(DataT);
    /** @brief Format description */
    std::string format_ = "";
    /** @brief Number of dimensions */
    static constexpr idx_t ndim_ = 1;
    /** @brief Multi-dimensional array shape */
    std::array<idx_t, 1> shape_ = { 0 };
    /** @brief Array strides */
    static const std::array<idx_t, 1> strides_;
    /** @brief Number of elements currently stored in the buffer. */
    idx_t size_ = 0;
};

/** @brief Initialize the strides */
template<typename DataT>
const std::array<idx_t, 1> Array<DataT>::strides_ = { 1 };

} // namespace scalar

/* Index-type traits for scalar::Array are centralized in
   feta/buffer/detail/IndexTraits.h (one IsIntIndex/IsBoolIndex partial spec
   keyed on the element-type predicate covers every Array<T>). */
} // namespace buffer
} // namespace feta