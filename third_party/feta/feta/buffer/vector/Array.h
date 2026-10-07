#pragma once

#include "feta/buffer/Slice.h"
#include "feta/core/memory.h"

namespace feta {
namespace buffer {
namespace vector {

template<typename DataT_, idx_t VectorDim>
using Container
    = core::memory::Container<DataT_, core::memory::Device::CPU, VectorDim>;

/** @brief Forward declarations */
template<typename DataT_, idx_t VectorDim>
class View;
template<typename DataT_, idx_t VectorDim>
class ConstView;

/** @brief Buffer vector item */
template<typename DataT_, idx_t VectorDim>
class Item {
    using ViewT      = View<DataT_, VectorDim>;
    using ConstViewT = ConstView<DataT_, VectorDim>;
    using InfoT      = Info<DataT_>;
    using CopyT      = Copy<DataT_>;
    using ContainerT = std::array<DataT_, VectorDim>;
    friend class View<DataT_, VectorDim>;
    friend class ConstView<DataT_, VectorDim>;

public:
    using DataT = DataT_;

    /** @brief Default constructor initializes all values to 0 */
    Item() = default;

    Item(const std::array<DataT, VectorDim>& arr) { *this = arr; }

    /** @brief Construct from data pointer */
    Item(const DataT* const ptr, const idx_t& stride = 1)
    {
        for (idx_t i = 0; i < VectorDim; i++)
            (*this)[i] = ptr[i * stride];
    }

    /** @brief Copy constructor */
    Item(const Item& other) { *this = other; }
    Item(const ViewT& other) { *this = other; }
    Item(const ConstViewT& other) { *this = other; }

    /** @brief Fill with the given value */
    void fill(const DataT& value)
    {
        for (idx_t i = 0; i < VectorDim; i++)
            (*this)[i] = value;
    }

    /** @brief Copy assignment operator */
    Item& operator=(const std::array<DataT, VectorDim>& other)
    {
        container_ = other;
        return *this;
    }
    Item& operator=(const Item& other)
    {
        container_ = other.container_;
        return *this;
    }
    Item& operator=(const ViewT& other)
    {
        for (idx_t i = 0; i < VectorDim; i++)
            (*this)[i] = other[i];
        return *this;
    }
    Item& operator=(const ConstViewT& other)
    {
        for (idx_t i = 0; i < VectorDim; i++)
            (*this)[i] = other[i];
        return *this;
    }

    /** @brief Return the vector dimension */
    idx_t size() const { return VectorDim; }

    /** @brief Expose the data pointer */
    inline DataT* data() { return container_.data(); }
    inline const DataT* data() const { return container_.data(); }

    /** @brief Access the vector elements */
    DataT& operator[](const int& i)
    {
        FETA_ASSERT(std::abs(i) < VectorDim, "Attempted out of bound access.");
        idx_t idx;
        if (i < 0)
            idx = static_cast<idx_t>(VectorDim + i);
        else
            idx = static_cast<idx_t>(i);
        return container_[idx];
    }
    const DataT& operator[](const int& i) const
    {
        FETA_ASSERT(std::abs(i) < VectorDim, "Attempted out of bound access.");
        idx_t idx;
        if (i < 0)
            idx = static_cast<idx_t>(VectorDim + i);
        else
            idx = static_cast<idx_t>(i);
        return container_[idx];
    }
    DataT& operator[](const idx_t& i) { return container_[i]; }
    const DataT& operator[](const idx_t& i) const { return container_[i]; }

    /** @brief Return a copy of itself */
    Item copy() const { return Item(data()); }

    /** @brief Return a buffer info structure */
    InfoT info() const
    {
        InfoT out;
        out.ptr     = const_cast<DataT*>(data());
        out.ndim    = 1;
        out.shape   = { VectorDim };
        out.strides = { 1 };
        return out;
    }

    /** @brief Return an std::vector containing a copy of the items */
    std::vector<DataT> stdVector() const
    {
        std::vector<DataT> out = std::vector<DataT>(VectorDim);
        CopyT::copy(data(), out);
        return out;
    }

    /** @brief Equality operator */
    friend bool operator==(const Item& a, const Item& b)
    {
        for (idx_t i = 0; i < VectorDim; i++) {
            if (a[i] != b[i])
                return false;
        }
        return true;
    }

    /** @brief Inequality operator */
    friend bool operator!=(const Item& a, const Item& b) { return !(a == b); }

protected:
    ContainerT container_;
};

/** @brief Item view */
template<typename DataT_, idx_t VectorDim>
class View {
    using ItemT      = Item<DataT_, VectorDim>;
    using ConstViewT = ConstView<DataT_, VectorDim>;
    using InfoT      = Info<DataT_>;

    friend class ConstView<DataT_, VectorDim>;

public:
    using DataT = DataT_;

    /** @brief Construct from pointer and stride */
    View(DataT* const ptr, const idx_t& stride)
        : ptr_{ ptr }
        , strides_{ stride }
    {
    }

    /** @brief Copy constructors */
    View(const ItemT& other)
        : ptr_{ const_cast<DataT*>(other.data()) }
        , strides_{ 1 }
    {
    }
    View(const View& other)
        : ptr_{ other.ptr_ }
        , strides_{ other.strides_ }
    {
    }

    /** @brief Copy assignment operator */
    View& operator=(const ItemT& other)
    {
        for (idx_t i = 0; i < VectorDim; i++)
            (*this)[i] = other[i];
        return *this;
    }
    View& operator=(const View& other)
    {
        for (idx_t i = 0; i < VectorDim; i++)
            (*this)[i] = other[i];
        return *this;
    }
    View& operator=(const ConstViewT& other)
    {
        for (idx_t i = 0; i < VectorDim; i++)
            (*this)[i] = other[i];
        return *this;
    }

    /** @brief Fill with the given value */
    void fill(const DataT& value)
    {
        for (idx_t i = 0; i < VectorDim; i++)
            (*this)[i] = value;
    }

    /** @brief Access the vector elements */
    DataT& operator[](const int& i)
    {
        FETA_ASSERT(std::abs(i) < VectorDim, "Attempted out of bound access.");
        idx_t idx;
        if (i < 0)
            idx = static_cast<idx_t>(VectorDim + i);
        else
            idx = static_cast<idx_t>(i);
        return ptr_[idx * strides_];
    }
    const DataT& operator[](const int& i) const
    {
        FETA_ASSERT(std::abs(i) < VectorDim, "Attempted out of bound access.");
        idx_t idx;
        if (i < 0)
            idx = static_cast<idx_t>(VectorDim + i);
        else
            idx = static_cast<idx_t>(i);
        return ptr_[idx * strides_];
    }
    DataT& operator[](const idx_t& i) { return ptr_[i * strides_]; }
    const DataT& operator[](const idx_t& i) const { return ptr_[i * strides_]; }

    /** @brief Expose the main data pointer */
    DataT* data() { return ptr_; }
    const DataT* data() const { return ptr_; }

    /** @brief Return the vector dimension */
    idx_t size() const { return VectorDim; }

    /** @brief Return a value-based item containing this view's values */
    ItemT yield() const { return ItemT(ptr_, strides_); }

    /** @brief Return a buffer info structure */
    InfoT info() const
    {
        InfoT out;
        out.ptr     = ptr_;
        out.ndim    = 1;
        out.shape   = { VectorDim };
        out.strides = { strides_ };
        return out;
    }

    /** @brief Return an std::vector containing a copy of the items */
    std::vector<DataT> stdVector() const { return yield().stdVector(); }

private:
    /** @brief Default constructor for internal use only */
    View() = default;

    DataT* ptr_          = nullptr;
    const idx_t strides_ = 1;
};

/** @brief Const item view */
/** @brief Item view */
template<typename DataT_, idx_t VectorDim>
class ConstView {
    using ItemT = Item<DataT_, VectorDim>;
    using ViewT = View<DataT_, VectorDim>;
    using InfoT = Info<DataT_>;

public:
    using DataT = DataT_;

    /** @brief Construct from pointer and stride */
    ConstView(const DataT* const ptr, const idx_t& stride)
        : ptr_{ const_cast<DataT*>(ptr) }
        , strides_{ stride }
    {
    }

    /** @brief Copy constructors */
    ConstView(const ItemT& other)
        : ptr_{ const_cast<DataT*>(other.data()) }
        , strides_{ 0 }
    {
    }
    ConstView(const ViewT& other)
        : ptr_{ other.ptr_ }
        , strides_{ other.strides_ }
    {
    }
    ConstView(const ConstView& other)
        : ptr_{ other.ptr_ }
        , strides_{ other.strides_ }
    {
    }

    /** @brief Access the vector elements */
    const DataT& operator[](const int& i) const
    {
        FETA_ASSERT(std::abs(i) < VectorDim, "Attempted out of bound access.");
        idx_t idx;
        if (i < 0)
            idx = static_cast<idx_t>(VectorDim + i);
        else
            idx = static_cast<idx_t>(i);
        return ptr_[idx * strides_];
    }
    const DataT& operator[](const idx_t& i) const { return ptr_[i * strides_]; }

    /** @brief Expose the main data pointer */
    const DataT* data() const { return ptr_; }

    /** @brief Return the vector dimension */
    idx_t size() const { return VectorDim; }

    /** @brief Return a value-based item containing this view's values */
    ItemT yield() const { return ItemT(ptr_, strides_); }

    /** @brief Return a buffer info structure */
    InfoT info() const
    {
        InfoT out;
        out.ptr     = ptr_;
        out.ndim    = 1;
        out.shape   = { VectorDim };
        out.strides = { strides_ };
        return out;
    }

    /** @brief Return an std::vector containing a copy of the items */
    std::vector<DataT> stdVector() const { return yield().stdVector(); }

private:
    /** @brief Default constructor for internal use only */
    ConstView() = default;

    DataT* ptr_          = nullptr;
    const idx_t strides_ = 1;
};

/** @brief Buffer array that owns the data */
template<typename DataT_, idx_t VectorDim>
class Array {
    using CopyT      = Copy<DataT_>;
    using Self       = Array;
    using ContainerT = Container<DataT_, VectorDim>;

public:
    /** @brief Expose the info type */
    using InfoT = Info<DataT_>;

    /** @brief Basic properties and data types */
    static constexpr bool IsVector = true;
    using DataT                    = DataT_;
    using ViewT                    = View<DataT_, VectorDim>;
    using ConstViewT               = ConstView<DataT_, VectorDim>;
    using ItemT                    = Item<DataT_, VectorDim>;
    static constexpr idx_t VecDims = VectorDim;

    /** @brief Iterators */
    using iterator       = detail::iterator<Self>;
    using const_iterator = detail::const_iterator<Self>;
    using value_iterator = detail::value_iterator<Self, IsVector>;

    /** @brief Slices */
    using SliceT      = detail::Slice<Self>;
    using ConstSliceT = detail::ConstSlice<Self>;

    /** @brief Factory method to construct from an std::vector */
    static Array make(
        const std::vector<DataT>& vec, const bool& transpose = false)
    {
        FETA_ASSERT(vec.size() % VectorDim == 0,
            "Flat input size is not compatible with this Vector Array type")
        Array out((idx_t)vec.size() / VectorDim);
        if (transpose) {
            for (idx_t i = 0; i < out.size(); i++)
                for (idx_t j = 0; j < VectorDim; j++)
                    out[i][j] = vec[i * VectorDim + j];
        } else {
            CopyT::copy(vec, out.data());
        }
        return out;
    }

    /** @brief Factory method to construct from an std::vector of std::vectors.
     * If the `treatRowsAsItems` flag is set to true and in case none of the two
     * dimension matches the vector dimension, the number of columns is
     * interpreted as the vector dimension. If it is set to false, the number of
     * rows is interpreted as the vector dimension
     */
    static Array make(const std::vector<std::vector<DataT>>& vec,
        const bool& treatRowsAsItems = true)
    {
        /* Assert that the inner dimension is uniform */
        idx_t ncols = vec[0].size();
        for (idx_t i = 1; i < vec.size(); i++)
            FETA_ASSERT(ncols == vec[i].size(), "Non-uniform inner dimension");
        /* Assert that at least one of the two dimensions equal the Vector
         * dimension */
        FETA_ASSERT(vec.size() == VectorDim || ncols == VectorDim,
            "At least one of the two dimensions must be equal to this Vector "
            "Dimension");
        /* Distinguish the cases */
        bool transpose = false;
        if (vec.size() == ncols) {
            /* We need to check whether the transposition preference is active
             * or not*/
            transpose |= treatRowsAsItems;
        } else {
            /* We need to transpose only if ncols == VectorDim*/
            if (ncols == VectorDim)
                transpose = true;
        }
        /* update the actual number of elements */
        idx_t sz = ncols;
        if (transpose)
            sz = vec.size();
        /* We now initialize and copy in the data */
        Array out(sz);
        if (!transpose) {
            DataT* outdata = out.data();
            for (idx_t i = 0; i < VectorDim; i++)
                CopyT::copy(vec[i], outdata + i * out.size_);
        } else {
            for (idx_t i = 0; i < sz; i++)
                for (idx_t j = 0; j < VectorDim; j++)
                    out[i][j] = vec[i][j];
        }
        return out;
    }

    /** @brief Factory method to construct by copying the data from the given
     * pointer and size */
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
        : container_{ sz }
        , size_{ sz }
    {
        init_(initVal);
    }
    Array(const int& sz, const DataT& initVal = Default<DataT>())
        : container_{ static_cast<idx_t>(sz) }
        , size_{ static_cast<idx_t>(sz) }
    {
        FETA_ASSERT(sz >= 0,
            "Negative size cannot be used to construct a buffer array");
        init_(initVal);
    }

    /** @brief Construct by allocating the memory for the given size, and
     * setting all elements to match the given item */
    Array(const idx_t& sz, const ItemT& initVal)
        : container_{ sz }
        , size_{ sz }
    {
        init_(initVal);
    }

    /** @brief Construct by allocating the memory for the given size, and
     * setting all elements to match the given item view */
    Array(const idx_t& sz, const ViewT& initVal)
        : container_{ sz }
        , size_{ sz }
    {
        init_(initVal);
    }

    /** @brief Construct by allocating the memory for the given size, and
     * setting all elements to match the given const item view */
    Array(const idx_t& sz, const ConstViewT& initVal)
        : container_{ sz }
    {
        init_(initVal);
    }

    /** @brief Construct by copying data from the given buffer info. By default,
     * treats the rows as vector dimensions (via the RowsAreVectors flag set to
     * false) if the actual shape cannot be detected from the buffer info. */
    Array(const InfoT& i, bool RowsAreVectors = false)
    {
        if (i.size() > 0) { /* Assert that info is valid */
            FETA_ASSERT(
                i.isValid(), "This info points to a non-allocated buffer");
            /* Assert the shape compatibility */
            FETA_ASSERT(i.ndim == 1 || i.ndim == 2,
                "Only 1D and 2D buffers are supported.");
            FETA_ASSERT(
                i.size() % VectorDim == 0, "Incompatible buffer info size.");
            /* Allocate the memory */
            size_      = i.size() / VectorDim;
            container_ = std::move(ContainerT(size_));
            updateShape_();
            /* Resolve, in element units, the source strides of the vector axis
             * (length size_) and the component axis (length VectorDim). The
             * gather below reads element (vector j, component k) from
             *     i.ptr[j * vecStride + k * compStride]
             * which is correct for any contiguity (C- or F-order) and any
             * shape, so it never re-derives the layout from the contiguity
             * pattern alone. */
            idx_t vecStride  = 1;
            idx_t compStride = 1;
            if (i.ndim == 1) {
                /* A flat buffer is either AoS (rows-are-vectors: each vector's
                 * components are contiguous) or SoA (component-major). */
                if (RowsAreVectors) {
                    vecStride  = VectorDim;
                    compStride = 1;
                } else {
                    vecStride  = 1;
                    compStride = size_;
                }
            } else {
                /* The component axis is the one whose length is VectorDim. When
                 * BOTH axes have length VectorDim (a square buffer) the axis
                 * assignment is genuinely ambiguous and the RowsAreVectors flag
                 * decides it: rows-are-vectors => axis 0 is the vector axis. */
                bool axis0IsComponent;
                if (i.shape[0] == VectorDim && i.shape[1] == VectorDim) {
                    axis0IsComponent = !RowsAreVectors;
                } else {
                    FETA_ASSERT(
                        i.shape[0] == VectorDim || i.shape[1] == VectorDim,
                        "At least one of the two dimensions in the buffer info "
                        "must be equal to this Vector Dimension");
                    axis0IsComponent = (i.shape[0] == VectorDim);
                }
                if (axis0IsComponent) {
                    compStride = i.strides[0];
                    vecStride  = i.strides[1];
                } else {
                    compStride = i.strides[1];
                    vecStride  = i.strides[0];
                }
            }
            /* Fast path: the source already matches the internal component-major
             * (SoA) layout contiguously, so a flat copy reproduces it exactly. */
            if (vecStride == 1 && compStride == size_) {
                CopyT::copy(i.ptr, data(), i.size());
            } else {
/* General strided gather (handles transposition / non-contiguous inputs) */
#pragma omp simd
                for (idx_t j = 0; j < size_; j++) {
                    for (idx_t k = 0; k < VectorDim; k++) {
                        (*this)[j][k] = i.ptr[j * vecStride + k * compStride];
                    }
                }
            }
        }
    }

    /** @brief Copy constructor physically copies the memory */
    Array(const Array& other)
        : Array{ std::move(Array(other.info())) }
    {
    }

    /** @brief Generic copy-constructor from size and type-compatible type */
    Array(const SliceT& other)
        : Array{ other.size() }
    {
        if (size() > 0)
            *this = other;
    }
    Array(const ConstSliceT& other)
        : Array{ other.size() }
    {
        if (size() > 0)
            *this = other;
    }


    /** @brief Move Constructor */
    Array(Array&& other) { *this = std::move(other); }

    /** @brief Fill with the given value */
    void fill(const DataT& value)
    {
        FETA_ASSERT(size_ > 0, "Size of this array is set to 0");
        std::fill(data(), data() + VecDims * size_, value);
    }

    /** @brief Fill with the given Item */
    void fill(const ItemT& item)
    {
        FETA_ASSERT(size_ > 0, "Size of this array is set to 0");
        for (idx_t i = 0; i < size_; i++)
            (*this)[i] = item;
    }
    void fill(const ViewT& item)
    {
        FETA_ASSERT(size_ > 0, "Size of this array is set to 0");
        for (idx_t i = 0; i < size_; i++)
            (*this)[i] = item;
    }
    void fill(const ConstViewT& item)
    {
        FETA_ASSERT(size_ > 0, "Size of this array is set to 0");
        for (idx_t i = 0; i < size_; i++)
            (*this)[i] = item;
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
        shape_     = std::exchange(other.shape_, { VecDims, 0 });
        strides_   = std::exchange(other.strides_, { 1, 1 });
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
        out.shape.reserve(2);
        out.strides.reserve(2);
        for (idx_t i = 0; i < 2; i++) {
            out.shape.push_back(shape_[i]);
            out.strides.push_back(strides_[i]);
        }
        return out;
    }

    /** @brief Return the info corresponding to the transposed array */
    InfoT transposedInfo() const
    {
        InfoT out;
        out.ptr    = const_cast<DataT*>(data());
        out.format = format_;
        out.ndim   = ndim_;
        out.shape.reserve(2);
        out.strides.reserve(2);
        for (idx_t i = 0; i < 2; i++) {
            out.shape.push_back(shape_[1 - i]);
            out.strides.push_back(strides_[1 - i]);
        }
        return out;
    }

    /** @brief Destruct by deallocating the memory */
    ~Array() { reset(); }

    /** @brief Create and copy the data to the given std::vector pf std::vector
     */
    std::vector<std::vector<DataT>> stdVector(
        const bool& transpose = true) const
    {
        /* By default the data is returned with a transposition */
        if (transpose) {
            std::vector<std::vector<DataT>> out
                = std::vector<std::vector<DataT>>(size_);
            for (idx_t i = 0; i < size_; i++) {
                std::vector<DataT> in = std::vector<DataT>(VectorDim);
                for (idx_t j = 0; j < VectorDim; j++) {
                    in[j] = (*this)[i][j];
                }
                out[i] = std::move(in);
            }
            return out;
        } else {
            std::vector<std::vector<DataT>> out;
            /* An unallocated (size-0) array has no data pointer; still return
             * VectorDim empty rows to preserve the component-major shape. */
            if (size_ == 0)
                return std::vector<std::vector<DataT>>(VectorDim);
            for (idx_t i = 0; i < VectorDim; i++) {
                const DataT* src      = i * size_ + data();
                std::vector<DataT> in = std::vector<DataT>(size_);
                CopyT::copy(src, in);
                out.push_back(std::move(in));
            }
            return out;
        }
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

    /** @brief Expose the container */
    inline ContainerT& container() { return container_; }
    inline const ContainerT& container() const { return container_; }

    /** @brief Access the data elements with allocation and bound checks */
    ViewT at(const int& i)
    {
        FETA_ASSERT(std::abs(i) < size_, "Attempted out of bound access.");
        return (*this)[i];
    }
    ConstViewT at(const int& i) const
    {
        FETA_ASSERT(std::abs(i) < size_, "Attempted out of bound access.");
        return (*this)[i];
    }
    ViewT at(const idx_t& i)
    {
        FETA_ASSERT(i < size_, "Attempted out of bound access.");
        return (*this)[i];
    }
    ConstViewT at(const idx_t& i) const
    {
        FETA_ASSERT(i < size_, "Attempted out of bound access.");
        return (*this)[i];
    }

    /** @brief Access the data elements as views of each vector element */
    ViewT operator[](const int& i)
    {
        idx_t idx;
        if (i < 0)
            idx = static_cast<idx_t>(size_ + i);
        else
            idx = static_cast<idx_t>(i);
        return (*this)[idx];
    }
    ConstViewT operator[](const int& i) const
    {
        idx_t idx;
        if (i < 0)
            idx = static_cast<idx_t>(size_ + i);
        else
            idx = static_cast<idx_t>(i);
        return (*this)[idx];
    }
    ViewT operator[](const idx_t& i) { return ViewT(data() + i, capacity()); }
    ConstViewT operator[](const idx_t& i) const
    {
        return ConstViewT(data() + i, capacity());
    }

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

    /** @brief Create a slice, according to the given input indexes, with bounds
     * checking */
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
        makeRoom_(1);
        (*this)[incrShape_()].fill(value);
    }
    void push_back(const ItemT& value)
    {
        makeRoom_(1);
        (*this)[incrShape_()] = value;
    }
    void push_back(const ViewT& value)
    {
        makeRoom_(1);
        (*this)[incrShape_()] = value;
    }
    void push_back(const ConstViewT& value)
    {
        makeRoom_(1);
        (*this)[incrShape_()] = value;
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
    /** @brief Return the 2D shape `[VectorDim, size]`. */
    const std::array<idx_t, 2>& shape() const { return shape_; }
    const std::array<idx_t, 2>& strides() const { return strides_; }
    /** @brief Return the total number of elements */
    idx_t totalSize() const { return VecDims * size_; }
    /** @brief Return the array capacity */
    idx_t capacity() const { return container_.size(); }

    /** @brief Clear the data and the size, without reducing the capacity */
    void clear()
    {
        fill(0);
        size_ = 0;
    }

    /** @brief Return whether the array is empty */
    bool empty() const { return size_ == 0; }

    /** @brief Reset to default, also deallocating the memory and setting the
     * capacity to 0 */
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
                copyInPlace_<VecDims - 1>(container_, newc, size_);
                container_ = std::move(newc);
            }
        }
    }

    /** @brief Expand the container of the given number of elements */
    void expand(const idx_t& addedCapacity)
    {
        reserve(addedCapacity);
        makeRoom_(addedCapacity);
        fillCapacity_(addedCapacity);
    }

protected:
    /** @brief Initialize the container: update shape according to size_ and
     * fill with the given initial value */
    template<typename T>
    void init_(const T& initVal)
    {
        if (size_ == 0)
            return;
        updateShape_();
        fill(initVal);
    }

    /** @brief Recursively copy the data in place for all the dimensions */
    template<idx_t Dim>
    static void copyInPlace_(
        const ContainerT& from, ContainerT& to, const idx_t& size)
    {
        if constexpr (Dim > 0) {
            copyInPlace_<Dim - 1>(from, to, size);
        }
        CopyT::copy(from.template data<Dim>(), to.template data<Dim>(), size);
    }

    /** @brief Resize the array increasing the capacity */
    void resize_()
    {
        ContainerT newc(capacity() + 1000);
        if (size_ > 0) {
            copyInPlace_<VecDims - 1>(container_, newc, size_);
        }
        container_ = std::move(newc);
    }

    /** @brief Shift the elements to make room for N new vector items
     */
    void makeRoom_(const idx_t& N)
    {
        /* Resize if required */
        if (capacity() == 0) {
            resize_();
        } else {
            while (capacity() - size_ < N)
                resize_();
        }
    }

    /** @brief Update shape according to the size */
    void updateShape_()
    {
        shape_[1]   = size_;
        strides_[1] = 1;
        strides_[0] = capacity();
    }

    /** @brief Increase the size by one, update the shape and return the
     * previous end of the array */
    idx_t incrShape_()
    {
        size_++;
        updateShape_();
        return size_ - 1;
    }

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
    static constexpr idx_t ndim_ = 2;
    /** @brief Multi-dimensional array shape */
    std::array<idx_t, 2> shape_ = { VecDims, 0 };
    /** @brief Array strides */
    std::array<idx_t, 2> strides_ = { 1, 1 };
    /** @brief Number of vectors currently stored in the buffer. */
    idx_t size_ = 0;
};

} // namespace vector
} // namespace buffer
} // namespace feta