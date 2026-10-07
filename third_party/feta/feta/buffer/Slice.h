#pragma once

#include "feta/buffer/Iter.h"
#include "feta/buffer/detail/IndexTraits.h"

namespace feta {
namespace buffer {
namespace detail {

/** @brief Boolean to proper index converter */
struct BoolToIntIdxs {

    /** @brief Return the number of trues */
    template<typename T>
    static idx_t countTrues(const T& idxs)
    {
        static_assert(IsBoolIndex<T>::value, "Not a boolean-typed index");
        /* Find the size */
        idx_t size = 0;
        for (idx_t i = 0; i < idxs.size(); i++) {
            if (idxs[i])
                size++;
        }
        return size;
    }

    /** @brief Do the conversion*/
    template<typename T>
    static std::vector<int> eval(const T& idxs)
    {
        idx_t size = countTrues<T>(idxs);
        std::vector<int> out;
        if (size > 0) {
            out.resize(size);
            idx_t count = 0;
            for (idx_t i = 0; i < idxs.size(); i++) {
                if (idxs[i])
                    out[count++] = static_cast<int>(i);
            }
        }
        return out;
    }
};
} // namespace detail

/** @brief Range - helps preparing indexes for slices */
class Range {
    using IdxsT = std::vector<int>;

public:
    /** @brief Construct from values */
    Range(const int& start, const int& stop, const int& step = 1)
        : start_{ start }
        , stop_{ stop }
        , step_{ step }
    {
    }

    /** @brief Return the number of elements in this range (exclusive stop semantics). */
    size_t size() const
    {
        /* Zero-length or invalid step yields empty range */
        if (step_ == 0)
            return 0;
        if (stop_ > start_) {
            if (step_ <= 0)
                return 0;
            const int diff = stop_ - start_;
            /* ceil(div) for integers: (diff + step - 1) / step */
            return static_cast<size_t>((diff + step_ - 1) / step_);
        } else if (stop_ < start_) {
            if (step_ >= 0)
                return 0;
            const int diff = start_ - stop_;
            const int negs = -step_;
            return static_cast<size_t>((diff + negs - 1) / negs);
        }
        return 0;
    }

    /** @brief Build and return the sorted list of integer indices described by this range. */
    IdxsT idxs() const
    {
        IdxsT out;
        /* Validate configuration: throw on invalid ranges per tests'
         * expectation */
        if (start_ != stop_) {
            FETA_ASSERT(step_ != 0, "Step must not be 0");
            if (stop_ < start_) {
                FETA_ASSERT(step_ < 0, "Step must be < 0 if stop < start");
            } else if (stop_ > start_) {
                FETA_ASSERT(step_ > 0, "Step must be > 0 if stop > start");
            }
        }
        if (size() > 0) {
            out.resize(size());
            int i       = start_;
            idx_t count = 0;
            if (stop_ < start_) {
                while (i > stop_) {
                    out[count++] = i;
                    i += step_; /* step_ is negative -> move towards stop_ */
                }
            } else {
                while (i < stop_) {
                    out[count++] = i;
                    i += step_;
                }
            }
        }
        return out;
    }

private:
    int start_;
    int stop_;
    int step_;
};

namespace detail {

/** @brief Assign range trait */
template<>
struct IsRange<Range> {
    static constexpr bool value = true;
};

/** @brief Index maker */
template<bool CheckBounds>
struct Indexes {

    /** @brief Create new indexes from other indexes */
    template<typename T>
    static std::vector<int> make(const T& ir, const std::vector<int>& idxs)
    {
        size_t sz = static_cast<size_t>(ir.size());
        std::vector<int> newidxs;
        if constexpr (IsRange<T>::value) {
            newidxs.resize(sz);
            idx_t count = 0;
            for (auto i : ir.idxs()) {
                if constexpr (CheckBounds) {
                    FETA_ASSERT(
                        i < idxs.size(), "Requested out-of-bounds slicing");
                }
                newidxs[count++] = idxs[i];
            }
        } else {
            static_assert(IsBoolIndex<T>::value || IsIntIndex<T>::value,
                "Invalid slicing index input type");
            if constexpr (IsBoolIndex<T>::value) {
                newidxs.resize(BoolToIntIdxs::countTrues<T>(ir));
                idx_t count = 0;
                if constexpr (CheckBounds) {
                    FETA_ASSERT(ir.size() <= idxs.size(),
                        "Requested out-of-bounds slicing");
                }
                for (idx_t i = 0; i < ir.size(); i++)
                    if (ir[i])
                        newidxs[count++] = idxs[i];
            } else {
                newidxs.resize(sz);
                idx_t count = 0;
                for (auto i : ir) {
                    if constexpr (CheckBounds) {
                        FETA_ASSERT(static_cast<size_t>(i) < idxs.size(),
                            "Requested out-of-bounds slicing");
                    }
                    newidxs[count++] = idxs[i];
                }
            }
        }
        return newidxs;
    }

    /** @brief Create new slice indexes from a full non-sliced array */
    template<typename T, typename ArrayT>
    static std::vector<int> make(const T& ir, const ArrayT& arr)
    {
        size_t sz = static_cast<size_t>(ir.size());
        std::vector<int> newidxs;
        if constexpr (IsRange<T>::value) {
            newidxs.resize(sz);
            idx_t count = 0;
            for (auto i : ir.idxs()) {
                if constexpr (CheckBounds) {
                    FETA_ASSERT(
                        i < arr.size(), "Requested out-of-bounds slicing");
                }
                newidxs[count++] = static_cast<int>(i);
            }
        } else {
            static_assert(IsBoolIndex<T>::value || IsIntIndex<T>::value,
                "Invalid slicing index input type");
            if constexpr (IsBoolIndex<T>::value) {
                newidxs.resize(BoolToIntIdxs::countTrues<T>(ir));
                idx_t count = 0;
                if constexpr (CheckBounds) {
                    FETA_ASSERT(ir.size() <= arr.size(),
                        "Requested out-of-bounds slicing");
                }
                for (idx_t i = 0; i < ir.size(); i++)
                    if (ir[i])
                        newidxs[count++] = static_cast<int>(i);
            } else {
                newidxs.resize(sz);
                idx_t count = 0;
                for (auto i : ir) {
                    if constexpr (CheckBounds) {
                        FETA_ASSERT(
                            i < arr.size(), "Requested out-of-bounds slicing");
                    }
                    newidxs[count++] = static_cast<int>(i);
                }
            }
        }
        return newidxs;
    }
};

/** @brief Forward declarations of ConstSlice */
template<typename ArrayT>
class ConstSlice;

/** @brief Array slice */
template<typename ArrayT>
class Slice {
    using IdxsT = std::vector<int>;
    using Self  = Slice;

    friend class ConstSlice<ArrayT>;

public:
    /** @brief Expose the data types */
    using ItemT      = typename ArrayT::ItemT;
    using ViewT      = typename ArrayT::ViewT;
    using ConstViewT = typename ArrayT::ConstViewT;

    /** @brief Expose the iterators */
    using iterator       = detail::iterator<Self>;
    using const_iterator = detail::const_iterator<Self>;
    using value_iterator = detail::value_iterator<Self, ArrayT::IsVector>;

    /** @brief Default constructor is forbidden */
    Slice() = delete;

    /** @brief Construct from the given array and the given indexes */
    Slice(ArrayT& arr, const IdxsT& idxs)
        : arr_{ arr }
        , idxs_{ idxs }
    {
    }

    /** @brief Copy constructor */
    Slice(Slice& other)
        : arr_{ other.arr_ }
        , idxs_{ other.idxs_ }
    {
    }

    /** @brief Move constructor */
    Slice(Slice&& other)
        : arr_{ other.arr_ }
        , idxs_{ std::move(other.idxs_) }
    {
    }

    /** @brief Assignment operator */
    template<typename T>
    Self& operator=(const T& other)
    {
        FETA_ASSERT(
            size() == other.size(), "Incompatible sizes for Slice Assignment!");

        for (idx_t i = 0; i < size(); i++)
            (*this)[i] = other[i];

        return *this;
    }

    /** @brief Move assignment operator */
    Self& operator=(Slice&& other)
    {
        arr_  = other.arr_;
        idxs_ = std::move(other.idxs_);
        return *this;
    }

    /** @brief Access the slice elements */
    ViewT operator[](const idx_t& i) { return arr_[idxs_[i]]; }
    ConstViewT operator[](const idx_t& i) const { return arr_[idxs_[i]]; }
    ViewT operator[](const int& i) { return arr_[idxs_[trueIdx(i)]]; }
    ConstViewT operator[](const int& i) const
    {
        return arr_[idxs_[trueIdx(i)]];
    }

    /** @brief Access the slice elements */
    ViewT at(const idx_t& i) { return arr_(idxs_.at(i)); }
    ConstViewT at(const idx_t& i) const { return arr_(idxs_.at(i)); }
    ViewT at(const int& i) { return arr_(idxs_.at(trueIdx(i))); }
    ConstViewT at(const int& i) const { return arr_(idxs_.at(trueIdx(i))); }

    /** @brief Create a new slice from this slice, according to the given input
     * indexes */
    template<typename T>
    Self operator[](const T& ir)
    {
        return Self(arr_, Indexes<false>::make<T>(ir, idxs_));
    }
    template<typename T>
    ConstSlice<ArrayT> operator[](const T& ir) const
    {
        return ConstSlice<ArrayT>(arr_, Indexes<false>::make<T>(ir, idxs_));
    }

    /** @brief Create a new slice from this slice, according to the given input
     * indexes, with bounds checking */
    template<typename T>
    Self at(const T& ir)
    {
        return Self(arr_, Indexes<true>::make<T>(ir, idxs_));
    }
    template<typename T>
    ConstSlice<ArrayT> at(const T& ir) const
    {
        return ConstSlice<ArrayT>(arr_, Indexes<true>::make<T>(ir, idxs_));
    }

    /** @brief Expose a const reference to the inner array */
    const ArrayT& inner() const { return arr_; }

    /** @brief Return the slice size */
    idx_t size() const { return static_cast<idx_t>(idxs_.size()); }

    /** @brief Iterators */
    iterator begin() { return iterator(*this, 0); }
    iterator end() { return iterator(*this, size()); }
    const_iterator begin() const { return const_iterator(*this, 0); }
    const_iterator end() const { return const_iterator(*this, size()); }
    const_iterator cbegin() const { return const_iterator(*this, 0); }
    const_iterator cend() const { return const_iterator(*this, size()); }
    value_iterator vbegin() const { return value_iterator(*this, 0); }
    value_iterator vend() const { return value_iterator(*this, size()); }

    /** @brief Reverse iterators */
    iterator rbegin() { return iterator(*this, size() - 1); }
    iterator rend() { return iterator(*this, -1); }
    const_iterator rbegin() const { return const_iterator(*this, size() - 1); }
    const_iterator rend() const { return const_iterator(*this, -1); }
    const_iterator crbegin() const { return const_iterator(*this, size() - 1); }
    const_iterator crend() const { return const_iterator(*this, -1); }
    value_iterator vrbegin() const { return value_iterator(*this, size() - 1); }
    value_iterator vrend() const { return value_iterator(*this, -1); }

    /** @brief Return whether the slice is empty */
    bool empty() const { return size() == 0; }

private:
    idx_t trueIdx(const int& i) const
    {
        if (i < 0) {
            return static_cast<idx_t>(idxs_.size() + i);
        } else
            return static_cast<idx_t>(i);
    }

    ArrayT& arr_;
    IdxsT idxs_;
};

/** @brief Array slice */
template<typename ArrayT>
class ConstSlice {
    using IdxsT = std::vector<int>;
    using Self  = ConstSlice;

public:
    /** @brief Expose the data types */
    using ItemT      = typename ArrayT::ItemT;
    using ViewT      = typename ArrayT::ViewT;
    using ConstViewT = typename ArrayT::ConstViewT;

    /** @brief Expose the iterators */
    using iterator       = detail::iterator<Self>;
    using const_iterator = detail::const_iterator<Self>;
    using value_iterator = detail::value_iterator<Self, ArrayT::IsVector>;

    /** @brief Default constructor is forbidden */
    ConstSlice() = delete;

    /** @brief Construct from the given array and the given indexes */
    ConstSlice(const ArrayT& arr, const IdxsT& idxs)
        : arr_{ arr }
        , idxs_{ idxs }
    {
    }

    /** @brief Copy constructor */
    ConstSlice(const ConstSlice& other)
        : arr_{ other.arr_ }
        , idxs_{ other.idxs_ }
    {
    }

    /** @brief Copy constructor */
    ConstSlice(const Slice<ArrayT>& other)
        : arr_{ other.arr_ }
        , idxs_{ other.idxs_ }
    {
    }

    /** @brief Move constructor */
    ConstSlice(ConstSlice&& other)
        : arr_{ other.arr_ }
        , idxs_{ std::move(other.idxs_) }
    {
    }

    /** @brief Move assignment operator */
    Self& operator=(ConstSlice&& other)
    {
        arr_  = other.arr_;
        idxs_ = std::move(other.idxs_);
        return *this;
    }

    /** @brief Access the slice elements */
    ConstViewT operator[](const idx_t& i) const { return arr_[idxs_[i]]; }
    ConstViewT operator[](const int& i) const
    {
        return arr_[idxs_[trueIdx(i)]];
    }

    /** @brief Access the slice elements */
    ConstViewT at(const int& i) const { return arr_(idxs_.at(trueIdx(i))); }

    /** @brief Create a new slice from this slice, according to the given input
     * indexes */
    template<typename T>
    Self operator[](const T& ir) const
    {
        return Self(arr_, Indexes<false>::make<T>(ir, idxs_));
    }


    /** @brief Create a new slice from this slice, according to the given input
     * indexes, with bounds checking */
    template<typename T>
    Self at(const T& ir) const
    {
        return Self(arr_, Indexes<true>::make<T>(ir, idxs_));
    }

    /** @brief Expose a const reference to the inner array */
    const ArrayT& inner() const { return arr_; }

    /** @brief Return the slice size */
    idx_t size() const { return static_cast<idx_t>(idxs_.size()); }

    /** @brief Iterators */
    const_iterator begin() const { return const_iterator(*this, 0); }
    const_iterator end() const { return const_iterator(*this, size()); }
    const_iterator cbegin() const { return const_iterator(*this, 0); }
    const_iterator cend() const { return const_iterator(*this, size()); }
    value_iterator vbegin() const { return value_iterator(*this, 0); }
    value_iterator vend() const { return value_iterator(*this, size()); }

    /** @brief Reverse iterators */
    const_iterator rbegin() const { return const_iterator(*this, size() - 1); }
    const_iterator rend() const { return const_iterator(*this, -1); }
    const_iterator crbegin() const { return const_iterator(*this, size() - 1); }
    const_iterator crend() const { return const_iterator(*this, -1); }
    value_iterator vrbegin() const { return value_iterator(*this, size() - 1); }
    value_iterator vrend() const { return value_iterator(*this, -1); }

    /** @brief Return whether the slice is empty */
    bool empty() const { return size() == 0; }

private:
    /** @brief Return the true positive index */
    idx_t trueIdx(const int& i) const
    {
        if (i < 0) {
            return static_cast<idx_t>(idxs_.size() + i);
        } else
            return static_cast<idx_t>(i);
    }

    const ArrayT& arr_;
    IdxsT idxs_;
};

} // namespace detail
} // namespace buffer
} // namespace feta