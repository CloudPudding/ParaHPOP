#pragma once

#include "feta/buffer/Copy.h"

namespace feta {
namespace buffer {
namespace detail {

/** @brief Reference based iterator */
template<typename ArrayT>
class iterator {
    using Self = iterator;

public:
    using ViewT = typename ArrayT::ViewT;

    iterator(ArrayT& arr, const int& index)
        : arr_{ arr }
        , index_{ index }
    {
    }
    ViewT operator*() { return arr_[index_]; };
    iterator& operator++()
    {
        index_++;
        return *this;
    }
    iterator operator++(int)
    {
        iterator tmp = *this;
        ++(*this);
        return tmp;
    }
    iterator& operator--()
    {
        index_--;
        return *this;
    }
    iterator operator--(int)
    {
        iterator tmp = *this;
        --(*this);
        return tmp;
    }
    bool operator!=(const Self& other) const { return index_ != other.index_; }
    bool operator==(const Self& other) const { return index_ == other.index_; }

private:
    ArrayT& arr_;
    int index_;
};

/** @brief Const Reference based iterator */
template<typename ArrayT>
class const_iterator {
    using Self = const_iterator;

public:
    using ConstViewT = typename ArrayT::ConstViewT;
    const_iterator(const ArrayT& arr, const int& index)
        : arr_{ arr }
        , index_{ index }
    {
    }
    ConstViewT operator*() { return arr_[index_]; };
    const_iterator& operator++()
    {
        index_++;
        return *this;
    }
    const_iterator operator++(int)
    {
        const_iterator tmp = *this;
        ++(*this);
        return tmp;
    }
    const_iterator& operator--()
    {
        index_--;
        return *this;
    }
    const_iterator operator--(int)
    {
        const_iterator tmp = *this;
        --(*this);
        return tmp;
    }
    bool operator!=(const Self& other) const { return index_ != other.index_; }
    bool operator==(const Self& other) const { return index_ == other.index_; }

private:
    const ArrayT& arr_;
    int index_;
};

/** @brief Value based iterator */
template<typename ArrayT, bool IsVector>
class value_iterator {
    using Self = value_iterator;

public:
    using ItemT = typename ArrayT::ItemT;
    value_iterator(const ArrayT& arr, const int& index)
        : arr_{ arr }
        , index_{ index }
    {
    }
    ItemT yield() const
    {
        if constexpr (IsVector)
            return arr_[index_].yield();
        else
            return arr_[index_];
    }
    ItemT operator*() { return yield(); };
    value_iterator& operator++()
    {
        index_++;
        return *this;
    }
    value_iterator operator++(int)
    {
        value_iterator tmp = *this;
        ++(*this);
        return tmp;
    }
    value_iterator& operator--()
    {
        index_--;
        return *this;
    }
    value_iterator operator--(int)
    {
        value_iterator tmp = *this;
        --(*this);
        return tmp;
    }
    bool operator!=(const Self& other) const { return index_ != other.index_; }
    bool operator==(const Self& other) const { return index_ == other.index_; }

private:
    const ArrayT& arr_;
    int index_;
};

} // namespace detail
} // namespace buffer
} // namespace feta