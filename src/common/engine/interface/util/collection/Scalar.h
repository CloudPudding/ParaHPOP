#pragma once

#include "interface/typedefs.h"
#include "interface/util/err.h"

namespace interface {
namespace util {
namespace collection {

/** @brief Generic scalar collection */
template<typename DataT>
class Scalar : public feta::buffer::scalar::Array<DataT> {
    using Self    = Scalar;
    using ParentT = typename feta::buffer::scalar::Array<DataT>;

public:
    /** @brief Expose the feta type */
    using FetaT = feta::scalar::Array<DataT>;
    /** @brief Inherit the assignment operators */
    using feta::buffer::scalar::Array<DataT>::operator=;
    /** @brief Inherit the square bracket operator */
    using feta::buffer::scalar::Array<DataT>::operator[];

    /** @brief Expose the collection item types */
    using ItemT      = DataT;
    using ViewT      = DataT&;
    using ConstViewT = const DataT&;

    /** @brief Slices */
    using SliceT      = feta::buffer::detail::Slice<Self>;
    using ConstSliceT = feta::buffer::detail::ConstSlice<Self>;

    /** @brief Re-expose default constructor */
    Scalar()
        : ParentT{}
    {
    }

    /** @brief Re-expose size-based constructor default constructor */
    Scalar(
        const idx_t& sz, const DataT& initVal = feta::buffer::Default<DataT>())
        : ParentT{ sz, initVal }
    {
    }

    /** @brief Construct from std::vector */
    Scalar(const std::vector<DataT>& values)
        : Scalar{ std::move(Scalar::make(values)) }
    {
    }

    /** @brief Construct from Feta */
    Scalar(const FetaT& other)
        : ParentT{ std::move(other.buffer()) }
    {
    }

    /** @brief Construct from Feta Reference */
    Scalar(typename FetaT::GRef other)
        : ParentT{ std::move(other.toBuffer()) }
    {
    }

    /** @brief Re-expose copy constructor */
    Scalar(const ParentT& other)
        : ParentT{ other }
    {
    }
    Scalar(const Scalar& other)
        : ParentT{ other }
    {
    }

    /** @brief Re-expose move constructor */
    Scalar(ParentT&& other)
        : ParentT{ std::move(other) }
    {
    }

    /** @brief Re-expose parent templated constructor */
    template<typename T>
    Scalar(const T& other)
        : ParentT{ other }
    {
    }

    /** @brief Re-expose slice-based constructors */
    Scalar(const typename ParentT::SliceT& other)
        : ParentT{ other }
    {
    }
    Scalar(const typename ParentT::ConstSliceT& other)
        : ParentT{ other }
    {
    }
    Scalar(const SliceT& other)
        : Scalar{ other.size() }
    {
        *this = other;
    }
    Scalar(const ConstSliceT& other)
        : Scalar{ other.size() }
    {
        *this = other;
    }

    /** @brief Re-expose copy assignment operator */
    Scalar& operator=(const ParentT& other)
    {
        ParentT::operator=(other);
        return *this;
    }
    Scalar& operator=(const Scalar& other)
    {
        ParentT::operator=(other);
        return *this;
    }

    /** @brief Re-expose move assignment operator */
    Scalar& operator=(ParentT&& other)
    {
        ParentT::operator=(std::move(other));
        return *this;
    }
    Scalar& operator=(Scalar&& other)
    {
        ParentT::operator=(std::move(other));
        return *this;
    }

    /** @brief Re-expose slice-based assignment operators */
    Scalar& operator=(const typename ParentT::SliceT& other)
    {
        ParentT::operator=(other);
        return *this;
    }
    Scalar& operator=(const typename ParentT::ConstSliceT& other)
    {
        ParentT::operator=(other);
        return *this;
    }
    Scalar& operator=(const SliceT& other)
    {
        PARAHPOP_ASSERT(this->size() == other.size(),
            "Incompatible sizes for Slice Assignment!");

        for (idx_t i = 0; i < this->size(); i++)
            (*this)[i] = other[i];

        return *this;
    }
    Scalar& operator=(const ConstSliceT& other)
    {
        PARAHPOP_ASSERT(this->size() == other.size(),
            "Incompatible sizes for Slice Assignment!");

        for (idx_t i = 0; i < this->size(); i++)
            (*this)[i] = other[i];

        return *this;
    }

    /** @brief Serialise to JSON array */
    json to_json() const
    {
        if (this->empty())
            return json::array();
        return json(this->stdVector());
    }

    /** @brief Push to the given feta reference */
    void pushTo(typename FetaT::GRef arr, const cudaStream_t& stream = 0) const
    {
        arr.fromBuffer(*this, stream);
    }
    void pushTo(typename FetaT::GRef::HandleT harr,
        const cudaStream_t& stream = 0) const
    {
        typename FetaT::GRef arr;
        arr.data_ = harr.data();
        arr.size_ = this->size();
        pushTo(arr, stream);
    }
};


} // namespace collection
} // namespace util
} // namespace interface