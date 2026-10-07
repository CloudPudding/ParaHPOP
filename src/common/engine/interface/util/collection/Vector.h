#pragma once

#include "interface/typedefs.h"
#include "interface/util/err.h"

namespace interface {
namespace util {
namespace collection {

/** @brief Vector - adds the direct construction from std tpyes */
template<typename DataT_, idx_t VecDims>
class Vector : public feta::buffer::vector::Array<DataT_, VecDims> {
    using Self    = Vector;
    using ParentT = feta::buffer::vector::Array<DataT_, VecDims>;

public:
    using DataT = DataT_;
    using feta::buffer::vector::Array<DataT_, VecDims>::operator=;
    using feta::buffer::vector::Array<DataT_, VecDims>::operator[];

    /** @brief Info */
    using InfoT = typename ParentT::InfoT;

    /** @brief Slices */
    using SliceT      = feta::buffer::detail::Slice<Self>;
    using ConstSliceT = feta::buffer::detail::ConstSlice<Self>;

    /** @brief Expose the item types */
    using ItemT      = typename feta::buffer::vector::Item<DataT_, VecDims>;
    using ViewT      = typename feta::buffer::vector::View<DataT_, VecDims>;
    using ConstViewT = typename feta::buffer::vector::ConstView<DataT, VecDims>;

    /** @brief Re-expose default constructor */
    Vector()
        : ParentT{}
    {
    }

    /** @brief Re-expose size-based constructor default constructor */
    Vector(
        const idx_t& sz, const DataT& initVal = feta::buffer::Default<DataT>())
        : ParentT{ sz, initVal }
    {
    }

    /** @brief Re-expose siz and item-based constructors */
    Vector(const idx_t& sz, const ItemT& initVal)
        : ParentT{ sz, initVal }
    {
    }
    Vector(const idx_t& sz, const ViewT& initVal)
        : ParentT{ sz, initVal }
    {
    }
    Vector(const idx_t& sz, const ConstViewT& initVal)
        : ParentT{ sz, initVal }
    {
    }

    /** @brief Re-expose info based constructor */
    Vector(const InfoT& info, const bool& RowsAreStates)
        : ParentT{ info, RowsAreStates }
    {
    }

    /** @brief construct from linearised std::vector */
    Vector(const std::vector<Real>& linvec, const bool& transpose = false)
        : ParentT{ std::move(ParentT::make(linvec, transpose)) }
    {
    }

    /** @brief construct from two-level nested std::vector */
    Vector(const std::vector<std::vector<Real>>& vecvec,
        const bool& TreatRowsAsVecs = true)
        : ParentT{ std::move(ParentT::make(vecvec, TreatRowsAsVecs)) }
    {
    }

    /** @brief Re-expose copy constructor */
    Vector(const ParentT& other)
        : ParentT{ other }
    {
    }
    Vector(const Vector& other)
        : ParentT{ other }
    {
    }

    /** @brief Construct from parent type */
    Vector(ParentT&& other)
        : ParentT{ std::move(other) }
    {
    }
    Vector(Vector&& other)
        : ParentT{ std::move(other) }
    {
    }

    /** @brief Re-expose slice-based constructors */
    Vector(const typename ParentT::SliceT& other)
        : ParentT{ other }
    {
    }
    Vector(const typename ParentT::ConstSliceT& other)
        : ParentT{ other }
    {
    }
    Vector(const SliceT& other)
        : Vector{ other.size() }
    {
        *this = other;
    }
    Vector(const ConstSliceT& other)
        : Vector{ other.size() }
    {
        *this = other;
    }

    /** @brief Re-expose copy assignment operator */
    Vector& operator=(const ParentT& other)
    {
        ParentT::operator=(other);
        return *this;
    }
    Vector& operator=(const Vector& other)
    {
        ParentT::operator=(other);
        return *this;
    }

    /** @brief Re-expose move assignment operator */
    Vector& operator=(ParentT&& other)
    {
        ParentT::operator=(std::move(other));
        return *this;
    }
    Vector& operator=(Vector&& other)
    {
        ParentT::operator=(std::move(other));
        return *this;
    }

    /** @brief Re-expose slice-based assignment operators */
    Vector& operator=(const typename ParentT::SliceT& other)
    {
        ParentT::operator=(other);
        return *this;
    }
    Vector& operator=(const typename ParentT::ConstSliceT& other)
    {
        ParentT::operator=(other);
        return *this;
    }
    Vector& operator=(const SliceT& other)
    {
        PARAHPOP_ASSERT(this->size() == other.size(),
            "Incompatible sizes for Slice Assignment!");

        for (idx_t i = 0; i < this->size(); i++)
            (*this)[i] = other[i];

        return *this;
    }
    Vector& operator=(const ConstSliceT& other)
    {
        PARAHPOP_ASSERT(this->size() == other.size(),
            "Incompatible sizes for Slice Assignment!");

        for (idx_t i = 0; i < this->size(); i++)
            (*this)[i] = other[i];

        return *this;
    }

    /** @brief Serialise to JSON array of arrays */
    json to_json() const
    {
        if (this->empty())
            return json::array();
        return json(this->stdVector());
    }

    /** @brief Re-expose size */
    const idx_t& size() const { return ParentT::size(); }
};

} // namespace collection
} // namespace util
} // namespace interface