/* Compile-time vector of vectors */
#pragma once

#include "paraHPOP/typedefs.h"
#include "paraHPOP/util/AugVecNT.h"
#include "interface/util/err.h"

namespace paraHPOP {
namespace util {

/** @brief Row major-style vector of vectors */
template<typename DataT, idx_t RowDim, idx_t ColDim>
class VecOfVec : public AugVecNT<DataT, RowDim * ColDim> {
    using ParentT = AugVecNT<DataT, RowDim * ColDim>;
    using Self    = VecOfVec<DataT, RowDim, ColDim>;

public:
    /* Scalar array type for packing/unpacking these data */
    using ArrayT    = feta::scalar::Array<DataT>;
    using VecArrayT = feta::vector::Array<DataT, ColDim>;

    /** @brief Row type */
    using RowT = AugVecNT<DataT, ColDim>;
    /** @brief Column type */
    using ColT = AugVecNT<DataT, RowDim>;

    /** @brief Factory method that creates a VecOfVec from a Reference Scalar
     * Array */
    DEVICEHOST()
    static VecOfVec<DataT, RowDim, ColDim> fromScalarArray(
        const typename ArrayT::GRef& arr)
    {
        return Self(std::move(ParentT::fromScalarArray(arr)));
    }

    /** @brief Default Constructor */
    DEVICEHOST()
    VecOfVec()
        : ParentT{}
    {
    }

    /** @brief Inherit constructors */
    DEVICEHOST()
    VecOfVec(ParentT&& other)
        : ParentT{ std::move(other) }
    {
    }

    /** @brief Override get method for the two-d logic */
    template<idx_t row, idx_t col>
    DEVICEHOST()
    DataT& get()
    {
        return ParentT::template get<row * ColDim + col>();
    }
    /** @brief Override get method for the two-d logic */
    template<idx_t row, idx_t col>
    DEVICEHOST()
    DataT get() const
    {
        return ParentT::template get<row * ColDim + col>();
    }

    /** @brief Extract the given row */
    template<idx_t row>
    DEVICEHOST()
    RowT getRow() const
    {
        return RowT(this->template segment<row * ColDim, ColDim>());
    }

    /** @brief Assign the given row */
    template<idx_t row>
    DEVICEHOST()
    Self& assignRow(const RowT& therow)
    {
        this->template segment<row * ColDim, ColDim>() = therow;
        return *this;
    }

    /** @brief Extract the given column */
    template<idx_t col>
    DEVICEHOST()
    ColT getCol() const
    {
        ColT thecol;
        // ColumnRecursion<col>::get(thecol, *this);
        this->recurseGetCol_<RowDim - 1, col>(thecol);
        return thecol;
    }

    /** @brief Assign the given column */
    template<idx_t col>
    DEVICEHOST()
    Self& assignCol(const ColT& thecol)
    {
        this->recurseAssignCol_<RowDim - 1, col>(thecol);
        return *this;
    }

protected:
    /** @brief Get column recursion */
    template<idx_t row, idx_t col>
    DEVICEHOST()
    void recurseGetCol_(ColT& out) const
    {
        if constexpr (row >= 1)
            this->recurseGetCol_<row - 1, col>(out);
        out.template get<row>() = this->template get<row, col>();
    }

    /** @brief Assign column recursion */
    template<idx_t row, idx_t col>
    DEVICEHOST()
    void recurseAssignCol_(const ColT& thecol)
    {
        if constexpr (row >= 1)
            this->recurseAssignCol_<row - 1, col>(thecol);
        this->template get<row, col>() = thecol.template get<row>();
    }

    /** @brief Check that the dimension of the given vector array are compatible
     * with this vec of vec */
    DEVICEHOST() static void dimCheck_(const typename VecArrayT::GRef arr)
    {
        static_assert(
            ColDim == VecArrayT::VecDims, "Incompatible column dimensions.");

#ifdef __CUDA_ARCH__
        PARAHPOP_GPU_ASSERT(
            arr.size() == Self::Size(), paraHPOP::err::INCOMPATIBLEVECSIZES);
#else
        PARAHPOP_ASSERT((idx_t)arr.size() == (idx_t)RowDim,
            "Incompatible VectorArray elements and Row Dimension.");
#endif
    }
};

} // namespace util
} // namespace paraHPOP