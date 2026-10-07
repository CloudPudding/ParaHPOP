/* Augmented feta::vector::Item */
#pragma once

#include "paraHPOP/typedefs.h"
#include "interface/util/err.h"

namespace paraHPOP {
namespace util {

/** @brief Augmented feta::vector::Item with unrolled assignment from scalar
 * array */
template<typename DataT, idx_t VecDim>
class AugVecNT : public feta::vector::Item<DataT, VecDim> {
    using ParentT = feta::vector::Item<DataT, VecDim>;
    using Self    = AugVecNT<DataT, VecDim>;
    using ArrayT  = feta::scalar::Array<DataT>;

public:
    /** @brief Factory method that creates a VecOfVec from a Reference Scalar
     * Array */
    DEVICEHOST()
    static AugVecNT<DataT, VecDim> fromScalarArray(
        const typename ArrayT::GRef& arr)
    {
        AugVecNT<DataT, VecDim> vov = arr;
        return vov;
    }

    /** @brief Default constructor */
    DEVICEHOST()
    AugVecNT()
        : ParentT{}
    {
    }

    /** @brief Constructor from parent Type */
    DEVICEHOST()
    AugVecNT(ParentT& parent)
        : ParentT{ parent }
    {
    }

    /** @brief Construct by evaluating an expression */
    template<typename Expr>
    DEVICEHOST()
    AugVecNT(
        const feta::vector::expr::Expression<Expr, typename Expr::ComponentT>&
            expr)
        : ParentT{ expr }
    {
    }

    /** @brief Construct from the given scalar array */
    DEVICEHOST()
    AugVecNT(const typename ArrayT::GRef arr)
    {
        /* Defer to assignment operator */
        *this = arr;
    }

    /** @brief Assign operator from the given scalar array */
    DEVICEHOST()
    Self& operator=(const typename ArrayT::GRef arr)
    {
        this->dimCheck(arr);
        if constexpr (VecDim > 0)
            this->unpackRange_<0, VecDim>(arr);
        return *this;
    }

    DEVICEHOST()
    void unpack(const typename ArrayT::GRef arr)
    {
        /* Call the assignment operator */
        *this = arr;
    }

    /** @brief Pack the values of this vector into the given scalar array */
    DEVICEHOST()
    void pack(typename ArrayT::GRef arr) const
    {
        this->dimCheck(arr);
        if constexpr (VecDim > 0)
            this->packRange_<0, VecDim>(arr);
    }

    /** @brief Return the total size of this VecOfVec */
    DEVICEHOST() static idx_t Size() { return VecDim; }

    /** @brief Return the sum of the elements of this vector */
    DEVICEHOST() DataT sum() { return this->sum_<VecDim - 1>(); }

protected:
    /** @brief Retrieve the elements from the given input scalar array.
     *
     * Divide-and-conquer over the half-open range [lo, hi): the instantiation
     * recursion is O(log VecDim) deep rather than O(VecDim), so the meta
     * vector (which flattens every event's layout — hundreds of entries wide)
     * stays well under the CUDA front-end's template-recursion ceiling. Each
     * component is assigned independently from arr[component], so the result
     * is identical to the former linear unroll regardless of split order. */
    template<idx_t lo, idx_t hi>
    DEVICEHOST()
    void unpackRange_(const typename ArrayT::GRef arr)
    {
        if constexpr (hi - lo == 1) {
            if (arr.size() > 0)
                this->template get<lo>() = arr[lo];
        } else {
            constexpr idx_t mid = lo + (hi - lo) / 2;
            this->unpackRange_<lo, mid>(arr);
            this->unpackRange_<mid, hi>(arr);
        }
    }

    /** @brief Push the elements to the given scalar array (divide-and-conquer
     * over [lo, hi); see unpackRange_ for the depth rationale). */
    template<idx_t lo, idx_t hi>
    DEVICEHOST()
    void packRange_(typename ArrayT::GRef arr) const
    {
        if constexpr (hi - lo == 1) {
            if (arr.size() > 0)
                arr[lo] = this->template get<lo>();
        } else {
            constexpr idx_t mid = lo + (hi - lo) / 2;
            this->packRange_<lo, mid>(arr);
            this->packRange_<mid, hi>(arr);
        }
    }

    /** @brief Return the reduced sum of the elements of this vector */
    template<idx_t row>
    DEVICEHOST()
    DataT sum_() const
    {
        DataT res = 0;
        if constexpr (row >= 1)
            res += this->sum_<row - 1>();
        res += this->template get<row>();
        return res;
    }


    /** @brief check the compatibility of the array dimensions */
    DEVICEHOST()
    void dimCheck(const typename ArrayT::GRef arr) const
    {
        /* if arr.size() == 0, then no event is required on this side. */
        /* set size = 1 to make the check pass even though it will not be
         * triggered */
        idx_t checksize = arr.size() != 0 ? arr.size() : 1;
/** TODO: Create a specific error */
#ifdef __CUDA_ARCH__
        PARAHPOP_GPU_ASSERT(
            checksize == Self::Size(), paraHPOP::err::INCOMPATIBLESIZES);
#else
        PARAHPOP_ASSERT(checksize == Self::Size(),
            "Incompatible Scalar Array and Compile-time Vector sizes.");
#endif
    }
};

} // namespace util
} // namespace paraHPOP