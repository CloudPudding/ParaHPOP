#pragma once

#include "brie/typedefs.h"

namespace brie {
namespace core {

/**
 * @brief Basic class for constant interval length ephemeris data
 *
 */
template<vecdim_t VecDims, bool UseTexture, bool work>
class ConstIntBodyUnit : public feta::scalar::texture::Array<Real,
                             UseTexture>::template Ref<work> {
protected:
    using ParentT       = typename feta::scalar::texture::Array<Real,
              UseTexture>::template Ref<work>;
    using RefVecRTArray = typename feta::vector::texture::Array<Real, VecDims,
        UseTexture>::template Ref<work>;

    using CoeffsT = typename feta::vector::texture::Array<Real, VecDims,
        UseTexture>::template Ref<work>;

    /* Coefficients segment */
    template<vecdim_t nDims>
    using CoeffsSegmentT = typename feta::vector::texture::Array<Real, nDims,
        UseTexture>::template Ref<work>;

public:
    /** @brief Expose the texture object */
    using TexT = typename ParentT::TexT;

    /**
     * @brief Factory method to construct the constant interval length body unit
     *
     */
    DEVICEHOST()
    static ConstIntBodyUnit make(Real* const data, const idx_t& size,
        const idx_t& bodyUnitOffset, const idx_t& nIntervals, const idx_t& pdeg,
        const TexT& tex = 0)
    {
        ConstIntBodyUnit out;
        out.data_           = data + bodyUnitOffset;
        out.size_           = size;
        out.bodyUnitOffset_ = bodyUnitOffset + 1 + nIntervals;
        out.nIntervals_     = nIntervals;
        out.pdeg_           = pdeg;
        out.tex_            = tex;
        out.texOffset_      = bodyUnitOffset;
        return out;
    }

    /**
     * @brief Return the interval radius (half length)
     *
     * @return intervalRadius
     *
     */
    DEVICEHOST() inline Real intervalRadius() const { return *(this->data()); }

    /**
     * @brief Return a pointer to the start of the initial interval epochs
     *
     * @return pointerToStartOfIntervals
     *
     */
    DEVICEHOST() inline Real* intStarts() const { return this->data() + 1; }

    /**
     * @brief Return a pointer the interpolating coefficients
     *
     * @return coefficientPointer
     *
     */
    DEVICEHOST() inline Real* coeffsStart() const
    {
        return intStarts() + nIntervals_;
    }

    /**
     * @brief Return a non-owning vector array reference containing the
     * interpolating coefficients.
     *
     * @return vectorArrayOfCoefficients
     *
     */
    DEVICEHOST() CoeffsT coeffs() const
    {
        idx_t totSize = nIntervals_ * (pdeg_ + 1);
        CoeffsT out;
        out.data_      = this->coeffsStart();
        out.nVecs_     = totSize;
        out.dimOffset_ = totSize;
        if constexpr (UseTexture) {
            out.tex_       = this->tex();
            out.texOffset_ = this->bodyUnitOffset_;
        }
        return out;
    }

    /** @brief Return a non-owning scalar array reference containing the start
     * of the intervals */
    DEVICEHOST() ParentT startOfIntervals() const
    {
        ParentT out;
        out.data_ = this->intStarts();
        out.size_ = nIntervals_;
        if constexpr (UseTexture) {
            out.tex_       = this->tex();
            out.texOffset_ = this->bodyUnitOffset_ - this->nIntervals_;
        }
        return out;
    }

    /** @brief Data members made public for PODification */

    /** @brief Data offset of this body unit in the full ephemeris data */
    idx_t bodyUnitOffset_ = 0;
    /** @brief Number of intervals in this body unit */
    idx_t nIntervals_ = 0;
    /** @brief Polynomial degree of the interpolants in this body unit */
    idx_t pdeg_ = 0;
};

} // namespace core
} // namespace brie