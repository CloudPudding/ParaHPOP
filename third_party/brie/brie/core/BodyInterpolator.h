#pragma once

#include "brie/typedefs.h"
#include "brie/util.h"

namespace brie {
namespace core {

/** @brief Helpers */
template<idx_t VectorDim, bool UseTexture>
struct ChebyshevHelper {

    /* The vector dimension */
    static constexpr idx_t VecDims = VectorDim;

    /* the vector coefficients type */
    using VecT =
        typename feta::vector::texture::Array<Real, VecDims, UseTexture>::GRef;


    /* The corresponding parm polynomial coefficients type */
    using CoeffsT =
        typename parm::interpolate::detail::PolynomialCoefficients<VecT>;

    /* The scalar interval type */
    using ScalarT =
        typename feta::scalar::texture::Array<Real, UseTexture>::GRef;

    /* The corresponding Parm interpolator type */
    using InterpolatorT =
        typename parm::interpolate::chebyshev::GlobalInterpolator<CoeffsT,
            ScalarT>;
};

/** @brief Direct BRIE interpolator. Replaces the Body-unit interpolation logic
 * with more lightweight routines */
template<bool UseTexture>
class BodyInterpolator : public ChebyshevHelper<6, UseTexture>::InterpolatorT {
    using ParentT = typename ChebyshevHelper<6, UseTexture>::InterpolatorT;
    using Self    = BodyInterpolator;
    using ScalarT = typename ChebyshevHelper<6, UseTexture>::ScalarT;
    using CoeffsT = typename ChebyshevHelper<6, UseTexture>::CoeffsT;
    using TexT    = typename ScalarT::TexT;

    /* Array types */
    template<bool work>
    using Vec3RArrT =
        typename feta::vector::Array<Real, 3>::template Ref<work>::HandleT;
    template<bool work>
    using Vec6RArrT =
        typename feta::vector::Array<Real, 6>::template Ref<work>::HandleT;

public:
    /** @brief Create the coefficients */
    DEVICEHOST()
    static CoeffsT makeCoeffs(const Real* const data,
        const idx_t& bodyUnitOffset, const idx_t& nIntervals, const idx_t& pdeg,
        const TexT& tex = 0)
    {
        typename CoeffsT::CoeffsT vecData;
        vecData.data_
            = const_cast<brie::Real*>(data + bodyUnitOffset + 1 + nIntervals);
        vecData.nVecs_     = nIntervals * (pdeg + 1);
        vecData.dimOffset_ = nIntervals * (pdeg + 1);
        if constexpr (UseTexture) {
            vecData.tex_       = tex;
            vecData.texOffset_ = bodyUnitOffset + 1 + nIntervals;
        }
        return CoeffsT::make(vecData, pdeg);
    }

    /** @brief Create the intervals */
    DEVICEHOST()
    static ScalarT makeIntervals(const Real* const data,
        const idx_t& bodyUnitOffset, const idx_t& nIntervals,
        const TexT& tex = 0)
    {
        ScalarT intervals;
        intervals.data_ = const_cast<brie::Real*>(data + bodyUnitOffset + 1);
        intervals.size_ = nIntervals;
        if constexpr (UseTexture) {
            intervals.tex_       = tex;
            intervals.texOffset_ = bodyUnitOffset + 1;
        }
        return intervals;
    }

    /** @brief Create */
    DEVICEHOST()
    static BodyInterpolator make(const Real* const data, const Real& radius,
        const idx_t& bodyUnitOffset, const idx_t& nIntervals, const idx_t& pdeg,
        const idx_t& type, const TexT& tex = 0)
    {
        ScalarT intervals
            = makeIntervals(data, bodyUnitOffset, nIntervals, tex);
        CoeffsT coeffs
            = makeCoeffs(data, bodyUnitOffset, nIntervals, pdeg, tex);
        return BodyInterpolator{ ParentT::make(coeffs, intervals, radius),
            type };
    }

    /** @brief Get the values */
    DEVICEHOST()
    void getValues(Vec3R& out, const Real& epoch) const
    {
        /* check that epoch is in range */
        assertEpochInRange(epoch);
        /* evaluate */
        this->template ieval<0, 3>(out, epoch);
    }

    /** @brief Get the values into the given array */
    template<bool work>
    DEVICEHOST()
    void getValues(const SampleIndex& index, Vec3RArrT<work>& out,
        const Real& epoch, const Real& sign) const
    {
        /* check that epoch is in range */
        assertEpochInRange(epoch);
        /* evaluate */
        this->template ieval<work, 0, 3>(index, out, epoch, sign);
    }

    /** @brief Packet (W-lane) position retrieval — accumulates W
     *         lanes of body positions at per-lane epochs into ``out``.
     *
     *  Position is the first 3 components for both Type 2 (position-
     *  only coefficients) and Type 3 (position + velocity).  Per-
     *  lane epochs may straddle chebyshev intervals; the underlying
     *  ``GlobalInterpolator::ievalPacket`` handles uniform-interval
     *  predicate + per-lane scalar fallback.
     *
     *  PR-6δ-B2 packet ephemeris query (chain-walker leaf). */
    template<idx_t W>
    inline void iGetValuesPacket(
        feta::vector::PacketItem<Real, 3, W>& out,
        const feta::simd::Packet<Real, W>& epochPkt) const
    {
        /* No range-check on the packet path — caller validates
         * epoch range scalar-side before entering the packet RHS.
         * (Per-lane range checks would be expensive to packetise.) */
        this->template ievalPacket<W, 0, 3>(out, epochPkt);
    }

    /** @brief Get the derivatives */
    DEVICEHOST()
    void getDerivatives(Vec3R& out, const Real& epoch) const
    {
        /* check that epoch is in range */
        assertEpochInRange(epoch);
        /* evaluate */
        if (type_ == 2) {
            this->template idEval<0, 3>(out, epoch);
        } else if (type_ == 3) {
            this->template ieval<3, 3>(out, epoch);
        } else {
#ifdef __CUDA_ARCH__
            BRIE_GPU_THROW(err::INVALID_BODY_UNIT_TYPE);
#else
            BRIE_THROW(std::runtime_error, "Invalid body unit type");
#endif
        }
    }

    /** @brief Get the derivatives into the given array */
    template<bool work>
    DEVICEHOST()
    void getDerivatives(const SampleIndex& index, Vec3RArrT<work>& out,
        const Real& epoch, const Real& sign) const
    {
        /* check that epoch is in range */
        assertEpochInRange(epoch);
        /* evaluate */
        if (type_ == 2) {
            this->template idEval<work, 0, 3>(index, out, epoch, sign);
        } else if (type_ == 3) {
            this->template ieval<work, 3, 3>(index, out, epoch, sign);
        } else {
#ifdef __CUDA_ARCH__
            BRIE_GPU_THROW(err::INVALID_BODY_UNIT_TYPE);
#else
            BRIE_THROW(std::runtime_error, "Invalid body unit type");
#endif
        }
    }

    /** @brief Get values and derivatives */
    DEVICEHOST()
    void getValuesAndDerivatives(Vec6R& out, const Real& epoch) const
    {
        /* check that epoch is in range */
        assertEpochInRange(epoch);
        /* evaluate */
        if (type_ == 2) {
            this->template iEvaldEval<0, 3>(out, epoch);
        } else if (type_ == 3) {
            this->template ieval<0, 6>(out, epoch);
        } else {
#ifdef __CUDA_ARCH__
            BRIE_GPU_THROW(err::INVALID_BODY_UNIT_TYPE);
#else
            BRIE_THROW(std::runtime_error, "Invalid body unit type");
#endif
        }
    }

    /** @brief Get values and derivatives into the given array */
    template<bool work>
    DEVICEHOST()
    void getValuesAndDerivatives(const SampleIndex& index, Vec6RArrT<work>& out,
        const Real& epoch, const Real& sign) const
    { /* check that epoch is in range */
        assertEpochInRange(epoch);
        /* evaluate */
        if (type_ == 2) {
            this->template iEvaldEval<work, 0, 3>(index, out, epoch, sign);
        } else if (type_ == 3) {
            this->template ieval<work, 0, 6>(index, out, epoch, sign);
        } else {
#ifdef __CUDA_ARCH__
            BRIE_GPU_THROW(err::INVALID_BODY_UNIT_TYPE);
#else
            BRIE_THROW(std::runtime_error, "Invalid body unit type");
#endif
        }
    }

    /** @brief Assert that the given epoch is in range */
    DEVICEHOST()
    inline void assertEpochInRange([[maybe_unused]] const Real& epoch) const
    {
#ifdef BRIE_DEBUG_MODE
        Real start = this->intervals()[0];
        Real end   = this->intervals()[this->intervals().size() - 1]
            + 2 * this->radius();
#ifdef __CUDA_ARCH__
        BRIE_GPU_ASSERT(
            epoch >= start && epoch <= end, err::OUT_OF_RANGE_TARGET_EPOCH);
#else
        if (epoch < start || epoch > end) {
            std::stringstream ss;
            ss << "Epoch " << epoch << " not in the available range [";
            ss << start << ", " << end << "]";
            BRIE_THROW(std::runtime_error, ss.str().c_str());
        }
#endif
#endif
    }

    /** @brief Operator != */
    DEVICEHOST() bool operator!=(const Self& other) const
    {
        return this->coeffs_.data_.data_ != other.coeffs_.data_.data_
            || this->coeffs_.data_.nVecs_ != other.coeffs_.data_.nVecs_
            || this->coeffs_.data_.dimOffset_ != other.coeffs_.data_.dimOffset_
            || this->intervals_.data_ != other.intervals_.data_
            || this->intervals_.size_ != other.intervals_.size_
            || this->radius() != other.radius() || this->type_ != other.type_;
    }

    /** @brief The unit type declarator */
    idx_t type_ = 0;
};

} // namespace core
} // namespace brie