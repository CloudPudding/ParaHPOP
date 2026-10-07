#pragma once

#include "brie/core/BodyUnit.h"

namespace brie {
namespace core {

/**
 * @brief Child class of `brie::ConstIntBodyUnit`. Manages SPK-like Type 3 Body
 * Unit data (position and velocity coefficients of Chebyshev polynomials with
 * equally-sized intervals)
 *
 * Example of corresponding SPICE kernel: jup344.bsp
 *
 */
template<bool UseTexture, bool work>
class Type3BodyUnit : public ConstIntBodyUnit<6, UseTexture, work> {
    using ParentT = ConstIntBodyUnit<6, UseTexture, work>;

    using CoeffsArrayT = typename ParentT::CoeffsT;
    using IntervalsT   = typename feta::scalar::texture::Array<Real,
          UseTexture>::template Ref<work>;
    using CoeffsT = typename parm::interpolate::detail::PolynomialCoefficients<
        CoeffsArrayT>;
    using InterpolatorT
        = parm::interpolate::chebyshev::GlobalInterpolator<CoeffsT, IntervalsT>;

    /* Half coefficients and interpolator */
    template<idx_t _DIMS>
    using CFSEGT           = typename ParentT::template CoeffsSegmentT<_DIMS>;
    using HalfCoeffsArrayT = CFSEGT<3>;
    using HalfCoeffsT =
        typename parm::interpolate::detail::PolynomialCoefficients<
            HalfCoeffsArrayT>;
    using HalfInterpolatorT
        = parm::interpolate::chebyshev::GlobalInterpolator<HalfCoeffsT,
            IntervalsT>;

public:
    /** @brief Expose the texture object */
    using TexT = typename ParentT::TexT;

    /**
     * @brief Factory method to constuct from data elments
     *
     */
    DEVICEHOST()
    static Type3BodyUnit make(Real* const data, const idx_t& size,
        const idx_t& bodyUnitOffset, const idx_t& nIntervals, const idx_t& pdeg,
        const TexT& tex = 0)
    {
        Type3BodyUnit out;
        out.data_           = data + bodyUnitOffset;
        out.size_           = size;
        out.bodyUnitOffset_ = bodyUnitOffset + 1 + nIntervals;
        out.nIntervals_     = nIntervals;
        out.pdeg_           = pdeg;
        out.tex_            = tex;
        out.texOffset_      = bodyUnitOffset;
        return out;
    }

    /** @brief Create the interpolator */
    DEVICEHOST() InterpolatorT interpolator() const
    {
        CoeffsT cfs     = CoeffsT::make(this->coeffs(), this->pdeg_);
        IntervalsT ints = this->startOfIntervals();
        Real r          = this->intervalRadius();
        return { cfs, ints, r };
    }

    /**
     * @brief Retrieve the position of this body at the given TDB epoch
     *
     * @param epoch
     * @return positionVector
     *
     */
    DEVICEHOST() inline Vec3R getValues(const Real& epoch)
    {
        return this->interpolator().template eval<0, 3>(epoch);
    }

    /**
     * @brief Retrieve the position of this body at the given TDB epoch
     *
     * @param out
     * @param epoch
     *
     */
    DEVICEHOST()
    inline void iGetValues(Vec3R& out, const Real& epoch)
    {
        return this->interpolator().template ieval<0, 3>(out, epoch);
    }

    /**
     * @brief Packet (W-lane) position retrieval — see
     *        ``Type2BodyUnit::iGetValuesPacket``.  Type 3 stores
     *        position + velocity in 6-component coefficients;
     *        ``ievalPacket<W, 0, 3>`` reads the position
     *        sub-range only. */
    template<idx_t W>
    inline void iGetValuesPacket(
        feta::vector::PacketItem<Real, 3, W>& out,
        const feta::simd::Packet<Real, W>& epochPkt)
    {
        return this->interpolator().template ievalPacket<W, 0, 3>(
            out, epochPkt);
    }

    /**
     * @brief Retrieve the velocity of this body at the given TDB epoch
     *
     * @param epoch
     * @return velocityVector
     *
     */
    DEVICEHOST() inline Vec3R getDerivatives(const Real& epoch)
    {
        return this->interpolator().template eval<3, 3>(epoch);
    }

    /**
     * @brief Retrieve the velocity of this body at the given TDB epoch
     *
     * @param out
     * @param epoch
     *
     */
    DEVICEHOST()
    inline void iGetDerivatives(Vec3R& out, const Real& epoch)
    {
        return this->interpolator().template ieval<3, 3>(out, epoch);
    }

    /**
     * @brief Retrieve the Cartesian state of this body at the given TDB epoch
     *
     * @param epoch
     * @return cartesianStateVector
     *
     */
    DEVICEHOST()
    inline Vec6R getValuesAndDerivatives(const Real& epoch)
    {
        return this->interpolator().eval(epoch);
    }

    /**
     * @brief Retrieve the Cartesian state of this body at the given TDB epoch
     *
     * @param out
     * @param epoch
     *
     */
    DEVICEHOST()
    inline void iGetValuesAndDerivatives(Vec6R& out, const Real& epoch)
    {
        return this->interpolator().ieval(out, epoch);
    }
};

} // namespace core
} // namespace brie
