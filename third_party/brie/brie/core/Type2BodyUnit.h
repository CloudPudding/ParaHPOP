#pragma once

#include "brie/core/BodyUnit.h"

namespace brie {
namespace core {

/**
 * @brief Child class of `brie::ConstIntBodyUnit`. Manages SPK-like Type 2 Body
 * Unit data (position-only coefficients of Chebyshev polynomials with
 * equally-sized intervals)
 *
 * Example of corresponding SPICE kernel: de430.bsp
 *
 * @note Numerical precision: `getValues`, `getDerivatives`, and
 *       `getValuesAndDerivatives` route through parm's
 *       `chebyshev::Evaluator`, whose backward Clenshaw recurrence has
 *       an empirical ~1e-13 relative forward-error floor (see
 *       `parm/interpolate/chebyshev/Interpolator.h::Evaluator::eval`
 *       docstring). This matches state-of-the-art production Chebyshev
 *       evaluation (SPICE/GSL/Boost.Math). Consumers needing tighter
 *       precision should either widen test tolerances accordingly or
 *       request compensated Clenshaw at the parm layer.
 */
template<bool UseTexture, bool work>
class Type2BodyUnit : public ConstIntBodyUnit<3, UseTexture, work> {
    using ParentT = ConstIntBodyUnit<3, UseTexture, work>;

    using CoeffsArrayT = typename ParentT::CoeffsT;

    using IntervalsT = typename feta::scalar::texture::Array<Real,
        UseTexture>::template Ref<work>;
    using CoeffsT = typename parm::interpolate::detail::PolynomialCoefficients<
        CoeffsArrayT>;
    using InterpolatorT
        = parm::interpolate::chebyshev::GlobalInterpolator<CoeffsT, IntervalsT>;

public:
    /** @brief Expose the texture object */
    using TexT = typename ParentT::TexT;

    /**
     * @brief Factory method to constuct from data elments
     *
     */
    DEVICEHOST()
    static Type2BodyUnit make(Real* const data, const idx_t& size,
        const idx_t& bodyUnitOffset, const uint_t& nIntervals,
        const uint_t& pdeg, const TexT& tex = 0,
        const idx_t& parentTexOffset = 0)
    {
        Type2BodyUnit out;
        out.data_           = data + bodyUnitOffset;
        out.size_           = size;
        out.bodyUnitOffset_ = bodyUnitOffset + 1 + nIntervals;
        out.nIntervals_     = nIntervals;
        out.pdeg_           = pdeg;
        out.tex_            = tex;
        out.texOffset_      = parentTexOffset + bodyUnitOffset;
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
        return this->interpolator().eval(epoch);
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
        return this->interpolator().ieval(out, epoch);
    }

    /**
     * @brief Packet (W-lane) position retrieval — accumulates W
     *        lanes of body positions at per-lane epochs into ``out``.
     *
     *  Per-lane epochs may straddle chebyshev intervals; the
     *  underlying ``GlobalInterpolator::ievalPacket`` handles the
     *  uniform-interval predicate + per-lane scalar fallback.
     *  PR-6δ-B2 packet ephemeris query.
     */
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
        return this->interpolator().dEval(epoch);
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
        return this->interpolator().idEval(out, epoch);
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
        return this->interpolator().evaldEval(epoch);
    }

    /**
     * @brief Retrieve the Cartesian state of this body at the given TDB epoch
     *
     * @param out
     * @param epoch
     *
     */
    DEVICEHOST()
    inline void getValuesAndDerivatives(Vec6R& out, const Real& epoch)
    {
        return this->interpolator().ievaldEval(out, epoch);
    }
};

} // namespace core
} // namespace brie
