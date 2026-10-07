#pragma once

#include "paraHPOP/typedefs.h"
#include "paraHPOP/util.h"

namespace paraHPOP {
namespace model {
namespace environment {
namespace eclipse {

/**
 * @file Shadow.h
 * @brief Dual-cone (umbra / penumbra / annular) occultation geometry.
 *
 * The model follows the conic shadow construction of Montenbruck & Gill,
 * *Satellite Orbits — Models, Methods and Applications* (§3.4): the Sun and
 * the occulting body are projected onto the spacecraft's celestial sphere as
 * two apparent disks with angular radii `a` and `b`, separated by an angle
 * `c`.  Their overlap fixes the fraction of the solar disk that remains
 * visible — the *occultation factor* used to scale solar radiation pressure —
 * and the two contact conditions `c = a + b` (penumbra) and `c = |a − b|`
 * (umbra / annular) provide the smooth boundary scalars consumed by the
 * eclipse event's root finder.
 *
 * The geometry is intentionally expressed as a small scalar pipeline rather
 * than a vector-valued object: the two input directions are reduced to three
 * scalars (`|rS|²`, `|rB|²`, `rS·rB`) on entry and never materialised as
 * cached vectors, keeping per-thread register pressure low in the device
 * kernels that evaluate it once per sample per RK stage.
 */

/**
 * @brief Apparent geometry of a Sun–body occultation as seen from a sample.
 *
 * All angles are in radians.  Populated by ::cone; consumed by ::factor,
 * ::penumbraScalar and ::innerScalar.
 */
struct Cone {
    /** @brief Apparent angular radius of the Sun, `asin(RSun / |rS|)`. */
    Real a;

    /** @brief Apparent angular radius of the occulting body,
     * `asin(RBody / |rB|)`. */
    Real b;

    /** @brief Apparent angular separation of the two disk centres,
     * `acos(r̂S · r̂B)`. */
    Real c;

    /** @brief True when the sample lies at or inside the body's surface
     * (`|rB| ≤ RBody`); the occultation factor is then identically zero. */
    bool insideBody;
};

namespace detail {

/** @brief Clamp `x` to the closed interval [`lo`, `hi`]. */
DEVICEHOST() inline Real clamp(const Real& x, const Real& lo, const Real& hi)
{
    return feta::math::fmin(feta::math::fmax(x, lo), hi);
}

} // namespace detail

/**
 * @brief Reduce two pointing directions to the apparent occultation geometry.
 *
 * @tparam ExprS Type of the Sun-relative direction expression.
 * @tparam ExprB Type of the body-relative direction expression.
 * @param rS    Vector from the Sun to the sample (sign-agnostic — only the
 *              magnitude and the angle relative to @p rB matter).
 * @param rB    Vector from the occulting body to the sample.
 * @param RSun  Physical radius of the Sun.
 * @param RBody Physical radius of the occulting body.
 * @return The populated ::Cone.
 *
 * @note The expressions are traversed in place to form the three scalar
 *       reductions; no intermediate vector is stored.  The `asin` arguments
 *       are clamped to [0, 1] and the `acos` argument to [−1, 1] so that a
 *       sample grazing or inside the body cannot drive the inverse
 *       trigonometric calls out of domain.
 */
template<typename ExprS, typename ExprB>
DEVICEHOST() inline Cone cone(
    const ExprS& rS, const ExprB& rB, const Real& RSun, const Real& RBody)
{
    /* Three scalar reductions — the only contact with the input vectors. */
    const Real s2 = rS.template squaredNorm<Real>(); /* |rS|² */
    const Real b2 = rB.template squaredNorm<Real>(); /* |rB|² */
    const Real sb = rS.dot(rB);                       /* rS · rB */

    const Real rInvS = Real{ 1 } / feta::math::sqrt(s2);
    const Real rInvB = Real{ 1 } / feta::math::sqrt(b2);

    Cone k;
    k.insideBody = (b2 <= RBody * RBody);
    k.a = feta::math::asin(detail::clamp(RSun * rInvS, Real{ 0 }, Real{ 1 }));
    k.b = feta::math::asin(detail::clamp(RBody * rInvB, Real{ 0 }, Real{ 1 }));
    k.c = feta::math::acos(
        detail::clamp(sb * rInvS * rInvB, Real{ -1 }, Real{ 1 }));
    return k;
}

/**
 * @brief Fraction of the solar disk visible from the sample (0 … 1).
 *
 * Returns 1 outside the penumbra, 0 in total eclipse (or when the sample is
 * inside the body), `1 − b²/a²` in an annular pass, and the lune-overlap
 * fraction (Montenbruck & Gill eq. 3.87) in the partial-eclipse band.
 */
DEVICEHOST() inline Real factor(const Cone& k)
{
    if (k.insideBody)
        return Real{ 0 }; /* sample inside the body — fully occulted */

    const Real a = k.a, b = k.b, c = k.c;

    if (c >= a + b)
        return Real{ 1 }; /* apparent disks disjoint — no occultation */

    if (c <= feta::math::abs(b - a)) {
        /* One disk lies entirely within the other. */
        if (b >= a)
            return Real{ 0 }; /* body covers the Sun — total eclipse */
        return Real{ 1 } - (b * b) / (a * a); /* annular: ring of light */
    }

    /* Partial eclipse — area of the lune common to both apparent disks. */
    const Real a2 = a * a;
    const Real b2 = b * b;
    const Real c2 = c * c;
    const Real x  = (c2 + a2 - b2) / (Real{ 2 } * c);
    const Real y  = feta::math::sqrt(feta::math::fmax(a2 - x * x, Real{ 0 }));
    const Real alpha = feta::math::acos(detail::clamp(x / a, Real{ -1 }, Real{ 1 }));
    const Real beta
        = feta::math::acos(detail::clamp((c - x) / b, Real{ -1 }, Real{ 1 }));

    /* Occulted area of the apparent solar disk, normalised by its own area. */
    const Real occluded = a2 * alpha + b2 * beta - c * y;
    constexpr Real pi    = Real{ 3.14159265358979323846 };
    return Real{ 1 } - occluded / (a2 * pi);
}

/**
 * @brief Penumbra boundary scalar: positive inside the penumbra.
 *
 * Equals `(a + b) − c`; its zero crossing marks first/last contact with the
 * penumbral cone (the start and end of any occultation).
 */
DEVICEHOST() inline Real penumbraScalar(const Cone& k)
{
    return (k.a + k.b) - k.c;
}

/**
 * @brief Inner (umbra / annular) boundary scalar: positive inside total or
 * annular obscuration.
 *
 * Equals `|a − b| − c`; its zero crossing marks the inner contact where the
 * occultation becomes total (`b > a`) or annular (`a > b`).  The two cases
 * share this scalar, so an annular pass registers on the inner boundary
 * exactly as a total eclipse does.
 */
DEVICEHOST() inline Real innerScalar(const Cone& k)
{
    return feta::math::abs(k.a - k.b) - k.c;
}

/**
 * @brief One-shot occultation factor for callers that need only the scaling.
 *
 * Equivalent to `factor(cone(rS, rB, RSun, RBody))`; this is the form used by
 * the SRP attenuation path, which never needs the boundary scalars.
 */
template<typename ExprS, typename ExprB>
DEVICEHOST() inline Real factor(
    const ExprS& rS, const ExprB& rB, const Real& RSun, const Real& RBody)
{
    return factor(cone(rS, rB, RSun, RBody));
}

} // namespace eclipse
} // namespace environment
} // namespace model
} // namespace paraHPOP
