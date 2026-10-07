#pragma once

#include "paraHPOP/typedefs.h"

namespace paraHPOP {
namespace model {
namespace environment {
namespace atmosphere {

/**
 * @brief Geodetic altitude above an oblate ellipsoid via the Napeos
 *        ORge_GeoToVec3 fixed-point iteration on z' (transcribed from
 *        ``godot/core/astro/Math.h::geodeticFromCartesian``).
 *
 *  @param r           Position vector in some inertial frame (km).
 *  @param pole        Unit vector along the body's polar (rotational)
 *                     axis, expressed in the SAME frame as ``r``.  For
 *                     Earth: ``pole ≈ ẑ_ICRF``; for Mars the pole is
 *                     ~37° off ICRF z.  Pole direction is rotation-
 *                     invariant about itself, so the prime-meridian
 *                     phase is irrelevant — only the pole direction
 *                     matters for altitude.
 *  @param R_eq        Equatorial radius of the body (km).
 *  @param flattening  Body flattening, dimensionless, in ``[0, 1)``.
 *  @param tol         Convergence tolerance on the iterate step.
 *
 *  Returns ``||r|| - R_eq`` when ``flattening == 0`` (the spherical
 *  fast path, bit-identical to the pre-flattening drag math; pole is
 *  unused in this branch).  For positive flattening, projects ``r``
 *  onto ``pole`` to recover the polar component and the squared
 *  perpendicular, then runs a small fixed-point loop on the polar
 *  iterate ``zd`` until consecutive iterates agree within ``tol``
 *  — typically 2-3 iterations at LEO altitudes.
 *
 *  Sign of ``zd`` tracks the sign of the polar component so the
 *  iteration handles northern and southern hemispheres symmetrically
 *  (matches GODOT's ``num::sign(value(z))`` term via ``copysign``).
 */
DEVICEHOST()
inline Real geodeticAltitude(const Vec3R& r, const Vec3R& pole,
    const Real& R_eq, const Real& flattening,
    const Real& tol = Real{ 1e-9 })
{
    if (flattening == Real{ 0 }) {
        return r.template norm<Real>() - R_eq;
    }

    /* Project r onto the body polar axis to recover the polar
       component ``z`` and the squared perpendicular ``xy``. */
    const Real r2     = r.dot(r);  /* ||r||² */
    const Real z      = r.dot(pole);
    const Real xy     = r2 - z * z;
    const Real flatsq = (Real{ 2 } - flattening) * flattening;

    /* Iterate: d := R_eq · flatsq · sqrt( sfq / (1 − flatsq · sfq) )
       with sfq = zd² / (xy + zd²) and zd = z + |d| · sign(z).
       Converges quadratically; stop when |d − d0| < tol. */
    Real d0  = Real{ 0 };
    Real d   = Real{ 1e10 };
    Real sfq = Real{ 0 };
    Real zd  = z;
    while (fabs(d - d0) > tol) {
        d0           = d;
        const Real z2 = zd * zd;
        sfq          = z2 / (xy + z2);
        d            = R_eq * flatsq * sqrt(sfq / (Real{ 1 } - flatsq * sfq));
        zd           = z + copysign(d, z);
    }

    return sqrt(xy + zd * zd) - R_eq / sqrt(Real{ 1 } - flatsq * sfq);
}

} // namespace atmosphere
} // namespace environment
} // namespace model
} // namespace paraHPOP
