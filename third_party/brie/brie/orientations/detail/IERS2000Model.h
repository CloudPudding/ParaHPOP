#pragma once

#include "brie/typedefs.h"
#include "brie/util/throw.h"

#include <feta/math/math.h>
#include <feta/vector/Item.h>

namespace brie {
namespace orientations {
namespace detail {

/** @brief Minimal row-major 3×3 matrix (9 contiguous `Real`, element `m[3*r+c]`).
 *  Replaces the bare `Real[9]` dense-rotation seam so length + row-major layout are
 *  structural (carried by the type), not a by-convention pointer + stride. It is an
 *  aggregate (no constructor → no forced zero-init, matching the old `Real[9]`
 *  locals); `operator[]` preserves the existing flat `3*i+j` indexing verbatim. */
struct Mat3 {
    Real m[9];
    DEVICEHOST() Real& operator[](idx_t i) { return m[i]; }
    DEVICEHOST() const Real& operator[](idx_t i) const { return m[i]; }
};

/**
 * @brief IERS2000 (CIO-based) ICRF→ITRF rotation model — the ModelPolicy
 * specialization of @ref RotationModelUnit for the native IPF path.
 *
 * Ports GODOT's `orient::IERS2000::rotation()` to device-safe `Real` math:
 * composes the polar-motion, Earth-rotation (ERA) and precession-nutation
 * matrices, `R = PM · ERA · PN`, plus the analytic time derivative `Ṙ`.
 * The inputs are the interpolated IERS channels and their per-day rates
 * (the @ref LagrangeInterp policy supplies them); this policy is stateless,
 * exactly like `euler313ToQuat`.
 *
 * Channel order (matching the IPF repack); the `Idx` enum is the single source
 * of this ordering, and `LagrangeInterp`'s `head<VD_NUT>`/`tail<VD_ERP>` fills it
 * into a typed `Item<Real, NCHANNELS>`:
 *   nodes/rates = [X, Y, s, xp, yp, ΔUT1, δX, δY]
 *   - X, Y, s          CIP coordinates + CIO locator   [rad]
 *   - xp, yp           polar motion                     [rad]
 *   - ΔUT1             UT1 − TT                          [days]
 *   - δX, δY           CIP corrections                  [rad]  (zero in this product)
 * Rates are per the interpolation key (per day); they are scaled to
 * per-second internally to match GODOT's seconds-based derivatives.
 *
 * The model time `ttMjd` is TT in MJD2000 days (the IPF node grid). The
 * Earth-rotation angle uses GODOT's integer-day / fractional-day split so
 * precision holds to ~1e-13 at the span edge (risk R2).
 *
 * `compose` emits the quaternion (Shepperd matrix→quat, scalar-first
 * `[w,x,y,z]`, whose active rotation `q⊗v⊗q⁻¹ = R·v` reproduces the cached
 * `Rotation::to_` convention) and ω = vee(Ṙᵀ R) — the angular velocity of
 * ITRF w.r.t. ICRF in inertial components, rad/s (the
 * `Frames::bodyAngularVelocity` contract; ω_z is the +sidereal rate).
 */
struct IERS2000Model {
    /** @brief Channel count: [X,Y,s,xp,yp,ΔUT1,δX,δY]. */
    static constexpr idx_t NCHANNELS = 8;

    /** @brief Named channel indices into the typed `Item<Real, NCHANNELS>` seam —
     *  the single source of the channel order. Nutation occupies `[iX, iS]`
     *  (LagrangeInterp's `head<VD_NUT>`); ERP occupies `[iXp, iDY]` (its
     *  `tail<VD_ERP>`). The two policies agree on this partition by construction;
     *  the `i`-prefix avoids colliding with the `X, Y, s, …` compose locals. */
    enum Idx : idx_t { iX = 0, iY, iS, iXp, iYp, iDut1, iDX, iDY };
    static_assert(iDY + 1 == NCHANNELS,
        "Idx must enumerate exactly NCHANNELS channels");

    /** @brief GODOT-faithful IERS2000 constants — single source (values ported
     *  verbatim from godot `orient/IERS2000.cpp`; do NOT change — HARD fidelity). */
    static constexpr Real cEraBias = 0.7790572732640;     ///< ERA bias term
    static constexpr Real cEraRate = 0.00273781191135448; ///< ERA rate term (/day)
    static constexpr Real cSpSlope = -6.24e-15;           ///< TIO locator s' slope (/day)
    static constexpr Real cSpDot   = -7.2205e-20;         ///< s' rate (rad/s)

    /** @brief Row-major 3×3 multiply: C = A · B.
     *  @note `C` MUST NOT alias `A` or `B`: the product is written element-by-element
     *  while the inputs are read in the same loop, so a self-aliasing call would read
     *  back an already-overwritten element and silently corrupt the result. */
    DEVICEHOST()
    static inline void mat3mul(Mat3& C, const Mat3& A, const Mat3& B)
    {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                C[3 * i + j] = A[3 * i + 0] * B[0 + j]
                    + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j];
    }

    /** @brief Debug-only guard: assert the interpolated rotation inputs are
     *  finite before they feed the (godot-faithful, deliberately unguarded) trig.
     *  Host-only + `BRIE_DEBUG_MODE` — release builds and the device compile pass
     *  expand to nothing, so the sm_61 omega/cacheRotation kernels are untouched.
     *  (Device-side reporting would need a dedicated `err` code in the committed
     *  `DeviceError.h`; the host assert is sufficient as a dev-build safety net.) */
    DEVICEHOST()
    static void checkInputsFinite(
        [[maybe_unused]] const feta::vector::Item<Real, NCHANNELS>& nodes,
        [[maybe_unused]] const feta::vector::Item<Real, NCHANNELS>& rates)
    {
#if defined(BRIE_DEBUG_MODE) && !defined(__CUDA_ARCH__)
        for (idx_t i = 0; i < NCHANNELS; ++i) {
            BRIE_ASSERT(feta::math::isfinite(nodes.data()[i]),
                "non-finite IERS2000 rotation node input");
            BRIE_ASSERT(feta::math::isfinite(rates.data()[i]),
                "non-finite IERS2000 rotation rate input");
        }
#endif
    }

    /** @brief The scalar prologue derived from the interpolated channels: the
     *  precession-nutation ZYZ angles (e, d, mems) + rates, the Earth-rotation
     *  angle θ + rate, and the TIO-locator s' + rate. */
    struct PreludeScalars {
        Real e, d, mems;          ///< PN ZYZ angles
        Real edot, ddot, memsdot; ///< PN ZYZ angle rates
        Real theta, thetadot;     ///< Earth-rotation angle + rate
        Real sp, spdot;           ///< TIO locator s' + rate
    };

    /** @brief Compute the scalar prologue SHARED by `composeMatrix` and
     *  `composeQuat` — everything both the dense matrices and the factor
     *  quaternions are built from — so the godot-faithful derivation (and its
     *  fidelity caveats) live in ONE place. `FORCEINLINE` so the returned
     *  aggregate is scalarized (SROA) and each caller's derive-then-consume
     *  interleave (the sm_61 omega-kernel register tuning, #28) is preserved. */
    DEVICEHOST() FORCEINLINE()
    static PreludeScalars computePrelude(
        const feta::vector::Item<Real, NCHANNELS>& nodes,
        const feta::vector::Item<Real, NCHANNELS>& rates, const Real& ttMjd,
        const Real& ttFrac)
    {
        constexpr Real SECDAY = 86400.0;
        constexpr Real invSec = 1.0 / SECDAY;
        constexpr Real TWO_PI = 2.0 * M_PI;

        /* nodes + per-second rates (Lagrange dEval is per day) */
        const Real X = nodes.get<iX>(), Y = nodes.get<iY>(), s = nodes.get<iS>();
        const Real dut1 = nodes.get<iDut1>();
        const Real dX = nodes.get<iDX>(), dY = nodes.get<iDY>();
        const Real Xd = rates.get<iX>() * invSec, Yd = rates.get<iY>() * invSec,
                   sd = rates.get<iS>() * invSec;
        const Real dut1d = rates.get<iDut1>() * invSec;
        const Real dXd = rates.get<iDX>() * invSec, dYd = rates.get<iDY>() * invSec;

        PreludeScalars p;

        /* ---------------- precession-nutation angles (e, d, mems) ---------- */
        const Real x = X + dX, y = Y + dY;
        const Real xdot = Xd + dXd, ydot = Yd + dYd, sdot = sd;
        const Real r2 = x * x + y * y;
        const Real rr = feta::math::sqrt(r2);
        const Real fr = 1.0 / feta::math::sqrt(1.0 - r2);
        p.e       = (r2 > 0.0) ? feta::math::atan2(y, x) : 0.0;
        p.d       = feta::math::atan(rr * fr);
        p.edot    = (r2 > 0.0) ? (ydot * x - xdot * y) / r2 : 0.0;
        p.ddot    = (x * xdot + y * ydot) * fr / rr;
        p.mems    = -p.e - s;
        p.memsdot = -p.edot - sdot;

        /* ---------------- Earth rotation angle θ + rate ---------------- */
        /* GODOT integer-day / fractional-day split. The ERA *phase* `f` is
         * formed from the high-precision O(1) day-fraction `ttFrac` carried
         * separately, NOT from `ttMjd − floor(ttMjd)`: a single ~1.8e4
         * absolute-time double quantizes the time-of-day to ULP(ttMjd)≈4e-12 d,
         * which the ERA (2π/day) turns into a ~2.5e-11 rad floor. GODOT keeps
         * an (int day, fraction) `jdPair` to avoid exactly this; `ttFrac`
         * mirrors that. The large `ut1jd` is used ONLY for the slow
         * `cRate·ut1jd` polynomial term, which tolerates the magnitude. */
        const Real ut1jd = ttMjd + dut1 - 0.5;
        const Real f     = feta::math::fmod(ttFrac + dut1 - 0.5, 1.0);
        const Real fdot  = invSec + dut1d; /* d(days)/d(sec) */
        Real theta = TWO_PI * (f + cEraBias + cEraRate * ut1jd);
        /* GODOT num::wrapZero2Pi: angle − 2π·floor(angle/2π). Reproducing
         * the exact reduction (rather than fmod, which rounds differently)
         * keeps |R − godot| at the same floor as godot's own ERA. */
        theta   = theta - TWO_PI * feta::math::floor(theta / TWO_PI);
        p.theta = theta;
        /* θ̇ — bit-faithful port of godot `orient/IERS2000.cpp`: the LOD/UT1-rate
         * correction `dut1d` is applied (via `fdot`) only to the dominant rotation
         * term; the slow 0.273 %/day ERA-precession term deliberately uses the
         * NOMINAL constant `cRate·invSec` (godot's `constexpr constRatePerSecond`),
         * NOT `cRate·fdot`. This is an intentional godot model approximation that
         * drops `2π·cRate·dut1d` (~1e-10 of the rate — negligible, far below UT1
         * knowledge). KEPT as-is for reference fidelity; do NOT "complete" the chain
         * rule here (it would diverge ~1e-10 from godot for no physical gain). */
        p.thetadot = TWO_PI * (fdot + cEraRate * invSec);

        /* ---------------- polar motion s' ---------------- */
        p.sp    = cSpSlope * (ttMjd - 0.5);
        p.spdot = cSpDot;
        return p;
    }

    /**
     * @brief Compose the ICRF→ITRF rotation matrix `R` and (if `Rdot` is
     * non-null) its per-second time derivative `Ṙ`, from the interpolated
     * channels `nodes` and their per-day `rates`, at TT model time
     * `ttMjd` (MJD2000 days).
     */
    DEVICEHOST()
    static void composeMatrix(Mat3& R, Mat3* const Rdot,
        const feta::vector::Item<Real, NCHANNELS>& nodes,
        const feta::vector::Item<Real, NCHANNELS>& rates, const Real& ttMjd,
        const Real& ttFrac)
    {
        checkInputsFinite(nodes, rates);
        constexpr Real invSec = 1.0 / 86400.0;
        const PreludeScalars p = computePrelude(nodes, rates, ttMjd, ttFrac);
        const Real e = p.e, d = p.d, mems = p.mems;
        const Real edot = p.edot, ddot = p.ddot, memsdot = p.memsdot;
        const Real theta = p.theta, thetadot = p.thetadot;
        const Real sp = p.sp, spdot = p.spdot;
        /* polar-motion channels (consumed directly by PM/PMd below) */
        const Real xp = nodes.get<iXp>(), yp = nodes.get<iYp>();
        const Real xpd = rates.get<iXp>() * invSec, ypd = rates.get<iYp>() * invSec;

        /* ---------------- precession-nutation (PN, PNd) ---------------- */
        Real c1, s1, c2, s2, c3, s3;
        feta::math::sincos(e, &s1, &c1);
        feta::math::sincos(d, &s2, &c2);
        feta::math::sincos(mems, &s3, &c3);

        Mat3 PN;
        PN[0] = c1 * c2 * c3 - s1 * s3;
        PN[1] = c1 * s3 + s1 * c2 * c3;
        PN[2] = -c3 * s2;
        PN[3] = -s1 * c3 - c1 * c2 * s3;
        PN[4] = c1 * c3 - s1 * c2 * s3;
        PN[5] = s2 * s3;
        PN[6] = s2 * c1;
        PN[7] = s2 * s1;
        PN[8] = c2;

        Mat3 PNd;
        {
            const Real c1d = -s1 * edot, s1d = c1 * edot;
            const Real c2d = -s2 * ddot, s2d = c2 * ddot;
            const Real c3d = -s3 * memsdot, s3d = c3 * memsdot;
            const Real c2s3 = c2 * s3, c2c3 = c2 * c3;
            const Real c2s3d = c2d * s3 + c2 * s3d;
            const Real c2c3d = c2d * c3 + c2 * c3d;
            PNd[0] = -s1d * s3 - s1 * s3d + c1d * c2c3 + c1 * c2c3d;
            PNd[1] = c1d * s3 + c1 * s3d + s1d * c2c3 + s1 * c2c3d;
            PNd[2] = -c3d * s2 - c3 * s2d;
            PNd[3] = -s1d * c3 - s1 * c3d - c1d * c2s3 - c1 * c2s3d;
            PNd[4] = c1d * c3 + c1 * c3d - s1d * c2s3 - s1 * c2s3d;
            PNd[5] = s2d * s3 + s2 * s3d;
            PNd[6] = s2d * c1 + s2 * c1d;
            PNd[7] = s2d * s1 + s2 * s1d;
            PNd[8] = c2d;
        }

        /* ---------------- Earth rotation (ER, ERd) ---------------- */
        Real ct, st;
        feta::math::sincos(theta, &st, &ct);
        Mat3 ER  = { { ct, st, 0.0, -st, ct, 0.0, 0.0, 0.0, 1.0 } };
        Mat3 ERd = { { -st * thetadot, ct * thetadot, 0.0, -ct * thetadot,
            -st * thetadot, 0.0, 0.0, 0.0, 0.0 } };

        /* ---------------- polar motion (PM, PMd) ---------------- */
        Mat3 PM  = { { 1.0, sp, xp, -sp, 1.0, -yp, -xp, yp, 1.0 } };
        Mat3 PMd = { { 0.0, spdot, xpd, -spdot, 0.0, -ypd, -xpd, ypd, 0.0 } };

        /* ---------------- compose R = PM·ER·PN ---------------- */
        Mat3 ERPN;
        mat3mul(ERPN, ER, PN);
        mat3mul(R, PM, ERPN);

        if (Rdot != nullptr) {
            /* Ṙ = PM·ERd·PN + PMd·(ER·PN) + PM·(ER·PNd) */
            Mat3 a, t1, t2, t3;
            mat3mul(a, ERd, PN);
            mat3mul(t1, PM, a);
            mat3mul(t2, PMd, ERPN);
            mat3mul(a, ER, PNd);
            mat3mul(t3, PM, a);
            for (int i = 0; i < 9; ++i)
                (*Rdot)[i] = t1[i] + t2[i] + t3[i];
        }
    }

    /** @brief Test/introspection boundary: fill caller-owned `Real[9]` `R` (and,
     *  if non-null, `Rdot`) from the typed `composeMatrix`. The production path
     *  (`compose` → cacheRotation) uses the typed `Mat3` form directly; this
     *  overload backs only `RotationModelUnit::matrices`. */
    DEVICEHOST()
    static void composeMatrix(Real* const R, Real* const Rdot,
        const feta::vector::Item<Real, NCHANNELS>& nodes,
        const feta::vector::Item<Real, NCHANNELS>& rates, const Real& ttMjd,
        const Real& ttFrac)
    {
        Mat3 Rm, Rdotm;
        composeMatrix(Rm, (Rdot != nullptr) ? &Rdotm : nullptr, nodes, rates,
            ttMjd, ttFrac);
        for (int i = 0; i < 9; ++i)
            R[i] = Rm[i];
        if (Rdot != nullptr)
            for (int i = 0; i < 9; ++i)
                Rdot[i] = Rdotm[i];
    }

    /** @brief Quaternion from a rotation matrix (Shepperd), scalar-first
     *  `[w,x,y,z]`, such that the active rotation `q⊗v⊗q⁻¹` equals `R·v`.
     *
     *  @note This is the *exact orthonormal projection* of `R`: the returned
     *  quaternion is a perfect, norm-preserving rotation (`|RᵀR−I|` of the
     *  quaternion's matrix is ~1e-16). A composed float64 `R = PM·ER·PN` is
     *  NOT perfectly orthonormal — GODOT's own reference matrix carries a
     *  ~3.2e-12 orthonormality defect — and GODOT applies that non-orthonormal
     *  matrix directly. So the residual ~1.6e-12 rad disagreement between this
     *  quaternion and GODOT's matrix (≈0.01 mm at Earth's surface) is GODOT's
     *  artifact, not a brie error; the quaternion is arguably the *more*-correct
     *  rotation. Bit-matching GODOT would require applying `R·v` directly (a
     *  matrix-direct path), which is invasive on the quaternion-based cudaj
     *  rotation cache and not worth it for 0.01 mm. Proven by the in-tree
     *  benchmark `RotationModelTest.OrthonormalityFloorIsGodotArtifact` and the
     *  mpmath oracle `ipf_native_poc/verify_shepperd_floor.py`
     *  (e_dbl == e_mpmath == 1.62e-12, e_on_orthonormal ~1e-16). */
    DEVICEHOST()
    static feta::Vec4T<Real> quatFromMatrix(const Mat3& R)
    {
        feta::Vec4T<Real> q;
        const Real m00 = R[0], m01 = R[1], m02 = R[2];
        const Real m10 = R[3], m11 = R[4], m12 = R[5];
        const Real m20 = R[6], m21 = R[7], m22 = R[8];
        const Real tr = m00 + m11 + m22;
        Real w, x, y, z;
        if (tr > 0.0) {
            Real S = feta::math::sqrt(tr + 1.0) * 2.0; /* 4w */
            w = 0.25 * S;
            x = (m21 - m12) / S;
            y = (m02 - m20) / S;
            z = (m10 - m01) / S;
        } else if (m00 > m11 && m00 > m22) {
            Real S = feta::math::sqrt(1.0 + m00 - m11 - m22) * 2.0; /* 4x */
            w = (m21 - m12) / S;
            x = 0.25 * S;
            y = (m01 + m10) / S;
            z = (m02 + m20) / S;
        } else if (m11 > m22) {
            Real S = feta::math::sqrt(1.0 + m11 - m00 - m22) * 2.0; /* 4y */
            w = (m02 - m20) / S;
            x = (m01 + m10) / S;
            y = 0.25 * S;
            z = (m12 + m21) / S;
        } else {
            Real S = feta::math::sqrt(1.0 + m22 - m00 - m11) * 2.0; /* 4z */
            w = (m10 - m01) / S;
            x = (m02 + m20) / S;
            y = (m12 + m21) / S;
            z = 0.25 * S;
        }
        q.template get<0>() = w;
        q.template get<1>() = x;
        q.template get<2>() = y;
        q.template get<3>() = z;
        return q;
    }

    /** @brief Test/introspection boundary overload from a raw `Real[9]`. */
    DEVICEHOST()
    static feta::Vec4T<Real> quatFromMatrix(const Real* const R)
    {
        Mat3 m;
        for (int i = 0; i < 9; ++i)
            m[i] = R[i];
        return quatFromMatrix(m);
    }

    /** @brief ω = vee(Ṙᵀ·R): angular velocity of ITRF w.r.t. ICRF, in
     *  inertial (ICRF) components [rad/s]. ω_z is the +sidereal rate. */
    DEVICEHOST()
    static feta::Vec3T<Real> omegaFromMatrices(
        const Mat3& R, const Mat3& Rdot)
    {
        /* M = Ṙᵀ·R  (skew-symmetric); take the antisymmetric part for
         * numerical robustness. M[3i+j] = sum_k Ṙ[k][i]·R[k][j]. */
        Mat3 M;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                M[3 * i + j] = Rdot[0 + i] * R[0 + j]
                    + Rdot[3 + i] * R[3 + j] + Rdot[6 + i] * R[6 + j];
        feta::Vec3T<Real> w;
        w.template get<0>() = 0.5 * (M[7] - M[5]); /* (M21 - M12)/2 */
        w.template get<1>() = 0.5 * (M[2] - M[6]); /* (M02 - M20)/2 */
        w.template get<2>() = 0.5 * (M[3] - M[1]); /* (M10 - M01)/2 */
        return w;
    }

    /** @brief Compose the quaternion + angular velocity directly. `ttFrac` is
     *  the high-precision O(1) TT MJD2000 day-fraction (see `composeMatrix`). */
    DEVICEHOST()
    static void compose(feta::Vec4T<Real>& q, feta::Vec3T<Real>& w,
        const feta::vector::Item<Real, NCHANNELS>& nodes,
        const feta::vector::Item<Real, NCHANNELS>& rates, const Real& ttMjd,
        const Real& ttFrac)
    {
        Mat3 R, Rdot;
        composeMatrix(R, &Rdot, nodes, rates, ttMjd, ttFrac);
        q = quatFromMatrix(R);
        w = omegaFromMatrices(R, Rdot);
    }

    /** @brief Unit quaternion (scalar-first `[w,x,y,z]`) for the `ZA(t)`
     *  Z-rotation `[[c,s,0],[-s,c,0],[0,0,1]]` (active rotation by `-t` about z).
     *  Body angular velocity of `ZA(t)` is `(0,0,-ṫ)`. */
    DEVICEHOST()
    static feta::Vec4T<Real> qAxisZ(const Real& t)
    {
        Real ch, sh;
        feta::math::sincos(0.5 * t, &sh, &ch);
        feta::Vec4T<Real> r;
        r.template get<0>() = ch;
        r.template get<1>() = 0.0;
        r.template get<2>() = 0.0;
        r.template get<3>() = -sh;
        return r;
    }

    /** @brief Unit quaternion for the `YA(t)` Y-rotation
     *  `[[c,0,-s],[0,1,0],[s,0,c]]` (active rotation by `-t` about y). Body
     *  angular velocity of `YA(t)` is `(0,-ṫ,0)`. */
    DEVICEHOST()
    static feta::Vec4T<Real> qAxisY(const Real& t)
    {
        Real ch, sh;
        feta::math::sincos(0.5 * t, &sh, &ch);
        feta::Vec4T<Real> r;
        r.template get<0>() = ch;
        r.template get<1>() = 0.0;
        r.template get<2>() = -sh;
        r.template get<3>() = 0.0;
        return r;
    }

    /** @brief Quaternion-native compose: the cache quaternion `q` (ICRF→ITRF,
     *  scalar-first) AND the angular velocity `w` (same frame/convention as
     *  @ref omegaFromMatrices) from ONE pass, WITHOUT materializing any dense
     *  rotation matrix or its derivative.
     *
     *  `R = PM·ER·PN` is decomposed into axis sub-rotations — `PN` is the
     *  ZYZ Euler `ZA(mems)·YA(d)·ZA(e)` (SOFA `c2ixys` form), `ER = ZA(θ)`, and
     *  `PM = I + skew(sp,xp,yp)` is the near-identity polar motion (small-angle
     *  quaternion). Each factor's body angular velocity is a closed form
     *  (`ω_ZA(t)=(0,0,-ṫ)`, `ω_YA(t)=(0,-ṫ,0)`, `ω_PM=(ypd,xpd,-spdot)`) and the
     *  total is composed in the body frame:
     *     ω = −[ q_PN*·(q_ER*·ω_PM + ω_ER) + ω_PN ].
     *  Matches the dense `vee(ṘᵀR)` to the orthonormality floor (~1e-13) — the
     *  quaternion is the EXACT rotation, so `w` is consistent with the cached
     *  quaternion `q` (see @ref quatFromMatrix). Peak live set is a handful of
     *  quaternions/vec3s — no 3×3 spill. */
    DEVICEHOST()
    static void composeQuat(feta::Vec4T<Real>& q, feta::Vec3T<Real>& w,
        const feta::vector::Item<Real, NCHANNELS>& nodes,
        const feta::vector::Item<Real, NCHANNELS>& rates, const Real& ttMjd,
        const Real& ttFrac)
    {
        checkInputsFinite(nodes, rates);
        constexpr Real invSec = 1.0 / 86400.0;
        const PreludeScalars p = computePrelude(nodes, rates, ttMjd, ttFrac);
        const Real e = p.e, d = p.d, mems = p.mems;
        const Real edot = p.edot, ddot = p.ddot, memsdot = p.memsdot;
        const Real theta = p.theta, thetadot = p.thetadot;
        const Real sp = p.sp, spdot = p.spdot;
        /* polar-motion channels (consumed by qPM and the ω_PM term) */
        const Real xp = nodes.get<iXp>(), yp = nodes.get<iYp>();
        const Real xpd = rates.get<iXp>() * invSec, ypd = rates.get<iYp>() * invSec;

        /* ---- factor quaternions ---- */
        const feta::Vec4T<Real> qZe = qAxisZ(e);
        const feta::Vec4T<Real> qYd = qAxisY(d);
        const feta::Vec4T<Real> qZm = qAxisZ(mems);
        const feta::Vec4T<Real> qER = qAxisZ(theta);
        feta::Vec4T<Real> qPMa;
        qPMa.template get<0>() = 1.0;
        qPMa.template get<1>() = 0.5 * yp;
        qPMa.template get<2>() = 0.5 * xp;
        qPMa.template get<3>() = -0.5 * sp;
        feta::Vec4T<Real> qPM;
        qPM = qPMa.unitVector(); /* normalise the small-angle quaternion */

        feta::Vec4T<Real> qPNa, qPN;
        qPNa = qZm.quatMul(qYd);
        qPN  = qPNa.quatMul(qZe);
        feta::Vec4T<Real> qa;
        qa = qPM.quatMul(qER);
        q  = qa.quatMul(qPN);

        /* ---- body angular velocities (closed form) ---- */
        feta::Vec3T<Real> wPM, wER, wZm, wYd, wZe;
        wPM.template get<0>() = ypd;
        wPM.template get<1>() = xpd;
        wPM.template get<2>() = -spdot;
        wER.template get<0>() = 0.0;
        wER.template get<1>() = 0.0;
        wER.template get<2>() = -thetadot;
        wZm.template get<0>() = 0.0;
        wZm.template get<1>() = 0.0;
        wZm.template get<2>() = -memsdot;
        wYd.template get<0>() = 0.0;
        wYd.template get<1>() = -ddot;
        wYd.template get<2>() = 0.0;
        wZe.template get<0>() = 0.0;
        wZe.template get<1>() = 0.0;
        wZe.template get<2>() = -edot;

        /* ω_PN = q_Ze*·(q_Yd*·ω_Zm + ω_Yd) + ω_Ze */
        feta::Vec3T<Real> acc, wPN;
        acc = qYd.quatConj().quatRotate(wZm);
        acc = qZe.quatConj().quatRotate(acc + wYd);
        wPN = acc + wZe;

        /* ω = −[ q_PN*·(q_ER*·ω_PM + ω_ER) + ω_PN ] */
        acc = qER.quatConj().quatRotate(wPM);
        acc = qPN.quatConj().quatRotate(acc + wER);
        w   = (acc + wPN) * (-1.0);
    }

    /** @brief Convert a TDB epoch (SPICE seconds past J2000) to the model
     *  time TT in MJD2000 days. Ports GODOT `tempo::tdbtt` (the periodic
     *  TDB−TT term, Fairhead & Bretagnon) — risk R4.
     *
     *  TDB-MJD2000 = etSec/86400 + 0.5 (MJD2000 origin is 2000-01-01T00:00,
     *  SPICE ET origin is the J2000 noon epoch, 0.5 day later). */
    DEVICEHOST()
    static Real epochToModelTime(
        const Real& tdbEtSec, Real* const dayFracOut = nullptr)
    {
        constexpr Real SECDAY = 86400.0;
        constexpr Real invSec = 1.0 / SECDAY;
        const Real tdbMjd     = tdbEtSec * invSec + 0.5;
        /* tdbtt: g = (357.53° − 0.9856003°·0.5) + 0.9856003°·tdbMjd,
         * the −0.5 folding the J2000-noon vs MJD2000-midnight offset. */
        constexpr Real DEG2RAD = M_PI / 180.0;
        constexpr Real alpha0  = 357.53;
        constexpr Real alpha   = 0.9856003;
        const Real g = DEG2RAD * (alpha0 + alpha * (tdbMjd - 0.5));
        const Real tdbMinusTtSec
            = 0.001658 * feta::math::sin(g) + 0.000014 * feta::math::sin(2.0 * g);
        const Real ttCorrDays = tdbMinusTtSec * invSec;
        /* Optionally emit the high-precision O(1) TT MJD2000 day-fraction the
         * ERA phase needs (see `composeMatrix`). Extract the TDB seconds-of-day
         * at O(1) magnitude (the exact remainder of the et double), turn it
         * into a day fraction, fold in the MJD2000 +0.5 offset, then subtract
         * the small TT correction — never forming the ~1.8e4 integer day, so
         * the time-of-day keeps full precision (capped only by ULP of the et
         * input itself, not by the much coarser ULP of the absolute day). */
        if (dayFracOut != nullptr) {
            const Real tdbSecOfDay = feta::math::fmod(tdbEtSec, SECDAY);
            Real tdbFrac = tdbSecOfDay * invSec + 0.5;
            tdbFrac -= feta::math::floor(tdbFrac);
            *dayFracOut = tdbFrac - ttCorrDays;
        }
        /* TT = TDB − (TDB−TT), in MJD2000 days (the large value: drives interp
         * and the slow ERA polynomial term). */
        return tdbMjd - ttCorrDays;
    }
};

} // namespace detail
} // namespace orientations
} // namespace brie
