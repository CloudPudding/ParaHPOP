#pragma once

#include "brie/frames/Rotation.h"

namespace brie {
namespace frames {

/** @brief Class that collects and connects the frames loaded via BRIE
 * rotation files, and more */
template<bool work, bool UseTexture>
class Frames {
    using RotEphUnitT =
        typename orientations::EphUnit<UseTexture>::template Ref<work>;
    using QuatT = feta::Vec4T<Real>;
    using Vec3R = feta::Vec3T<Real>;
    template<bool iwork>
    using VecRef =
        typename feta::vector::Array<Real, 3>::template Ref<iwork>::HandleT;

    using QuatRef = feta::vector::Array<Real, 4>::GRef;

    /** @brief Dim=3 global-ref handle for the inertial-frame
     *  ω(epoch, target) output of ``bodyAngularVelocity<iwork>``.
     *  Mirrors ``QuatRef`` (the global cache target the rotation API
     *  writes through). */
    using OmegaRef = feta::vector::Array<Real, 3>::GRef;

public:
    using RotationT = Rotation<false>;

    /** @brief Expose the inner rotation object */
    DEVICEHOST() RotEphUnitT rotations() const { return rotations_; }

    /** @brief Obliquity of the ecliptic at J2000 in radians: the IAU-1976
     *  constant 84381.448 arcsec that defines SPICE's ECLIPJ2000 frame.
     *  The arcsec→rad operation order (x/3600 first, ·π/180 after)
     *  reproduces SPICE's ECLIPJ2000 rotation matrix bit-for-bit — the
     *  reversed order lands 1 ULP off in sin(ε). Pinned in test_Frames. */
    DEVICEHOST() static constexpr Real eclipticObliquityJ2000()
    {
        return 84381.448 / 3600.0 * degToRad();
    }

    /** @brief The constant passive ICRF→ECLIPJ2000 bias: a frame rotation
     *  about +X by the J2000 obliquity. ECLIPJ2000 is inertial, so the
     *  bias is epoch-free and its angular velocity is zero. Also the
     *  composition factor for Type-3 binary-frame units whose
     *  INERT_FRAME is 17 (ECLIPJ2000-based PCKs, e.g. the NAIF Earth
     *  high-precision kernels). */
    DEVICEHOST() static RotationT eclipticBias()
    {
        return RotationT::make(RotationT::xRot(eclipticObliquityJ2000()));
    }

    /** @brief Extract a Rotation object for the local body axes at the given
     * epoch. Always ICRF→body: units whose base frame is ECLIPJ2000
     * (Type-3 INERT_FRAME == 17, e.g. the NAIF Earth high-precision
     * kernels) get the constant ecliptic bias composed —
     * R(ICRF→body) = R(ecl→body)·E(ICRF→ecl). */
    DEVICEHOST()
    RotationT bodyAxesRotation(const Real& epoch, const NaifId& target) const
    {
        if (isModelUnit_(target)) {
            return RotationT::make(
                rotations_.getQuaternionDirect(epoch, target));
        }
        RotationT rot
            = RotationT::fromRADecPM(getRADecPMInRad_(epoch, target));
        if (eclipticBased_(target)) {
            rot = RotationT::make(rot.eval().quatMul(eclipticBias().eval()));
        }
        return rot;
    }

    /** @brief Extract a Rotation object for the local body axes at the given
     * epoch */
    template<bool iwork>
    DEVICEHOST()
    RotationT bodyAxesRotation(const SampleIndex& idx, VecRef<iwork>& buffer,
        const Real& epoch, const NaifId& target) const
    {
        if (isModelUnit_(target)) {
            return RotationT::make(
                rotations_.getQuaternionDirect(epoch, target));
        }
        getRADecPMInRad_<iwork>(idx, buffer, epoch, target);
        RotationT rot = RotationT::fromRADecPM(buffer[idx]);
        if (eclipticBased_(target)) {
            rot = RotationT::make(rot.eval().quatMul(eclipticBias().eval()));
        }
        return rot;
    }

    /** @brief Extract the quaternion into the given global array */
    template<bool iwork>
    DEVICEHOST()
    void extractBodyAxesQuaternion(const SampleIndex& idx, QuatRef& quats,
        VecRef<iwork>& buffer, const Real& epoch, const NaifId& target) const
    {
        if (isModelUnit_(target)) {
            /* Direct Shepperd quaternion straight into the cache slot
             * (already ICRF→target; no RA/Dec/PM, no ecliptic bias). */
            quats[idx] = rotations_.getQuaternionDirect(epoch, target);
            return;
        }
        getRADecPMInRad_<iwork>(idx, buffer, epoch, target);
        if (eclipticBased_(target)) {
            /* Compose the constant ecliptic bias so the cached quaternion
             * is ICRF→body, like every other entry in the cache. Passive
             * composition stacks the later rotation on the left:
             * q_total = q(ecl→body) ⊗ q(ICRF→ecl). */
            quats[idx] = RotationT::raDecPMToQuat(buffer[idx])
                             .quatMul(eclipticBias().eval());
        } else {
            RotationT::raDecPMToQuat<iwork>(idx, buffer, quats);
        }
    }

    /** @brief Body angular velocity expressed in the inertial frame, rad/s.
     *
     *  Derived from IAU rotational kinematics for the Euler 3-1-3 passive
     *  sequence (RA + π/2, π/2 - Dec, PM) used by Rotation::raDecPMToQuat.
     *  Expressed in the inertial (J2000) frame so consumers can apply
     *  ω × r directly to inertial-frame positions without an intermediate
     *  body-fixed rotation. With α ≡ RA, δ ≡ DEC, W ≡ PM:
     *
     *    ω_x =  δ̇ sin α + Ẇ cos α cos δ
     *    ω_y = -δ̇ cos α + Ẇ sin α cos δ
     *    ω_z =  α̇ + Ẇ sin δ
     *
     *  Verification: for Earth (α ≈ 0, δ ≈ 90°), ω → (0, 0, PM_rate) which
     *  matches the sidereal rate ≈ 7.29e-5 rad/s along inertial ẑ. The FD
     *  parity test in test_Frames.* is the regression gate.
     *
     *  The Euler kinematics produce ω in the unit's base frame; for
     *  ECLIPJ2000-based Type-3 units the components are mapped back to
     *  ICRF through the constant bias (ω_icrf = Eᵀ·ω_ecl), so the answer
     *  is always ICRF-relative. */
    DEVICEHOST()
    Vec3R bodyAngularVelocity(const Real& epoch, const NaifId& target) const
    {
        if (isModelUnit_(target)) {
            return rotations_.getAngularVelocityDirect(epoch, target);
        }
        const Vec3R rdp     = getRADecPMInRad_(epoch, target);
        const Vec3R rdpRate = getRADecPMRateInRadPerSec_(epoch, target);

        const Real alpha    = rdp.template get<0>();
        const Real delta    = rdp.template get<1>();
        const Real alphaDot = rdpRate.template get<0>();
        const Real deltaDot = rdpRate.template get<1>();
        const Real WDot     = rdpRate.template get<2>();

        Real sinA, cosA, sinD, cosD;
        feta::math::sincos(alpha, &sinA, &cosA);
        feta::math::sincos(delta, &sinD, &cosD);

        Vec3R omega;
        omega.template get<0>() = deltaDot * sinA + WDot * cosA * cosD;
        omega.template get<1>() = -deltaDot * cosA + WDot * sinA * cosD;
        omega.template get<2>() = alphaDot + WDot * sinD;
        if (eclipticBased_(target)) {
            return eclipticBias().applyInverse(omega);
        }
        return omega;
    }

    /** @brief ``<iwork>`` shmem-scratch overload of ``bodyAngularVelocity``.
     *
     *  Pre-computes RA/Dec/PM angles into ``angleScratch[idx]`` and the
     *  matching rates into ``rateScratch[idx]`` via the existing
     *  ``<iwork>`` dispatcher cascade (``getRADecPMInRad_<iwork>`` +
     *  ``getRADecPMRateInRadPerSec_<iwork>``).  Both inputs land in
     *  block-shared memory before the Euler 3-1-3 kinematics formula
     *  runs, so the four ``sin``/``cos`` intermediates plus the three
     *  output components are the only live FP values during the final
     *  FMA chain — well within the register cap at
     *  ``EnvKernelLaunch`` (128/8 → reg 64) and below.
     *
     *  Bit-identity contract: the same FMA chain in the same order as
     *  the scalar ``bodyAngularVelocity`` above, only the input transport
     *  changes (scalar return → shmem-resident slot).  ``EXPECT_EQ`` is
     *  the right gate on the parity test. */
    template<bool iwork>
    DEVICEHOST()
    void bodyAngularVelocity(const SampleIndex& idx, OmegaRef& omegaOut,
        VecRef<iwork>& angleScratch, VecRef<iwork>& rateScratch,
        const Real& epoch, const NaifId& target) const
    {
        if (isModelUnit_(target)) {
            omegaOut[idx] = rotations_.getAngularVelocityDirect(epoch, target);
            return;
        }
        getRADecPMInRad_<iwork>(idx, angleScratch, epoch, target);
        getRADecPMRateInRadPerSec_<iwork>(idx, rateScratch, epoch, target);

        const Real alpha = angleScratch.template get<0>(idx);
        const Real delta = angleScratch.template get<1>(idx);
        const Real alphaDot = rateScratch.template get<0>(idx);
        const Real deltaDot = rateScratch.template get<1>(idx);
        const Real WDot     = rateScratch.template get<2>(idx);

        Real sinA, cosA, sinD, cosD;
        feta::math::sincos(alpha, &sinA, &cosA);
        feta::math::sincos(delta, &sinD, &cosD);

        Vec3R omega;
        omega.template get<0>() = deltaDot * sinA + WDot * cosA * cosD;
        omega.template get<1>() = -deltaDot * cosA + WDot * sinA * cosD;
        omega.template get<2>() = alphaDot + WDot * sinD;
        if (eclipticBased_(target)) {
            omega = eclipticBias().applyInverse(omega);
        }
        omegaOut[idx] = omega;
    }

    /** @brief Create a clone of this frame object */
    DEVICEHOST() Frames clone() const { return *this; }

    /** @brief Inner rotation eph unit - made public for PODification */
    RotEphUnitT rotations_;

protected:
    /** @brief Retrieve a Vec3R containing Right
     * Ascension, Declination, And Prime meridian angle on the 0,1,2
     * components converted in radians, if necessary from the given metadata
     * unit type */
    DEVICEHOST()
    inline Vec3R getRADecPMInRad_(const Real& epoch, const NaifId& target) const
    {
        Vec3R radecpm = rotations_.getRADecPM(epoch, target);

        /* get the body unit for the target body */
        idx_t tc = rotations_.metadata().getTargetBodyCount(target);

        /* if unit type is degrees, convert to radians */
        if (rotations_.metadata().getUnitType(tc) == 1) {
            radecpm *= degToRad();
        }

        return radecpm;
    }
    template<bool iwork>
    DEVICEHOST()
    inline void getRADecPMInRad_(const SampleIndex& idx, VecRef<iwork>& out,
        const Real& epoch, const NaifId& target) const
    {
        rotations_.template getRADecPM<iwork>(idx, out, epoch, target);

        /* get the body unit for the target body */
        idx_t tc = rotations_.metadata().getTargetBodyCount(target);

        /* if unit type is degrees, convert to radians */
        if (rotations_.metadata().getUnitType(tc) == 1) {
            out[idx] *= degToRad();
        }
    }

    /** @brief RA/DEC/PM rates uniformly in rad/s.
     *  Mirror of getRADecPMInRad_ on the derivative path: Type 1 bodies
     *  return deg/s natively (PCK polynomials are deg-valued) and get a
     *  degToRad() scale; Type 3 binary frames already produce rad/s. */
    DEVICEHOST()
    inline Vec3R getRADecPMRateInRadPerSec_(
        const Real& epoch, const NaifId& target) const
    {
        Vec3R rates = rotations_.getRADecPMRate(epoch, target);

        idx_t tc = rotations_.metadata().getTargetBodyCount(target);

        if (rotations_.metadata().getUnitType(tc) == 1) {
            rates *= degToRad();
        }

        return rates;
    }

    /** @brief ``<iwork>`` shmem-scratch overload of
     *  ``getRADecPMRateInRadPerSec_``. Mirrors ``getRADecPMInRad_<iwork>``
     *  on the rate path — writes the (rad/s)-normalised rate vector into
     *  ``out[idx]`` via the existing ``getRADecPMRate_<iwork>`` dispatcher
     *  (which routes to Type1/Type3 internally). */
    template<bool iwork>
    DEVICEHOST()
    inline void getRADecPMRateInRadPerSec_(const SampleIndex& idx,
        VecRef<iwork>& out, const Real& epoch, const NaifId& target) const
    {
        rotations_.template getRADecPMRate<iwork>(idx, out, epoch, target);

        idx_t tc = rotations_.metadata().getTargetBodyCount(target);
        if (rotations_.metadata().getUnitType(tc) == 1) {
            out[idx] *= degToRad();
        }
    }

    /** @brief Degrees to radians conversion */
    DEVICEHOST() static constexpr Real degToRad() { return M_PI / 180.0; }

    /** @brief True when the target's unit evaluates against ECLIPJ2000
     *  (Type-3 INERT_FRAME == 17): the public rotation/ω entry points
     *  compose the constant ecliptic bias on top, so answers stay
     *  ICRF-relative for every unit. J2000-based units (Type 1, and
     *  Type 3 with INERT_FRAME == 1 such as moon_pa) take the untouched
     *  legacy path — bit-identical by construction. */
    DEVICEHOST()
    inline bool eclipticBased_(const NaifId& target) const
    {
        return rotations_.baseInertFrame(
                   rotations_.metadata().getTargetBodyCount(target))
            == 17;
    }

    /** @brief True when the target is a model-based orientation unit
     *  (``unit_type == 4``, the native IPF → IERS2000 family). Such units
     *  compose a rotation *matrix* and expose a direct quaternion + ω
     *  (Shepperd / analytic derivative), so the public Frames entry points
     *  take the direct escape hatch instead of the RA/Dec/PM round-trip
     *  (pole-singular for Earth) and its Euler-rate kinematics. The base
     *  frame is ICRF by construction, so no ecliptic bias is composed. */
    DEVICEHOST()
    inline bool isModelUnit_(const NaifId& target) const
    {
        return rotations_.metadata().getUnitType(
                   rotations_.metadata().getTargetBodyCount(target))
            == 4;
    }
};
} // namespace frames
} // namespace brie