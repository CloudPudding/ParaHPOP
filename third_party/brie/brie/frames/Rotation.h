#pragma once

#include "brie/orientations.h"
#include <type_traits>

namespace brie {
namespace frames {

/** @brief Axis convention for a local-orbital frame built from a Cartesian
 * state (see Rotation::localFrameRotation). All conventions share the same
 * orbit plane (normal h = r x v); they differ only in which physical
 * direction maps to which body axis:
 *
 *  - RTN / RSW : x = radial (r), z = orbit normal (h), y = h x r (transverse)
 *  - TNW       : x = tangential (v), z = orbit normal (h), y = h x v (in-plane)
 *  - VNB       : x = velocity (v), y = orbit normal (h), z = v x h (binormal)
 *  - LVLH      : z = -r (nadir), y = -h, x = h x r (local horizontal)
 *
 * RTN/RSW (radial primary) and TNW (velocity primary) are the two "base"
 * triads (x -> primary, z -> h); VNB and LVLH are fixed axis relabelings of
 * those bases.
 */
enum class AxesSelection { RTN, RSW, TNW, VNB, LVLH };

/** @brief Individual rotation */
template<bool Direct = true>
class Rotation {

    /** @brief Feta quaternion type  */
    using QuatT = feta::Vec4T<Real>;
    /** @brief Feta 3D vector type */
    using Vec3R = feta::Vec3T<Real>;
    /** @brief Feta 6D vector type: head<3> represents the position components,
     * tail<3> the velocity components */
    using Vec6R = feta::Vec6T<Real>;

    /* buffer */
    template<bool iwork>
    using VecRef =
        typename feta::vector::Array<Real, 3>::template Ref<iwork>::HandleT;
    using QuatRef = feta::vector::Array<Real, 4>::GRef;

public:
    /** @brief Return the default quaternion */
    DEVICEHOST() static QuatT NoRot()
    {
        QuatT out;
        out.get<0>() = 1;
        return out;
    }

    /** @brief Principal rotation axis = the imaginary quaternion component
     *  (the ``get<>`` index) that carries ``sin(θ/2)``. */
    enum class Axis : idx_t { X = 1, Y = 2, Z = 3 };

    /** @brief Return a principal-axis rotation quaternion from the given
     *  angle, about the principal @ref Axis ``A``. Single-sources the former
     *  ``xRot`` / ``yRot`` / ``zRot`` triplet (identical save for that one
     *  component index), so the three stay in lock-step; bit-identical to the
     *  originals by construction (``A`` is a compile-time index → the same two
     *  stores in the same order). */
    template<Axis A>
    DEVICEHOST() static inline QuatT axisRot(const Real& theta)
    {
        QuatT out;
        Real halfTheta = theta * 0.5;
        Real s, c;
        feta::math::sincos(
            halfTheta, &s, &c); // single call for better accuracy & perf
        out.get<0>()                              = c;
        out.template get<static_cast<idx_t>(A)>() = s;
        if constexpr (Direct) {
            return out;
        } else {
            return out.quatConj();
        }
    }

    /** @brief Return a "z"-rotation quaternion from the given angle */
    DEVICEHOST() static inline QuatT zRot(const Real& theta)
    {
        return axisRot<Axis::Z>(theta);
    }

    /** @brief Return a "y"-rotation from the given angle */
    DEVICEHOST() static inline QuatT yRot(const Real& theta)
    {
        return axisRot<Axis::Y>(theta);
    }

    /** @brief Return a "x"-rotation from the given angle */
    DEVICEHOST() static inline QuatT xRot(const Real& theta)
    {
        return axisRot<Axis::X>(theta);
    }

    /** @brief Return a quaternion from the given angle and feta
     * expression-based principal axis */
    template<typename Expr>
    DEVICEHOST()
    static inline QuatT axisAngle(const Expr& axis, const Real& angle)
    {
        static_assert(feta::core::expr::isExpression<Expr>::value,
            "Input must be a feta vector expression");
        static_assert(Expr::VecDims == 3, "Input axis must be 3D vector");
        QuatT out;
        Real halfAngle = angle * 0.5;
        Real s, c;
        feta::math::sincos(halfAngle, &s, &c);
        out.get<0>()  = c;
        out.tail<3>() = axis.unitVector() * s;
        if constexpr (Direct) {
            return out;
        } else {
            return out.quatConj();
        }
    }

    /** @brief Return a full quaternion from the given Vec3R containing
     * Right Ascension, Declination, And Prime meridian angle on the 0,1,2
     * components
     */
    DEVICEHOST()
    static inline QuatT raDecPMToQuat(const Vec3R& radecpm)
    {
        static_assert(Direct == false);

        Vec3R eulerAngles;
        eulerAngles.get<0>() = M_PI / 2 + radecpm.get<0>(); // RA
        eulerAngles.get<1>() = M_PI / 2 - radecpm.get<1>(); // Dec
        eulerAngles.get<2>() = radecpm.get<2>();            // PM
        return Rotation<false>::euler313ToQuat(eulerAngles);
    }
    template<bool iwork>
    DEVICEHOST()
    static inline void raDecPMToQuat(
        const SampleIndex& idx, VecRef<iwork>& radecpm, QuatRef& quat)
    {
        static_assert(Direct == false);

        Vec3R eulerAngles;
        eulerAngles.get<0>() = M_PI / 2 + radecpm.template get<0>(idx); // RA
        eulerAngles.get<1>() = M_PI / 2 - radecpm.template get<1>(idx); // Dec
        eulerAngles.get<2>() = radecpm.template get<2>(idx);            // PM
        quat[idx] = Rotation<false>::euler313ToQuat(eulerAngles);
    }

    /** @brief Quaternion exchanger */
    DEVICEHOST() static inline QuatT qmul(const QuatT& q)
    {
        if constexpr (Direct) {
            return q;
        } else {
            return q.quatConj();
        }
    }

    /** @brief Return a quaternion for a 3-1-3 (Z-X-Z) Euler rotation using
     * the angles provided in the order which the rotations are to be applied
     * (first, second, third).
     */
    DEVICEHOST() static inline QuatT euler313ToQuat(const Vec3R& angles)
    {
        // Define quaternions for rotations about fixed axes (Z-X-Z) with
        // angles (first, second, third). For active rotations on vectors
        // with v' = q v q^{-1}, composing R = Rz(third) * Rx(second) *
        // Rz(first) corresponds to q_total = q1 ⊗ q2 ⊗ q3.
        //
        // IMPORTANT: For passive (frame) rotations (Direct == false), the
        // correct overall rotation is the inverse of the active one:
        // q_passive = (q_active)^{-1} = q3^{-1} ⊗ q2^{-1} ⊗ q1^{-1}.
        // Since zRot/xRot/yRot already return conjugated quaternions when
        // Direct == false, we must also REVERSE the multiplication order to
        // obtain the proper inverse of the composed rotation.

        const QuatT q1 = Rotation<Direct>::zRot(angles.get<0>()); // Rz(first)
        const QuatT q2 = Rotation<Direct>::xRot(angles.get<1>()); // Rx(second)
        const QuatT q3 = Rotation<Direct>::zRot(angles.get<2>()); // Rz(third)
        if constexpr (Direct) {
            /* Active rotation: compose in the natural order */
            return q1.quatMul(q2).quatMul(q3);
        } else {
            /* Passive/frame rotation: reverse order to build the inverse */
            return q3.quatMul(q2).quatMul(q1);
        }
    }
    template<bool iwork>
    DEVICEHOST()
    static inline void euler313ToQuat(
        const SampleIndex& idx, VecRef<iwork>& angles, QuatRef& quat)
    {
        // Define quaternions for rotations about fixed axes (Z-X-Z) with
        // angles (first, second, third). For active rotations on vectors
        // with v' = q v q^{-1}, composing R = Rz(third) * Rx(second) *
        // Rz(first) corresponds to q_total = q1 ⊗ q2 ⊗ q3.
        //
        // IMPORTANT: For passive (frame) rotations (Direct == false), the
        // correct overall rotation is the inverse of the active one:
        // q_passive = (q_active)^{-1} = q3^{-1} ⊗ q2^{-1} ⊗ q1^{-1}.
        // Since zRot/xRot/yRot already return conjugated quaternions when
        // Direct == false, we must also REVERSE the multiplication order to
        // obtain the proper inverse of the composed rotation.

        const QuatT q1
            = Rotation<Direct>::zRot(angles.template get<0>(idx)); // Rz(first)
        const QuatT q2
            = Rotation<Direct>::xRot(angles.template get<1>(idx)); // Rx(second)
        const QuatT q3
            = Rotation<Direct>::zRot(angles.template get<2>(idx)); // Rz(third)
        if constexpr (Direct) {
            /* Active rotation: compose in the natural order */
            quat[idx] = q1.quatMul(q2).quatMul(q3);
        } else {
            /* Passive/frame rotation: reverse order to build the inverse */
            quat[idx] = q3.quatMul(q2).quatMul(q1);
        }
    }

    /** @brief Shortest arc quaternion rotation from the given Vec3R (from)
     * to the given Vec3R (t0) */
    DEVICEHOST()
    static QuatT shortestArcRotation(const Vec3R& from, const Vec3R& to)
    {
        Vec3R f = from.unitVector();
        Vec3R t = to.unitVector();

        Real cosTheta = f.dot(t);

        Real theta = feta::math::acos(cosTheta);

        /* Special case when vectors are the same: no rotation needed */
        /* should be changed for single precision */
        if (cosTheta > 1.0 - 1e-12)
            return Rotation<Direct>::NoRot();
        /* Special case when vectors are the opposite (180 deg rotation) */
        /* should be changed for single precision */
        if (cosTheta < -1.0 + 1e-12) {
            QuatT out;
            out.get<2>() = 1;
            if constexpr (Direct) {
                return out;
            } else {
                return out.quatConj();
            }
        }

        /* go on with the computation otherwise */
        return QuatT(Rotation<Direct>::axisAngle(f.cross(t), theta));
    }

    /** @brief Factory method that takes a Vec3R containing Right Ascension,
     * Declination, And Prime Meridian and constructs a rotation object,
     * that can perform rotations from the implicit inertial frame toward
     * the target frame */
    template<typename Expr>
    DEVICEHOST()
    static Rotation fromRADecPM(const Expr& radecpm)
    {
        static_assert(Direct == false);

        return Rotation::make(Rotation<false>::raDecPMToQuat(radecpm));
    }

    /** @brief Factory: build a rotation from Euler Z-X-Z (3-1-3) angles
     * given as (first, second, third). Direct active rotation. */
    template<typename Expr>
    DEVICEHOST()
    static Rotation fromEuler313(const Expr& angles)
    {
        return Rotation::make(Rotation::euler313ToQuat(angles));
    }

    /** @brief Build the coordinate (passive) rotation from the implicit
     * inertial frame into the local-orbital frame defined by the given
     * Cartesian state (position in head<3>, velocity in tail<3>), for the
     * requested axis convention @p Sel (see AxesSelection).
     *
     * The state is assumed expressed relative to the reference body
     * (from -> to); the orbit-plane normal is h = (r x v). The construction
     * aligns the x axis with the convention's "primary" direction (radial r
     * for RTN/RSW/LVLH, velocity v for TNW/VNB), then rotates about that new
     * x axis to align z with h. RTN/RSW and TNW therefore reproduce the
     * legacy localPosFrameRotation / localVelFrameRotation bit-for-bit; VNB
     * and LVLH apply a constant axis relabeling on top of those bases.
     */
    template<AxesSelection Sel, typename Expr>
    DEVICEHOST()
    static Rotation localFrameRotation(const Expr& cartState)
    {
        static_assert(feta::core::expr::isExpression<Expr>::value,
            "Input must be a feta vector expression");
        static_assert(Expr::VecDims == 6, "Input axis must be 6D vector");
        static_assert(Direct == false,
            "Local frames are coordinate (passive) rotations of the frame, "
            "not of vectors");

        /* radial-primary conventions align x with r; the rest align x with v */
        constexpr bool radialPrimary
            = (Sel == AxesSelection::RTN || Sel == AxesSelection::RSW
                || Sel == AxesSelection::LVLH);

        const auto r = cartState.template head<3>();
        const auto v = cartState.template tail<3>();
        /* compute the angular momentum (orbit normal) vector expression */
        const auto h = (r.cross(v)).unitVector();

        /* Vec3R for the identity x axis */
        Vec3R x;
        x.get<0>() = 1;
        /* Vec3R for the identity z axis */
        Vec3R z;
        z.get<2>() = 1;

        /* primary direction aligned to the x axis */
        Vec3R primary;
        if constexpr (radialPrimary) {
            primary = r;
        } else {
            primary = v;
        }

        /* first rotation: shortest angle rotation to align x with the primary */
        Rotation<true> r1 = Rotation<true>::make(
            Rotation<true>::shortestArcRotation(x, primary));

        /* Second rotation: Rotate around the newly defined x axis to align
         * z with h */
        Vec3R zRotated    = r1.rotate(z);
        Rotation<true> r2 = Rotation<true>::make(
            Rotation<true>::shortestArcRotation(zRotated, h));

        /* composite active rotation (identity axes -> base triad) */
        QuatT _to = r2.eval().quatMul(r1.eval());

        /* base coordinate rotation: invert so we rotate the coordinate system
         * rather than vectors. For RTN/RSW/TNW this is the final result. */
        Rotation base = Rotation::make(_to).inverse();

        if constexpr (Sel == AxesSelection::RTN || Sel == AxesSelection::RSW
            || Sel == AxesSelection::TNW) {
            return base;
        } else if constexpr (Sel == AxesSelection::VNB) {
            /* VNB relabels the TNW base axes (x, y, z) -> (x, z, -y): an
             * active Rx(-90 deg) applied to the base coordinate quaternion. */
            const QuatT relabel = Rotation<true>::xRot(-M_PI / 2);
            return Rotation::make(relabel.quatMul(base.eval()));
        } else { /* AxesSelection::LVLH */
            /* LVLH relabels the RTN base axes (x, y, z) -> (y, -z, -x): an
             * active 120 deg rotation about (1, 1, -1)/sqrt(3), i.e. the unit
             * quaternion (w, x, y, z) = (1, 1, 1, -1)/2. */
            QuatT relabel;
            relabel.get<0>() = 0.5;
            relabel.get<1>() = 0.5;
            relabel.get<2>() = 0.5;
            relabel.get<3>() = -0.5;
            return Rotation::make(relabel.quatMul(base.eval()));
        }
    }

    /** @brief Local-orbital RTN frame rotation (radial primary): aligns the
     * x axis with the position vector, then the z axis with the orbit normal.
     * Thin wrapper over localFrameRotation<AxesSelection::RTN>. */
    DEVICEHOST()
    static Rotation localPosFrameRotation(const Vec6R& cartState)
    {
        return localFrameRotation<AxesSelection::RTN>(cartState);
    }

    /** @brief Local-orbital TNW frame rotation (velocity primary): aligns the
     * x axis with the velocity vector, then the z axis with the orbit normal.
     * Thin wrapper over localFrameRotation<AxesSelection::TNW>. */
    template<typename Expr>
    DEVICEHOST()
    static Rotation localVelFrameRotation(const Expr& cartState)
    {
        return localFrameRotation<AxesSelection::TNW>(cartState);
    }

    /** @brief Inertial angular velocity (rad/s, expressed in the inertial
     * frame) of the local-orbital frame defined by the Cartesian state
     * @p cartState (position in head<3>, velocity in tail<3>, target relative
     * to centre): @f$\boldsymbol\omega = (\mathbf r\times\mathbf v)/|\mathbf r|^2@f$.
     *
     * This is the orbital angular velocity of the radius vector — the EXACT
     * rigid-rotation rate of the radial-primary local-orbital triads (RTN/RSW
     * and LVLH, whose orbit normal h = r x v is constant for unperturbed
     * motion), and the exact rate of ANY convention on a circular orbit (the
     * synodic / CR3BP case). Paired with localFrameRotation it transports a
     * state into the co-rotating frame; see rotatingFrameVelocity. For the
     * velocity-primary conventions (TNW/VNB) on an eccentric orbit it is the
     * orbital-rate transport (the velocity direction's exact rate additionally
     * needs the acceleration, which is not available kinematically). */
    template<typename Expr>
    DEVICEHOST()
    static Vec3R localFrameAngularVelocity(const Expr& cartState)
    {
        static_assert(feta::core::expr::isExpression<Expr>::value,
            "Input must be a feta vector expression");
        static_assert(Expr::VecDims == 6, "Input state must be a 6D vector");
        const Vec3R r = cartState.template head<3>();
        const Vec3R v = cartState.template tail<3>();
        return Vec3R(r.cross(v) * (Real(1.0) / r.squaredNorm()));
    }

    /** @brief Two-body / CR3BP mean motion @f$n = \sqrt{(gm_1+gm_2)/a^3}@f$
     * (rad/s): the constant rotation rate of the synodic frame of two primaries
     * with gravitational parameters @p gm1, @p gm2 separated by @p separation.
     * For a circular relative orbit this equals |localFrameAngularVelocity|;
     * for an eccentric orbit it is the mean (time-averaged) rate. */
    DEVICEHOST() static inline Real meanMotion(
        const Real& gm1, const Real& gm2, const Real& separation)
    {
        return feta::math::sqrt(
            (gm1 + gm2) / (separation * separation * separation));
    }

    /** @brief Rotating-frame velocity modifier: given this passive ICRF->frame
     * rotation, the frame's inertial angular velocity @p omega (e.g. from
     * localFrameAngularVelocity), and an inertial position @p r / velocity
     * @p v (relative to the frame centre), return the velocity as measured in
     * the co-rotating frame, expressed in frame axes: @f$v' = R\,(v-\omega\times
     * r)@f$. Inverse of inertialFromRotatingVelocity. */
    template<typename P, typename V, typename W>
    DEVICEHOST()
    Vec3R rotatingFrameVelocity(
        const P& r, const V& v, const W& omega) const
    {
        return Vec3R(this->rotate(v - omega.cross(r)));
    }

    /** @brief Recover the inertial velocity from a co-rotating-frame velocity:
     * given the inertial position @p rInertial, the frame-axes velocity
     * @p vFrame, and the frame's inertial angular velocity @p omega,
     * @f$v = R^{\!\top} v' + \omega\times r@f$. Inverse of
     * rotatingFrameVelocity. */
    template<typename P, typename V, typename W>
    DEVICEHOST()
    Vec3R inertialFromRotatingVelocity(
        const P& rInertial, const V& vFrame, const W& omega) const
    {
        return Vec3R(this->applyInverse(vFrame) + omega.cross(rInertial));
    }

    /** @brief Factory method to Construct a rotation with `to` only - given as
     * an expression */
    template<typename Expr>
    DEVICEHOST()
    static Rotation make(const Expr& to)
    {
        static_assert(feta::core::expr::isExpression<Expr>::value,
            "Input must be a feta vector expression");
        static_assert(
            Expr::VecDims == 4, "Input axis must be 4D vector / Quaternion");
        return { to.unitVector() };
    }

    /** @brief do the direct rotation */
    template<typename Expr>
    DEVICEHOST()
    std::conditional_t<(Expr::VecDims == 4), QuatT, Vec3R> rotate(
        const Expr& vec) const
    {
        /* Assert that the dimension is either 3 or 4 and a vector expression */
        static_assert(feta::core::expr::isExpression<Expr>::value,
            "Input must be a feta vector expression");
        static_assert(Expr::VecDims == 3 || Expr::VecDims == 4,
            "Input axis must be 3D or 4D vector");

        if constexpr (Expr::VecDims == 4) {
            /* if 4D, assume it is a quaternion and perform quaternion
             * multiplication */
            return eval().quatMul(vec).quatMul(inverseEval());
        } else {
            /* if 3D, assume it is a pure vector and perform rotation */
            /* We decompose the product right to left */
            return eval()
                .quatMul(vec.asPureQuaternion())
                .quatMul(inverseEval())
                .asBack3DVector();
        }
    }

    /** @brief operator/ alias - alias for inverse rotations: applyInverse
     */
    template<typename Expr>
    DEVICEHOST()
    std::conditional_t<(Expr::VecDims == 4), QuatT, Vec3R> applyInverse(
        const Expr& vec) const
    {
        /* Assert that the dimension is either 3 or 4 and a vector expression */
        static_assert(feta::core::expr::isExpression<Expr>::value,
            "Input must be a feta vector expression");
        static_assert(Expr::VecDims == 3 || Expr::VecDims == 4,
            "Input axis must be 3D or 4D vector");
        if constexpr (Expr::VecDims == 4) {
            /* if 4D, assume it is a quaternion and perform quaternion
             * multiplication */
            return inverseEval().quatMul(vec).quatMul(eval());
        } else {
            /* if 3D, assume it is a pure vector and perform rotation */
            return inverseEval()
                .quatMul(vec.asPureQuaternion())
                .quatMul(eval())
                .asBack3DVector();
        }
    }

    /** @brief Evaluate the direct composite rotation and return the
     * resulting quaternion */
    DEVICEHOST() inline const QuatT& eval() const { return to_; }

    /** @brief Evaluate the inverse composite rotation and return the
     * resulting quaternion */
    DEVICEHOST() QuatT inverseEval() const { return to_.quatReciprocal(); }

    /** @brief Return an explict version of the inverse rotation */
    DEVICEHOST() Rotation inverse() const
    {
        return Rotation::make(to_.quatConj());
    }

    /** @brief Data members made public for PODification */
    /** @brief "to" - base rotation from a given frame to a common inertial
     * frame (ICRF), representing the reference frame to rotate "to" */
    QuatT to_ = NoRot();
};

} // namespace frames
} // namespace brie