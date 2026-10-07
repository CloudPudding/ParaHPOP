#pragma once

#include "paraHPOP/model/environment/Environment.h"

#include "paraHPOP/typedefs.h"

namespace paraHPOP {
namespace model {
namespace accelerations {

using EnvT    = environment::Env;
using OrientationsT = environment::Orientations<false>;

/** @brief Collection of J2 acceleration functions */
struct J2 {

    /** @brief Low level evaluation function
     * See Wakker, Fundamentals of Astrodynamics, eq. 20.6
     */
    DEVICEHOST()
    static inline Vec3R basicEval(const Vec3R& pos, const Real& gm,
        const Real& j2, const Real& bodyRadius)
    {
        /* Inverse radius terms */
        const Real invr  = pos.rNorm();
        const Real invr2 = invr * invr;
        const Real invr5 = invr2 * invr2 * invr;
        Vec3R vc(1.0 - 5.0 * (pos.get<2>() * pos.get<2>()) * invr2);
        const Real c = -1.5 * gm * j2 * invr5 * bodyRadius * bodyRadius;

        /* add 2 to the z component */
        vc.get<2>() += 2.0;

        /* Return the evaluated result */
        return c * pos * vc;
    }

    /** @brief Compute the J2 acceleration of ``body`` on the sample.  ``pos``
     *  is the sample position relative to ``body`` (the caller subtracts the
     *  body position).  ``rotTarget`` is the orientation frame the body rotates
     *  under (its own id under the IAU default, or a resolved binary-frame
     *  target like ITRF93's 3000); the GM / J2 / radius are read from
     *  ``body``'s constants entry. */
    DEVICEHOST()
    static Vec3R eval(const SampleIndex& i, const Vec3R& pos,
        const mReal_t& epoch, const brie::NaifId& body,
        const brie::NaifId& rotTarget, const EnvT::GRef& env)
    {
        return eval_<false>(i, pos, epoch, body, rotTarget, env);
    }
    DEVICEHOST()
    static Vec3R eval(const SampleIndex& i, const Vec3R& pos,
        const mReal_t& epoch, const brie::NaifId& body,
        const brie::NaifId& rotTarget, const EnvT::WRef& env)
    {
        return eval_<true>(i, pos, epoch, body, rotTarget, env);
    }

protected:
    /** @brief Compute the J2 acceleration */
    template<bool work>
    DEVICEHOST()
    static inline Vec3R
        eval_(const SampleIndex& i, const Vec3R& pos, const mReal_t& epoch,
            const brie::NaifId& body, const brie::NaifId& rotTarget,
            const EnvT::Ref<work>& env)
    {
        Vec3R out;
        ieval_<work>(i, out, pos, epoch, body, rotTarget, env);
        return out;
    }

    /** @brief Compute in-place the J2 acceleration of ``body``.  ``pos`` is
     *  already ``body``-relative (the caller subtracts the body position). */
    template<bool work>
    DEVICEHOST()
    static inline void ieval_([[maybe_unused]] const SampleIndex& i, Vec3R& acc,
        const Vec3R& pos, const mReal_t& epoch, const brie::NaifId& body,
        const brie::NaifId& rotTarget, const EnvT::Ref<work>& env)
    {
        auto bc = env.constants().body(body);

        if (pos.norm() > bc.soi())
            return;

        typename EnvT::Ref<work>::OrientationsT::RotationT rot
            = env.orientations().bodyAxesRotation(epoch, rotTarget);
        const Vec3R iPos = rot.rotate(pos);

        /* evaluate otherwise */
        acc += rot.applyInverse(basicEval(iPos, bc.gm(), bc.j2(), bc.r()));
    }
};

namespace kernel {

/** @brief Compile-time launch metadata for ::j2.  See ``LaunchTraits.h``. */
using J2Launch = KernelLaunchTraits<256, 4>;

__global__ void j2(feta::vector::Array<Real, 3>::GRef bodyAcc,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef pos,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef bodyPos,
    GRID_CONSTANT() feta::vector::Array<Real, 4>::GRef rotCache,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated,
    GRID_CONSTANT() Real gm, GRID_CONSTANT() Real j2,
    GRID_CONSTANT() Real radius, GRID_CONSTANT() Real soi);
}

} // namespace accelerations
} // namespace model
} // namespace paraHPOP