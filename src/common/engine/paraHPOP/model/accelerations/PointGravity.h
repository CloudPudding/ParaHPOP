#pragma once

#include "paraHPOP/model/environment/Environment.h"
#include "paraHPOP/typedefs.h"

namespace paraHPOP {
namespace model {
namespace accelerations {

using EnvT = ::paraHPOP::model::environment::Env;

/** @brief Collection of point gravity acceleration functions */
struct PointGravity {

    /** @brief Basic Two-Body acceleration */
    template<typename Expr>
    DEVICEHOST()
    static FORCEINLINE() decltype(auto)
        twoBody(const Expr& deltaPos, const Real& gm)
    {
        return -gm * deltaPos * deltaPos.rCubedNorm();
    }

    /** @brief Basic Two-Body acceleration */
    template<typename Expr>
    DEVICEHOST()
    static FORCEINLINE() decltype(auto)
        twoBody(const SampleIndex& i, const Expr& deltaPos, const Real& gm)
    {
        return -gm * deltaPos[i] * deltaPos.rCubedNorm(i);
    }

    /**
     * @brief Compute cumulative acceleration for a set of bodies
     *
     */
    template<typename Expr>
    DEVICEHOST()
    static decltype(auto) eval(const SampleIndex& i, const Expr& pos,
        const mReal_t& epoch, const brie::NaifId& COI, const EnvT::GRef& env)
    {
        return eval_<Expr, false>(i, pos, epoch, COI, env);
    }
    template<typename Expr>
    DEVICEHOST()
    static decltype(auto) eval(const SampleIndex& i, const Expr& pos,
        const mReal_t& epoch, const brie::NaifId& COI, const EnvT::WRef& env)
    {
        return eval_<Expr, true>(i, pos, epoch, COI, env);
    }

    /** @brief Barycentric Third body acceleration */
    template<typename L, typename R>
    DEVICEHOST()
    static decltype(auto)
        thirdBody_B(const L& scPos, const R& bodyPos, const Real& gm)
    {
        /* default case is barycentric */
        return twoBody(scPos - bodyPos, gm);
    }

    /** @brief Barycentric Third body acceleration */
    template<typename L, typename R>
    DEVICEHOST()
    static decltype(auto) thirdBody_B(
        const SampleIndex& i, const L& scPos, const R& bodyPos, const Real& gm)
    {
        /* default case is barycentric */
        return twoBody(i, scPos - bodyPos, gm);
    }

    /** @brief NonBarycentric Third body acceleration */
    template<typename L, typename R>
    DEVICEHOST()
    static decltype(auto)
        thirdBody_NB(const L& scPos, const R& bodyPos, const Real& gm)
    {
        /* Barycentric term + Tidal term */
        return twoBody(scPos - bodyPos, gm) + twoBody(bodyPos, gm);
    }

    /** @brief NonBarycentric Third body acceleration */
    template<typename L, typename R>
    DEVICEHOST()
    static decltype(auto) thirdBody_NB(
        const SampleIndex& i, const L& scPos, const R& bodyPos, const Real& gm)
    {
        /* Barycentric term + Tidal term */
        return twoBody(i, scPos - bodyPos, gm) + twoBody(i, bodyPos, gm);
    }

    /** @brief Inner evaluation for a single body */
    DEVICEHOST()
    static FORCEINLINE() Vec3R eval(const Vec3R& deltaPos, const Vec3R& bodyPos,
        const Real& gm, const bool& onBodyOrBarycenter)
    {
        if (onBodyOrBarycenter) {
            /* Simple two-body acceleration: same as twoBody() helper */
            return twoBody(deltaPos, gm);
        } else {
            /* Non-barycentric third-body term: barycentric + tidal.
               using the existing helpers keeps the sign consistent. */
            return twoBody(deltaPos, gm) + twoBody(bodyPos, gm);
        }
    }

protected:
    /** @brief work-templated evaluation */
    template<typename Expr, bool work>
    DEVICEHOST()
    static inline Vec3R
        eval_(const SampleIndex& i, const Expr& pos, const mReal_t& epoch,
            const brie::NaifId& COI, const EnvT::Ref<work>& env)
    {
        /* Initialize Null vector */
        Vec3R acc;
        ieval_<Expr, work>(i, acc, pos, epoch, COI, env);
        return acc;
    }

    /** @brief work-templated in-place evaluation */
    template<typename Expr, bool work>
    DEVICEHOST()
    static inline void ieval_(const SampleIndex& idx, Vec3R& acc,
        const Expr& pos, const mReal_t& epoch, const brie::NaifId& COI,
        const EnvT::Ref<work>& env)
    {

        /* If COI is barycenter, no tidal term */
        if (COI == 0) { /* TODO: Generalize to any system (e.g. planetary
                           without Sun's perturbation )*/
            /* Compute sum for all bodies */
            for (mSize_t i = 0; i < env.bodies().size(); i++) {
                acc += baryEval<Expr, work>(
                    idx, pos, epoch, env.bodies()[i], env);
            }
        } else {
            /* This case is not barycentric, */
            /* Compute sum for all bodies */
            for (mSize_t i = 0; i < env.bodies().size(); i++) {
                acc += nonBaryEval<Expr, work>(
                    idx, pos, epoch, env.bodies()[i], COI, env);
            }
        }
    }

    /** @brief One single evaluation - barycentric */
    template<typename Expr, bool work>
    DEVICEHOST()
    static inline Vec3R
        baryEval(const SampleIndex& i, const Expr& pos, const mReal_t& epoch,
            const brie::NaifId& body, const EnvT::Ref<work>& env)
    {
        Vec3R bodyPos = env.getPosition(i, epoch, body, 0);
        return thirdBody_B(pos, bodyPos, env.constants().body(body).gm());
    }

    /** @brief One single evaluation - non barycentric */
    template<typename Expr, bool work>
    DEVICEHOST()
    static inline Vec3R nonBaryEval(const SampleIndex& i, const Expr& pos,
        const mReal_t& epoch, const brie::NaifId& body, const brie::NaifId& COI,
        const EnvT::Ref<work>& env)
    {
        if (body != COI) {
            Vec3R bodyPos = env.getPosition(i, epoch, body, COI);
            return thirdBody_NB(pos, bodyPos, env.constants().body(body).gm());
        } else {
            return twoBody(pos, env.constants().body(body).gm());
        }
    }
};

/** @brief Evaluation kernel for a given body */
namespace kernel {

/** @brief Compile-time launch metadata for ::pointGravity.
 *
 *  Single source of truth: the kernel definition references the same
 *  constants in its ``__launch_bounds__``, and the dispatcher reads
 *  them at compile time via ``computeBlocks<PointGravityLaunch>(...)``.
 *  See ``paraHPOP/model/physics/reduced/LaunchTraits.h`` for the rationale. */
using PointGravityLaunch = KernelLaunchTraits<256, 4>;

__global__ void pointGravity(feta::vector::Array<Real, 3>::GRef bodyAcc,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef pos,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef bodyPos,
    GRID_CONSTANT() feta::scalar::Array<NaifId>::GRef::HandleT COI,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated,
    GRID_CONSTANT() NaifId bodyID, GRID_CONSTANT() Real gm);
} // namespace kernel

} // namespace accelerations
} // namespace model
} // namespace paraHPOP