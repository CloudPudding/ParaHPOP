#pragma once

#include "paraHPOP/model/environment/Environment.h"

#include "paraHPOP/typedefs.h"

namespace paraHPOP {
namespace model {
namespace accelerations {

using EnvT = ::paraHPOP::model::environment::Env;

/** @brief Collection of SRP acceleration functions */
struct SRP {
    /** @brief Mean solar radiation pressure at 1 AU (kN / km^2).
     *  (Taken from Godot on 28/04/2025.) */
    static constexpr mReal_t AUmeanSRP = 4.56e-3;

    /** @brief Compute the SRP acceleration */
    template<typename Expr>
    DEVICEHOST()
    static Vec3R
        eval(const SampleIndex& i, const Expr& pos, const mReal_t& epoch,
            const mReal_t& AMS, const brie::NaifId& COI, const EnvT::GRef& env)
    {
        return eval_<Expr, false>(i, pos, epoch, AMS, COI, env);
    }
    template<typename Expr>
    DEVICEHOST()
    static Vec3R
        eval(const SampleIndex& i, const Expr& pos, const mReal_t& epoch,
            const mReal_t& AMS, const brie::NaifId& COI, const EnvT::WRef& env)
    {
        return eval_<Expr, true>(i, pos, epoch, AMS, COI, env);
    }

    /** @brief Evaluation with the correct position already available */
    template<typename Expr, bool work>
    DEVICEHOST()
    static inline decltype(auto)
        eval(const Expr& pos, const mReal_t& AMS, const EnvT::Ref<work>& env)
    {
        mReal_t factor = env.constants().au();
        factor *= factor * AUmeanSRP * AMS;

        return (factor * pos.rCubedNorm()) * pos;
    }

    /** @brief Evaluation with the correct position already available */
    template<typename Expr, bool work>
    DEVICEHOST()
    static inline decltype(auto) eval(const SampleIndex& i, const Expr& pos,
        const mReal_t& AMS, const EnvT::Ref<work>& env)
    {
        mReal_t factor = env.constants().au();
        factor *= factor * AUmeanSRP * AMS;

        return (factor * pos.rCubedNorm(i)) * pos[i];
    }

    /** @brief Lowest level evaluation with available AU constant */
    DEVICEHOST()
    static FORCEINLINE() Vec3R
        eval(const Vec3R& pos, const mReal_t& AMS, const mReal_t& AU)
    {
        mReal_t factor = AU * AU * AUmeanSRP * AMS;

        return (factor * pos.rCubedNorm()) * pos;
    }

protected:
    /** @brief Internal evaluation */
    template<typename Expr, bool work>
    DEVICEHOST()
    static inline Vec3R eval_(const SampleIndex i, const Expr& pos,
        const mReal_t epoch, const mReal_t AMS, const brie::NaifId COI,
        const EnvT::Ref<work> env)
    {
        Vec3R sunPos = Vec3R::Zeros();
        if (COI != 10) {
            sunPos = env.getPosition(i, epoch, 10, COI);
        }
        auto deltapos = pos - sunPos;
        return eval<decltype(deltapos), work>(deltapos, AMS, env);
    }
};

/** @brief Evaluation kernel for a given body */
namespace kernel {

/** @brief Compile-time launch metadata for ::srp.  See ``LaunchTraits.h``. */
using SRPLaunch = KernelLaunchTraits<256, 8>;

__global__ void srp(feta::vector::Array<Real, 3>::GRef bodyAcc,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef pos,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef bodyPos,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT mass,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT area,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT cr,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated,
    GRID_CONSTANT() Real AU);

/** @brief ::srp variant scaled by the combined per-sample occultation
 *  factor (written by ``occultationFactors`` — see
 *  ``environment/eclipse/Occultation.h``).  Launched in place of ::srp
 *  only when at least one occulting body shadows an active radiation
 *  source; ``occ[i] == 1`` (full sunlight) reproduces ::srp bitwise. */
__global__ void srpShadow(feta::vector::Array<Real, 3>::GRef bodyAcc,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef pos,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef bodyPos,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT mass,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT area,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT cr,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated,
    GRID_CONSTANT() Real AU,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT occ);
} // namespace kernel

} // namespace accelerations
} // namespace model
} // namespace paraHPOP