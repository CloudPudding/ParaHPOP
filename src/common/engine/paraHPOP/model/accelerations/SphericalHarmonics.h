#pragma once

#include "paraHPOP/model/accelerations/sphericalharmonics/Evaluation.h"
#include "paraHPOP/model/environment/Environment.h"

#include <cstddef>

namespace paraHPOP {
namespace model {
namespace accelerations {

using EnvT    = ::paraHPOP::model::environment::Env;
using OrientationsT = ::paraHPOP::model::environment::Orientations<false>;

/** @brief Collection of spherical harmonics acceleration functions */
struct SphericalHarmonics {
    /** @brief SH coefficient ref types */
    using SHCoeffsT = ::paraHPOP::model::environment::Env::SHCoeffsT;
    template<bool work>
    using SHCoeffsRefT = SHCoeffsT::Ref<work>;

    /** @brief Evaluate SH acceleration given a body-relative position and
     *  per-body coefficients. Rotates into body-fixed frame, evaluates,
     *  and rotates back.  ``rotTarget`` is the orientation frame the body
     *  rotates under — the caller passes the resolved per-body target (its
     *  own id under the IAU default, or a binary-frame target like ITRF93's
     *  3000 under auto-best). */
    template<bool work>
    DEVICEHOST()
    static Vec3R eval(Vec3R scratch, const mReal_t& epoch,
        const brie::NaifId& rotTarget, const SHCoeffsRefT<work>& shCoeffs,
        const typename EnvT::Ref<work>::OrientationsT& orientations)
    {
        using EvalT = sphericalharmonics::Evaluation<work>;
        scratch
            = orientations.bodyAxesRotation(epoch, rotTarget).rotate(scratch);
        scratch = EvalT::eval(scratch, shCoeffs);
        scratch = orientations.bodyAxesRotation(epoch, rotTarget)
                      .applyInverse(scratch);
        return scratch;
    }
};

/** @brief evaluation kernels */
namespace kernel {

/** @brief Launch bounds and dynamic scratch size for ::sphericalHarmonics. */
struct SphericalHarmonicsLaunch : KernelLaunchTraits<256, 4> {
    /** @brief Per-thread accumulator (3), phasor (4), and quaternion (4). */
    static constexpr std::size_t sharedBytes(idx_t threads)
    {
        return sizeof(Real) * 11 * threads;
    }
};

/** @brief Evaluate spherical harmonics acceleration using cached rotations.
 *
 * Reads pre-computed rotation quaternions from @p rotCache instead of
 * calling into the brie rotation chain, eliminating all call-frame
 * stack usage and shared-memory rotation caching overhead.
 *
 * Launch with SphericalHarmonicsLaunch::sharedBytes(blockSize) dynamic
 * shared-memory bytes. The scratch layout uses the actual blockDim.x.
 *
 * @param[in,out] acc       Accumulation array for accelerations (ICRF)
 * @param[in]     pos       Body-relative positions (ICRF)
 * @param[in]     rotCache  Pre-computed rotation quaternions
 * @param[in]     shCoeffs  Spherical harmonics coefficients
 */
__global__ void sphericalHarmonics(feta::vector::Array<Real, 3>::GRef bodyAcc,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef pos,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef bodyPos,
    GRID_CONSTANT() feta::vector::Array<Real, 4>::GRef rotCache,
    GRID_CONSTANT() EnvT::SHCoeffsT::GRef shCoeffs,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated,
    GRID_CONSTANT() Real soi,
    GRID_CONSTANT() bool useRadialCache = false,
    GRID_CONSTANT() bool useCorrectedDivision = false);

} // namespace kernel

} // namespace accelerations
} // namespace model
} // namespace paraHPOP
