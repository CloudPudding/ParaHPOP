/** @file SphericalHarmonics.cu
 * @brief Spherical harmonics kernels: rotation caching + SH evaluation.
 *
 * The rotation cache kernel (`cacheRotation`) pre-computes body-axes
 * rotation quaternions to a global-memory array so that the SH kernel
 * (`sphericalHarmonics`) can read them cheaply instead of recomputing
 * the full IAU rotation call chain.
 *
 */

#include "paraHPOP/model/accelerations/SphericalHarmonics.h"

#include "brie/frames/RefRotation.h"

namespace paraHPOP {
namespace model {
namespace accelerations {
namespace kernel {

/* SH block-sizing: matches the 256-thread regime of pointGravity/srp/j2.
 * Launch bounds live in SphericalHarmonics.h as
 * ``SphericalHarmonicsLaunch`` so the dispatcher can read them at
 * compile time without any cudart query. */

/* ================================================================== */
/*  sphericalHarmonics — cache-aware SH evaluation kernel             */
/* ================================================================== */

__global__ __launch_bounds__(SphericalHarmonicsLaunch::maxBlockSize,
    SphericalHarmonicsLaunch::minBlocksPerSM) void sphericalHarmonics(
    feta::vector::Array<Real, 3>::GRef bodyAcc,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef pos,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef bodyPos,
    GRID_CONSTANT() feta::vector::Array<Real, 4>::GRef rotCache,
    GRID_CONSTANT() EnvT::SHCoeffsT::GRef shCoeffs,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated,
    GRID_CONSTANT() Real soi,
    GRID_CONSTANT() bool useRadialCache,
    GRID_CONSTANT() bool useCorrectedDivision)
{
    SampleIndex i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);

#if __CUDA_ARCH__ < 700
    /** @brief Copy coefficient ref to shared (block-wide, once) for sm_61
     *  where GRID_CONSTANT is unsupported. */
    __shared__ EnvT::SHCoeffsT::GRef sharedSHCoeffs;
    if (i.work() == 0) {
        sharedSHCoeffs = shCoeffs;
    }
    __syncthreads();
#endif

    /* Three SoA regions sized to the launched block, in Real elements:
     * accumulator [0, 3B), phasor [3B, 7B), quaternion [7B, 11B).
     * The Real array and its element offsets preserve Real alignment.
     * Static shared handles remain separate from this dynamic scratch. */
    extern __shared__ Real s_scratch[];

    /* Per-thread shmem-backed Dim=3 accumulator for the SH evaluation.
     * Threading the running ``acc`` through shmem instead of a register
     * Vec3R frees 24 B of register pressure across the Legendre m-loop,
     * which competes with the 4-double recurrence state at the 64-reg
     * cap on sm_61.  Same shmem-dependency-chain trick that broke the
     * parm::adaptive::advance over-parallelisation; see
     * [[parm-advance-shmem-done]] for the canonical pattern.
     * Cost: blockDim.x × 3 × sizeof(Real) bytes/block. */
    __shared__ feta::vector::Array<Real, 3>::WRef::HandleT accHandle;

    /* Parallel per-thread shmem-backed Dim=4 slot for the PseudoPhasor
     * recurrence (sinL, cosL, sinML, cosML).  Moves the 32 B of
     * persistent phasor state out of the register file across the SH
     * order loop.  Cost: blockDim.x × 4 × sizeof(Real) bytes/block. See
     * @ref sphericalharmonics::RefPseudoPhasor for the layout. */
    __shared__ feta::vector::Array<Real, 4>::WRef::HandleT phasorHandle;

    /* Third per-thread shmem-backed Dim=4 slot for the rotation
     * quaternion.  The kernel rotates twice (forward + inverse), and
     * the residual 24 B sm_61 stack at B-v2 was the compiler caching
     * the per-component LEA addresses of rotCache[i] across the SH
     * eval to skip recomputation on the second access.  Pulling the
     * quaternion into shmem once and reading from it via a
     * @ref brie::frames::RefRotation view eliminates the pointer-spill:
     * the second rotation reads from shmem (LDS) not global+cache
     * (LDG), and the rotCache base pointer no longer needs to stay
     * live in the register file.
     * Cost: blockDim.x × 4 × sizeof(Real) bytes/block. */
    __shared__ feta::vector::Array<Real, 4>::WRef::HandleT rotCacheHandle;

    if (i.work() == 0) {
        accHandle.ptr          = s_scratch;
        accHandle.dOffset      = blockDim.x;
        phasorHandle.ptr       = s_scratch + 3 * blockDim.x;
        phasorHandle.dOffset   = blockDim.x;
        rotCacheHandle.ptr     = s_scratch + 7 * blockDim.x;
        rotCacheHandle.dOffset = blockDim.x;
    }
    __syncthreads();

    /* Tail threads also participate in shared initialization barriers. */
    if (i.global() >= bodyAcc.size())
        return;

    if (terminated[i])
        return;

    Vec3R scratch = (pos - bodyPos)[i];

    /* Do not compute if outside the SOI (will anyway be well below machine
     * epsilon) */
    if (scratch.norm() > soi)
        return;

    /* Pull rotCache[i] from global into the per-thread shmem slot via
     * feta expression-template assignment (View::operator=); unfolds
     * to four per-component stores with no Vec4R intermediate. */
    rotCacheHandle[i] = rotCache[i];

    /* Construct the shmem-backed rotation view once and reuse for
     * both forward and inverse rotations.  The view itself is two
     * references; the 32 B quaternion is materialised into a local
     * Vec4R inside rotate()/applyInverse() and dropped at return. */
    using RefRotT = brie::frames::RefRotation<false, decltype(rotCacheHandle)>;
    RefRotT rot{ rotCacheHandle, i };

    scratch = rot.rotate(scratch);

    /* Evaluate spherical harmonics acceleration in body-fixed frame.
     * The shmem-backed eval writes the result to accHandle[i]. */
    using EvalT = sphericalharmonics::Evaluation<false>;

#if __CUDA_ARCH__ < 700
    EvalT::evalDevice(i, scratch, sharedSHCoeffs, accHandle, phasorHandle,
        -1, -1, useRadialCache, useCorrectedDivision);
#else
    EvalT::evalDevice(i, scratch, shCoeffs, accHandle, phasorHandle,
        -1, -1, useRadialCache, useCorrectedDivision);
#endif

    scratch = rot.applyInverse(accHandle[i]);

    /* Non-atomic feta `+=` into this body's per-sample scratch slot.
     * See PointGravity.cu for the cross-kernel ordering rationale. */
    bodyAcc[i] += scratch;
}

} // namespace kernel

} // namespace accelerations
} // namespace model
} // namespace paraHPOP
