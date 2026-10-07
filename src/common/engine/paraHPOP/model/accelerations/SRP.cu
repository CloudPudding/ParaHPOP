#include "paraHPOP/model/accelerations/SRP.h"

namespace paraHPOP {
namespace model {
namespace accelerations {
namespace kernel {

/* Launch bounds live in SRP.h as ``SRPLaunch`` so the dispatcher can
   read them at compile time without any cudart query. */
__global__ __launch_bounds__(SRPLaunch::maxBlockSize,
    SRPLaunch::minBlocksPerSM) void srp(
    feta::vector::Array<Real, 3>::GRef bodyAcc,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef pos,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef sunPos,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT mass,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT area,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT cr,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated,
    GRID_CONSTANT() Real AU)
{
    SampleIndex i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);

    if (i.global() >= bodyAcc.size())
        return;

    if (terminated[i])
        return;

    /* Inline the packed AMS = cr * area / mass at the call site (no cache).
     * See plan's Bandwidth analysis for the empirical justification. */
    const Real effectiveAMS = cr[i] * area[i] / mass[i];

    /* Compute acceleration */
    Vec3R dp = pos[i] - sunPos[i];
    dp       = SRP::eval(dp, effectiveAMS, AU);

    /* Non-atomic feta `+=` into this body's per-sample scratch slot
     * (chained intra-body via the per-step CUDA graph; reduce kernel
     * later sums across bodies in fixed order — see PointGravity.cu
     * for the rationale). */
    bodyAcc[i] += dp;
}

/* Body kept textually parallel to ::srp — the ONLY difference is the
   occultation scaling on the accumulate, so the two kernels stay
   bit-comparable (occ == 1 multiplies bitwise-identically). */
__global__ __launch_bounds__(SRPLaunch::maxBlockSize,
    SRPLaunch::minBlocksPerSM) void srpShadow(
    feta::vector::Array<Real, 3>::GRef bodyAcc,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef pos,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef sunPos,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT mass,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT area,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT cr,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated,
    GRID_CONSTANT() Real AU,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT occ)
{
    SampleIndex i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);

    if (i.global() >= bodyAcc.size())
        return;

    if (terminated[i])
        return;

    const Real effectiveAMS = cr[i] * area[i] / mass[i];

    /* Compute acceleration */
    Vec3R dp = pos[i] - sunPos[i];
    dp       = SRP::eval(dp, effectiveAMS, AU);

    /* Occultation-scaled accumulate into this body's scratch slot. */
    bodyAcc[i] += occ[i] * dp;
}
} // namespace kernel

} // namespace accelerations
} // namespace model
} // namespace paraHPOP