#include "paraHPOP/model/accelerations/PointGravity.h"

namespace paraHPOP {
namespace model {
namespace accelerations {
namespace kernel {

/* Launch bounds live in PointGravity.h as ``PointGravityLaunch`` so the
   dispatcher can read them at compile time without any cudart query.
   The minBlocksPerSM=4 setting was relaxed from 8 to raise the register
   budget from 32 -> 64 regs/thread, eliminating local-memory spills
   observed in SASS (LDL.LU.64 reloads). */
__global__ __launch_bounds__(PointGravityLaunch::maxBlockSize,
    PointGravityLaunch::minBlocksPerSM) void pointGravity(
    feta::vector::Array<Real, 3>::GRef bodyAcc,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef pos,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef bodyPos,
    GRID_CONSTANT() feta::scalar::Array<NaifId>::GRef::HandleT COI,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated,
    GRID_CONSTANT() NaifId bodyID, GRID_CONSTANT() Real gm)
{
    SampleIndex i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);

    if (i.global() >= bodyAcc.size())
        return;

    if (terminated[i])
        return;

    /* Load both positions into registers */
    Vec3R bp                = bodyPos[i];
    Vec3R dp                = pos[i] - bp;
    NaifId center           = COI[i];
    bool onBodyOrBarycenter = (center == bodyID) || (center == 0);

    /* Accumulate result back into dp: once eval() is entered dp is consumed
       as input, so the compiler can reuse those 6 double registers for the
       output, reducing peak register pressure before the write. */
    dp = PointGravity::eval(dp, bp, gm, onBodyOrBarycenter);

    /* Non-atomic feta `+=` into this body's per-sample scratch slot.
     * Within a body, gravity / SRP / SH-or-J2 are graph-edge-serialised
     * against the same `bodyAccScratch[bodyID]` slot, so the writes
     * are race-free without atomics.  Across bodies, each body's
     * scratch is a distinct allocation, so kernels in parallel
     * branches cannot alias.  A single deterministic reduce kernel
     * sums across bodies in fixed body order (see
     * `AccumulateBodies.h::reduceBodyAccScratch`). */
    bodyAcc[i] += dp;
}
} // namespace kernel

} // namespace accelerations
} // namespace model
} // namespace paraHPOP