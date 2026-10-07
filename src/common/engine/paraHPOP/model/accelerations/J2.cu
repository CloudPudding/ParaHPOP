#include "paraHPOP/model/accelerations/J2.h"

namespace paraHPOP {
namespace model {
namespace accelerations {
namespace kernel {

/* Launch bounds live in J2.h as ``J2Launch`` so the dispatcher can read
   them at compile time without any cudart query. */
__global__ __launch_bounds__(J2Launch::maxBlockSize,
    J2Launch::minBlocksPerSM) void j2(
    feta::vector::Array<Real, 3>::GRef bodyAcc,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef pos,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef bodyPos,
    GRID_CONSTANT() feta::vector::Array<Real, 4>::GRef rotCache,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated,
    GRID_CONSTANT() Real gm, GRID_CONSTANT() Real j2,
    GRID_CONSTANT() Real radius, GRID_CONSTANT() Real soi)
{
    SampleIndex i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);

    if (i.global() >= bodyAcc.size())
        return;

    if (terminated[i])
        return;

    /* Compute acceleration */
    Vec3R scratch = (pos - bodyPos)[i];

    /* Do not compute if outside the SOI (will anyway be well below machine
     * epsilon) */
    if (scratch.norm() > soi)
        return;

    OrientationsT::RotationT rot = { rotCache[i] };
    scratch                = rot.rotate(scratch);
    scratch                = J2::basicEval(scratch, gm, j2, radius);
    scratch = rot.applyInverse(scratch);

    /* Non-atomic feta `+=` into this body's per-sample scratch slot.
     * See PointGravity.cu for the cross-kernel ordering rationale. */
    bodyAcc[i] += scratch;
}

} // namespace kernel
} // namespace accelerations
} // namespace model
} // namespace paraHPOP