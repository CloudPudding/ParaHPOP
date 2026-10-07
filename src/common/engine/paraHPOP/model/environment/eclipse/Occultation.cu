#include "paraHPOP/model/environment/eclipse/Occultation.h"

namespace paraHPOP {
namespace model {
namespace environment {
namespace eclipse {
namespace kernel {

/* Launch bounds live in Occultation.h as ``OccultationLaunch`` so the
   dispatcher can read them at compile time without any cudart query. */
__global__ __launch_bounds__(OccultationLaunch::maxBlockSize,
    OccultationLaunch::minBlocksPerSM) void occultationFactors(
    feta::scalar::Array<Real>::GRef::HandleT occ,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef pos,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef sunPos,
    GRID_CONSTANT() OcculterPack pack,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated,
    GRID_CONSTANT() Real RSun)
{
    SampleIndex i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);

    if (i.global() >= pos.size())
        return;

    if (terminated[i])
        return;

    /* Sun→sample direction is shared by every occulter factor; the
     * per-occulter direction is consumed as an expression so only the
     * three scalar reductions of cone() ever materialise. */
    const Vec3R rS = pos[i] - sunPos[i];

    Real f = Real{ 1 };
    for (idx_t k = 0; k < pack.n; k++) {
        f *= factor(rS, pos[i] - pack.eph[k][i], RSun, pack.r[k]);
    }

    occ[i] = f;
}

} // namespace kernel
} // namespace eclipse
} // namespace environment
} // namespace model
} // namespace paraHPOP
