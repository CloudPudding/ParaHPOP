#include "paraHPOP/model/environment/Environment.h"

namespace paraHPOP {
namespace model {
namespace environment {
namespace kernel {

/* coiResolveKernel — resolve native-frame cache to COI-relative slot.
 *
 * __launch_bounds__ pins the max block size; shared buffer is sized
 * for that max and stride uses blockDim.x so any launched block size
 * in [32, EnvKernelLaunch::maxBlockSize] is correct. Bound and shmem
 * extent both reference the same constexpr ``EnvKernelLaunch`` cap.
 *
 * No chebyshev — reads pre-cached native-frame slot values populated
 * by the upstream per-body fill kernel. */
__global__ void __launch_bounds__(EnvKernelLaunch::maxBlockSize,
    EnvKernelLaunch::minBlocksPerSM)
    coiResolveKernel(feta::vector::Array<Real, 3>::GRef ephCache,
        GRID_CONSTANT() EphNativeCacheT::GRef cache,
        GRID_CONSTANT() NaifId target,
        GRID_CONSTANT() feta::scalar::Array<NaifId>::GRef::HandleT centers,
        GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT epochs,
        GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated)
{
    SampleIndex i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);

    if (i.global() >= ephCache.size())
        return;

    /* Prepare shared memory cache — sized for the max block size. */
    __shared__ Real shmem[EnvKernelLaunch::maxBlockSize * 3];
    __shared__ feta::vector::Array<Real, 3>::WRef::HandleT thisCache;
    if (i.work() == 0) {
        thisCache.ptr     = shmem;
        thisCache.dOffset = blockDim.x;
    }
    __syncthreads();

    if (terminated[i])
        return;

    /* Initialize the shared memory buffer to zero */
    thisCache[i] = feta::vector::Item<Real, 3>::Zeros();

    /* Walker reads native-frame slot values from `cache` (no chebyshev). */
    Real epoch    = epochs[i];
    NaifId center = centers[i];

    cache.template iGetPosition<true>(i, thisCache, epoch, target, center);

    /* place back into global memory */
    ephCache[i] = thisCache;
}

/* coiResolveVelocityKernel — resolve native-frame Dim=6 cache to a
 * COI-relative velocity slot.  Same launch envelope + shared-memory
 * layout as coiResolveKernel; only the walker call differs
 * (iGetVelocity instead of iGetPosition). */
__global__ void __launch_bounds__(EnvKernelLaunch::maxBlockSize,
    EnvKernelLaunch::minBlocksPerSM)
    coiResolveVelocityKernel(feta::vector::Array<Real, 3>::GRef velCache,
        GRID_CONSTANT() EphNativeCacheT::GRef cache,
        GRID_CONSTANT() NaifId target,
        GRID_CONSTANT() feta::scalar::Array<NaifId>::GRef::HandleT centers,
        GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT epochs,
        GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated)
{
    SampleIndex i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);

    if (i.global() >= velCache.size())
        return;

    __shared__ Real shmem[EnvKernelLaunch::maxBlockSize * 3];
    __shared__ feta::vector::Array<Real, 3>::WRef::HandleT thisCache;
    if (i.work() == 0) {
        thisCache.ptr     = shmem;
        thisCache.dOffset = blockDim.x;
    }
    __syncthreads();

    if (terminated[i])
        return;

    thisCache[i] = feta::vector::Item<Real, 3>::Zeros();

    Real epoch    = epochs[i];
    NaifId center = centers[i];

    cache.template iGetVelocity<true>(i, thisCache, epoch, target, center);

    velCache[i] = thisCache;
}

/* copyNativeCacheKernel — per-sample conditional native slot copy.
 * Skips on terminated[i]. Explicitly instantiated for
 * Dim == 6. */
template<idx_t Dim>
__global__ void __launch_bounds__(EnvKernelLaunch::maxBlockSize,
    EnvKernelLaunch::minBlocksPerSM)
    copyNativeCacheKernel(
        typename feta::vector::Array<Real, Dim>::GRef pinnedSlot,
        GRID_CONSTANT() typename feta::vector::Array<Real, Dim>::GRef
            variableSlot,
        GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated)
{
    SampleIndex i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);

    if (i.global() >= pinnedSlot.size())
        return;
    if (terminated[i])
        return;

    pinnedSlot[i] = variableSlot[i];
}

/* Explicit instantiation: Dim == 6 matches EphNativeCacheT.
 * GRID_CONSTANT() annotations MUST match the template declaration. */
template __global__ void copyNativeCacheKernel<6>(
    typename feta::vector::Array<Real, 6>::GRef pinnedSlot,
    GRID_CONSTANT() typename feta::vector::Array<Real, 6>::GRef variableSlot,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated);

/* omegaResolveKernel — pre-compute the inertial-frame body angular
 * velocity ω(epoch_i, target) for each sample.  Uses the
 * ``bodyAngularVelocity<true>`` brie shmem-scratch cascade:
 * RA/Dec/PM angles + rates are pre-computed into two per-thread
 * Dim=3 shmem buffers before the Euler 3-1-3 kinematics formula
 * runs, so the rotation polynomial walker's live-value set lands
 * in shared memory rather than registers — same pattern that lets
 * ``cacheRotation`` hit STACK ≤ 40 B at register cap 64. */
__global__ void __launch_bounds__(EnvKernelLaunch::maxBlockSize,
    EnvKernelLaunch::minBlocksPerSM)
    omegaResolveKernel(feta::vector::Array<Real, 3>::GRef omegaCache,
        GRID_CONSTANT() OrientationsT orientations,
        GRID_CONSTANT() NaifId target,
        GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT epochs,
        GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated)
{
    SampleIndex i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (i.global() >= omegaCache.size())
        return;

    /* Two Dim=3 shmem scratches — one for RA/Dec/PM angles, one for
     * their rates.  Total static shmem: 2 × 128 × 3 × 8 = 6 KiB / block
     * at EnvKernelLaunch (128/8 → 12 KiB/block budget on sm_61). */
    using VecRef = feta::vector::Array<Real, 3>::WRef::HandleT;
    __shared__ Real shmemA[EnvKernelLaunch::maxBlockSize * 3];
    __shared__ Real shmemR[EnvKernelLaunch::maxBlockSize * 3];
    __shared__ VecRef angleScratch;
    __shared__ VecRef rateScratch;
    if (i.work() == 0) {
        angleScratch.ptr     = shmemA;
        angleScratch.dOffset = blockDim.x;
        rateScratch.ptr      = shmemR;
        rateScratch.dOffset  = blockDim.x;
    }
    __syncthreads();

    if (terminated[i])
        return;

    /* Initialize the shared memory buffers to zero (matches the
     * cacheRotation pattern so the first read in any dispatcher sees
     * a defined value). */
    angleScratch[i] = feta::vector::Item<Real, 3>::Zeros();
    rateScratch[i]  = feta::vector::Item<Real, 3>::Zeros();

    /* Epochs are already SPICE time post-Propagator::prepare(); no
     * mjdToSpice conversion needed (same convention every other kernel
     * uses). */
    orientations.template bodyAngularVelocity<true>(
        i, omegaCache, angleScratch, rateScratch, epochs[i], target);
}

/* coiResolveStateKernel — fused position + velocity resolve.  Reads the
 * Dim=6 native cache once (one COI chain walk via
 * `iGetPositionAndVelocity`) and writes both the Dim=3 position and
 * Dim=3 velocity outputs.  Used for atmosphere-active bodies where both
 * outputs are needed; non-atmosphere bodies stay on the pos-only
 * `coiResolveKernel`. */
__global__ void __launch_bounds__(EnvKernelLaunch::maxBlockSize,
    EnvKernelLaunch::minBlocksPerSM)
    coiResolveStateKernel(feta::vector::Array<Real, 3>::GRef ephCache,
        feta::vector::Array<Real, 3>::GRef velCache,
        GRID_CONSTANT() EphNativeCacheT::GRef cache,
        GRID_CONSTANT() NaifId target,
        GRID_CONSTANT() feta::scalar::Array<NaifId>::GRef::HandleT centers,
        GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT epochs,
        GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated)
{
    SampleIndex i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (i.global() >= ephCache.size())
        return;

    /* Dim=6 shmem scratch — 2× the pos-only kernel.  At maxBlockSize=256
     * this is 256*6*8 = 12 KiB / block, well within the per-SM
     * shared-memory budget on all four target arches. */
    __shared__ Real shmem[EnvKernelLaunch::maxBlockSize * 6];
    __shared__ feta::vector::Array<Real, 6>::WRef::HandleT thisCache;
    if (i.work() == 0) {
        thisCache.ptr     = shmem;
        thisCache.dOffset = blockDim.x;
    }
    __syncthreads();

    if (terminated[i])
        return;

    thisCache[i]  = feta::vector::Item<Real, 6>::Zeros();
    Real epoch    = epochs[i];
    NaifId center = centers[i];

    cache.template iGetPositionAndVelocity<true>(
        i, thisCache, epoch, target, center);

    /* Split the Dim=6 result into the two parallel Dim=3 outputs.
     * Storage stays as two separate containers — only the fill kernel is
     * fused, not the storage. */
    ephCache[i] = thisCache[i].template segment<0, 3>();
    velCache[i] = thisCache[i].template segment<3, 3>();
}

/* cacheRotation — inlined rotation pre-computation kernel. */
__global__ void __launch_bounds__(EnvKernelLaunch::maxBlockSize,
    EnvKernelLaunch::minBlocksPerSM)
    cacheRotation(feta::vector::Array<Real, 4>::GRef rotCache,
        GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT epochs,
        GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated,
        GRID_CONSTANT() OrientationsT orientations,
        GRID_CONSTANT() NaifId bodyId)
{
    SampleIndex i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);

    if (i.global() >= rotCache.size())
        return;

    const Real epoch = epochs[i];

    /* Buffer for the rotation call chain — sized for the max block size. */
    using VecRef = feta::vector::Array<Real, 3>::WRef::HandleT;
    __shared__ Real shmem[EnvKernelLaunch::maxBlockSize * 3];
    __shared__ VecRef buffer;
    if (i.work() == 0) {
        buffer.ptr     = shmem;
        buffer.dOffset = blockDim.x;
    }
    __syncthreads();

    if (terminated[i])
        return;

    /* Initialize the shared memory buffer to zero */
    buffer[i] = feta::vector::Item<Real, 3>::Zeros();

    /* store the quaternion rotation components */
    orientations.template extractBodyAxesQuaternion<true>(
        i, rotCache, buffer, epoch, bodyId);
}


} // namespace kernel
} // namespace environment
} // namespace model
} // namespace paraHPOP
