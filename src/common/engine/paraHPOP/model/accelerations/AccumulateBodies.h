#pragma once

#include "paraHPOP/typedefs.h"

namespace paraHPOP {
namespace model {
namespace accelerations {
namespace kernel {

using VecGRef = feta::vector::Array<Real, 3>::GRef;

/** Host policy resolved for each graph's batch size before capture.
 * Explicit diagnostic overrides are parsed once per process. No device
 * allocation or policy lookup occurs during graph replay. */
// Scheduling is fixed: serial forces within each parallel body branch.
struct ForceKernelOptions {
    idx_t shThreads = 0;
    idx_t nrlThreads = 0;
    bool shRadialCache = false;
    bool shCorrectedDivision = false;
};
ForceKernelOptions forceKernelOptions(idx_t sampleCount);

/** @brief One body's independently produced acceleration contributions.
 *  The owning Environment uploads these device-pointer views once per
 *  force layout; their order is the deterministic body summation order. */
struct BodyAccTerm {
    VecGRef base{};
    VecGRef shape{};
    bool hasShape = false;

    /* FETA's owning scalar array compares its initializer with the default
     * value. Compare view metadata, never the pointed-to device contents. */
    DEVICEHOST() static bool sameView(const VecGRef& a, const VecGRef& b)
    {
        return a.data_ == b.data_ && a.nVecs_ == b.nVecs_
            && a.dimOffset_ == b.dimOffset_ && a.tex_ == b.tex_
            && a.texOffset_ == b.texOffset_;
    }
    DEVICEHOST() bool operator==(const BodyAccTerm& other) const
    {
        return hasShape == other.hasShape && sameView(base, other.base)
            && sameView(shape, other.shape);
    }
    DEVICEHOST() bool operator!=(const BodyAccTerm& other) const
    { return !(*this == other); }
};
using BodyAccTermsArrayT = feta::scalar::Array<BodyAccTerm>;

/** @brief Compile-time launch metadata for the body-scratch zero and
 *  per-body reduce kernels.  Both are pure-bandwidth kernels with no
 *  shared memory, no atomics, and no chebyshev-style heavy compute,
 *  so we keep the same per-block geometry as the existing per-body
 *  acceleration kernels (matches `EnvKernelLaunch`). */
using AccumulateBodiesLaunch = KernelLaunchTraits<256, 4>;

/** @brief Zero one body's per-sample acceleration scratch slot.
 *  Issued once per body at the entry of every RHS-graph build so the
 *  per-body acceleration kernels can accumulate via non-atomic feta
 *  ``+=`` into a known-zero slot.  Skips terminated samples for
 *  consistency with the per-body kernels. */
__global__ void zeroBodyAccScratch(
    feta::vector::Array<Real, 3>::GRef bodyAcc,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated);

/** @brief Add one body's per-sample partial acceleration into the
 *  per-sample total ``acc`` (``acc[i] += bodyAcc[i]`` via feta
 *  expression-template ``+=``).  Replaces the legacy ``atomicSum``
 *  accumulation path: per-body acceleration kernels write
 *  non-atomically into ``bodyAcc`` (no cross-kernel race because
 *  bodies' scratch slots don't alias), and one of these reduce
 *  kernels is issued per body in fixed body order, chained serially
 *  in the per-step CUDA graph.
 *
 *  Each thread does one feta ``+=`` on its own sample's slot — single
 *  writer per address per launch, deterministic FP order across
 *  replays, identical to the host RHS pattern in
 *  ``PrivateAccelerations.h::eval_``. */
__global__ void reduceBodyAccScratch(
    feta::vector::Array<Real, 3>::GRef acc,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef bodyAcc,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated);

/** @brief Reduce all independently produced force scratch in one pass.
 *  Each sample starts with acc[i], then adds (base + optional shape) in
 *  term order, followed by the optional Earth correction.  No scratch
 *  read occurs for terminated samples; zero terms supports corrections
 *  without any active per-body force. */
__global__ void reduceForceAccScratch(VecGRef acc,
    GRID_CONSTANT() BodyAccTermsArrayT::GRef::HandleT terms,
    GRID_CONSTANT() idx_t nTerms, GRID_CONSTANT() VecGRef corrections,
    GRID_CONSTANT() bool hasCorrections,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated);

} // namespace kernel
} // namespace accelerations
} // namespace model
} // namespace paraHPOP
