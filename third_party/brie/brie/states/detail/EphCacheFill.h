#pragma once

#ifndef BRIE_CPU_ONLY

#include <cuda.h>

#include <feta/feta.h>

#include "brie/states/EphUnit.h"
#include "brie/typedefs.h"

namespace brie {
namespace states {
namespace kernel {

/** @brief Per-body native-frame cache fill kernel.
 *
 *  One thread per sample.  Reads `source.interpolators_[bodyIdx]`
 *  (the body's chebyshev `BodyInterpolator`), evaluates at
 *  `epochs[i]`, and writes the result into `slot[i]`.  The result is
 *  in the body's NATIVE frame (parent-relative as encoded in the SPK)
 *  — no walker, no COI subtraction.
 *
 *  For `Dim == 3` writes position; for `Dim == 6` writes position +
 *  velocity. */
template<bool UseTexture, idx_t Dim>
__global__ void fillBodyKernel(
    typename feta::vector::Array<Real, Dim>::GRef slot,
    GRID_CONSTANT() typename EphUnit<UseTexture>::GRef source,
    GRID_CONSTANT() idx_t bodyIdx,
    GRID_CONSTANT() typename feta::scalar::Array<Real>::GRef::HandleT epochs)
{
    feta::SampleIndex i
        = feta::SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);

    if (i.global() >= slot.size())
        return;

    if constexpr (Dim == 3) {
        Vec3R val;
        val.setZero();
        source.interpolators_[bodyIdx].getValues(val, epochs[i]);
        slot[i] = val;
    } else { /* Dim == 6 */
        Vec6R val;
        val.setZero();
        source.interpolators_[bodyIdx].getValuesAndDerivatives(
            val, epochs[i]);
        slot[i] = val;
    }
}

} // namespace kernel
} // namespace states
} // namespace brie

#endif // BRIE_CPU_ONLY
