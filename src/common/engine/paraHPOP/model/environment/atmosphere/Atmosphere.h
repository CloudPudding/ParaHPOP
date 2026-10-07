#pragma once

#include "paraHPOP/typedefs.h"

namespace paraHPOP {
namespace model {
namespace environment {
namespace atmosphere {

/**
 * @brief Concept header for atmosphere density models.
 *
 * An atmosphere density policy used by the drag kernel exposes the band
 * contract — ``DEVICEHOST()`` ``idx_t selectBand(const Real& altKm)``,
 * ``Real cutoffOf(idx_t band)`` (per-band early-exit altitude), and
 * ``Real densityInBand(const Real& altKm, idx_t band)`` — plus a
 * convenience ``Real density(const Real& altKm)``.  The single-block
 * @ref ExpBlock and the layered @ref PiecewiseExponentialAtmosphere::GRef
 * both satisfy it, so the two drag kernels share one templated body.
 *
 * Density is returned in ``kg/km^3`` (i.e. SI density × 1e9) to stay
 * self-consistent with paraHPOP's km / km·s⁻² unit system, so the
 * drag kernel never has to apply a unit-conversion factor inline.
 *
 * Concrete models live alongside this header (e.g.
 * ``PiecewiseExponentialAtmosphere.h``).  New atmosphere models should be
 * added as sibling headers and re-exported through the umbrella
 * ``environment/atmosphere.h``.
 */
struct Atmosphere {
    /** @brief Cutoff altitude in km — above this altitude the drag
     *  kernel skips the sample (and the density model is allowed to
     *  return 0 or an undefined value). */
    Real hCutoff = Real{ 0 };
};

} // namespace atmosphere
} // namespace environment
} // namespace model
} // namespace paraHPOP
