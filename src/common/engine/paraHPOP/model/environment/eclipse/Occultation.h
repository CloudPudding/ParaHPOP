#pragma once

#include "paraHPOP/model/environment/Environment.h"
#include "paraHPOP/model/environment/eclipse/Shadow.h"

namespace paraHPOP {
namespace model {
namespace environment {
namespace eclipse {

/**
 * @file Occultation.h
 * @brief Combined per-sample occultation factor precompute.
 *
 * SRP occultation is scene-level N→1: every body flagged ``occulting``
 * attenuates the single Sun radiation source, and the per-occulter
 * dual-cone factors (see Shadow.h) combine as a product — exact for
 * non-overlapping occultations.  One kernel launch writes ONE scalar
 * per sample into the environment's combined occultation cache, which
 * the ``srpShadow`` kernel then consumes.  This is independent of the
 * eclipse *event* (1:1:1 Sun → one named body → spacecraft), which
 * shares only the Shadow.h geometry core.
 */

/** @brief Hard cap on the number of simultaneous occulters baked into
 *  the by-value kernel pack.  Asserted at graph-build time. */
constexpr idx_t MAXOCCULTERS = 8;

/** @brief By-value pack of per-occulter resolved-cache handles and
 *  physical radii.  Slots ``[0, n)`` are valid; the cache GRefs point
 *  at the per-body COI-relative resolved ephemeris slots (pinned tier
 *  at stage 0, variable tier at stages > 0) and the radii come from
 *  the constants set (validated > 0 at model build). */
struct OcculterPack {
    feta::vector::Array<Real, 3>::GRef eph[MAXOCCULTERS];
    Real r[MAXOCCULTERS];
    idx_t n = 0;
};

namespace kernel {

/** @brief Compile-time launch metadata for ::occultationFactors.  See
 *  ``LaunchTraits.h``. */
using OccultationLaunch = KernelLaunchTraits<256, 8>;

/** @brief Write the combined occultation factor for every sample.
 *
 *  Per sample ``i``: ``occ[i] = Π_k factor(pos[i] − sunPos[i],
 *  pos[i] − occPos_k[i], RSun, r_k)`` over the packed occulters.
 *  Fully sunlit samples get exactly 1.0 (each plain-Real factor is
 *  exactly 1 outside the penumbra), so the downstream ``srpShadow``
 *  multiply is a bitwise no-op in full sunlight.
 *
 * @param[out] occ        Combined per-sample factor (0 … 1)
 * @param[in]  pos        Per-sample spacecraft position (COI-relative)
 * @param[in]  sunPos     Sun resolved position cache (COI-relative)
 * @param[in]  pack       Occulter resolved-cache handles + radii
 * @param[in]  terminated Per-sample termination flags (skip if set)
 * @param[in]  RSun       Physical radius of the Sun
 */
__global__ void occultationFactors(
    feta::scalar::Array<Real>::GRef::HandleT occ,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef pos,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef sunPos,
    GRID_CONSTANT() OcculterPack pack,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated,
    GRID_CONSTANT() Real RSun);

} // namespace kernel
} // namespace eclipse
} // namespace environment
} // namespace model
} // namespace paraHPOP
