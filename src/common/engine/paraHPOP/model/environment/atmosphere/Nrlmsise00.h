#pragma once

#include "paraHPOP/model/environment/atmosphere/Nrlmsise00Atmosphere.h"

namespace paraHPOP {
namespace model {
namespace environment {
namespace atmosphere {
namespace kernel {

using NrlPreprocessLaunch = KernelLaunchTraits<128, 4>;
/* NRLMSISE-00 carries substantial per-thread working state.  The lower
 * minimum block count leaves the compiler enough registers for the model. */
using NrlDensityLaunch = KernelLaunchTraits<128, 1>;

/** Convert ET/J2000 state to UTC, geodetic coordinates and weather inputs. */
__global__ void prepareNrlmsise00(
    NrlInputArrayT::GRef inputs,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef pos,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef ephcache,
    GRID_CONSTANT() feta::vector::Array<Real, 4>::GRef quatcache,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT epochs,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated,
    GRID_CONSTANT() NrlWeatherArrayT::GRef weather,
    GRID_CONSTANT() NrlSettings settings,
    GRID_CONSTANT() Real equatorialRadius,
    GRID_CONSTANT() Real flattening);

/** Evaluate total mass density; output units are paraHPOP kg/km^3. */
__global__ void evaluateNrlmsise00(
    GRID_CONSTANT() NrlInputArrayT::GRef inputs,
    feta::scalar::Array<Real>::GRef::HandleT density,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated,
    GRID_CONSTANT() Real densityScale);

} // namespace kernel

/** Evaluate NRLMSISE-00 total mass density on the CPU.
 *
 * @param fixedPosition Spacecraft position relative to Earth, expressed in
 *        the configured Earth-fixed frame [km].
 * @param epochEt       SPICE ET/TDB seconds past J2000.
 * @param atmosphere    Host-side weather/settings view.
 * @param equatorialRadius Earth equatorial radius [km].
 * @param flattening    Earth ellipsoid flattening.
 * @return Density in paraHPOP units [kg/km^3], zero above the configured
 *         cutoff, or NaN when inputs/weather coverage are invalid.
 */
Real evaluateNrlmsise00HostDensity(const Vec3R& fixedPosition,
    Real epochEt, const Nrlmsise00Atmosphere::GRef& atmosphere,
    Real equatorialRadius, Real flattening);

} // namespace atmosphere
} // namespace environment
} // namespace model
} // namespace paraHPOP
