#pragma once

#include "paraHPOP/model/environment/Environment.h"
#include "paraHPOP/model/environment/atmosphere.h"
#include "paraHPOP/model/environment/atmosphere/Geodetic.h"
#include "paraHPOP/typedefs.h"

namespace paraHPOP {
namespace model {
namespace accelerations {

/* Match J2.h's idiom — the drag kernel constructs a RotationT from
 * the cached quaternion to extract the body polar axis (geodetic
 * altitude) on bodies with flattening != 0.  Re-declaring the same
 * alias as J2.h is fine: identical typedefs in the same namespace
 * are allowed by C++. */
using OrientationsT = environment::Orientations<false>;

/** @brief Atmospheric drag acceleration functions.
 *
 *  Math (all quantities in INERTIAL — J2000 — frame; km / km·s⁻¹ /
 *  km·s⁻² / kg / km² / kg·km⁻³):
 *
 *      r_rel  = r_sc − r_body          (== pos − ephcache, COI-frame)
 *      h      = ‖r_rel‖ − R_eq         (spherical;  flattening == 0)
 *      h      = geodeticAltitude(r_rel, k̂_inertial, R_eq,
 *                                flattening)
 *                                      (oblate ellipsoid; flattening > 0.
 *                                       Matches GODOT geodeticFromCartesian;
 *                                       see Geodetic.h.  k̂_inertial is the
 *                                       body polar axis reconstructed from
 *                                       the cached body-fixed→inertial quat.
 *                                       ``flattening`` is a body shape
 *                                       property, not an atmosphere param.)
 *      v_atm  = v_body + ω_inertial × r_rel
 *      v_rel  = v_sc − v_atm           (v_body == velcache, COI-frame)
 *      ρ      = atmosphere.density(h)
 *      BC     = cd · area / mass
 *      a_drag = −½ · ρ · BC · ‖v_rel‖ · v_rel
 *
 *  Off-COI is fully supported: ephcache + velcache are the COI-relative
 *  body position + velocity caches, both filled by parallel
 *  ``coiResolveKernel`` / ``coiResolveVelocityKernel`` launches.  When
 *  the drag body happens to be the sample's COI body, both caches
 *  evaluate to zero by construction and the math reduces to the
 *  classic ``v_rel = vel − ω × r``.
 *
 *  The helpers below decompose the math so a future STM-aware partials
 *  kernel can build derivatives without re-deriving the chain rule. */
struct Drag {
    /** @brief Compute v_rel = v_sc − v_body − ω × r_rel as a pure ET
     *  chain.  ω must be in inertial frame (which is what brie's
     *  ``bodyAngularVelocity`` returns post-PR-1.5.x). */
    template<typename ER, typename EV, typename EVb, typename EO>
    DEVICEHOST()
    static auto evalRelVelocity(const ER& r_rel, const EV& v_sc,
        const EVb& v_body, const EO& omega_inertial)
    {
        return v_sc - v_body - omega_inertial.cross(r_rel);
    }

    /** @brief a_drag = −½ · ρ · BC · ‖v_rel‖ · v_rel.
     *  BC = cd·area/mass inlined per sample (mirrors SRP's inline
     *  effective-AMS pattern from PR-2). */
    template<typename EV>
    DEVICEHOST()
    static Vec3R evalAcceleration(const Real& rho, const Real& cd,
        const Real& area, const Real& mass, const EV& v_rel)
    {
        const Real bc     = cd * area / mass;
        const Real factor = Real{ -0.5 } * rho * bc * v_rel.template norm<Real>();
        return factor * v_rel;
    }

    /** @brief Full drag acceleration given pre-resolved body position +
     *  velocity + ω, body polar axis k̂ in inertial frame, the body's
     *  equatorial-ish radius, and the atmosphere model.  Returns zero
     *  if the spacecraft is above the atmosphere cutoff.
     *
     *  Altitude is geodetic when ``flattening > 0`` (matches GODOT's
     *  ``geodeticFromCartesian``: projects r onto ``pole`` to recover
     *  the polar component) and falls back to the bit-identical
     *  spherical ``||r|| − R_eq`` when ``flattening == 0``.  ``flattening``
     *  is a *body* shape property (``RefEnvironment::flattening(body)``),
     *  not an atmosphere parameter. */
    template<typename Atmosphere>
    DEVICEHOST()
    static Vec3R eval(const Vec3R& r_rel, const Vec3R& v_sc,
        const Vec3R& v_body, const Vec3R& omega_inertial, const Vec3R& pole,
        const Real& R_eq, const Real& flattening,
        const Real& mass, const Real& area, const Real& cd,
        const Atmosphere& atm)
    {
        const Real h = ::paraHPOP::model::environment::atmosphere::geodeticAltitude(
            r_rel, pole, R_eq, flattening);
        /* Select the altitude band once, then both the cutoff early-exit and
         * the density read use it (the cutoff is per-band). */
        const idx_t band = atm.selectBand(h);
        if (h > atm.cutoffOf(band))
            return Vec3R::Zeros();
        const Real rho   = atm.densityInBand(h, band);
        const Vec3R vRel = evalRelVelocity(r_rel, v_sc, v_body, omega_inertial);
        return evalAcceleration(rho, cd, area, mass, vRel);
    }
};

/** @brief Evaluation kernel for drag on a given body */
namespace kernel {

/** @brief Compile-time launch metadata for ::drag.  Mirrors SRPLaunch. */
/* 256/4 matches J2/SH launch bounds (~64 reg/thread cap on sm_61).
 * Originally tightened to 256/8 when drag's spill cascade brought
 * STACK to 0; the quat-to-pole post-refactor reintroduced 16-40 B
 * STACK at 256/8, so the higher register budget at 256/4 is the
 * right trade — at sm_61 the 6160 B shmem footprint already pushed
 * effective occupancy below 8 blocks/SM (8×6160 = 49 KB > 48 KB
 * shared budget), so the nominal occupancy loss is mostly cosmetic
 * and the eliminated spill is a real wall-time win. */
using DragLaunch = KernelLaunchTraits<256, 4>;

/* The drag kernel takes the per-body atmosphere as a feta-handle GRef (the
 * piecewise segment array's kernel-facing handle), passed by value.  The
 * owning ``PiecewiseExponentialAtmosphere`` lives in the Environment; only
 * this small trivially-copyable handle crosses into the kernel. */
using DragAtmT = ::paraHPOP::model::environment::atmosphere::
    PiecewiseExponentialAtmosphere::GRef;

/* The single-block (1-segment) kernel takes the four block params as a small
 * trivially-copyable ``ExpBlock`` value (constant-bank, no per-body
 * segment-array load, no band scan).  Selected at graph-construction time
 * when a body has exactly one atmosphere segment — see PrivateDimensional
 * and ``Environment::expBlock``. */
using DragExpT = ::paraHPOP::model::environment::atmosphere::ExpBlock;

/** @brief Drag kernels — one per atmosphere flavour, selected at
 *  graph-construction time by the body's segment count (see
 *  PrivateDimensional).  Per-sample contract (identical for both; only the
 *  density policy differs):
 *    - r_rel = pos − ephcache (body position in COI frame).
 *    - h = geodeticAltitude(r_rel, pole, R_eq, flattening).  Above the
 *      (per-band) cutoff → no contribution; the early-exit still skips the
 *      ω / velcache / v_rel reads for above-atmosphere samples.
 *    - Read ω (inertial frame) from ``omegacache[i]`` — pre-computed by the
 *      Tier-2c ``omegaResolveKernel`` so the body does NOT carry the
 *      polynomial walk + sin/cos kinematics inline (that fold took the
 *      stack spill from 352 B on sm_61 to ~0 B).
 *    - v_rel = vel − velcache − ω × r_rel.  velcache is the body's
 *      COI-relative velocity (zero when drag-body == COI body).
 *    - a_drag = −½ · ρ(h) · (cd · area / mass) · ‖v_rel‖ · v_rel.
 *    - Non-atomic ``bodyAcc[i] += a_drag``.
 *
 *  ``dragPiecewise`` resolves the altitude band from the per-body segment
 *  array (@ref DragAtmT, branchless count).  ``dragExp`` is the single-block
 *  fast path: its density comes from four scalar params (@ref DragExpT) with
 *  no band scan and no segment-array load.  Both share one templated device
 *  body, so the hand-tuned shmem-``v_rel`` STACK:0 structure is identical
 *  and the single-block result is bit-identical to the legacy single
 *  exponential. */
__global__ void dragPiecewise(feta::vector::Array<Real, 3>::GRef bodyAcc,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef pos,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef vel,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef ephcache,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef velcache,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef omegacache,
    GRID_CONSTANT() feta::vector::Array<Real, 4>::GRef quatcache,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT mass,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT area,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT cd,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated,
    GRID_CONSTANT() Real R_eq,
    GRID_CONSTANT() Real flattening,
    GRID_CONSTANT() DragAtmT atm);

/** @brief Single-block (1-segment) drag fast path — see ``dragPiecewise``
 *  for the per-sample contract.  Atmosphere arrives as four scalar params
 *  (@ref DragExpT), no band machinery. */
__global__ void dragExp(feta::vector::Array<Real, 3>::GRef bodyAcc,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef pos,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef vel,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef ephcache,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef velcache,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef omegacache,
    GRID_CONSTANT() feta::vector::Array<Real, 4>::GRef quatcache,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT mass,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT area,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT cd,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated,
    GRID_CONSTANT() Real R_eq,
    GRID_CONSTANT() Real flattening,
    GRID_CONSTANT() DragExpT atm);

/** @brief Drag using a density cache produced by the two NRLMSISE-00
 *  graph nodes (input preparation and density evaluation). */
__global__ void dragNrlmsise00(feta::vector::Array<Real, 3>::GRef bodyAcc,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef pos,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef vel,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef ephcache,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef velcache,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef omegacache,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT density,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT mass,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT area,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT cd,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated);
} // namespace kernel

} // namespace accelerations
} // namespace model
} // namespace paraHPOP
