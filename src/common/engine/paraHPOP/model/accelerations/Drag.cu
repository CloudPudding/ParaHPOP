#include "paraHPOP/model/accelerations/Drag.h"

namespace paraHPOP {
namespace model {
namespace accelerations {
namespace kernel {

/* Shared per-sample drag body, templated on the atmosphere density policy
 * (``DragAtmT`` GRef for the piecewise kernel, ``DragExpT`` ExpBlock for the
 * single-block kernel).  ``__forceinline__`` so it is folded into each
 * __global__ wrapper below: the wrappers' GRID_CONSTANT params are then used
 * directly from the constant bank (the by-value pass here is elided) and the
 * hand-tuned shmem-``v_rel`` STACK:0 structure is identical for both
 * kernels.  Only ``atm.selectBand`` differs (a runtime branchless scan for
 * piecewise vs a folded constant 0 for the single block), so the single-block
 * result is bit-identical to the legacy single ``drag`` kernel.  Launch
 * bounds live in Drag.h as ``DragLaunch`` (parallel to SRP). */
template<typename Atm>
__device__ __forceinline__ void dragBody(
    feta::vector::Array<Real, 3>::GRef bodyAcc,
    feta::vector::Array<Real, 3>::GRef pos,
    feta::vector::Array<Real, 3>::GRef vel,
    feta::vector::Array<Real, 3>::GRef ephcache,
    feta::vector::Array<Real, 3>::GRef velcache,
    feta::vector::Array<Real, 3>::GRef omegacache,
    feta::vector::Array<Real, 4>::GRef quatcache,
    feta::scalar::Array<Real>::GRef::HandleT mass,
    feta::scalar::Array<Real>::GRef::HandleT area,
    feta::scalar::Array<Real>::GRef::HandleT cd,
    feta::scalar::Array<bool>::GRef::HandleT terminated, Real R_eq,
    Real flattening, const Atm& atm)
{
    /* Per-thread Dim=3 shmem slot for ``v_rel`` — the single read-only
     * @c Vec3R that lives across the ``atm.densityInBand`` exp call (the
     * call clobbers caller-saved registers, forcing the compiler to
     * spill the value on Volta+; cubin pre-D.1: 8 B STACK + 1 LDL +
     * 1 STL on sm_70/80/90, 0 on sm_61).  The three short-lived
     * @c Vec3Rs (r_rel, omega, v_body) used 1-2 times within a tight
     * span stay register-resident — pulling them to registers for a
     * couple of math ops is cheaper than the LDS round-trip.  Cost:
     * 3 × 8 × maxBlockSize bytes per block (6 KiB at default 256). */
    __shared__ Real s_vRel[DragLaunch::maxBlockSize * 3];
    __shared__ feta::vector::Array<Real, 3>::WRef::HandleT vRelHandle;

    SampleIndex i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);

    if (threadIdx.x == 0) {
        vRelHandle.ptr     = s_vRel;
        vRelHandle.dOffset = blockDim.x;
    }
    __syncthreads();

    if (i.global() >= bodyAcc.size())
        return;

    if (terminated[i])
        return;

    /* r_rel = pos − ephcache (COI-relative spacecraft-to-body vector).
     * Off-COI is fully supported — ephcache is the resolved
     * (COI-relative) body position cache, zero by construction when
     * the drag body == COI body and non-zero otherwise. */
    Vec3R r_rel  = (pos - ephcache)[i];
    /* Body polar axis k̂ in inertial frame: reconstructed from the
       body-fixed→inertial quaternion stored in quatcache (filled by
       cacheRotation, same RA/Dec/PM polynomial that omega uses).
       Applying the rotation to ẑ_body gives the body z-axis in
       inertial coordinates, i.e. the pole.  Only the pole direction
       matters for geodetic altitude (the formula is invariant under
       prime-meridian rotation about the polar axis).  J2.cu idiom:
       construct a register-resident RotationT once, apply.  STACK=0
       across all four archs at the DragLaunch 256/4 cap. */
    Vec3R zBody;
    zBody.template get<0>() = Real{ 0 };
    zBody.template get<1>() = Real{ 0 };
    zBody.template get<2>() = Real{ 1 };
    OrientationsT::RotationT rot = { quatcache[i] };
    const Vec3R pole = rot.applyInverse(zBody);
    /* Altitude: geodetic when flattening > 0, spherical fast path
       (||r||−R_eq) otherwise — bit-identical to the pre-flattening
       behaviour on flattening==0 configs.  flattening is a body shape
       property passed by value (env_.flattening(body)), not an atmosphere
       parameter. */
    const Real h = ::paraHPOP::model::environment::atmosphere::geodeticAltitude(
        r_rel, pole, R_eq, flattening);

    /* Select the altitude band once; its per-band cutoff drives the
     * high-altitude early-exit (so above-atmosphere samples still skip the
     * ω / v_body / v_rel reads below) and feeds the density eval.  For the
     * single-block policy ``selectBand`` folds to the constant 0. */
    const idx_t band = atm.selectBand(h);
    if (h > atm.cutoffOf(band))
        return;

    /* ω in inertial frame — read from the pre-computed Dim=3 cache
     * filled once per substage by ``omegaResolveKernel``.  Moving this
     * out of the drag kernel removed the polynomial walk + sin/cos
     * kinematics that previously dominated drag's register footprint
     * (352-byte stack spill on sm_61 → ~0 expected post-fold). */
    const Vec3R omega = omegacache[i];

    /* v_rel = v_sc − v_body − ω × r_rel.  velcache[i] is the body's
     * COI-relative velocity (zero when drag-body == COI body by
     * construction).  Single-frame inertial math throughout.  Written
     * directly into the per-thread shmem slot via feta ET — no Vec3R
     * register-resident mirror; the View's @c operator= unfolds the
     * expression to three per-component stores. */
    const Vec3R v_body = velcache[i];
    vRelHandle[i]      = vel[i] - v_body - omega.cross(r_rel);

    /* a_drag = −½ · ρ · (cd·area/mass) · ‖v_rel‖ · v_rel.
     * Inline BC mirrors SRP's inline effective-AMS pattern from PR-2.
     * Both reads of v_rel below (norm + final accumulate) go through
     * the shmem View; the compiler never has to materialise v_rel as
     * a register-resident @c Vec3R surviving the @c atm.densityInBand call. */
    const Real rho = atm.densityInBand(h, band);
    const Real bc  = cd[i] * area[i] / mass[i];
    const Real factor
        = Real{ -0.5 } * rho * bc * vRelHandle[i].template norm<Real>();

    /* Non-atomic feta `+=` into this body's per-sample scratch slot
     * (chained intra-body via the per-step CUDA graph; reduce kernel
     * later sums across bodies in deterministic order). */
    bodyAcc[i] += factor * vRelHandle[i];
}

/** @brief Piecewise drag kernel — band resolved on device from the per-body
 *  segment array (@ref DragAtmT).  Thin wrapper over @ref dragBody. */
__global__ __launch_bounds__(DragLaunch::maxBlockSize,
    DragLaunch::minBlocksPerSM) void dragPiecewise(
    feta::vector::Array<Real, 3>::GRef bodyAcc,
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
    GRID_CONSTANT() DragAtmT atm)
{
    dragBody(bodyAcc, pos, vel, ephcache, velcache, omegacache, quatcache,
        mass, area, cd, terminated, R_eq, flattening, atm);
}

/** @brief Single-block drag kernel — density from four scalar params
 *  (@ref DragExpT), no band scan, no segment-array load.  Thin wrapper over
 *  @ref dragBody; bit-identical to the legacy single exponential. */
__global__ __launch_bounds__(DragLaunch::maxBlockSize,
    DragLaunch::minBlocksPerSM) void dragExp(
    feta::vector::Array<Real, 3>::GRef bodyAcc,
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
    GRID_CONSTANT() DragExpT atm)
{
    dragBody(bodyAcc, pos, vel, ephcache, velcache, omegacache, quatcache,
        mass, area, cd, terminated, R_eq, flattening, atm);
}

__global__ __launch_bounds__(DragLaunch::maxBlockSize,
    DragLaunch::minBlocksPerSM) void dragNrlmsise00(
    feta::vector::Array<Real, 3>::GRef bodyAcc,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef pos,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef vel,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef ephcache,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef velcache,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef omegacache,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT density,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT mass,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT area,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT cd,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated)
{
    const SampleIndex i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (i.global() >= bodyAcc.size() || terminated[i]) return;
    const Vec3R rRel = (pos - ephcache)[i];
    const Vec3R vRel = vel[i] - velcache[i] - omegacache[i].cross(rRel);
    bodyAcc[i] += Drag::evalAcceleration(
        density[i], cd[i], area[i], mass[i], vRel);
}
} // namespace kernel

} // namespace accelerations
} // namespace model
} // namespace paraHPOP
