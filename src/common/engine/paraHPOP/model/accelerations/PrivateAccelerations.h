#pragma once

#include "paraHPOP/model/accelerations/Accelerations.h"
#include "paraHPOP/model/accelerations/Drag.h"
#include "paraHPOP/model/accelerations/EarthCorrections.h"
#include "paraHPOP/model/accelerations/J2.h"
#include "paraHPOP/model/accelerations/PointGravity.h"
#include "paraHPOP/model/accelerations/RefAccelerations.h"
#include "paraHPOP/model/accelerations/SRP.h"
#include "paraHPOP/model/accelerations/SphericalHarmonics.h"
#include "paraHPOP/model/environment/eclipse/Shadow.h"
#include "paraHPOP/model/environment/atmosphere/Nrlmsise00.h"

namespace paraHPOP {
namespace model {
namespace accelerations {

/** @brief Internal evaluation of the total acceleration from raw
 *  per-sample fields.
 *
 *  AMS (``cr · area / mass``, reflectivity-weighted) and BC
 *  (``cd · area / mass``, drag-coefficient-weighted) are derived
 *  once at the top of the body loop so each force consumes the
 *  combination tied to its physics.  The two derived scalars stay
 *  in registers across all bodies — same cost as passing them as
 *  parameters, with no risk of cd↔cr swap at the call site. */
template<bool work, bool MaybeVolatile>
DEVICEHOST()
Vec3R RefAccelerations<work, MaybeVolatile>::eval_(const SampleIndex& i,
    const Vec3R& pos, [[maybe_unused]] const Vec3R& vel, const mReal_t& epoch,
    const mReal_t& mass, const mReal_t& area, const mReal_t& cr,
    const mReal_t& cd, const brie::NaifId& COI,
    const typename EnvT::template Ref<work, MaybeVolatile>& env) const
{
    Vec3R acc = Vec3R::Zeros();

    /* Derived once per sample, reused across all bodies in the loop.
     * AMS is SRP-only (includes reflectivity); BC is drag-only
     * (includes drag coefficient).  Naming them apart at this scope
     * keeps the per-body branches unambiguous. */
    const mReal_t AMS = cr * area / mass;
    const mReal_t BC  = cd * area / mass;

    /* Loop over the active bodies to minimise the ephemeris calls.  One
     * getPosition() per body, results reused for all acceleration
     * computations. */
    for (idx_t bodyi = 0; bodyi < env.bodies().size(); bodyi++) {
        brie::NaifId body = env.bodies()[bodyi];
        /* Rotation frame for this body's force-model rotations (SH, J2,
         * drag): the resolved per-body orientation target — its own id under
         * the IAU default, or a binary-frame target like ITRF93's 3000 under
         * auto-best.  Ephemeris/GM/radius stay keyed on ``body`` itself. */
        const brie::NaifId rotTarget = env.orientationTarget(bodyi);
        /* For atmosphere bodies fetch pos + vel in one chain walk via
         * ``getPositionAndVelocity`` (brie internally evaluates the COI
         * chain once and returns Vec6R); pos-only bodies stay on the
         * pos-only call.  Halves the brie chain-walk cost for drag
         * bodies on the host RHS. */
        Vec3R bodyPos;
        Vec3R bodyVel;
        bool haveBodyVel = false;
        if (active<src::ATMOSPHERE>(bodyi) && COI != body) {
            const auto state = env.getPositionAndVelocity(
                i, epoch, body, COI);
            bodyPos     = state.template segment<0, 3>();
            bodyVel     = state.template segment<3, 3>();
            haveBodyVel = true;
        } else {
            bodyPos = env.getPosition(i, epoch, body, COI);
        }

        /* A body can be present only for radiation, drag or occultation.
         * Match the device graph: its presence does not enable point gravity. */
        if (active<src::POINTGRAVITY>(bodyi)) {
            mReal_t gm = env.constants().body(body).gm();
            if (COI == 0) {
                /* barycentric */
                acc += PointGravity::thirdBody_B(pos, bodyPos, gm);
            } else if (body != COI) {
                /* non-barycentric third-body */
                acc += PointGravity::thirdBody_NB(pos, bodyPos, gm);
            } else {
                /* body == COI: simple two-body */
                acc += PointGravity::twoBody(pos, gm);
            }
        }

        /* Spherical harmonics or J2 (mutually exclusive per body) */
        if constexpr (!work) {
            if (active<src::SHAPE>(bodyi)) {
                auto deltaPos = (COI != body) ? pos - bodyPos : pos;
                acc += SphericalHarmonics::eval<work>(deltaPos, epoch,
                    rotTarget, env.shCoeffs(bodyi), env.orientations());
            } else if (active<src::OBLATENESS>(bodyi)) {
                auto deltaPos = (COI != body) ? pos - bodyPos : pos;
                acc += J2::eval(i, deltaPos, epoch, body, rotTarget, env);
            }
        }

        /* Radiation pressure (uses SRP-specific AMS = cr · area / mass) */
        if (active<src::RADIATION>(bodyi)) {
            auto deltaPos = (COI != 10) ? pos - bodyPos : pos;
            /* Occultation: scene-level N→1 — every occulting body (the
             * Sun excluded) attenuates this single radiation source, the
             * dual-cone factors combining as a product.  The flag scan
             * keeps the no-occulter path on the exact original
             * expression (bit-identity with shadow-less builds). */
            bool anyOcc = false;
            for (idx_t oj = 0; oj < env.bodies().size(); oj++) {
                if (active<src::OCCULTING>(oj) && env.bodies()[oj] != 10) {
                    anyOcc = true;
                    break;
                }
            }
            if (!anyOcc) {
                acc += SRP::eval<Vec3R, work>(deltaPos, AMS, env);
            } else {
                /* `body` here is the Sun (radiation is validated
                 * Sun-only), so its constants entry carries RSun. */
                const mReal_t rSun = env.constants().body(body).r();
                mReal_t occ        = mReal_t{ 1 };
                for (idx_t oj = 0; oj < env.bodies().size(); oj++) {
                    const brie::NaifId ob = env.bodies()[oj];
                    if (!active<src::OCCULTING>(oj) || ob == 10)
                        continue;
                    Vec3R occPos = Vec3R::Zeros();
                    if (COI != ob)
                        occPos = env.getPosition(i, epoch, ob, COI);
                    occ *= ::paraHPOP::model::environment::eclipse::factor(
                        deltaPos, pos - occPos,
                        rSun, env.constants().body(ob).r());
                }
                acc += occ * SRP::eval<Vec3R, work>(deltaPos, AMS, env);
            }
        }

        /* Atmospheric drag (uses drag-specific BC = cd · area / mass).
         * COI-relative body velocity comes from brie directly (cheaper
         * than re-rolling a host velocity cache; same numerics as the
         * device's velcache fed by ``coiResolveVelocityKernel``).
         * ``r_rel`` reduces to ``pos`` when ``body == COI`` (bodyPos
         * == 0 by construction). */
        if (active<src::ATMOSPHERE>(bodyi)) {
            const Vec3R r_rel = (COI != body) ? pos - bodyPos : pos;
            /* bodyVel populated by the fused getPositionAndVelocity call
             * above for off-COI; reduces to zero by construction when
             * drag-body == COI body. */
            if (!haveBodyVel)
                bodyVel = Vec3R::Zeros();
            const Vec3R omega
                = env.orientations().bodyAngularVelocity(epoch, rotTarget);
            /* Body polar axis k̂ in inertial frame.  ``bodyAxesRotation``
               returns the ICRF→body-fixed transform (per brie's
               ``test_Frames.cpp::rot.rotate(icrfs)``).  We want body
               z-axis (0,0,1) expressed in ICRF — that's the INVERSE
               transform applied to (0,0,1).  Only the pole direction
               matters for geodetic altitude (the formula is invariant
               under prime-meridian rotation). */
            Vec3R zBody;
            zBody.template get<0>() = mReal_t{ 0 };
            zBody.template get<1>() = mReal_t{ 0 };
            zBody.template get<2>() = mReal_t{ 1 };
            const auto axesRotation
                = env.orientations().bodyAxesRotation(epoch, rotTarget);
            const Vec3R pole = axesRotation.applyInverse(zBody);
            const mReal_t R_eq       = env.constants().body(body).r();
            const mReal_t flattening = env.flattening(bodyi);

#ifndef __CUDA_ARCH__
            /* CPU NRLMSISE-00 path.  Unlike the CUDA backend's three graph
             * nodes, the host fuses input preparation, density evaluation
             * and drag into this per-sample RHS call.  The surrounding host
             * propagator already distributes samples across OpenMP threads. */
            if (env.isNrlmsise00(bodyi)) {
                const Vec3R fixed = axesRotation.rotate(r_rel);
                const mReal_t rho = ::paraHPOP::model::environment::atmosphere::
                    evaluateNrlmsise00HostDensity(fixed, epoch,
                        env.nrlAtmosphere(bodyi), R_eq, flattening);
                const Vec3R v_rel = vel - bodyVel - omega.cross(r_rel);
                acc += Drag::evalAcceleration(
                    rho, cd, area, mass, v_rel);
                continue;
            }
#endif
            const auto atm           = env.atmosphere(bodyi);

            /* Altitude: geodetic when flattening > 0 (matches the GODOT
               side and the device drag kernel), spherical ||r||−R_eq fast
               path otherwise.  flattening is a body shape property
               (env.flattening(bodyi)), not an atmosphere parameter. */
            const mReal_t h = ::paraHPOP::model::environment::atmosphere::
                geodeticAltitude(r_rel, pole, R_eq, flattening);
            const idx_t band = atm.selectBand(h);
            if (h <= atm.cutoffOf(band)) {
                const Vec3R v_rel = vel - bodyVel - omega.cross(r_rel);
                const mReal_t rho = atm.densityInBand(h, band);
                const mReal_t factor
                    = mReal_t{ -0.5 } * rho * BC
                      * v_rel.template norm<mReal_t>();
                acc += factor * v_rel;
            }
        }
    }

    // Earth corrections are scene-level additions, evaluated exactly once.
    if (env.earthCorrections().any())
        acc += EarthCorrections::eval(i, pos, vel, epoch, COI, env);
    return acc;
}

/** @brief Packet (W-lane) total acceleration evaluation.
 *
 *  Two paths:
 *  - Pure packet (PointGravity + SRP only): per-body contributions
 *    computed lane-wise via packet arithmetic.  Body positions retrieved
 *    via ``env.getPositionPacket<W>``.
 *  - Per-lane scalar fallback (any SHAPE/OBLATENESS active): extract
 *    each lane's scalar state, dispatch the existing scalar ``eval_``,
 *    transpose results back into a packet.  Correctness-equivalent;
 *    loses SIMD on this packet. */
template<bool work, bool MaybeVolatile>
template<idx_t W>
feta::vector::PacketItem<Real, 3, W>
RefAccelerations<work, MaybeVolatile>::evalPacket(
    const feta::PacketIndex<W>& pi,
    const feta::vector::PacketItem<Real, 3, W>& pos,
    [[maybe_unused]] const feta::vector::PacketItem<Real, 3, W>& vel,
    const feta::simd::Packet<Real, W>& epoch,
    const feta::simd::Packet<Real, W>& mass,
    const feta::simd::Packet<Real, W>& area,
    const feta::simd::Packet<Real, W>& cr,
    const feta::simd::Packet<Real, W>& cd, const brie::NaifId& COI,
    const typename EnvT::template Ref<work, MaybeVolatile>& env) const
{
    using PacketT = feta::simd::Packet<Real, W>;
    using PItem3  = feta::vector::PacketItem<Real, 3, W>;

    /* Path-selection predicate: any J2 / SH / ATMOSPHERE active?
     * Drag falls back to per-lane scalar because brie's
     * ``bodyAngularVelocity`` is scalar-only — same precedent as
     * J2/SH whose chebyshev evaluators are scalar-only on host. */
    bool hasScalarOnlyForce = env.earthCorrections().any();
    if constexpr (!work) {
        for (idx_t b = 0; b < env.bodies().size(); b++) {
            if (this->template active<src::SHAPE>(b)
                || this->template active<src::OBLATENESS>(b)
                || this->template active<src::ATMOSPHERE>(b)) {
                hasScalarOnlyForce = true;
                break;
            }
        }
    }

    /* Occultation forces the per-lane scalar fallback too: the factor
     * loop needs per-occulter scalar chain walks (env.getPosition) and
     * constants lookups.  Only relevant when a radiation source is
     * active — an occulting flag with no SRP never changes physics. */
    if (!hasScalarOnlyForce) {
        bool anyOcc = false, anyRad = false;
        for (idx_t b = 0; b < env.bodies().size(); b++) {
            anyOcc |= (this->template active<src::OCCULTING>(b)
                && env.bodies()[b] != 10);
            anyRad |= this->template active<src::RADIATION>(b);
        }
        hasScalarOnlyForce = anyOcc && anyRad;
    }

    /* Per-lane fallback: unpack to scalar Items, call scalar eval per lane,
     * pack the lane-major Items back into a packet output. */
    if (hasScalarOnlyForce) {
        Vec3R posLanes[W], velLanes[W], outLanes[W];
        alignas(64) Real epochLanes[W], massLanes[W], areaLanes[W],
            crLanes[W], cdLanes[W];
        feta::cpu::unpackLanes(pos, posLanes);
        feta::cpu::unpackLanes(vel, velLanes);
        feta::cpu::unpackLanes(epoch, epochLanes);
        feta::cpu::unpackLanes(mass, massLanes);
        feta::cpu::unpackLanes(area, areaLanes);
        feta::cpu::unpackLanes(cr, crLanes);
        feta::cpu::unpackLanes(cd, cdLanes);

        for (idx_t k = 0; k < W; ++k) {
            SampleIndex idxK = SampleIndex::make(pi.base_ + k);
            outLanes[k] = this->eval(idxK, posLanes[k], velLanes[k],
                epochLanes[k], massLanes[k], areaLanes[k], crLanes[k],
                cdLanes[k], COI, env);
        }

        PItem3 out;
        feta::cpu::packLanes(out, outLanes);
        return out;
    }

    /* Pure-packet path uses SRP only (no J2/SH/drag).  Compute AMS
     * once inline so the per-body loop can broadcast the SRP scaling
     * across lanes the same way the scalar path does. */
    const PacketT AMS = cr * area / mass;

    /* Pure packet path — PointGravity + SRP only.  Uses feta's packet
     * primitives (``packetRCubedNorm`` + scaled ``packetIAdd``) with
     * expression-template ``pos - bodyPos`` for native, low-verbosity
     * syntax. */
    PItem3 acc;
    acc.setZero();

    const Real AU = env.constants().au();
    const PacketT srpFactorAMS
        = PacketT::broadcast(AU * AU * SRP::AUmeanSRP) * AMS;

    for (idx_t bodyi = 0; bodyi < env.bodies().size(); bodyi++) {
        const brie::NaifId body = env.bodies()[bodyi];
        const PItem3 bodyPos
            = env.template getPositionPacket<W>(pi, epoch, body, COI);
        /* PointGravity contribution.  Branches on COI like scalar
         * ``eval_``: barycentric (single twoBody on delta);
         * non-barycentric third-body (twoBody on delta + twoBody on
         * bodyPos); body == COI (twoBody on pos directly). */
        if (this->template active<src::POINTGRAVITY>(bodyi)) {
            const Real gmScalar   = env.constants().body(body).gm();
            const PacketT minusGm = PacketT::broadcast(-gmScalar);
            if (COI == 0 || body != COI) {
                const auto delta = pos - bodyPos;
                /* acc += -gm * delta / |delta|³ */
                const PacketT factor
                    = minusGm * feta::cpu::packetRCubedNorm(delta, pi);
                feta::cpu::packetIAdd(acc, factor, delta, pi);
                if (COI != 0 && body != COI) {
                    /* Non-barycentric tidal term: -gm * bodyPos / |bodyPos|³ */
                    const PacketT factorB
                        = minusGm * feta::cpu::packetRCubedNorm(bodyPos, pi);
                    feta::cpu::packetIAdd(acc, factorB, bodyPos, pi);
                }
            } else {
                /* body == COI: ``twoBody(pos, gm)``. */
                const PacketT factor
                    = minusGm * feta::cpu::packetRCubedNorm(pos, pi);
                feta::cpu::packetIAdd(acc, factor, pos, pi);
            }
        }

        /* SRP contribution.  ``SRP::eval(deltaPos, AMS) = +factor * AMS *
         * deltaPos / |deltaPos|³`` (sign: outward push) where
         * ``deltaPos = pos - bodyPos`` if ``COI != 10`` (non-heliocentric),
         * else ``pos`` directly. */
        if (this->template active<src::RADIATION>(bodyi)) {
            if (COI != 10) {
                const auto delta       = pos - bodyPos;
                const PacketT srpScale = srpFactorAMS
                    * feta::cpu::packetRCubedNorm(delta, pi);
                feta::cpu::packetIAdd(acc, srpScale, delta, pi);
            } else {
                const PacketT srpScale
                    = srpFactorAMS * feta::cpu::packetRCubedNorm(pos, pi);
                feta::cpu::packetIAdd(acc, srpScale, pos, pi);
            }
        }
    }

    return acc;
}

} // namespace accelerations
} // namespace model
} // namespace paraHPOP
