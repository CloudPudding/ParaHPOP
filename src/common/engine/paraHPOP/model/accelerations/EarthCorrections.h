#pragma once

#include "paraHPOP/model/accelerations/hpop_compat/HpopCorrections.h"
#include "paraHPOP/model/environment/Environment.h"

namespace paraHPOP::model::accelerations {

/** Native C++/CUDA port of the local MATLAB HPOP 4.2.2 corrections.
 * The SI conversion is confined to this boundary; paraHPOP uses km and
 * seconds in its dimensional RHS. No MATLAB runtime or per-call allocation.
 * Corrected IERS 2010/FES2004 equations; no MATLAB runtime dependency.
 */
struct EarthCorrections {
    template<class Rotation>
    DEVICEHOST() static void setRotation(hpop::Input& in, const Rotation& rot)
    {
        // Columns of E are the rotated inertial basis vectors.
        for (idx_t k = 0; k < 3; ++k) {
            Vec3R basis = Vec3R::Zeros();
            basis.data()[k] = 1.0;
            const Vec3R col = rot.rotate(basis);
            for (idx_t j = 0; j < 3; ++j)
                in.E[3 * j + k] = col.data()[j];
        }
    }

    DEVICEHOST() static void setState(hpop::Input& in,
        const Vec3R& r, const Vec3R& v)
    {
        for (idx_t k = 0; k < 3; ++k) {
            in.r[k] = 1000.0 * r.data()[k];
            in.v[k] = 1000.0 * v.data()[k];
        }
    }

    DEVICEHOST() static void setBodies(hpop::Input& in,
        const Vec3R& sun, const Vec3R& moon)
    {
        for (idx_t k = 0; k < 3; ++k) {
            in.sun[k] = 1000.0 * sun.data()[k];
            in.moon[k] = 1000.0 * moon.data()[k];
        }
    }

    DEVICEHOST() static bool setTime(hpop::Input& in, Real epoch,
        const environment::earthcorrections::View& model)
    {
        const auto t = model.timeAt(epoch);
        in.mjdUtc = t.mjdUtc;
        in.ut1Utc = t.ut1Utc;
        in.ttUtc = t.ttUtc;
        in.xpArcsec = t.xpArcsec;
        in.ypArcsec = t.ypArcsec;
        return t.valid;
    }

    DEVICEHOST() static Vec3R evaluateInput(const hpop::Input& in,
        const environment::earthcorrections::Flags& flags)
    {
        hpop::Constants constants;
        constants.zeroTide = flags.zeroTide;
        const auto out = hpop::evaluate(in, constants,
            flags.solidEarthTides, flags.oceanTides, flags.relativity);
        Vec3R acc;
        for (idx_t k = 0; k < 3; ++k)
            acc.data()[k] = 1.0e-3 *
                (out.solid[k] + out.ocean[k] + out.relativity[k]);
        return acc;
    }

    template<class EnvRef>
    DEVICEHOST() static Vec3R eval(const SampleIndex& i,
        const Vec3R& pos, const Vec3R& vel, Real epoch, NaifId coi,
        const EnvRef& env)
    {
        const auto model = env.earthCorrections();
        if (!model.any())
            return Vec3R::Zeros();

        hpop::Input in{};
        Vec3R r = pos, v = vel;
        if (coi != NaifId{399}) {
            const auto earth = env.getPositionAndVelocity(i, epoch, 399, coi);
            r -= earth.template segment<0, 3>();
            v -= earth.template segment<3, 3>();
        }
        setState(in, r, v);
        if (model.tidesActive()) {
            if (!setTime(in, epoch, model)) {
                Vec3R invalid;
                for (idx_t k = 0; k < 3; ++k) invalid.data()[k] = NAN;
                return invalid;
            }
            setBodies(in, env.getPosition(i, epoch, 10, 399),
                env.getPosition(i, epoch, 301, 399));
            NaifId frame = 399;
            for (idx_t b = 0; b < env.bodies().size(); ++b)
                if (env.bodies()[b] == NaifId{399})
                    frame = env.orientationTarget(b);
            setRotation(in, env.orientations().bodyAxesRotation(epoch, frame));
        }
        return evaluateInput(in, model.flags);
    }
};

namespace kernel {
using EarthCorrectionsLaunch = KernelLaunchTraits<128, 1>;

using CorrectionVec = feta::vector::Array<Real, 3>::GRef;
using CorrectionEpochs = feta::scalar::Array<Real>::GRef::HandleT;
using CorrectionMask = feta::scalar::Array<bool>::GRef::HandleT;
using CorrectionRotation = feta::vector::Array<Real, 4>::GRef;

// Each model writes its own SI acceleration; no concurrent += writes.
__global__ void solidEarthTides(CorrectionVec outputSI,
    GRID_CONSTANT() CorrectionVec pos,
    GRID_CONSTANT() CorrectionEpochs epochs,
    GRID_CONSTANT() CorrectionMask terminated,
    GRID_CONSTANT() environment::earthcorrections::View model,
    GRID_CONSTANT() CorrectionRotation rotations,
    GRID_CONSTANT() CorrectionVec sun, GRID_CONSTANT() CorrectionVec moon);

__global__ void oceanTides(CorrectionVec outputSI,
    GRID_CONSTANT() CorrectionVec pos,
    GRID_CONSTANT() CorrectionEpochs epochs,
    GRID_CONSTANT() CorrectionMask terminated,
    GRID_CONSTANT() environment::earthcorrections::View model,
    GRID_CONSTANT() CorrectionRotation rotations);

// Geocentric Schwarzschild correction: stage position and velocity only.
__global__ void relativisticCorrection(CorrectionVec outputSI,
    GRID_CONSTANT() CorrectionVec pos, GRID_CONSTANT() CorrectionVec vel,
    GRID_CONSTANT() CorrectionMask terminated);

__global__ void accumulateEarthCorrections(CorrectionVec acc,
    GRID_CONSTANT() CorrectionVec solidSI, GRID_CONSTANT() CorrectionVec oceanSI,
    GRID_CONSTANT() CorrectionVec relativitySI,
    GRID_CONSTANT() CorrectionMask terminated,
    GRID_CONSTANT() environment::earthcorrections::Flags flags);
}
}
