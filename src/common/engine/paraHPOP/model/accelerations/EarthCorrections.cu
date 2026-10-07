#include "paraHPOP/model/accelerations/EarthCorrections.h"
#include "paraHPOP/model/accelerations/hpop_compat/HpopDeviceTables.cuh"

namespace paraHPOP::model::accelerations::kernel {
namespace {
__device__ bool prepareTideInput(hpop::Input& in, const SampleIndex& i,
    CorrectionVec pos, CorrectionEpochs epochs,
    environment::earthcorrections::View model, CorrectionRotation rotations)
{
    EarthCorrections::setState(in, pos[i], Vec3R::Zeros());
    if (!EarthCorrections::setTime(in, epochs[i], model)) return false;
    environment::OrientationsT::RotationT rot = { rotations[i] };
    EarthCorrections::setRotation(in, rot);
    return true;
}
__device__ Vec3R invalidAcceleration()
{
    Vec3R invalid;
    for (idx_t k = 0; k < 3; ++k) invalid.data()[k] = NAN;
    return invalid;
}
}

__global__ __launch_bounds__(EarthCorrectionsLaunch::maxBlockSize,
    EarthCorrectionsLaunch::minBlocksPerSM)
void solidEarthTides(CorrectionVec outputSI,
    GRID_CONSTANT() CorrectionVec pos,
    GRID_CONSTANT() CorrectionEpochs epochs,
    GRID_CONSTANT() CorrectionMask terminated,
    GRID_CONSTANT() environment::earthcorrections::View model,
    GRID_CONSTANT() CorrectionRotation rotations,
    GRID_CONSTANT() CorrectionVec sun, GRID_CONSTANT() CorrectionVec moon)
{
    const auto i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (i.global() >= outputSI.size() || terminated[i]) return;
    hpop::Input in{};
    if (!prepareTideInput(in, i, pos, epochs, model, rotations)) {
        outputSI[i] = invalidAcceleration(); return;
    }
    EarthCorrections::setBodies(in, sun[i], moon[i]);
    hpop::Constants constants; constants.zeroTide = model.flags.zeroTide;
    hpop::Coefficients coefficients;
    hpop::solidCoefficients(in, constants, coefficients);
    Vec3R result;
    hpop::acceleration(in, constants, coefficients, result.data(), 4);
    outputSI[i] = result;
}

__global__ __launch_bounds__(EarthCorrectionsLaunch::maxBlockSize,
    EarthCorrectionsLaunch::minBlocksPerSM)
void oceanTides(CorrectionVec outputSI,
    GRID_CONSTANT() CorrectionVec pos,
    GRID_CONSTANT() CorrectionEpochs epochs,
    GRID_CONSTANT() CorrectionMask terminated,
    GRID_CONSTANT() environment::earthcorrections::View model,
    GRID_CONSTANT() CorrectionRotation rotations)
{
    const auto i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (i.global() >= outputSI.size() || terminated[i]) return;
    hpop::Input in{};
    if (!prepareTideInput(in, i, pos, epochs, model, rotations)) {
        outputSI[i] = invalidAcceleration(); return;
    }
    hpop::Constants constants; constants.zeroTide = model.flags.zeroTide;
    hpop::Coefficients coefficients;
    hpop::oceanCoefficients(in, constants, coefficients);
    Vec3R result;
    hpop::acceleration(in, constants, coefficients, result.data(), 6);
    outputSI[i] = result;
}

__global__ __launch_bounds__(EarthCorrectionsLaunch::maxBlockSize,
    EarthCorrectionsLaunch::minBlocksPerSM)
void relativisticCorrection(CorrectionVec outputSI,
    GRID_CONSTANT() CorrectionVec pos, GRID_CONSTANT() CorrectionVec vel,
    GRID_CONSTANT() CorrectionMask terminated)
{
    const auto i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (i.global() >= outputSI.size() || terminated[i]) return;
    hpop::Input in{};
    EarthCorrections::setState(in, pos[i], vel[i]);
    hpop::Constants constants;
    Vec3R result;
    hpop::relativity(in, constants, result.data());
    outputSI[i] = result;
}

__global__ __launch_bounds__(EarthCorrectionsLaunch::maxBlockSize,
    EarthCorrectionsLaunch::minBlocksPerSM)
void accumulateEarthCorrections(CorrectionVec acc,
    GRID_CONSTANT() CorrectionVec solidSI, GRID_CONSTANT() CorrectionVec oceanSI,
    GRID_CONSTANT() CorrectionVec relativitySI,
    GRID_CONSTANT() CorrectionMask terminated,
    GRID_CONSTANT() environment::earthcorrections::Flags flags)
{
    const auto i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (i.global() >= acc.size() || terminated[i]) return;
    Vec3R solid = Vec3R::Zeros(), ocean = Vec3R::Zeros(), relativity = Vec3R::Zeros();
    if (flags.solidEarthTides) solid = solidSI[i];
    if (flags.oceanTides) ocean = oceanSI[i];
    if (flags.relativity) relativity = relativitySI[i];
    Vec3R addition;
    for (idx_t k = 0; k < 3; ++k)
        // The previous kernel stored the km/s^2 correction before reducing
        // it. Keep that rounding boundary; do not fuse the scale with +=.
        addition.data()[k] = __dmul_rn(1.0e-3,
            solid.data()[k] + ocean.data()[k] + relativity.data()[k]);
    acc[i] += addition;
}
}
