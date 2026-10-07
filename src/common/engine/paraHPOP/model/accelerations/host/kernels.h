#pragma once

#include "paraHPOP/model/accelerations/J2.h"
#include "paraHPOP/model/accelerations/PointGravity.h"
#include "paraHPOP/model/accelerations/SRP.h"
#include "paraHPOP/model/accelerations/SphericalHarmonics.h"
#include "paraHPOP/typedefs.h"
#include "interface/config/model/sphericalharmonics/Coefficients.h"
#include <parm/util/HostLaunch.h>

namespace paraHPOP {
namespace model {
namespace accelerations {
namespace host {
namespace kernel {

using Vec3Ref      = feta::vector::Array<Real, 3>::GRef;
using Vec4Ref      = feta::vector::Array<Real, 4>::GRef;
using BoolHandle   = feta::scalar::Array<bool>::GRef::HandleT;
using RealHandle   = feta::scalar::Array<Real>::GRef::HandleT;
using NaifIdHandle = feta::scalar::Array<NaifId>::GRef::HandleT;
using EnvRef       = environment::Env::GRef;
using OrientationsRef = environment::Orientations<false>;
using SHCoeffsRef
    = interface::config::model::sphericalharmonics::Coefficients::GRef;

template<typename EphCacheHostT>
inline void fillEphNativeHost(const EphCacheHostT& cache,
    const environment::EphT::GRef& source, const RealHandle& epochs,
    const BoolHandle& terminated, idx_t bytesPerSample = 0)
{
    for (idx_t b = 0; b < cache.nBodyUnits(); b++) {
        auto leaf = source.interpolators_[b];
        auto slot = cache.slotRef(b);
        feta::cpu::packetBatchedFor<Real>(
            idx_t{ 0 }, slot.size(),
            [&](const auto& pi) {
                constexpr idx_t W = std::remove_cvref_t<decltype(pi)>::width;
                auto active       = parm::util::Host::applyTail<W>(
                    ~parm::util::Host::loadMask<Real, W>(terminated, pi), pi);
                if (!active.anyTrue())
                    return;
                parm::util::Host::dispatchMasked(
                    pi, active, [&](const SampleIndex& i) {
                        Vec6R val;
                        val.setZero();
                        leaf.getValuesAndDerivatives(val, epochs[i.global()]);
                        slot[i] = val;
                    });
            },
            bytesPerSample,
            false);
    }
}

template<typename EphCacheHostT>
inline void fillEphNativeHostAtOffset(const EphCacheHostT& cache,
    const environment::EphT::GRef& source, const RealHandle& baseEpochs,
    const RealHandle& offsets, const BoolHandle& terminated,
    idx_t bytesPerSample = 0)
{
    for (idx_t b = 0; b < cache.nBodyUnits(); b++) {
        auto leaf = source.interpolators_[b];
        auto slot = cache.slotRef(b);
        feta::cpu::packetBatchedFor<Real>(
            idx_t{ 0 }, slot.size(),
            [&](const auto& pi) {
                constexpr idx_t W = std::remove_cvref_t<decltype(pi)>::width;
                auto active       = parm::util::Host::applyTail<W>(
                    ~parm::util::Host::loadMask<Real, W>(terminated, pi), pi);
                if (!active.anyTrue())
                    return;
                parm::util::Host::dispatchMasked(
                    pi, active, [&](const SampleIndex& i) {
                        const Real targetEpoch
                            = baseEpochs[i.global()] + offsets[i.global()];
                        Vec6R val;
                        val.setZero();
                        leaf.getValuesAndDerivatives(val, targetEpoch);
                        slot[i] = val;
                    });
            },
            bytesPerSample,
            false);
    }
}

inline void cacheRotation(Vec4Ref cache, const RealHandle& epochs,
    const BoolHandle& terminated, const OrientationsRef& orientations,
    NaifId bodyId, idx_t bytesPerSample = 0)
{
    feta::cpu::packetBatchedFor<Real>(
        idx_t{ 0 }, cache.size(),
        [&](const auto& pi) {
            constexpr idx_t W = std::remove_cvref_t<decltype(pi)>::width;
            auto active       = parm::util::Host::applyTail<W>(
                ~parm::util::Host::loadMask<Real, W>(terminated, pi), pi);
            if (!active.anyTrue())
                return;

            parm::util::Host::dispatchMasked(
                pi, active, [&](const SampleIndex& i) {
                    auto rot = orientations.bodyAxesRotation(
                        epochs[i.global()], bodyId);
                    cache[i] = rot.eval();
                });
        },
        bytesPerSample,
        false);
}

} // namespace kernel
} // namespace host
} // namespace accelerations
} // namespace model
} // namespace paraHPOP
