#pragma once

#include "paraHPOP/model/environment/atmosphere/PiecewiseExponentialAtmosphere.h"
#include "paraHPOP/model/environment/atmosphere/Nrlmsise00Atmosphere.h"
#include "paraHPOP/model/environment/earthcorrections/EarthCorrections.h"
#include "paraHPOP/typedefs.h"
#include "paraHPOP/util.h"
#include "interface/config/model/sphericalharmonics/Coefficients.h"

namespace paraHPOP {
namespace model {
namespace environment {

using Parser = brie::gravity::Parser;

using EphT = brie::states::EphUnit<true>;

using EphNativeCacheT = brie::states::EphStateCache<true,
    feta::core::memory::Device::CUDA_DEVICE>;

using EphNativeCacheHostT = brie::states::EphStateCache<true,
    feta::core::memory::Device::CPU>;

using RotationsT = brie::orientations::EphUnit<false>;

using CT = brie::Constants;

using GTreeT = brie::gravity::Tree;

template<bool work>
using Orientations = brie::frames::Frames<work, false>;

template<bool work, bool MaybeVolatile = false>
class RefEnvironment {
    using BodyArrT = typename NaifIdArray::template Ref<work, MaybeVolatile>;
    using EphT     = typename EphT::template Ref<work, MaybeVolatile>;
    using CT       = typename CT::template Ref<work, MaybeVolatile>;
    using GTreeT   = typename GTreeT::template Ref<work, MaybeVolatile>;

public:
    
    static constexpr bool Volatile = MaybeVolatile && work;
    using OrientationsT = Orientations<work>;
    
    using EphNativeCacheGRefT = EphNativeCacheT::GRef;

    using SHCoeffsGRefT
        = interface::config::model::sphericalharmonics::Coefficients::GRef;
    
    using SHRefsT = typename feta::scalar::Array<
        SHCoeffsGRefT>::template Ref<work, MaybeVolatile>;
    
    using AtmT = atmosphere::PiecewiseExponentialAtmosphere;
    
    using AtmsRefT = typename feta::scalar::Array<
        AtmT::GRef>::template Ref<work, MaybeVolatile>;
    
    using NrlAtmT = atmosphere::Nrlmsise00Atmosphere;
    using NrlAtmsRefT = typename feta::scalar::Array<
        NrlAtmT::GRef>::template Ref<work, MaybeVolatile>;
    
    using FlatRefT =
        typename feta::scalar::Array<Real>::template Ref<work, MaybeVolatile>;
    
    using OrientTargetsRefT = typename feta::scalar::Array<
        NaifId>::template Ref<work, MaybeVolatile>;

    DEVICEHOST()
    static RefEnvironment make(const BodyArrT& bodies, const EphT& ephs,
        const CT& cs, const OrientationsT& orientations, const GTreeT& tree,
        const SHRefsT& shCoeffsRefs               = {},
        const EphNativeCacheGRefT& ephCachePinned = {},
        const EphNativeCacheGRefT& ephCacheVariable = {},
        const AtmsRefT& atmospheresRefs           = {},
        const NrlAtmsRefT& nrlAtmospheresRefs     = {},
        const FlatRefT& flatteningsRef            = {},
        const OrientTargetsRefT& orientTargetsRef = {},
        const earthcorrections::View& earthCorrections = {})
    {
        return { bodies,
                 ephs,
                 cs,
                 orientations,
                 tree,
                 shCoeffsRefs,
                 ephCachePinned,
                 ephCacheVariable,
                 atmospheresRefs,
                 nrlAtmospheresRefs,
                 flatteningsRef,
                 orientTargetsRef,
                 earthCorrections };
    }

    DEVICEHOST()
    Vec3R getPosition([[maybe_unused]] const SampleIndex& i, const Real& epoch,
        const NaifId& target, const NaifId& center) const
    {
        Vec3R x;
        ephs_.iGetPosition(x, epoch, target, center);
        return x;
    }

    template<idx_t W>
    inline feta::vector::PacketItem<Real, 3, W> getPositionPacket(
        [[maybe_unused]] const feta::PacketIndex<W>& pi,
        const feta::simd::Packet<Real, W>& epoch,
        const NaifId& target, const NaifId& center) const
    {
        feta::vector::PacketItem<Real, 3, W> x;
        x.setZero();  // PacketItem default-ctor is uninitialised; the
                      // brie chain-walker accumulates onto x, so we
                      // must zero before the first chain-step write.
        ephs_.template iGetPositionPacket<W>(x, epoch, target, center);
        return x;
    }

    DEVICEHOST()
    Vec3R getVelocity([[maybe_unused]] const SampleIndex& i, const Real& epoch,
        const NaifId& target, const NaifId& center) const
    {
        Vec3R v;
        ephs_.iGetVelocity(v, epoch, target, center);
        return v;
    }

    DEVICEHOST()
    Vec6R getPositionAndVelocity([[maybe_unused]] const SampleIndex& i,
        const Real& epoch, const NaifId& target, const NaifId& center) const
    {
        Vec6R x;
        ephs_.iGetPositionAndVelocity(x, epoch, target, center);
        return x;
    }

    DEVICEHOST() const BodyArrT& bodies() const { return bodies_; }

    DEVICEHOST() const EphT& ephemeris() const { return ephs_; }

    DEVICEHOST() const CT& constants() const { return cs_; }

    DEVICEHOST() const OrientationsT& orientations() const { return orientations_; }

    DEVICEHOST() const GTreeT& gravityTree() const { return tree_; }

    DEVICEHOST() SHCoeffsGRefT shCoeffs(idx_t bodyIdx) const
    {
        return shCoeffsRefs_[bodyIdx];
    }

    DEVICEHOST() AtmT::GRef atmosphere(idx_t bodyIdx) const
    {
        return atmospheresRefs_[bodyIdx];
    }

    DEVICEHOST() NrlAtmT::GRef nrlAtmosphere(idx_t bodyIdx) const
    {
        return nrlAtmospheresRefs_[bodyIdx];
    }

    DEVICEHOST() bool isNrlmsise00(idx_t bodyIdx) const
    {
        return bodyIdx < nrlAtmospheresRefs_.size()
            && nrlAtmospheresRefs_[bodyIdx].active;
    }

    DEVICEHOST() const earthcorrections::View& earthCorrections() const
    {
        return earthCorrections_;
    }

    DEVICEHOST() Real flattening(idx_t bodyIdx) const
    {
        return flatteningsRef_[bodyIdx];
    }

    DEVICEHOST() NaifId orientationTarget(idx_t bodyIdx) const
    {
        return orientTargetsRef_[bodyIdx];
    }

    DEVICEHOST()
    const EphNativeCacheGRefT& ephNativePinned() const
    {
        return ephCachePinned_;
    }
    DEVICEHOST()
    const EphNativeCacheGRefT& ephNativeVariable() const
    {
        return ephCacheVariable_;
    }

    DEVICEHOST() RefEnvironment clone() const { return *this; }

    BodyArrT bodies_;
    EphT ephs_;
    CT cs_;
    OrientationsT orientations_;
    GTreeT tree_;
    SHRefsT shCoeffsRefs_;
    
    EphNativeCacheGRefT ephCachePinned_;
    EphNativeCacheGRefT ephCacheVariable_;
    
    AtmsRefT atmospheresRefs_;
    
    NrlAtmsRefT nrlAtmospheresRefs_;
    
    FlatRefT flatteningsRef_;
    
    OrientTargetsRefT orientTargetsRef_;

    earthcorrections::View earthCorrections_{};
};

} // namespace environment
} // namespace model
} // namespace paraHPOP
