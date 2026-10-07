#pragma once

#include "paraHPOP/model/accelerations/AccumulateBodies.h"
#include "paraHPOP/model/environment/RefEnvironment.h"
#include "paraHPOP/model/environment/atmosphere.h"

#include "interface/config/model/Environment.h"
#include "interface/naming/orientations/Resolver.h"

#include <algorithm>
#include <array>

namespace paraHPOP {
namespace model {
namespace environment {

using OrientationsT = Orientations<false>;

class Environment {
    using Self = Environment;

public:
    
    using VecCacheT = feta::vector::Array<Real, 3>;
    
    using BodyAccTermsArrayT = accelerations::kernel::BodyAccTermsArrayT;
    
    using QuatCacheT = feta::vector::Array<Real, 4>;
    
    using VecCacheStoreT  = feta::core::memory::Container<Real,
        feta::core::memory::Device::CUDA_DEVICE, 3>;
    using QuatCacheStoreT = feta::core::memory::Container<Real,
        feta::core::memory::Device::CUDA_DEVICE, 4>;
    
    using QuatCacheHostStoreT = feta::core::memory::Container<Real,
        feta::core::memory::Device::CPU, 4>;
    
    using ScalarCacheStoreT = feta::core::memory::Container<Real,
        feta::core::memory::Device::CUDA_DEVICE, 1>;
    
    using ScalarCacheT = feta::scalar::Array<Real>;
    
    enum class CacheTarget : int { HOST, DEVICE, BOTH };
    using SHCoeffsT
        = interface::config::model::sphericalharmonics::Coefficients;
    
    using SHRefsArrayT = feta::scalar::Array<SHCoeffsT::GRef>;
    using InterfaceT   = interface::config::model::Environment;
    
    using AtmT = atmosphere::PiecewiseExponentialAtmosphere;
    using NrlAtmT = atmosphere::Nrlmsise00Atmosphere;
    
    using AtmsRefArrayT = feta::scalar::Array<AtmT::GRef>;
    
    using NrlAtmsRefArrayT = feta::scalar::Array<NrlAtmT::GRef>;
    
    using FlatArrayT = feta::scalar::Array<Real>;
    
    using OrientTargetArrayT = feta::scalar::Array<NaifId>;

    template<bool work, bool MaybeVolatile = false>
    using Ref = RefEnvironment<work, MaybeVolatile>;
    
    using GRef = Ref<false>;
    
    using WRef = Ref<true>;
    
    using VolatileRef = Ref<true, true>;

    Environment() = delete;

    Environment(const InterfaceT& ienv)
        : Environment(ienv, preparePropagationData_(ienv))
    {
    }

    Environment(const InterfaceT& ienv, bool )
        : tree_{ std::move(ienv.makeGravityTree()) }
        , bodies_{ std::move(
              const_cast<InterfaceT&>(ienv).bodies().make().dump()) }
        , ephs_{ std::move(
              const_cast<InterfaceT&>(ienv).ephemeris().make(bodies_).dump()) }
        , cs_{ std::move(
              const_cast<InterfaceT&>(ienv).constants().make().dump()) }
        , rotations_{ const_cast<InterfaceT&>(ienv).orientations().make().dump() }
        , shCoeffs_{ extractAllShCoeffs_(ienv) }
        , shCoeffsRefs_{ buildShCoeffsRefs_(shCoeffs_, false) }
        , atmospheres_{ extractAtmospheres_(ienv) }
        , nrlAtmospheres_{ extractNrlAtmospheres_(ienv) }
        , earthCorrections_{ ienv.earthCorrections() }
        , atmospheresRefs_{ buildAtmospheresRefs_(atmospheres_, false) }
        , nrlAtmospheresRefs_{
              buildNrlAtmospheresRefs_(nrlAtmospheres_, false) }
        , flattenings_{ extractFlattenings_(ienv) }
        , anyOcculting_{ extractAnyOcculting_(ienv) }
    {
        
        ienv.validateRadiationOnlyOnSun();

        orientationTargets_ = resolveOrientationTargets_(ienv);

        if (earthCorrections_.any()) {
            const auto bodyRef = bodies_.hostRef();
            PARAHPOP_ASSERT(bodyRef.contains(NaifId{399}),
                "Earth corrections require Earth (NAIF 399) among active bodies.");
            if (earthCorrections_.tidesActive()) {
                for (idx_t i = 0; i < bodyRef.size(); ++i) {
                    if (bodyRef[i] != NaifId{399}) continue;
                    PARAHPOP_ASSERT(rotations_.hostRef().metadata().hasBody(
                            orientationTargets_.hostRef()[i]),
                        "Earth tides require loaded Earth-fixed orientation data.");
                }
                const auto traverser = ephs_.hostRef().traverser();
                for (NaifId id : {NaifId{10}, NaifId{301}, NaifId{399}}) {
                    bool found = false;
                    for (idx_t k = 0; k < traverser.size(); ++k)
                        found = found || traverser[k].id_ == id;
                    PARAHPOP_ASSERT(found,
                        "Earth tides require loaded Sun, Moon, and Earth ephemerides.");
                }
            }
        }

        for (idx_t i = 0; i < nrlAtmospheres_.size(); ++i) {
            if (!nrlAtmospheres_[i].active())
                continue;
            PARAHPOP_ASSERT(bodies_.hostRef()[i] == NaifId{ 399 },
                "NRLMSISE-00 may only be configured on Earth.");
            const NaifId rotTarget = orientationTargets_.hostRef()[i];
            PARAHPOP_ASSERT(rotations_.hostRef().metadata().hasBody(rotTarget),
                "NRLMSISE-00 requires loaded Earth-fixed orientation data.");
        }

        for (idx_t i = 0; i < shCoeffs_.size(); i++) {
            NaifId shBody = shCoeffs_[i].hostRef(true).body();
            if (shBody != 0) {
                bool isDefaultOrientations = ienv.orientations().files().empty();
                PARAHPOP_ASSERT(!isDefaultOrientations,
                    "Active spherical harmonics require fully defined "
                    "orientations.");
                PARAHPOP_ASSERT(bodies_.hostRef().contains(shBody),
                    "Spherical harmonics body " + std::to_string(shBody)
                        + " not found among active bodies.");
                const NaifId rotTarget = orientationTargets_.hostRef()[i];
                PARAHPOP_ASSERT(rotations_.hostRef().metadata().hasBody(rotTarget),
                    "Spherical harmonics body " + std::to_string(shBody) + " ("
                        + brie::gravity::Parser::parsedName(shBody)
                        + ") rotates under orientation target "
                        + std::to_string(rotTarget)
                        + ", which has no loaded body-fixed orientation data.");
            }
        }
    }

    Environment(Environment& other)       = delete;
    Environment(const Environment& other) = delete;

    Environment(Environment&& other)
        : bodies_{ std::move(other.bodies_) }
        , ephs_{ std::move(other.ephs_) }
        , cs_{ std::move(other.cs_) }
        , tree_{ std::move(other.tree_) }
        , rotations_{ std::move(other.rotations_) }
        , shCoeffs_{ std::move(other.shCoeffs_) }
        , shCoeffsRefs_{ std::move(other.shCoeffsRefs_) }
        , atmospheres_{ std::move(other.atmospheres_) }
        , nrlAtmospheres_{ std::move(other.nrlAtmospheres_) }
        , earthCorrections_{ std::move(other.earthCorrections_) }
        , atmospheresRefs_{ std::move(other.atmospheresRefs_) }
        , nrlAtmospheresRefs_{ std::move(other.nrlAtmospheresRefs_) }
        , flattenings_{ std::move(other.flattenings_) }
        , orientationTargets_{ std::move(other.orientationTargets_) }
        , anyOcculting_{ other.anyOcculting_ }
        , native_{ std::move(other.native_) }
        , resolved_{ std::move(other.resolved_) }
        , rotation_{ std::move(other.rotation_) }
        , hostScratch_{ std::move(other.hostScratch_) }
    {
        
    }

    Environment(NaifIdArray&& bodies, EphT&& ephs, CT&& cs, GTreeT&& gtree,
        RotationsT&& rotations            = RotationsT::empty(),
        std::vector<SHCoeffsT>&& shCoeffs = {})
        : bodies_{ std::move(bodies) }
        , ephs_{ std::move(ephs) }
        , cs_{ std::move(cs) }
        , tree_{ std::move(gtree) }
        , rotations_{ std::move(rotations) }
        , shCoeffs_{ std::move(shCoeffs) }
        , shCoeffsRefs_{ buildShCoeffsRefs_(shCoeffs_, false) }
        , atmospheresRefs_{ buildAtmospheresRefs_(atmospheres_, false) }
        , nrlAtmospheresRefs_{
              buildNrlAtmospheresRefs_(nrlAtmospheres_, false) }
    {
        
        orientationTargets_ = defaultOrientationTargets_(bodies_);
    }

    Environment& operator=(Environment&& other)
    {
        this->bodies_                = std::move(other.bodies_);
        this->ephs_                  = std::move(other.ephs_);
        this->cs_                    = std::move(other.cs_);
        this->tree_                  = std::move(other.tree_);
        this->rotations_             = std::move(other.rotations_);
        this->shCoeffs_              = std::move(other.shCoeffs_);
        this->shCoeffsRefs_          = std::move(other.shCoeffsRefs_);
        this->atmospheres_           = std::move(other.atmospheres_);
        this->atmospheresRefs_       = std::move(other.atmospheresRefs_);
        this->nrlAtmospheres_        = std::move(other.nrlAtmospheres_);
        this->earthCorrections_      = std::move(other.earthCorrections_);
        this->nrlAtmospheresRefs_    = std::move(other.nrlAtmospheresRefs_);
        this->flattenings_           = std::move(other.flattenings_);
        this->orientationTargets_    = std::move(other.orientationTargets_);
        this->anyOcculting_ = other.anyOcculting_;
        this->native_       = std::move(other.native_);
        this->resolved_     = std::move(other.resolved_);
        this->rotation_     = std::move(other.rotation_);
        this->hostScratch_  = std::move(other.hostScratch_);
        
        return *this;
    }

    idx_t numShCoeffs() const { return shCoeffs_.size(); }

    void upload(const cudaStream_t& stream = 0)
    {
        bodies_.upload(stream);
        ephs_.upload(stream);
        cs_.upload(stream);
        tree_.upload(stream);
        rotations_.upload(stream);
        for (auto& sh : shCoeffs_)
            sh.upload(stream);
        
        rebuildShCoeffsRefs_(true);
        shCoeffsRefs_.upload(stream);
        rebuildShCoeffsRefs_(false);
        
        for (auto& atm : atmospheres_)
            atm.upload(stream);
        for (auto& atm : nrlAtmospheres_)
            atm.upload(stream);
        earthCorrections_.upload(stream);
        rebuildAtmospheresRefs_(true);
        atmospheresRefs_.upload(stream);
        rebuildAtmospheresRefs_(false);
        rebuildNrlAtmospheresRefs_(true);
        nrlAtmospheresRefs_.upload(stream);
        rebuildNrlAtmospheresRefs_(false);
        
        flattenings_.upload(stream);
        
        orientationTargets_.upload(stream);
        
    }

    void download(const cudaStream_t& stream = 0)
    {
        bodies_.download(stream);
        ephs_.download(stream);
        cs_.download(stream);
        tree_.download(stream);
        rotations_.download(stream);
        for (auto& sh : shCoeffs_)
            sh.download(stream);
    }

    void clearDevice()
    {
        bodies_.clearDevice();
        ephs_.clearDevice();
        cs_.clearDevice();
        tree_.clearDevice();
        rotations_.clearDevice();
        for (auto& sh : shCoeffs_)
            sh.clearDevice();
        shCoeffsRefs_.clearDevice();
        for (auto& atm : atmospheres_)
            atm.clearDevice();
        for (auto& atm : nrlAtmospheres_)
            atm.clearDevice();
        earthCorrections_.clearDevice();
        atmospheresRefs_.clearDevice();
        nrlAtmospheresRefs_.clearDevice();
        flattenings_.clearDevice();
        orientationTargets_.clearDevice();
        
        native_.pinned   = EphNativeCacheT();
        native_.variable = EphNativeCacheT();
        resolved_.posVariable.clear();
        resolved_.posPinned.clear();
        resolved_.accScratch.clear();
        resolved_.shapeAccScratch.clear();
        resolved_.correctionScratchSI = {};
        resolved_.forceReduction = ForceReductionPlan{};
        resolved_.velVariable.clear();
        resolved_.velPinned.clear();
        resolved_.omegaVariable.clear();
        resolved_.omegaPinned.clear();
        resolved_.occVariable.clear();
        resolved_.occPinned.clear();
        resolved_.nrlInputs.clear();
        resolved_.nrlDensity.clear();
        rotation_.variable.clear();
        rotation_.pinned.clear();
    }

    NaifIdArray& bodies() { return bodies_; }
    const NaifIdArray& bodies() const { return bodies_; }

    earthcorrections::View earthCorrections() const
    { return earthCorrections_.hostRef(); }

    void assertEarthCorrectionsEpochCoverageEt(Real lo, Real hi) const
    { earthCorrections_.assertEpochCoverageEt(lo, hi); }

    EphT& ephemeris() { return ephs_; }
    const EphT& ephemeris() const { return ephs_; }

    CT& constants() { return cs_; }
    const CT& constants() const { return cs_; }

    GTreeT& gravityTree() { return tree_; }
    const GTreeT& gravityTree() const { return tree_; }

    RotationsT& rotations() { return rotations_; }
    const RotationsT& rotations() const { return rotations_; }

    SHCoeffsT& shCoeffs(idx_t bodyIdx) { return shCoeffs_[bodyIdx]; }
    const SHCoeffsT& shCoeffs(idx_t bodyIdx) const
    {
        return shCoeffs_[bodyIdx];
    }

    AtmT::GRef atmosphere(idx_t bodyIdx) const
    {
        return atmospheres_[bodyIdx].deviceRef();
    }

    idx_t nAtmosphereSegments(idx_t bodyIdx) const
    {
        return atmospheres_[bodyIdx].nSegments();
    }

    atmosphere::ExpBlock expBlock(idx_t bodyIdx) const
    {
        return atmospheres_[bodyIdx].block0();
    }

    bool isNrlmsise00(idx_t bodyIdx) const
    {
        return bodyIdx < nrlAtmospheres_.size()
            && nrlAtmospheres_[bodyIdx].active();
    }

    atmosphere::NrlWeatherArrayT::GRef nrlWeather(idx_t bodyIdx) const
    {
        return nrlAtmospheres_[bodyIdx].deviceWeather();
    }

    atmosphere::NrlSettings nrlSettings(idx_t bodyIdx) const
    {
        return nrlAtmospheres_[bodyIdx].settings();
    }

    atmosphere::NrlInputArrayT::GRef nrlInputs(idx_t bodyIdx) const
    {
        PARAHPOP_ASSERT(bodyIdx < resolved_.nrlInputs.size()
                && resolved_.nrlInputs[bodyIdx].size() > 0,
            "NRLMSISE-00 input cache is not allocated for this body.");
        return deviceCacheRef_<atmosphere::NRL_INPUT_COMPONENTS>(
            resolved_.nrlInputs[bodyIdx]);
    }

    feta::scalar::Array<Real>::GRef::HandleT nrlDensity(idx_t bodyIdx) const
    {
        PARAHPOP_ASSERT(bodyIdx < resolved_.nrlDensity.size()
                && resolved_.nrlDensity[bodyIdx].size() > 0,
            "NRLMSISE-00 density cache is not allocated for this body.");
        feta::scalar::Array<Real>::GRef::HandleT h;
        h.ptr = resolved_.nrlDensity[bodyIdx].data();
        return h;
    }

    Real flattening(idx_t bodyIdx) const
    {
        return flattenings_.hostRef()[bodyIdx];
    }

    NaifId orientationTarget(idx_t bodyIdx) const
    {
        return orientationTargets_.hostRef()[bodyIdx];
    }

    bool anyOcculting() const { return anyOcculting_; }

    bool anyAtmosphere() const
    {
        for (const auto& atm : atmospheres_) {
            if (atm.active())
                return true;
        }
        for (const auto& atm : nrlAtmospheres_)
            if (atm.active()) return true;
        return false;
    }

    bool anyNrlmsise00() const
    {
        for (const auto& atm : nrlAtmospheres_)
            if (atm.active()) return true;
        return false;
    }

    std::vector<SHCoeffsT> releaseShCoeffs() { return std::move(shCoeffs_); }

    EphNativeCacheT& ephNativePinned() { return native_.pinned; }
    const EphNativeCacheT& ephNativePinned() const
    {
        return native_.pinned;
    }
    EphNativeCacheT& ephNativeVariable() { return native_.variable; }
    const EphNativeCacheT& ephNativeVariable() const
    {
        return native_.variable;
    }

    bool nativeSlotUnused(idx_t b) const
    {
        return b < native_.unusedSlotMask.size() && native_.unusedSlotMask[b];
    }
    
    idx_t nUnusedNativeSlots() const
    {
        idx_t n = 0;
        for (char c : native_.unusedSlotMask)
            n += (c != 0);
        return n;
    }

    struct EphCacheGeometry {
        idx_t       resolvedSlots = 0; ///< one resolved slot per ACTIVE body
        idx_t       nativeUnits   = 0; ///< native slots (active + chain parents)
        idx_t       nSamples      = 0; ///< per-sample cache length
        std::size_t deviceBytes   = 0; ///< realized device bytes across all tiers
    };

    EphCacheGeometry deviceCacheGeometry() const
    {
        EphCacheGeometry g;
        g.resolvedSlots = static_cast<idx_t>(resolved_.posVariable.size());
        g.nativeUnits   = native_.pinned.nBodyUnits();
        g.nSamples      = native_.pinned.nSamples();

        std::size_t bytes = 0;
        
        for (idx_t b = 0; b < native_.pinned.nBodyUnits(); b++)
            bytes += static_cast<std::size_t>(
                         native_.pinned.slot(b).totalSize()) * sizeof(Real);
        for (idx_t b = 0; b < native_.variable.nBodyUnits(); b++)
            bytes += static_cast<std::size_t>(
                         native_.variable.slot(b).totalSize()) * sizeof(Real);
        
        const auto sumVec = [&bytes](const auto& vec) {
            for (const auto& c : vec)
                bytes += static_cast<std::size_t>(c.totalSize()) * sizeof(Real);
        };
        sumVec(resolved_.posVariable);   sumVec(resolved_.posPinned);
        sumVec(resolved_.accScratch);    sumVec(resolved_.shapeAccScratch);
        sumVec(resolved_.correctionScratchSI);
        bytes += static_cast<std::size_t>(resolved_.forceReduction.terms.size())
            * sizeof(accelerations::kernel::BodyAccTerm);
        sumVec(rotation_.variable);  sumVec(rotation_.pinned);
        sumVec(resolved_.velVariable);   sumVec(resolved_.velPinned);
        sumVec(resolved_.omegaVariable); sumVec(resolved_.omegaPinned);
        sumVec(resolved_.occVariable);   sumVec(resolved_.occPinned);
        sumVec(resolved_.nrlInputs);      sumVec(resolved_.nrlDensity);
        g.deviceBytes = bytes;
        return g;
    }

    EphNativeCacheHostT& ephNativePinnedHost()
    {
        return native_.pinnedHost;
    }
    const EphNativeCacheHostT& ephNativePinnedHost() const
    {
        return native_.pinnedHost;
    }
    EphNativeCacheHostT& ephNativeVariableHost()
    {
        return native_.variableHost;
    }
    const EphNativeCacheHostT& ephNativeVariableHost() const
    {
        return native_.variableHost;
    }

    typename VecCacheT::GRef ephVariable(idx_t i) const
    {
        PARAHPOP_ASSERT(i < resolved_.posVariable.size(),
            "ephemeris variable cache not allocated for body index "
                + std::to_string(i));
        return deviceCacheRef_<3>(resolved_.posVariable[i]);
    }

    typename VecCacheT::GRef bodyAccScratch(idx_t i) const
    {
        PARAHPOP_ASSERT(i < resolved_.accScratch.size(),
            "body-acceleration scratch not allocated for body index "
                + std::to_string(i));
        return deviceCacheRef_<3>(resolved_.accScratch[i]);
    }

    typename VecCacheT::GRef shapeAccScratch(idx_t i) const
    {
        PARAHPOP_ASSERT(i < resolved_.shapeAccScratch.size()
                && resolved_.shapeAccScratch[i].size() > 0,
            "shape-acceleration scratch not allocated for body index "
                + std::to_string(i));
        return deviceCacheRef_<3>(resolved_.shapeAccScratch[i]);
    }

    // Slots: solid tides, ocean tides, relativity. Disabled slots are empty.
    typename VecCacheT::GRef correctionScratchSI(idx_t term) const
    {
        PARAHPOP_ASSERT(term < 3, "Invalid correction scratch slot");
        if (resolved_.correctionScratchSI[term].size() == 0) return {};
        return deviceCacheRef_<3>(resolved_.correctionScratchSI[term]);
    }

    bool needsTidePosition(idx_t body) const
    {
        const NaifId id = bodies_.hostRef()[body];
        return earthCorrections_.hostRef().flags.solidEarthTides
            && (id == NaifId{10} || id == NaifId{301});
    }

    void prepareForceReduction(const std::vector<idx_t>& activeBodies,
        const std::vector<idx_t>& shapeBodies, const cudaStream_t& stream) const
    {
        auto& plan = resolved_.forceReduction;
        if (plan.initialized && plan.activeBodies == activeBodies
            && plan.shapeBodies == shapeBodies)
            return;

        if (activeBodies.empty()) {
            plan.terms = BodyAccTermsArrayT::flexible();
        } else {
            BodyAccTermsArrayT terms(static_cast<idx_t>(activeBodies.size()),
                accelerations::kernel::BodyAccTerm{});
            auto href = terms.hostRef();
            for (idx_t t = 0; t < activeBodies.size(); ++t) {
                const idx_t bodyIdx = activeBodies[t];
                accelerations::kernel::BodyAccTerm term;
                term.base = bodyAccScratch(bodyIdx);
                term.hasShape = std::find(shapeBodies.begin(), shapeBodies.end(),
                                    bodyIdx) != shapeBodies.end();
                if (term.hasShape)
                    term.shape = shapeAccScratch(bodyIdx);
                href[t] = term;
            }
            terms.upload(stream);
            plan.terms = std::move(terms);
        }
        plan.activeBodies = activeBodies;
        plan.shapeBodies = shapeBodies;
        plan.initialized = true;
    }

    BodyAccTermsArrayT::GRef::HandleT forceReductionTerms() const
    {
        if (resolved_.forceReduction.terms.size() == 0)
            return {};
        return resolved_.forceReduction.terms.deviceRef().handle();
    }

    typename VecCacheT::GRef ephPinned(idx_t i) const
    {
        PARAHPOP_ASSERT(i < resolved_.posPinned.size(),
            "ephemeris pinned cache not allocated for body index "
                + std::to_string(i));
        return deviceCacheRef_<3>(resolved_.posPinned[i]);
    }

    typename VecCacheT::GRef velVariable(idx_t i) const
    {
        PARAHPOP_ASSERT(i < resolved_.velVariable.size(),
            "velocity variable cache not allocated for body index "
                + std::to_string(i));
        return deviceCacheRef_<3>(resolved_.velVariable[i]);
    }

    typename VecCacheT::GRef velPinned(idx_t i) const
    {
        PARAHPOP_ASSERT(i < resolved_.velPinned.size(),
            "velocity pinned cache not allocated for body index "
                + std::to_string(i));
        return deviceCacheRef_<3>(resolved_.velPinned[i]);
    }

    typename VecCacheT::GRef omegaVariable(idx_t i) const
    {
        PARAHPOP_ASSERT(i < resolved_.omegaVariable.size(),
            "omega variable cache not allocated for body index "
                + std::to_string(i));
        return deviceCacheRef_<3>(resolved_.omegaVariable[i]);
    }

    typename VecCacheT::GRef omegaPinned(idx_t i) const
    {
        PARAHPOP_ASSERT(i < resolved_.omegaPinned.size(),
            "omega pinned cache not allocated for body index "
                + std::to_string(i));
        return deviceCacheRef_<3>(resolved_.omegaPinned[i]);
    }

    typename ScalarCacheT::GRef::HandleT occVariable() const
    {
        PARAHPOP_ASSERT(!resolved_.occVariable.empty(),
            "occultation-factor variable cache not allocated — no "
            "occulting body was configured at allocation time");
        typename ScalarCacheT::GRef::HandleT h;
        h.ptr = resolved_.occVariable[0].data();
        return h;
    }

    typename ScalarCacheT::GRef::HandleT occPinned() const
    {
        PARAHPOP_ASSERT(!resolved_.occPinned.empty(),
            "occultation-factor pinned cache not allocated — no "
            "occulting body was configured at allocation time");
        typename ScalarCacheT::GRef::HandleT h;
        h.ptr = resolved_.occPinned[0].data();
        return h;
    }

    typename QuatCacheT::GRef quatVariable(idx_t i) const
    {
        PARAHPOP_ASSERT(i < rotation_.variable.size(),
            "quaternion variable cache not allocated for body index "
                + std::to_string(i));
        return deviceCacheRef_<4>(rotation_.variable[i]);
    }

    typename QuatCacheT::GRef quatPinned(idx_t i) const
    {
        PARAHPOP_ASSERT(i < rotation_.pinned.size(),
            "quaternion pinned cache not allocated for body index "
                + std::to_string(i));
        return deviceCacheRef_<4>(rotation_.pinned[i]);
    }

    typename QuatCacheT::GRef quatVariableHost(idx_t i) const
    {
        PARAHPOP_ASSERT(i < rotation_.variableHost.size(),
            "host quaternion variable cache not allocated for body index "
                + std::to_string(i));
        return hostCacheRef_<4>(rotation_.variableHost[i]);
    }

    typename QuatCacheT::GRef quatPinnedHost(idx_t i) const
    {
        PARAHPOP_ASSERT(i < rotation_.pinnedHost.size(),
            "host quaternion pinned cache not allocated for body index "
                + std::to_string(i));
        return hostCacheRef_<4>(rotation_.pinnedHost[i]);
    }

    VecCacheT& ephCache([[maybe_unused]] const idx_t& i = 0)
    {
        PARAHPOP_ASSERT(!hostScratch_.slot.empty(),
            "host ephemeris scratch not allocated — call allocateEphCache "
            "with CacheTarget::HOST before using ephCache");
        return hostScratch_.slot[0];
    }
    const VecCacheT& ephCache([[maybe_unused]] const idx_t& i = 0) const
    {
        PARAHPOP_ASSERT(!hostScratch_.slot.empty(),
            "host ephemeris scratch not allocated — call allocateEphCache "
            "with CacheTarget::HOST before using ephCache");
        return hostScratch_.slot[0];
    }

    void allocateEphCache(const idx_t& nSamples, const idx_t& numRotations = 0,
        CacheTarget target                 = CacheTarget::DEVICE,
        const std::vector<NaifId>& centres = {})
    {
        
        releaseEphCache(target);

        PARAHPOP_ASSERT(ephs_.nBodyUnits() >= bodies_.size(),
            "EphUnit body count must cover at least the active bodies");

        const bool wantHost   = (target == CacheTarget::HOST
            || target == CacheTarget::BOTH);
        const bool wantDevice = (target == CacheTarget::DEVICE
            || target == CacheTarget::BOTH);

        if (wantDevice)
            ephs_.upload();

        if (wantDevice)
            allocateDeviceCache_(nSamples, numRotations, centres);
        if (wantHost)
            allocateHostCache_(nSamples, numRotations);
    }

    void releaseEphCache(CacheTarget target = CacheTarget::BOTH)
    {
        const bool wantHost   = (target == CacheTarget::HOST
            || target == CacheTarget::BOTH);
        const bool wantDevice = (target == CacheTarget::DEVICE
            || target == CacheTarget::BOTH);

        if (wantDevice) {
            native_.pinned   = EphNativeCacheT();
            native_.variable = EphNativeCacheT();
            resolved_.posVariable.clear();
            resolved_.posPinned.clear();
            resolved_.velVariable.clear();
            resolved_.velPinned.clear();
            resolved_.omegaVariable.clear();
            resolved_.omegaPinned.clear();
            resolved_.occVariable.clear();
            resolved_.occPinned.clear();
            resolved_.nrlInputs.clear();
            resolved_.nrlDensity.clear();
            rotation_.variable.clear();
            rotation_.pinned.clear();
            resolved_.accScratch.clear();
            resolved_.shapeAccScratch.clear();
            resolved_.correctionScratchSI = {};
            resolved_.forceReduction = ForceReductionPlan{};
        }
        if (wantHost) {
            native_.pinnedHost   = EphNativeCacheHostT();
            native_.variableHost = EphNativeCacheHostT();
            rotation_.variableHost.clear();
            rotation_.pinnedHost.clear();
            
            hostScratch_.slot.clear();
        }
    }

    GRef hostRef() const
    {
        return GRef::make(bodies_.hostRef(), ephs_.hostRef(), cs_.hostRef(),
            Orientations<false>(rotations_.hostRef()), tree_.hostRef(),
            shCoeffsRefs_.hostRef(), native_.pinnedHost.ref(),
            native_.variableHost.ref(), atmospheresRefs_.hostRef(),
            nrlAtmospheresRefs_.hostRef(),
            flattenings_.hostRef(), orientationTargets_.hostRef(),
            earthCorrections_.hostRef());
    }

    GRef deviceRef() const
    {
        return GRef::make(bodies_.deviceRef(), ephs_.deviceRef(),
            cs_.deviceRef(), Orientations<false>(rotations_.deviceRef()),
            tree_.deviceRef(), shCoeffsRefs_.deviceRef(),
            native_.pinned.ref(), native_.variable.ref(),
            atmospheresRefs_.deviceRef(), nrlAtmospheresRefs_.deviceRef(),
            flattenings_.deviceRef(), orientationTargets_.deviceRef(),
            earthCorrections_.deviceRef());
    }

private:
    
    static bool preparePropagationData_(const InterfaceT& ienv)
    {
        const_cast<InterfaceT&>(ienv).preparePropagationData();
        return true;
    }

    static std::vector<AtmT> extractAtmospheres_(const InterfaceT& ienv)
    {
        const auto& atms = ienv.bodies().atmospheres();
        std::vector<AtmT> out;
        out.reserve(atms.size());
        for (idx_t i = 0; i < static_cast<idx_t>(atms.size()); i++) {
            const auto& segs = atms[i].segments();
            if (segs.empty()) {
                AtmT atm(idx_t{ 1 });
                atm.setSegment(0, Real{ 0 }, Real{ 0 }, Real{ 1 }, Real{ 0 });
                out.push_back(std::move(atm));
            } else {
                const idx_t n = static_cast<idx_t>(segs.size());
                AtmT atm(n);
                for (idx_t s = 0; s < n; s++) {
                    atm.setSegment(s, segs[s].rho0, segs[s].h0,
                        segs[s].scaleHeight, segs[s].hCutoff);
                }
                out.push_back(std::move(atm));
            }
        }
        return out;
    }

    static std::vector<NrlAtmT> extractNrlAtmospheres_(const InterfaceT& ienv)
    {
        const auto& configs = ienv.bodies().atmospheres();
        std::vector<NrlAtmT> out;
        out.reserve(configs.size());
        for (const auto& config : configs)
            out.emplace_back(config);
        return out;
    }

    static AtmsRefArrayT buildAtmospheresRefs_(
        const std::vector<AtmT>& atms, bool useDevice)
    {
        AtmsRefArrayT arr(atms.size(), AtmT::GRef{});
        auto href = arr.hostRef();
        for (idx_t i = 0; i < static_cast<idx_t>(atms.size()); i++) {
            href[i] = useDevice ? atms[i].deviceRef() : atms[i].hostRef();
        }
        return arr;
    }

    void rebuildAtmospheresRefs_(bool useDevice) const
    {
        if (atmospheres_.empty())
            return;
        auto href = atmospheresRefs_.hostRef();
        for (idx_t i = 0; i < static_cast<idx_t>(atmospheres_.size()); i++) {
            href[i] = useDevice ? atmospheres_[i].deviceRef()
                                : atmospheres_[i].hostRef();
        }
    }

    static NrlAtmsRefArrayT buildNrlAtmospheresRefs_(
        const std::vector<NrlAtmT>& atms, bool useDevice)
    {
        NrlAtmsRefArrayT arr(atms.size(), NrlAtmT::GRef{});
        auto href = arr.hostRef();
        for (idx_t i = 0; i < static_cast<idx_t>(atms.size()); ++i)
            href[i] = useDevice ? atms[i].deviceRef() : atms[i].hostRef();
        return arr;
    }

    void rebuildNrlAtmospheresRefs_(bool useDevice) const
    {
        if (nrlAtmospheres_.empty())
            return;
        auto href = nrlAtmospheresRefs_.hostRef();
        for (idx_t i = 0;
             i < static_cast<idx_t>(nrlAtmospheres_.size()); ++i) {
            href[i] = useDevice ? nrlAtmospheres_[i].deviceRef()
                                : nrlAtmospheres_[i].hostRef();
        }
    }

    static FlatArrayT extractFlattenings_(const InterfaceT& ienv)
    {
        const auto fref = ienv.bodies().flattenings().ref();
        const idx_t n   = static_cast<idx_t>(ienv.bodies().size());
        const idx_t m   = fref.size();
        FlatArrayT arr(n, Real{ 0 });
        auto href = arr.hostRef();
        for (idx_t i = 0; i < n; i++)
            href[i] = (i < m) ? static_cast<Real>(fref[i]) : Real{ 0 };
        return arr;
    }

    OrientTargetArrayT resolveOrientationTargets_(const InterfaceT& ienv) const
    {
        namespace orient  = interface::naming::orientations;
        const auto bref   = bodies_.hostRef();
        const idx_t n     = bref.size();
        const auto& names = ienv.bodies().orientationNames();
        const idx_t m     = static_cast<idx_t>(names.size());
        const auto loaded = [this](const orient::OrientationId& t) {
            return rotations_.hostRef().metadata().hasBody(
                static_cast<NaifId>(t));
        };
        OrientTargetArrayT arr(n, NaifId{ 0 });
        auto href = arr.hostRef();
        for (idx_t i = 0; i < n; i++) {
            const NaifId body      = bref[i];
            const std::string name = (i < m) ? names[i] : std::string{};
            href[i]                = static_cast<NaifId>(
                orient::resolveBodyOrientationTarget(body, name, loaded));
            
            if (href[i] != body) {
                const auto entry = orient::Registry::instance().findByTarget(
                    static_cast<orient::OrientationId>(href[i]));
                PARAHPOP_INFO("Body %s (%i) force-model rotation uses the %s "
                           "frame (orientation target %i), not its IAU "
                           "body-fixed default",
                    brie::gravity::Parser::parsedName(body).c_str(),
                    static_cast<int>(body),
                    entry ? entry->name.c_str() : "custom",
                    static_cast<int>(href[i]));
            }
        }
        return arr;
    }

    static OrientTargetArrayT defaultOrientationTargets_(
        const NaifIdArray& bodies)
    {
        const auto bref = bodies.hostRef();
        const idx_t n   = bref.size();
        OrientTargetArrayT arr(n, NaifId{ 0 });
        auto href = arr.hostRef();
        for (idx_t i = 0; i < n; i++)
            href[i] = bref[i];
        return arr;
    }

    static bool extractAnyOcculting_(const InterfaceT& ienv)
    {
        namespace ibody = interface::config::model::environment::body;
        constexpr NaifId SUN_NAIF_ID = 10;
        const auto& bodies = ienv.bodies();
        const auto flags   = bodies.features().flags().ref();
        bool any           = false;
        for (idx_t i = 0; i < bodies.size(); i++) {
            if (!flags.template get<ibody::OCCULTING>(i))
                continue;
            if (bodies.ids()[i] == SUN_NAIF_ID) {
                PARAHPOP_WARN(
                    "Body feature 'occulting' is set on the Sun and will "
                    "be ignored: the Sun cannot occult its own radiation.");
                continue;
            }
            any = true;
        }
        return any;
    }

    static std::vector<SHCoeffsT> extractAllShCoeffs_(const InterfaceT& ienv)
    {
        using SHInterfaceT
            = interface::config::model::environment::SphericalHarmonics;
        auto& bodies             = ienv.bodies();
        const auto& fieldConfigs = bodies.fieldConfigs();
        std::vector<SHCoeffsT> out;
        out.reserve(fieldConfigs.size());
        for (idx_t i = 0; i < fieldConfigs.size(); i++) {
            if (!fieldConfigs[i].files().empty()) {
                auto& shRef = const_cast<SHInterfaceT&>(fieldConfigs[i]);
                out.push_back(std::move(shRef.make().dump()));
            } else {
                out.push_back(SHCoeffsT());
            }
        }
        return out;
    }

    static SHRefsArrayT buildShCoeffsRefs_(
        const std::vector<SHCoeffsT>& shCoeffs, bool useDevice)
    {
        SHRefsArrayT arr(shCoeffs.size(), SHCoeffsT::GRef{});
        auto href = arr.hostRef();
        for (idx_t i = 0; i < shCoeffs.size(); i++) {
            href[i] = useDevice ? shCoeffs[i].deviceRef(true)
                                : shCoeffs[i].hostRef(true);
        }
        return arr;
    }

    void rebuildShCoeffsRefs_(bool useDevice) const
    {
        if (shCoeffs_.empty())
            return;
        auto href = shCoeffsRefs_.hostRef();
        for (idx_t i = 0; i < shCoeffs_.size(); i++) {
            href[i] = useDevice ? shCoeffs_[i].deviceRef(true)
                                : shCoeffs_[i].hostRef(true);
        }
    }

    void checkDeviceCacheBudget_(
        const idx_t& nSamples, const idx_t& numRotations) const
    {
        const idx_t nNative = ephs_.nBodyUnits();
        const idx_t nActive = bodies_.size();
        const std::size_t bytesNative
            = 2 * static_cast<std::size_t>(nNative) * nSamples * 6
              * sizeof(Real);
        const std::size_t bytesResolved
            = 2 * static_cast<std::size_t>(nActive) * nSamples * 3
              * sizeof(Real);
        const std::size_t bytesQuat
            = (numRotations > 0)
                  ? 2 * static_cast<std::size_t>(nActive) * nSamples * 4
                        * sizeof(Real)
                  : 0;
        

        
        const std::size_t bytesAccScratch
            = static_cast<std::size_t>(nActive) * nSamples * 3
              * sizeof(Real);
        
        constexpr std::size_t bytesShapeScratch = 0;
        const auto correctionFlags = earthCorrections_.hostRef().flags;
        const unsigned correctionCount = unsigned(correctionFlags.solidEarthTides)
            + unsigned(correctionFlags.oceanTides) + unsigned(correctionFlags.relativity);
        const std::size_t bytesCorrectionScratch
            = correctionCount * static_cast<std::size_t>(nSamples) * 3 * sizeof(Real);
        constexpr std::size_t bytesReductionTerms = 0;
        
        const std::size_t bytesVel
            = anyAtmosphere()
                  ? 4 * static_cast<std::size_t>(nActive) * nSamples * 3
                        * sizeof(Real)
                  : 0;
        
        const std::size_t bytesOcc
            = anyOcculting_
                  ? 2 * static_cast<std::size_t>(nSamples) * sizeof(Real)
                  : 0;
        std::size_t nNrlBodies = 0;
        for (const auto& nrl : nrlAtmospheres_)
            nNrlBodies += nrl.active() ? 1u : 0u;
        const std::size_t bytesNrl = nNrlBodies
            * static_cast<std::size_t>(nSamples)
            * (atmosphere::NRL_INPUT_COMPONENTS + 1) * sizeof(Real);
        const std::size_t bytesTotal
            = bytesNative + bytesResolved + bytesQuat + bytesAccScratch
              + bytesShapeScratch + bytesCorrectionScratch + bytesReductionTerms
              + bytesVel + bytesOcc + bytesNrl;

        std::size_t freeBytes = 0, totalBytes = 0;
        PARAHPOP_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));

        PARAHPOP_ASSERT(bytesTotal * 10 <= freeBytes * 6,
            "Ephemeris cache allocation would exceed 60% of free GPU memory: "
            "required=" + std::to_string(bytesTotal)
                + " bytes (native=" + std::to_string(bytesNative)
                + " + resolved=" + std::to_string(bytesResolved)
                + " + quat=" + std::to_string(bytesQuat)
                + " + accScratch=" + std::to_string(bytesAccScratch)
                + " + shapeScratch=" + std::to_string(bytesShapeScratch)
                + " + correctionScratch=" + std::to_string(bytesCorrectionScratch)
                + " + reductionTerms=" + std::to_string(bytesReductionTerms)
                + " + vel=" + std::to_string(bytesVel)
                + " + occ=" + std::to_string(bytesOcc)
                + " + nrl=" + std::to_string(bytesNrl)
                + "), free=" + std::to_string(freeBytes)
                + " bytes, total=" + std::to_string(totalBytes)
                + " bytes. Reduce nSamples (currently "
                + std::to_string(nSamples)
                + ") or split into multiple simulations.");
    }

    void allocateDeviceCache_(const idx_t& nSamples, const idx_t& numRotations,
        const std::vector<NaifId>& centres = {})
    {
        const idx_t nActive = bodies_.size();
        const bool  anyAtm  = anyAtmosphere();

        checkDeviceCacheBudget_(nSamples, numRotations);

        native_.pinned   = ephs_.makeCache<6>(nSamples);
        native_.variable = ephs_.makeCache<6>(nSamples);

        resolved_.posVariable.reserve(nActive);
        resolved_.posPinned.reserve(nActive);
        resolved_.accScratch.reserve(nActive);
        resolved_.shapeAccScratch.reserve(nActive);
        for (idx_t i = 0; i < nActive; i++) {
            resolved_.posVariable.emplace_back(nSamples);
            resolved_.posPinned.emplace_back(nSamples);
            resolved_.accScratch.emplace_back(nSamples);
            // SH now accumulates into this body's accScratch in order.
            resolved_.shapeAccScratch.emplace_back();
        }
        const auto flags = earthCorrections_.hostRef().flags;
        const bool enabled[] = { flags.solidEarthTides, flags.oceanTides, flags.relativity };
        for (idx_t term = 0; term < 3; ++term)
            if (enabled[term]) resolved_.correctionScratchSI[term] = VecCacheStoreT(nSamples);

        if (numRotations > 0) {
            rotation_.variable.reserve(nActive);
            rotation_.pinned.reserve(nActive);
            for (idx_t i = 0; i < nActive; i++) {
                rotation_.variable.emplace_back(nSamples);
                rotation_.pinned.emplace_back(nSamples);
            }
        }

        if (anyAtm) {
            resolved_.velVariable.reserve(nActive);
            resolved_.velPinned.reserve(nActive);
            resolved_.omegaVariable.reserve(nActive);
            resolved_.omegaPinned.reserve(nActive);
            for (idx_t i = 0; i < nActive; i++) {
                resolved_.velVariable.emplace_back(nSamples);
                resolved_.velPinned.emplace_back(nSamples);
                resolved_.omegaVariable.emplace_back(nSamples);
                resolved_.omegaPinned.emplace_back(nSamples);
            }
        }

        resolved_.nrlInputs.reserve(nActive);
        resolved_.nrlDensity.reserve(nActive);
        for (idx_t i = 0; i < nActive; ++i) {
            if (isNrlmsise00(i)) {
                resolved_.nrlInputs.emplace_back(nSamples);
                resolved_.nrlDensity.emplace_back(nSamples);
            } else {
                resolved_.nrlInputs.emplace_back();
                resolved_.nrlDensity.emplace_back();
            }
        }

        if (anyOcculting_) {
            resolved_.occVariable.emplace_back(nSamples);
            resolved_.occPinned.emplace_back(nSamples);
        }

        computeUnusedSlotMask_(centres);
    }

    void computeUnusedSlotMask_(const std::vector<NaifId>& centres = {})
    {
        const idx_t nNative = ephs_.nBodyUnits();
        native_.unusedSlotMask.assign(nNative, 1); 
        if (nNative == 0)
            return;
        const auto traverser = ephs_.hostRef().traverser();

        auto clearChain = [&](idx_t start) {
            idx_t p = start;
            while (p != traverser.size() && native_.unusedSlotMask[p] != 0) {
                native_.unusedSlotMask[p] = 0;
                p                       = traverser[p].centerPos_;
            }
        };

        auto bref = bodies_.hostRef();
        for (idx_t i = 0; i < bref.size(); i++) {
            const NaifId id = bref[i];
            if (id == 0) 
                continue;
            clearChain(traverser.at(id).pos_);
        }

        // Tide-driving bodies must be current at every RK substage even when
        // they have no point-gravity feature and were loaded only for tides.
        if (earthCorrections_.tidesActive())
            for (NaifId id : {NaifId{10}, NaifId{301}, NaifId{399}})
                clearChain(traverser.at(id).pos_);

        for (const NaifId& id : centres) {
            if (id == 0)
                continue;
            for (idx_t p = 0; p < traverser.size(); p++) {
                if (traverser[p].id_ == id) {
                    clearChain(p);
                    break;
                }
            }
        }
    }

    void allocateHostCache_(
        const idx_t& nSamples, const idx_t& numRotations)
    {
        const idx_t nActive = bodies_.size();
        native_.pinnedHost   = ephs_.template makeCache<6,
            feta::core::memory::Device::CPU>(nSamples);
        native_.variableHost = ephs_.template makeCache<6,
            feta::core::memory::Device::CPU>(nSamples);

        if (numRotations > 0) {
            rotation_.variableHost.reserve(nActive);
            rotation_.pinnedHost.reserve(nActive);
            for (idx_t i = 0; i < nActive; i++) {
                rotation_.variableHost.emplace_back(nSamples);
                rotation_.pinnedHost.emplace_back(nSamples);
            }
        }

    }

    template<idx_t N>
    static typename feta::vector::Array<Real, N>::GRef deviceCacheRef_(
        const feta::core::memory::Container<Real,
            feta::core::memory::Device::CUDA_DEVICE, N>& c)
    {
        typename feta::vector::Array<Real, N>::GRef r;
        r.data_      = c.data();
        r.nVecs_     = c.size();
        r.dimOffset_ = c.size();
        r.tex_       = 0;
        r.texOffset_ = 0;
        return r;
    }

    template<idx_t N>
    static typename feta::vector::Array<Real, N>::GRef hostCacheRef_(
        const feta::core::memory::Container<Real,
            feta::core::memory::Device::CPU, N>& c)
    {
        typename feta::vector::Array<Real, N>::GRef r;
        r.data_      = c.data();
        r.nVecs_     = c.size();
        r.dimOffset_ = c.size();
        r.tex_       = 0;
        r.texOffset_ = 0;
        return r;
    }

    NaifIdArray bodies_;
    
    EphT ephs_;
    
    CT cs_;
    
    GTreeT tree_;
    
    RotationsT rotations_ = RotationsT::empty();
    
    std::vector<SHCoeffsT> shCoeffs_;
    
    mutable SHRefsArrayT shCoeffsRefs_;
    
    std::vector<AtmT> atmospheres_;
    
    std::vector<NrlAtmT> nrlAtmospheres_;
    earthcorrections::EarthCorrections earthCorrections_;
    
    mutable AtmsRefArrayT atmospheresRefs_ = AtmsRefArrayT{ 0, AtmT::GRef{} };
    
    mutable NrlAtmsRefArrayT nrlAtmospheresRefs_
        = NrlAtmsRefArrayT{ 0, NrlAtmT::GRef{} };
    
    FlatArrayT flattenings_ = FlatArrayT{ 0, Real{ 0 } };
    
    OrientTargetArrayT orientationTargets_ = OrientTargetArrayT{ 0, NaifId{ 0 } };

    bool anyOcculting_ = false;

    struct NativeEphemerisCache {
        EphNativeCacheT     pinned       = EphNativeCacheT();
        EphNativeCacheT     variable     = EphNativeCacheT();
        EphNativeCacheHostT pinnedHost   = EphNativeCacheHostT();
        EphNativeCacheHostT variableHost = EphNativeCacheHostT();
        
        std::vector<char>   unusedSlotMask = {};
    };

    struct ForceReductionPlan {
        BodyAccTermsArrayT terms = BodyAccTermsArrayT::flexible();
        std::vector<idx_t> activeBodies = {};
        std::vector<idx_t> shapeBodies = {};
        bool initialized = false;
    };

    struct ResolvedEphemerisCache {
        std::vector<VecCacheStoreT>    posVariable   = {};
        std::vector<VecCacheStoreT>    posPinned     = {};
        std::vector<VecCacheStoreT>    accScratch    = {};
        
        std::vector<VecCacheStoreT>    shapeAccScratch = {};
        
        std::array<VecCacheStoreT, 3> correctionScratchSI;
        mutable ForceReductionPlan    forceReduction;
        std::vector<VecCacheStoreT>    velVariable   = {};
        std::vector<VecCacheStoreT>    velPinned     = {};
        std::vector<VecCacheStoreT>    omegaVariable = {};
        std::vector<VecCacheStoreT>    omegaPinned   = {};
        std::vector<ScalarCacheStoreT> occVariable   = {};
        std::vector<ScalarCacheStoreT> occPinned     = {};
        std::vector<atmosphere::NrlInputCacheStoreT> nrlInputs = {};
        std::vector<ScalarCacheStoreT> nrlDensity = {};
    };

    struct BodyRotationCache {
        std::vector<QuatCacheStoreT>     variable     = {};
        std::vector<QuatCacheStoreT>     pinned       = {};
        std::vector<QuatCacheHostStoreT> variableHost = {};
        std::vector<QuatCacheHostStoreT> pinnedHost   = {};
    };

    struct HostEphemerisScratch {
        std::vector<VecCacheT> slot = {};
    };

    NativeEphemerisCache   native_;
    ResolvedEphemerisCache resolved_;
    BodyRotationCache      rotation_;
    HostEphemerisScratch   hostScratch_;
};

using Env = model::environment::Environment;

namespace kernel {

using EnvKernelLaunch = KernelLaunchTraits<128, 8>;

__global__ void coiResolveKernel(feta::vector::Array<Real, 3>::GRef ephCache,
    GRID_CONSTANT() EphNativeCacheT::GRef cache, GRID_CONSTANT() NaifId target,
    GRID_CONSTANT() feta::scalar::Array<NaifId>::GRef::HandleT centers,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT epochs,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated);

__global__ void coiResolveVelocityKernel(
    feta::vector::Array<Real, 3>::GRef velCache,
    GRID_CONSTANT() EphNativeCacheT::GRef cache, GRID_CONSTANT() NaifId target,
    GRID_CONSTANT() feta::scalar::Array<NaifId>::GRef::HandleT centers,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT epochs,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated);

__global__ void coiResolveStateKernel(
    feta::vector::Array<Real, 3>::GRef ephCache,
    feta::vector::Array<Real, 3>::GRef velCache,
    GRID_CONSTANT() EphNativeCacheT::GRef cache, GRID_CONSTANT() NaifId target,
    GRID_CONSTANT() feta::scalar::Array<NaifId>::GRef::HandleT centers,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT epochs,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated);

__global__ void omegaResolveKernel(
    feta::vector::Array<Real, 3>::GRef omegaCache,
    GRID_CONSTANT() OrientationsT orientations, GRID_CONSTANT() NaifId target,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT epochs,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated);

template<idx_t Dim>
__global__ void copyNativeCacheKernel(
    typename feta::vector::Array<Real, Dim>::GRef pinnedSlot,
    GRID_CONSTANT() typename feta::vector::Array<Real, Dim>::GRef variableSlot,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated);

__global__ void cacheRotation(feta::vector::Array<Real, 4>::GRef rotCache,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT epochs,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated,
    GRID_CONSTANT() OrientationsT orientations, GRID_CONSTANT() NaifId bodyId);

} // namespace kernel
} // namespace environment
} // namespace model
} // namespace paraHPOP
