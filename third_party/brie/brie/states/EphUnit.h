#pragma once

#ifndef BRIE_CPU_ONLY
#include <cuda.h>
#endif

#include <feta/feta.h>
#include <parm/util/Observable.h>
#include <sstream>
#include <string>

#include "brie/states/RefEphUnit.h"
#include "brie/states/metadata/EphUnit.h"
#include "brie/typedefs.h"
#include "brie/util/cbor.h"
#include "brie/util/json.h"

#include "brie/util/DeviceError.h"
#include "brie/util/throw.h"

namespace brie {
namespace states {

/* Forward declaration so `makeCache` can return an EphCache; the full
 * definition lives in `brie/states/EphCache.h`. */
template<bool UseTexture, idx_t Dim,
    feta::core::memory::Device DeviceT>
class EphCache;

/**
 * @brief Ephemeris Unit - class to manage ephemeris evaluations
 *
 */
template<bool UseTexture>
class EphUnit : public parm::util::Observable<EphUnit<UseTexture>> {
    using MetadataT = metadata::EphUnit;
    using DataT     = feta::scalar::texture::Array<Real, UseTexture>;

public:
    /** @brief Stream and texture types */
    using StreamT = typename DataT::StreamT;
    using TexT    = typename DataT::TexT;

    /** @brief reference types */
    template<bool work, bool MaybeVolatile = false>
    using Ref = RefEphUnit<work, UseTexture, MaybeVolatile>;
    /** @brief global reference type */
    using GRef = Ref<false>;
    /** @brief work reference type */
    using WRef = Ref<true>;
    /** @brief Volatile work reference */
    using VolatileRef = Ref<true, true>;

    /** @brief The chain interpolator type */
    using ChainInterpolatorT = core::EphChainInterpolator<UseTexture>;

    /** @brief The traverser type */
    using TraverserT = core::Traverser;

    /** @brief Factory method to create an empty eph unit object */
    static EphUnit<UseTexture> empty()
    {
        MetadataT metadata(0);
        DataT data(0, 0.0);
        return EphUnit<UseTexture>(std::move(metadata), std::move(data));
    }

    /**
     * @brief Factory method to create a `brie::EphUnit` object from a .brie
     * file, or a `nlohmann::json::array_t` of .brie files
     *
     * @param brieFile file(s) to be read
     * @param NaifIds (Optional) subset of NAIF ids to be loaded from the .brie
     * file
     * @param pahts (Optional) file path manager
     * @return EphUnit
     */
    static EphUnit<UseTexture> fromBrie(const nlohmann::json& brieFile,
        const feta::scalar::Array<NaifId>& NaifIDs
        = feta::scalar::Array<NaifId>(0),
        const util::Paths& paths = util::paths::brieDefaultPath())
    {
        if (brieFile.is_string())
            if (NaifIDs.size() == 0)
                return fromBrieFileString_(brieFile, paths);
            else
                return fromBrieFileString_(brieFile, paths)
                    .makeFullSubset(NaifIDs);
        else if (brieFile.is_array())
            if (NaifIDs.size() == 0)
                return fromBrieFileArray_(brieFile, paths);
            else
                return fromBrieFileArray_(brieFile, paths)
                    .makeFullSubset(NaifIDs);
        else
            BRIE_THROW(std::runtime_error,
                "Brie files must be given either as a "
                "`nlohmann::json::string_t` or as a `nlohmann::json::array_t` "
                "of strings.")
    }

    /**
     * @brief Factory method to create a `brie::EphUnit` object from a
     * `nlohmann::json` object
     *
     * @param j `nlohmann::json` object
     * @param NaifIds (Optional) subset of NAIF ids to be loaded from the .brie
     * file
     * @return EphUnit
     */
    static EphUnit<UseTexture> fromJson(const nlohmann::json& j,
        const feta::scalar::Array<NaifId>& NaifIDs
        = feta::scalar::Array<NaifId>(0))
    {
        if (j.is_object())
            if (NaifIDs.size() == 0)
                return fromJson_(j);
            else
                return fromJson_(j).makeFullSubset(NaifIDs);
        else if (j.is_array())
            if (NaifIDs.size() == 0)
                return fromJsonArray_(j);
            else
                return fromJsonArray_(j).makeFullSubset(NaifIDs);
        else
            BRIE_THROW(std::runtime_error,
                "`nlohman::json` data must be given either as "
                "`nlohmann::json::object_t` or as a `nlohmann::json::array_t`");
    }

    /**
     * @brief Copy constructor is forbidden
     *
     */
    EphUnit(EphUnit& other)       = delete;
    EphUnit(const EphUnit& other) = delete;

    /**
     * @brief Move constructor is allowed - from data members
     *
     */
    EphUnit(MetadataT&& metadata, DataT&& data)
        : metadata_{ std::move(metadata) }
        , data_{ std::move(data) }
    {
    }

    /**
     * @brief Move constructor is allowed - from other EphUnit
     *
     */
    EphUnit(EphUnit&& other)
        : metadata_{ std::move(other.metadata_) }
        , data_{ std::move(other.data_) }
        /* transfer host/device chain state without reallocating */
        , hostChain_{ std::move(other.hostChain_) }
        , hostTraverser_{ std::move(other.hostTraverser_) }
        , hostChainReady_{ other.hostChainReady_ }
#ifndef BRIE_CPU_ONLY
        , deviceChain_{ std::move(other.deviceChain_) }
        , deviceTraverser_{ std::move(other.deviceTraverser_) }
        , deviceChainReady_{ other.deviceChainReady_ }
#endif
    {
        /* after move the source must no longer report ready */
        other.hostChainReady_ = false;
#ifndef BRIE_CPU_ONLY
        other.deviceChainReady_ = false;
#endif
        /* Inherit observers from `other` and notify them of our address. */
        this->finalizeMove(std::move(other));
        /* the moved-from chain objects are in a valid empty state after the
           std::move; their destructors will not free any cuda memory */
    }

    /**
     * @brief Basic constructor - host only
     *
     */
    EphUnit(const vecdim_t& Nbodies, const idx_t& totDataSize)
        : metadata_{ Nbodies }
        , data_{ totDataSize }
    {
    }

    /**
     * @brief Move assignment operator
     *
     */
    EphUnit<UseTexture>& operator=(EphUnit&& other)
    {
        if (this != &other) {
            /* Move the primary members first */
            metadata_ = std::move(other.metadata_);
            data_     = std::move(other.data_);

            /* Transfer prepared chains/traversers and readiness flags */
            hostChain_      = std::move(other.hostChain_);
            hostTraverser_  = std::move(other.hostTraverser_);
            hostChainReady_ = other.hostChainReady_;
#ifndef BRIE_CPU_ONLY
            deviceChain_      = std::move(other.deviceChain_);
            deviceTraverser_  = std::move(other.deviceTraverser_);
            deviceChainReady_ = other.deviceChainReady_;
#endif
            /* make sure the moved-from object no longer thinks it owns them */
            other.hostChainReady_ = false;
#ifndef BRIE_CPU_ONLY
            other.deviceChainReady_ = false;
#endif
            /* Inherit observers from `other` and notify them of our address. */
            this->finalizeMove(std::move(other));
        }
        return *this;
    }

    /**
     * @brief Read data and metadata from .brie file (Layout 1)
     *
     * @param brieEphUnit Data read from a .brie file
     *
     */
    void readBrie1(const nlohmann::json& brieEphUnit)
    {
        /* read metadata */
        metadata_.readBrie1(brieEphUnit);

        /* get reference to metadata and this data */
        MetadataT::GRef newMref = metadata_.hostRef();

        /* read data based on offsets */
        idx_t bodyCount = 0;
        for (auto bodyUnit : brieEphUnit) {
            /* offsets */
            std::copy(bodyUnit["data"].begin(), bodyUnit["data"].end(),
                data_.getHostData() + newMref.getDataOffset(bodyCount));
            /* advance body count */
            bodyCount++;
        }
    }

    /**
     * @brief Read data and metadata from .brie file (Layout 2)
     *
     * @param brieEphUnit Data read from a .brie file
     *
     */
    void readBrie2(const nlohmann::json& brieEphUnit)
    {
        /* read metadata */
        metadata_.readBrie2(brieEphUnit["metadata"]);

        /* read data */
        std::copy(brieEphUnit["data"].begin(), brieEphUnit["data"].end(),
            data_.getHostData());
    }

    /**
     * @brief Make a `brie::EphUnit` that contains ONLY of the given Naif IDs
     *
     * @warning A plain subset carries ONLY the listed bodies: target-center
     * chains are looked up by NaifId at evaluation time, so queries through
     * a missing center fail. Prefer `makeFullSubset`, which includes all
     * the required centers automatically.
     *
     * @param NaifIDs NAIF ids for which to create the subset
     * @param stream (Optional) CUDA stream for asynchronous memcpy
     * @return EphUnit New subset ephemeris unit
     *
     */
    EphUnit<UseTexture> makeSubset(
        const feta::scalar::Array<NaifId>& NaifIDs) const
    {
        /* Create metadata subset */
        MetadataT metadata = metadata_.makeSubset(NaifIDs);

        /* Create data subset */
        DataT data(metadata.totalDataSize());

        /* create ephs, read and return */
        EphUnit<UseTexture> ephs(std::move(metadata), std::move(data));
        copyTo(ephs);
        return ephs;
    }

    /**
     * @brief Make a `brie::EphUnit` that contains the given Naif IDs AND all
     * the parent bodies in the ephemeris tree, i.e. the necessary ones to
     * compute all the possible combinations target-centre of the given NAIF
     * IDs.
     *
     * @param NaifIDs NAIF ids for which to create the subset
     * @param stream (Optional) CUDA stream for asynchronous memcpy
     * @return EphUnit New subset ephemeris unit
     *
     */
    EphUnit<UseTexture> makeFullSubset(
        const feta::scalar::Array<NaifId>& NaifIDs) const
    {
        feta::scalar::Array<NaifId> trueIDs
            = metadata_.makeTrueNaifIDs(NaifIDs);
        return makeSubset(trueIDs);
    }

    /**
     * @brief Merge this `brie::EphUnit` with another `brie::EphUnit`.
     * The bodies in the other `brie::EphUnit` that are already found in this
     * `brie::EphUnit` are omitted from the merge.
     *
     * @param other Other `brie::EphUnit`
     * @param stream (Optional) CUDA stream for asynchronous memcpy
     * @return EphUnit New merged `brie::EphUnit`
     *
     */
    EphUnit merge(const EphUnit& other);

#ifndef BRIE_CPU_ONLY
    /**
     * @brief Copy the data in this ephemeris unit from host to device
     *
     * @param stream (Optional) CUDA stream for asynchronous copy
     *
     */
    void upload(const StreamT& stream = 0)
    {
        metadata_.upload(stream);
        data_.upload(stream);
        /* We can now finalize the device chains */
        makeDeviceChain_();
    }

    /**
     * @brief Copy the data in this ephemeris unit from device to host
     *
     * @param stream (Optional) CUDA stream for asynchronous copy
     *
     */
    void download(const StreamT& stream = 0)
    {
        metadata_.download(stream);
        data_.download(stream);
        // cache_.download(stream);
        deviceChain_.download(stream);
        deviceTraverser_.download(stream);
    }

    /** @brief Clear data from the GPU */
    void clearDevice()
    {
        metadata_.clearDevice();
        data_.clearDevice();
        // cache_.clearDevice();
        deviceChain_.clearDevice();
        deviceTraverser_.clearDevice();
    }
#endif

    /**
     * @brief Return number of bodies in this `brie::EphUnit`
     *
     * @return Nbodies
     *
     */
    vecdim_t nBodyUnits() const { return metadata_.size(); }

    /**
     * @brief Return the total size of the data in this `brie::EphUnit`, as the
     * total number of `brie::Real` scalars contained in the
     * `brie::EphUnit.data_` private member
     *
     * @return size
     *
     */
    idx_t size() const { return data_.size(); }

    /**
     * @brief Return a non-owning `brie::RefEphUnit` that allows for operations
     * with the host data
     *
     * @return hostRefEphUnit
     *
     */
    GRef hostRef() const
    {
        const_cast<EphUnit*>(this)->makeHostChain_();
        return GRef{ hostTraverser_.hostRef(), hostChain_.hostRef() };
    }
    GRef ref() const { return hostRef(); }

#ifndef BRIE_CPU_ONLY
    /**
     * @brief Return a non-owning `brie::RefEphUnit` that allows for operations
     * with the device data
     *
     * @return deviceRefEphUnit
     *
     */
    GRef deviceRef() const
    {
        /* device chain is created upon `upload` call - we just return */
        return GRef{ deviceTraverser_.deviceRef(), deviceChain_.deviceRef() };
    }
#endif

    /**
     * @brief Copy the `brie::EphUnit.data_` from another `brie::EphUnit`,
     * filtering based on the `brie::EphUnit.metadata_` contained in this
     * `brie::EphUnit`
     *
     * @param other Other ephemeris unit
     * @param stream (Optional) CUDA stream for asynchronous copy
     *
     */
    void copyFrom(const EphUnit& otherEph)
    {
        copyFrom_(otherEph.metadata_.hostRef(), otherEph.data_);
    }

    /**
     * @brief Copy the `brie::EphUnit.data_` to another `brie::EphUnit`,
     * filtering based on the `brie::EphUnit.metadata_` contained in the other
     * `brie::EphUnit`
     *
     * @param other Other ephemeris unit
     * @param stream (Optional) CUDA stream for asynchronous copy
     *
     */
    void copyTo(EphUnit& otherEph) const
    {
        copyTo_(otherEph.metadata_.hostRef(), otherEph.data_);
    }

    /** @brief Expose the metadata contained in this
     * `brie::states::EphUnit`
     */
    MetadataT& metadata() { return metadata_; }
    const MetadataT& metadata() const { return metadata_; }

    /** @brief Expose the raw data contained in this
     * `brie::states::EphUnit`
     */
    DataT& data() { return data_; }
    const DataT& data() const { return data_; }

    /** @brief Clone this object */
    EphUnit clone() const
    {
        return EphUnit(std::move(metadata_.clone()), std::move(data_.clone()));
    }

    /** @brief Emit a native-frame cache mirroring this `EphUnit`'s body
     *  set, sized for `nSamples` per-body slots.  The returned cache has
     *  its traverser copied in place from this `EphUnit` and its chain
     *  bound to this `EphUnit`'s metadata, ready for walker queries
     *  (after the slot data is filled).  The cache registers as an
     *  observer of this `EphUnit`, so subsequent moves are tracked
     *  automatically.
     *
     *  Out-of-line definition lives in `brie/states/EphCache.h`. */
    template<idx_t Dim,
        feta::core::memory::Device DeviceT
        = feta::core::memory::Device::CUDA_DEVICE>
    EphCache<UseTexture, Dim, DeviceT> makeCache(const idx_t& nSamples);

protected:
    /**
     * @brief Factory method to create a `brie::EphUnit` object from a given
     * .brie-formatted input (Layout 1)
     *
     */
    static EphUnit<UseTexture> fromBrie1_(const nlohmann::json& j)
    {
        /* read number of body units */
        vecdim_t Nbodies = j.size();
        /* read data size */
        idx_t totDataSize = 0;
        for (auto it : j) {
            totDataSize += it["data"].size();
        }
        /* create ephemeris object */
        EphUnit<UseTexture> ephs(Nbodies, totDataSize);
        /* read brie file */
        ephs.readBrie1(j);

        return ephs;
    }

    /**
     * @brief Factory method to create a `brie::EphUnit` object from a given
     * .brie-formatted input (Layout 2)
     *
     */
    static EphUnit<UseTexture> fromBrie2_(const nlohmann::json& j)
    {
        /* read data size */
        idx_t totDataSize = j["data"].size();
        /* get number of body units */
        vecdim_t Nbodies = j["metadata"]["nBodyUnits"];
        /* create ephemeris object */
        EphUnit<UseTexture> ephs(Nbodies, totDataSize);
        /* read brie file */
        ephs.readBrie2(j);

        return ephs;
    }

    /**
     * @brief Internal factory method to read a single .brie file into a
     * `brie::EphUnit` object
     *
     * @param brieFile Path to the .brie file
     * @param paths (Optional) Path manager
     *
     */
    static EphUnit<UseTexture> fromBrieFileString_(const std::string& brieFile,
        const util::Paths& paths = util::paths::brieDefaultPath())
    {
        cbor::EphFileBuffer fb = cbor::peekEphFile(brieFile, paths);

        if (fb.sizes.layout == 2) {
            const vecdim_t Nbodies
                = static_cast<vecdim_t>(fb.sizes.nBodyUnits);
            const idx_t totDataSize
                = static_cast<idx_t>(fb.sizes.dataLen);

            EphUnit<UseTexture> ephs(Nbodies, totDataSize);

            int*  intDst    = ephs.metadata_.intMembers().hostRef().data();
            Real* doubleDst = (fb.sizes.doubleMetadataLen != 0)
                ? ephs.metadata_.realMembers().hostRef().data()
                : nullptr;
            Real* dataDst   = ephs.data_.getHostData();

            cbor::loadEphLayout2(fb,                                 //
                intDst,    fb.sizes.intMetadataLen,                  //
                doubleDst, fb.sizes.doubleMetadataLen,               //
                dataDst,   fb.sizes.dataLen);

            ephs.metadata_.hostRef().updateCenterPositions();
            return ephs;
        }

        nlohmann::json j = Load::cbor(brieFile, paths);
        return fromJson_(j);
    }

    /** @brief Internal factory method to read an array of `nlohmann::json`
     * objects into a `brie::EphUnit` object
     *
     * @param jarr `nlohmann::json::array_t` object
     *
     */
    static EphUnit<UseTexture> fromJsonArray_(
        const nlohmann::json::array_t& jarr)
    {
        /* load first eph */
        EphUnit<UseTexture> ephs = fromJson_(jarr[0]);
        if (jarr.size() > 1) {
            for (idx_t i = 1; i < jarr.size(); i++) {
                /* Load this eph unit */
                EphUnit<UseTexture> thisEph = fromJson_(jarr[i]);
                /* Merge into new eph unit */
                ephs = ephs.merge(thisEph);
            }
        }

        return ephs;
    }

    /** @brief Internal factory method to read a single `nlohmann::json` object
     * into a `brie::EphUnit` object
     *
     * @param j `nlohmann::json` object
     *
     */
    static EphUnit<UseTexture> fromJson_(const nlohmann::json& j)
    {
        /* distinguish layout types */
        if (j["layout"] == 1)
            return fromBrie1_(j["core"]);
        else if (j["layout"] == 2)
            return fromBrie2_(j["core"]);
        else
            BRIE_THROW(std::runtime_error, "Unknown BRIE layout type");
    }

    /**
     * @brief Internal factory method to create a single `brie::EphUnit` object
     * from multiple .brie files
     *
     * @param brieFiles `nlohmann::json::array_t`, list of .brie files to be
     * read
     * @param paths (Optional) Path manager
     *
     */
    static EphUnit<UseTexture> fromBrieFileArray_(
        const nlohmann::json::array_t& brieFiles,
        const util::Paths& paths = util::paths::brieDefaultPath())
    {

        /* load first eph */
        EphUnit<UseTexture> ephs = fromBrieFileString_(brieFiles[0], paths);

        if (brieFiles.size() > 1) {
            for (idx_t i = 1; i < brieFiles.size(); i++) {
                /* Load this eph unit */
                EphUnit<UseTexture> thisEph
                    = fromBrieFileString_(brieFiles[i], paths);
                /* Merge into new eph unit */
                ephs = ephs.merge(thisEph);
            }
        }

        return ephs;
    }

    /**
     * @brief Internal copy from method
     *
     */
    void copyFrom_(const MetadataT::GRef& otherRef, const DataT& otherData);

    void copyTo_(const MetadataT::GRef& otherRef, DataT& otherData) const;

    /** @brief Finalize the allocation of chains an
                   d traverser */
    void hostChainMalloc_()
    {
        const idx_t nBodies = metadata_.size();
        if (nBodies != 0) {
            hostChain_     = std::move(ChainInterpolatorT(nBodies));
            hostTraverser_ = std::move(TraverserT(nBodies));
        }
    }

    /** @brief Fill `chain` + `traverser` from the metadata. The `Device`
     *  template selects the device-buffer + texture `make` overload; the host
     *  path uses the plain overload. The radius (and, on the host path, the
     *  interpolator data) is always read from host-resident memory. Build-time
     *  (run once per `make*Chain_`), so the branch is a compile-time
     *  `if constexpr` over otherwise-identical bodies. */
    template<bool Device>
    void fillChains_(ChainInterpolatorT& chain, TraverserT& traverser)
    {
        MetadataT::GRef mref                   = metadata_.hostRef();
        typename ChainInterpolatorT::GRef cref = chain.hostRef();
        TraverserT::GRef tref                  = traverser.hostRef();

        using BodyLocatorT      = core::BodyLocator;
        using BodyInterpolatorT = core::BodyInterpolator<UseTexture>;

        const Real* const hostPtr = data_.getHostData();

        /* Fill one by one */
        for (idx_t bodyCount = 0; bodyCount < mref.size(); bodyCount++) {
            /* Interpolator stuff */
            const idx_t dataOffset = mref.getDataOffset(bodyCount);
            const idx_t nIntervals = mref.getNintervals(bodyCount);
            const idx_t pdeg       = mref.getPdeg(bodyCount);
            const idx_t dtype      = mref.getDtype(bodyCount);
            const Real radius      = hostPtr[dataOffset];
            if constexpr (Device) {
#ifndef BRIE_CPU_ONLY
                cref[bodyCount] = BodyInterpolatorT::make(data_.getDeviceData(),
                    radius, dataOffset, nIntervals, pdeg, dtype,
                    data_.deviceRef().tex());
#endif
            } else {
                cref[bodyCount] = BodyInterpolatorT::make(
                    hostPtr, radius, dataOffset, nIntervals, pdeg, dtype);
            }

            /* Locator/traverser stuff */
            const NaifId target   = mref.getTarget(bodyCount);
            const idx_t targetPos = bodyCount;
            const idx_t centerPos = mref.getCenterPos(bodyCount);
            tref[bodyCount]       = BodyLocatorT(target, targetPos, centerPos);
        }
    }

    /** @brief Fill the host chains */
    void fillHostChains_() { fillChains_<false>(hostChain_, hostTraverser_); }

    /** @brief Make the host chain (complete) */
    void makeHostChain_()
    {
        if (!hostChainReady_) {
            hostChainMalloc_();
            fillHostChains_();
            hostChain_.bind(metadata_.handleMembers().getHostData());
            hostChainReady_ = true;
        }
    }

#ifndef BRIE_CPU_ONLY

    /** @brief Finalize the allocation of chains and traverser */
    void deviceChainMalloc_()
    {
        const idx_t nBodies = metadata_.size();
        if (nBodies != 0) {
            deviceChain_     = std::move(ChainInterpolatorT(nBodies));
            deviceTraverser_ = std::move(TraverserT(nBodies));
        }
    }

    /** @brief Fill the device chains */
    void fillDeviceChains_()
    {
        fillChains_<true>(deviceChain_, deviceTraverser_);
    }

    /** @brief make the device chain */
    void makeDeviceChain_()
    {
        if (!deviceChainReady_) {
            deviceChainMalloc_();
            fillDeviceChains_();
            deviceChain_.bind(metadata_.handleMembers().getDeviceData());
            /* Upload AFTER filling with the correct interpolators */
            deviceChain_.upload();
            deviceTraverser_.upload();
            deviceChainReady_ = true;
        }
    }

#endif

    /** @brief Clear the chains - reset to flexible and set the flags to false
     */
    void resetChains_()
    {
        hostChain_      = std::move(ChainInterpolatorT::flexible());
        hostTraverser_  = std::move(TraverserT::flexible());
        hostChainReady_ = false;
#ifndef BRIE_CPU_ONLY
        deviceChain_      = std::move(ChainInterpolatorT::flexible());
        deviceTraverser_  = std::move(TraverserT::flexible());
        deviceChainReady_ = false;
#endif
    }

    /** @brief metadata in this `brie::EphUnit` **/
    MetadataT metadata_;
    /** @brief data in this `brie::EphUnit` **/
    DataT data_;
    /** @brief The Host chain interpolator */
    ChainInterpolatorT hostChain_ = ChainInterpolatorT::flexible();
    /** @brief The Host traverser */
    TraverserT hostTraverser_ = TraverserT::flexible();
    /** @brief Whether the host chain is ready */
    bool hostChainReady_ = false;
#ifndef BRIE_CPU_ONLY
    /** @brief The Device chain interpolator */
    ChainInterpolatorT deviceChain_ = ChainInterpolatorT::flexible();
    /** @brief The Device traverser */
    TraverserT deviceTraverser_ = TraverserT::flexible();
    /** @brief Whether the device chain is ready */
    bool deviceChainReady_ = false;
#endif
};

/* Explicit instantiation */

extern template class EphUnit<true>;
extern template class EphUnit<false>;

} // namespace states
} // namespace brie
