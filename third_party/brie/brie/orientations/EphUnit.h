#pragma once

#ifndef BRIE_CPU_ONLY
#include <cuda.h>
#endif

#include <feta/feta.h>
#include <sstream>
#include <string>

#include "brie/orientations/RefEphUnit.h"
#include "brie/orientations/metadata/EphUnit.h"
#include "brie/typedefs.h"
#include "brie/util/cbor.h"
#include "brie/util//json.h"

#include "brie/util//DeviceError.h"
#include "brie/util//throw.h"

namespace brie {
namespace orientations {

/**
 * @brief Ephemeris Unit - class to manage ephemeris evaluations
 *
 */
template<bool UseTexture>
class EphUnit {
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

    /** @brief Factory method to create an empty eph unit object */
    static EphUnit<UseTexture> empty()
    {
        MetadataT metadata = MetadataT::flexible();
        DataT data         = DataT::flexible();
        return EphUnit<UseTexture>(std::move(metadata), std::move(data));
    }

    /**
     * @brief Factory method to create a `brie::orientations::EphUnit` object from
     * a .brie file, or a `nlohmann::json::array_t` of .brie files
     *
     * @param brieFile file(s) to be read
     * @param NaifIds (Optional) subset of NAIF ids to be loaded from the .brie
     * file
     * @param pahts (Optional) file path manager
     * @return EphUnit
     */
    static EphUnit fromBrie(const nlohmann::json& brieFile,
        const feta::scalar::Array<NaifId>& NaifIDs
        = feta::scalar::Array<NaifId>(0),
        const util::Paths& paths = util::paths::brieDefaultPath())
    {
        if (brieFile.is_string()) {
            if (NaifIDs.size() == 0) {
                return fromBrieFileString_(brieFile, paths);
            } else {
                return fromBrieFileString_(brieFile, paths)
                    .makeFullSubset(NaifIDs);
            }
        } else if (brieFile.is_array()) {
            if (NaifIDs.size() == 0) {
                return fromBrieFileArray_(brieFile, paths);
            } else {
                return fromBrieFileArray_(brieFile, paths)
                    .makeFullSubset(NaifIDs);
            }
        } else {
            BRIE_THROW(std::runtime_error,
                "Brie files must be given either as a "
                "`nlohmann::json::string_t` or as a "
                "`nlohmann::json::array_t` "
                "of strings.")
        }
    }

    /**
     * @brief Copy constructor is forbidden
     *
     */
    EphUnit(EphUnit& other) = delete;

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
        : EphUnit{ std::move(other.metadata_), std::move(other.data_) }
    {
    }

    /**
     * @brief Basic constructor - host only
     *
     */
    EphUnit(const vecdim_t Nbodies, const idx_t totDataSize)
        : metadata_{ Nbodies }
        , data_{ totDataSize }
    {
    }

    /**
     * @brief Move assignment operator
     *
     */
    EphUnit& operator=(EphUnit&& other)
    {
        metadata_ = std::move(other.metadata_);
        data_     = std::move(other.data_);

        return *this;
    }


    /**
     * @brief Read data and metadata from .brie file (Layout 1)
     *
     * @param brieRotEphUnit Data read from a .brie file
     *
     */
    void readBrie1(const nlohmann::json& brieRotEphUnit)
    {
        /* read metadata */
        metadata_.readBrie1(brieRotEphUnit);

        /* get reference to metadata and this data */
        MetadataT::GRef newMref = metadata_.hostRef();

        /* read data based on offsets */
        idx_t bodyCount = 0;
        for (auto bodyUnit : brieRotEphUnit) {
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
     * @param brieRotEphUnit Data read from a .brie file
     *
     */
    void readBrie2(const nlohmann::json& brieRotEphUnit)
    {
        /* read metadata */
        metadata_.readBrie2(brieRotEphUnit["metadata"]);

        /* read data */
        std::copy(brieRotEphUnit["data"].begin(), brieRotEphUnit["data"].end(),
            data_.getHostData());
    }

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
    }

    /** @brief Clear data from the GPU */
    void clearDevice()
    {
        metadata_.clearDevice();
        data_.clearDevice();
    }
#endif

    /**
     * @brief Return number of bodies in this `brie::orientations::EphUnit`
     *
     * @return Nbodies
     *
     */
    vecdim_t nBodyUnits() const { return metadata_.size(); }

    /**
     * @brief Return the total size of the data in this
     * `brie::orientations::EphUnit`, as the total number of `brie::Real`
     * scalars contained in the `brie::orientations::EphUnit.data_` private
     * member
     *
     * @return size
     *
     */
    idx_t size() const { return data_.size(); }

    /**
     * @brief Return a non-owning `brie::RefRotEphUnit` that allows for
     * operations with the host data
     *
     * @return hostRefRotEphUnit
     *
     */
    GRef hostRef() const
    {
        return GRef::make(metadata_.hostRef(), data_.hostRef());
    }
    GRef ref() const { return hostRef(); }

#ifndef BRIE_CPU_ONLY
    /**
     * @brief Return a non-owning `brie::RefRotEphUnit` that allows for
     * operations with the device data
     *
     * @return deviceRefRotEphUnit
     *
     */
    GRef deviceRef() const
    {
        return GRef::make(metadata_.deviceRef(), data_.deviceRef());
    }
#endif

    /**
     * @brief Copy the `brie::orientations::EphUnit.data_` from another
     * `brie::orientations::EphUnit`, filtering based on the
     * `brie::orientations::EphUnit.metadata_` contained in this
     * `brie::orientations::EphUnit`
     *
     * @param other Other ephemeris unit
     * @param stream (Optional) CUDA stream for asynchronous copy
     *
     */
    void copyFrom(const EphUnit& otherEph);

    /**
     * @brief Copy the `brie::orientations::EphUnit.data_` to another
     * `brie::orientations::EphUnit`, filtering based on the
     * `brie::orientations::EphUnit.metadata_` contained in the other
     * `brie::orientations::EphUnit`
     *
     * @param other Other ephemeris unit
     * @param stream (Optional) CUDA stream for asynchronous copy
     *
     */
    void copyTo(EphUnit& otherEph) const;

    /**
     * @brief Make a `brie::orientations::EphUnit` that contains ONLY the given
     * rotation targets
     *
     * @warning A plain subset carries ONLY the listed targets. Type-1 body
     * units look up their barycenter nutation-precession unit by NaifId at
     * evaluation time and SILENTLY SKIP the correction if that unit is
     * absent — the rotation changes without any error. Prefer
     * `makeFullSubset`, which includes the barycenter parents
     * automatically.
     *
     * @param NaifIDs rotation targets for which to create the subset
     * @return EphUnit New subset rotation ephemeris unit
     *
     */
    EphUnit makeSubset(const feta::scalar::Array<NaifId>& NaifIDs) const
    {
        /* Create metadata subset */
        MetadataT metadata = metadata_.makeSubset(NaifIDs);

        /* Create data subset */
        DataT data(metadata.totalDataSize());

        /* create ephs, read and return */
        EphUnit ephs(std::move(metadata), std::move(data));
        copyTo(ephs);
        return ephs;
    }

    /**
     * @brief Make a `brie::orientations::EphUnit` that contains the given
     * rotation targets AND the barycenter parent units of every requested
     * Type-1 body unit, i.e. the necessary ones to evaluate the requested
     * rotations including their nutation-precession corrections.
     *
     * @param NaifIDs rotation targets for which to create the subset
     * @return EphUnit New subset rotation ephemeris unit
     *
     */
    EphUnit makeFullSubset(const feta::scalar::Array<NaifId>& NaifIDs) const
    {
        feta::scalar::Array<NaifId> trueIDs
            = metadata_.makeTrueNaifIDs(NaifIDs);
        return makeSubset(trueIDs);
    }

    /**
     * @brief Merge this `brie::orientations::EphUnit` with another
     * `brie::orientations::EphUnit`. The targets in the other
     * `brie::orientations::EphUnit` that are already found in this
     * `brie::orientations::EphUnit` are omitted from the merge.
     *
     * @param other Other `brie::orientations::EphUnit`
     * @return EphUnit New merged `brie::orientations::EphUnit`
     *
     */
    EphUnit merge(const EphUnit& other) const;

    /** @brief Expose the metadata contained in this
     * `brie::orientations::EphUnit`
     */
    MetadataT& metadata() { return metadata_; }
    const MetadataT& metadata() const { return metadata_; }

    /** @brief Expose the raw data contained in this
     * `brie::orientations::EphUnit`
     */
    DataT& data() { return data_; }
    const DataT& data() const { return data_; }

    /** @brief Clone this object */
    EphUnit clone() const
    {
        return EphUnit(std::move(metadata_.clone()), std::move(data_.clone()));
    }

private:
    /**
     * @brief Factory method to create a `brie::orientations::EphUnit` object
     * from a given .brie-formatted input (Layout 1)
     *
     */
    static EphUnit fromBrie1_(const nlohmann::json& j)
    {
        /* read number of body units */
        vecdim_t Nbodies = j.size();
        /* read data size */
        idx_t totDataSize = 0;
        for (auto it : j) {
            totDataSize += it["data"].size();
        }
        /* create ephemeris object */
        EphUnit ephs(Nbodies, totDataSize);
        /* read brie file */
        ephs.readBrie1(j);

        return ephs;
    }

    /**
     * @brief Factory method to create a `brie::orientations::EphUnit` object
     * from a given .brie-formatted input (Layout 2)
     *
     */
    static EphUnit fromBrie2_(const nlohmann::json& j)
    {
        /* read data size */
        idx_t totDataSize = j["data"].size();
        /* get number of body units */
        vecdim_t Nbodies = j["metadata"]["nBodyUnits"];
        /* create ephemeris object */
        EphUnit ephs(Nbodies, totDataSize);
        /* read brie file */
        ephs.readBrie2(j);

        return ephs;
    }

    /**
     * @brief Internal factory method to read a single .brie file into a
     * `brie::orientations::EphUnit` object
     *
     * @param brieFile Path to the .brie file
     * @param paths (Optional) Path manager
     *
     */
    static EphUnit fromBrieFileString_(std::string brieFile,
        const util::Paths& paths = util::paths::brieDefaultPath())
    {
        cbor::EphFileBuffer fb = cbor::peekEphFile(brieFile, paths);

        if (fb.sizes.layout == 2) {
            const vecdim_t Nbodies
                = static_cast<vecdim_t>(fb.sizes.nBodyUnits);
            const idx_t totDataSize
                = static_cast<idx_t>(fb.sizes.dataLen);

            EphUnit ephs(Nbodies, totDataSize);

            int*  intDst    = ephs.metadata_.intMembers().hostRef().data();
            Real* doubleDst = (fb.sizes.doubleMetadataLen != 0)
                ? ephs.metadata_.realMembers().hostRef().data()
                : nullptr;
            Real* dataDst   = ephs.data_.getHostData();

            cbor::loadEphLayout2(fb,                        //
                intDst,    fb.sizes.intMetadataLen,         //
                doubleDst, fb.sizes.doubleMetadataLen,      //
                dataDst,   fb.sizes.dataLen);

            return ephs;
        }

        nlohmann::json j = Load::cbor(brieFile, paths);
        if (j["layout"] == 1) {
            return fromBrie1_(j["core"]);
        } else if (j["layout"] == 2) {
            return fromBrie2_(j["core"]);
        } else {
            BRIE_THROW(std::runtime_error, "Unknown BRIE layout type");
        }
    }

    /**
     * @brief Internal factory method to read an array of .brot files into a
     * single merged `brie::orientations::EphUnit` object
     *
     * @param brieFiles Paths to the .brot files
     * @param paths (Optional) Path manager
     *
     */
    static EphUnit fromBrieFileArray_(const nlohmann::json::array_t& brieFiles,
        const util::Paths& paths = util::paths::brieDefaultPath())
    {
        /* load first eph */
        EphUnit ephs = fromBrieFileString_(brieFiles[0], paths);

        if (brieFiles.size() > 1) {
            for (idx_t i = 1; i < brieFiles.size(); i++) {
                /* Load this eph unit */
                EphUnit thisEph = fromBrieFileString_(brieFiles[i], paths);
                /* Merge into new eph unit */
                ephs = ephs.merge(thisEph);
            }
        }

        return ephs;
    }


    /** @brief metadata in this `brie::orientations::EphUnit` **/
    MetadataT metadata_;
    /** @brief data in this `brie::EphUnit` **/
    DataT data_;
};

/* Explicit instantiation */
extern template class EphUnit<true>;
extern template class EphUnit<false>;

} // namespace orientations
} // namespace brie