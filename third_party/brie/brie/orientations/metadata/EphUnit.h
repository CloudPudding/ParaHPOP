#pragma once

#include <nlohmann/json.hpp>

#include "brie/orientations/metadata/RefEphUnit.h"
#include "brie/util/throw.h"


namespace brie {
namespace orientations {
namespace metadata {

/**
 * @brief Class to manage the Ephemeris metadata
 *
 */
class EphUnit
    : public core::Metadata<common::Sizes::Int, 1, common::Sizes::Real> {
    using ParentT = core::Metadata<common::Sizes::Int, 1, common::Sizes::Real>;
    /* Data types aliases */
    using ParentT::IntArrayT;
    using ParentT::RealArrayT;
    using Self       = EphUnit;

public:
    /** @brief reference types **/
    template<bool work, bool MaybeVolatile = false>
    using Ref = RefEphUnit<Self, work, MaybeVolatile>;
    /** @brief global reference type */
    using GRef = Ref<false>;
    /** @brief work reference type */
    using WRef = Ref<true>;
    /** @brief Volatile work reference */
    using VolatileRef = Ref<true, true>;

    /** @brief basic host constructor */
    EphUnit(const idx_t& nBodies)
        : ParentT{ nBodies }
    {
    }

    /**
     * @brief Inherit constructors
     *
     */
    EphUnit(ParentT&& parent)
        : EphUnit::ParentT{ std::move(parent) }
    {
    }

    /**
     * @brief Read the metadata of a .brie file (Layout 1)
     *
     * @param brieEphemerisData
     *
     */
    void readBrie1(const nlohmann::json& brieEphUnit);

    /**
     * @brief Read the metadata of a .brie file (Layout 2)
     *
     * @param brieEphemerisData
     *
     */
    void readBrie2(const nlohmann::json& brieEphUnitMetaData);

    /**
     * @brief Create new metadata for the given NAIF IDs (rotation targets)
     *
     * @warning A plain subset carries ONLY the listed targets. Type-1 body
     * units look up their barycenter nutation-precession unit by NaifId at
     * evaluation time and SILENTLY SKIP the correction if that unit is
     * absent — the rotation changes without any error. Prefer
     * `makeTrueNaifIDs` / `EphUnit::makeFullSubset`, which include the
     * barycenter parents automatically.
     *
     * @param NaifIDs
     * @return newMetadata
     *
     */
    EphUnit makeSubset(const feta::scalar::Array<NaifId>& NaifIDs) const
    {
        /* initialize */
        EphUnit out(NaifIDs.size());
        /* get references to work with */
        GRef thisref                           = this->hostRef();
        GRef outref                            = out.hostRef();
        feta::scalar::Array<NaifId>::GRef iref = NaifIDs.hostRef();
        /* fill in data */
        for (idx_t i = 0; i < NaifIDs.size(); i++) {
            /* find in current metadata */
            idx_t bodyCount = thisref.getTargetBodyCount(iref[i]);
            if (bodyCount >= thisref.size())
                BRIE_THROW(std::runtime_error,
                    "rotation target " + std::to_string(iref[i])
                        + " is not contained in this rotation ephemeris "
                          "unit");
            /* fill in corresponding int data */
            outref.assignIntVec(i, thisref.getIntVec(bodyCount));
            /* fill in epochs */
            outref.assignRealVec(i, thisref.getRealVec(bodyCount));
        }

        /* update data offsets and return */
        outref.updateDataOffsets();

        return out;
    }

    /**
     * @brief Merge this and another `brie::orientations::metadata::EphUnit`
     * objects into a new `brie::orientations::metadata::EphUnit` object
     *
     * @param otherMetadata
     * @return newMetadata
     *
     */
    EphUnit merge(const EphUnit& other) const
    {
        /* Create new empty metadata */
        EphUnit metadata(this->size() + other.size());
        GRef mref       = metadata.hostRef();
        GRef thisref    = this->hostRef();
        GRef missingref = other.hostRef();

        /* Fill in data for `this` */
        idx_t count = 0;
        for (idx_t i = 0; i < thisref.size(); i++) {
            /* fill in corresponding int data */
            mref.assignIntVec(count, thisref.getIntVec(i));
            /* fill in epochs */
            mref.assignRealVec(count, thisref.getRealVec(i));
            /* advance on the merged ephs */
            count++;
        }
        /* Fill in data for `missing` */
        for (idx_t i = 0; i < missingref.size(); i++) {
            /* fill in corresponding int data */
            mref.assignIntVec(count, missingref.getIntVec(i));
            /* fill in epochs */
            mref.assignRealVec(count, missingref.getRealVec(i));
            /* advance on the merged ephs */
            count++;
        }

        /* adjust data offsets and return */
        mref.updateDataOffsets();

        return metadata;
    }

    /**
     * @brief Return the NAIF IDs (rotation targets) contained in this
     * metadata set
     *
     * @return NaifIDs
     *
     */
    feta::scalar::Array<NaifId> getNaifIDs() const
    {
        /* Allocate array */
        feta::scalar::Array<NaifId> out(this->size());

        /* Assign elements */
        feta::scalar::Array<NaifId>::GRef outRef = out.hostRef();
        GRef ref                                 = this->hostRef();

        for (idx_t i = 0; i < size(); i++) {
            outRef[i] = ref.getTarget(i);
        }

        /* return */
        return out;
    }

    /**
     * @brief Return the NAIF IDs NOT contained in this metadata set
     *
     * @param referenceNaifIDs
     * @return missingNaifIDs
     *
     */
    feta::scalar::Array<NaifId> getMissingNaifIDs(
        const feta::scalar::Array<NaifId>& NaifIDs) const
    {
        /* Find size */
        idx_t i, j;
        idx_t missingSize                      = 0;
        GRef ref                               = this->hostRef();
        feta::scalar::Array<NaifId>::GRef iref = NaifIDs.hostRef();
        for (i = 0; i < NaifIDs.size(); i++) {
            if (!ref.hasBody(iref[i]))
                missingSize++;
        }

        /* Allocate, fill and return (a zero-size feta array is not
         * allocated and must not be referenced) */
        feta::scalar::Array<NaifId> out(missingSize);
        if (missingSize != 0) {
            feta::scalar::Array<NaifId>::GRef outref = out.hostRef();
            j                                        = 0;
            for (i = 0; i < NaifIDs.size(); i++) {
                if (!ref.hasBody(iref[i])) {
                    outref[j] = iref[i];
                    j++;
                }
            }
        }

        return out;
    }

    /**
     * @brief Return the "true" subset array, containing the NAIF ids of all
     * the given rotation targets plus the barycenter parents of every
     * requested Type-1 body unit: their nutation-precession units are looked
     * up by NaifId at evaluation time and the correction silently drops if
     * the parent unit is absent.
     *
     * @param referenceNaifIDs
     * @return trueNaifIDs
     *
     */
    feta::scalar::Array<NaifId> makeTrueNaifIDs(
        const feta::scalar::Array<NaifId>& NaifIDs) const;

    /**
     * @brief Return the total data size
     *
     * @return totalDataSize
     *
     */
    idx_t totalDataSize() const
    {
        idx_t out = 0;
        GRef ref  = this->hostRef();
        for (idx_t i = 0; i < this->size(); i++)
            out += ref.getDataSize(i);

        return out;
    }

    /**
     * @brief Return a non-owning `brie::states::metadata::RefEphUnit` that
     * allows for operations with the host data
     *
     * @return hostRefEphUnit
     *
     */
    GRef hostRef() const { return GRef::make(ParentT::hostRef()); }
    GRef ref() const { return hostRef(); }

#ifndef BRIE_CPU_ONLY
    /**
     * @brief Return a non-owning `brie::states::metadata::RefEphUnit` that
     * allows for operations with the device data
     *
     * @return deviceRefEphUnit
     *
     */
    GRef deviceRef() const { return GRef::make(ParentT::deviceRef()); }
#endif

    /** @brief Clone this object */
    EphUnit clone() const { return EphUnit(std::move(ParentT::clone())); }
};

} // namespace metadata
} // namespace orientations
} // namespace brie