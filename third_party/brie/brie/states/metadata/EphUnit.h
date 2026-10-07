#pragma once

#include "brie/states/metadata/RefEphUnit.h"
#include "brie/util/throw.h"
#include <nlohmann/json.hpp>

namespace brie {
namespace states {
namespace metadata {

/**
 * @brief Class to manage the Ephemeris metadata
 *
 */
class EphUnit : public core::Metadata<INTSIZE, HANDLESIZE, REALSIZE> {
    using ParentT = core::Metadata<INTSIZE, HANDLESIZE, REALSIZE>;
    /* Data types aliases */
    using ParentT::HandleArrayT;
    using ParentT::IntArrayT;
    using ParentT::RealArrayT;
    using Self       = EphUnit;

public:
    /** @brief reference types **/
    template<bool work, bool MaybeVolatile = false>
    using Ref = RefEphUnit<work, MaybeVolatile>;
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
        : ParentT{ std::move(parent) }
    {
    }

    /**
     * @brief Read the metadata of a .brie file (Layout 1)
     *
     * @param brieEphemerisData
     *
     */
    void readBrie1(const nlohmann::json& brieEphUnit)
    {
        /* work with reference */
        GRef ref        = this->hostRef();
        idx_t bodyCount = 0;
        for (auto bodyUnit : brieEphUnit) {
            /* read int members */
            ref.getFrame(bodyCount)      = bodyUnit["metadata"]["frame"];
            ref.getDtype(bodyCount)      = bodyUnit["metadata"]["dtype"];
            ref.getTarget(bodyCount)     = bodyUnit["metadata"]["target"];
            ref.getCenter(bodyCount)     = bodyUnit["metadata"]["center"];
            ref.getNintervals(bodyCount) = bodyUnit["metadata"]["nintervals"];
            ref.getPdeg(bodyCount)       = bodyUnit["metadata"]["pdeg"];
            /* data size */
            ref.getDataSize(bodyCount) = bodyUnit["data"].size();
            /* determine data offset for this body unit */
            if (bodyCount == 0) {
                ref.getDataOffset(bodyCount) = 0;
            } else {
                ref.getDataOffset(bodyCount) = ref.getDataOffset(bodyCount - 1)
                    + ref.getDataSize(bodyCount - 1);
            }
            /* read double members */
            ref.getInitialEpoch(bodyCount) = bodyUnit["metadata"]["startEpoch"];
            ref.getFinalEpoch(bodyCount)   = bodyUnit["metadata"]["finalEpoch"];
            /* increase count */
            bodyCount += 1;
        }

        /* update the center positions */
        ref.updateCenterPositions();
    }

    /**
     * @brief Read the metadata of a .brie file (Layout 2)
     *
     * @param brieEphemerisData
     *
     */
    void readBrie2(const nlohmann::json& brieEphUnit)
    {
        /* read integer members */
        std::copy(brieEphUnit["intMetadata"].begin(),
            brieEphUnit["intMetadata"].end(), intMembers_.hostRef().data());

        /* read double members */
        std::copy(brieEphUnit["doubleMetadata"].begin(),
            brieEphUnit["doubleMetadata"].end(), realMembers_.hostRef().data());

        /* Update the center positions */
        this->hostRef().updateCenterPositions();
    }

    /**
     * @brief Create new metadata for the given NAIF IDs
     *
     * @warning A plain subset carries ONLY the listed bodies: target-center
     * chains are looked up by NaifId at evaluation time, so queries through
     * a missing center fail. Prefer `makeTrueNaifIDs` /
     * `EphUnit::makeFullSubset`, which include all the required centers
     * automatically.
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
            /* find in current metadata (miss returns size()+1; size()
             * itself denotes the SSB root, which holds no data row) */
            idx_t bodyCount = thisref.getTargetBodyCount(iref[i]);
            if (bodyCount >= thisref.size())
                BRIE_THROW(std::runtime_error,
                    "target " + std::to_string(iref[i])
                        + " is not contained in this ephemeris unit");
            /* fill in corresponding int data */
            outref.assignIntVec(i, thisref.getIntVec(bodyCount));
            /* fill in epochs */
            outref.assignRealVec(i, thisref.getRealVec(bodyCount));
        }

        /* update data offset, center locations and return */
        outref.updateDataOffsets();
        outref.updateCenterPositions();

        return out;
    }

    /**
     * @brief Merge this and another `brie::states::metadata::EphUnit` objects
     * into a new `brie::states::metadata::EphUnit` object
     *
     * @param otherMetadata
     * @return newMetadata
     *
     */
    EphUnit merge(const EphUnit& other)
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
            // fill in epochs
            mref.assignRealVec(count, missingref.getRealVec(i));
            /* advance on the merged ephs */
            count++;
        }

        /* adjust data offsets, center locations and return */
        mref.updateDataOffsets();
        mref.updateCenterPositions();

        return metadata;
    }

    /**
     * @brief Static method to find the center of the given target, for the
     * given `brie::states::metadata::EphUnit::GRef`
     *
     * @param targetNaifId
     * @param metadataReference
     * @return centreNaifId
     *
     */
    static NaifId findCenter(const NaifId& target, const GRef& ref)
    {
        return ref.getCenter(ref.getTargetBodyCount(target));
    }

    /**
     * @brief Static method to find the center of the given target, for the
     * given .brie-formatted input metadata
     *
     * @param targetNaifId
     * @param brieMetadata
     * @return centreNaifId
     *
     */
    static NaifId findCenter(
        const NaifId& target, const nlohmann::json& brieEphUnit)
    {
        NaifId out = 0;
        for (auto bodyUnit : brieEphUnit) {
            if (bodyUnit["target"] == target) {
                out = bodyUnit["center"];
                break;
            }
        }
        return out;
    }

    /** @brief Static method to check whether a given unit contains the given
     * body (Ref Eph Unit)*/
    static bool isContained(const NaifId& target, const GRef& ref)
    {
        return ref.getTargetBodyCount(target) <= ref.size();
    }

    /** @brief Static method to check whether a given unit contains the given
     * body (Ref Eph Unit)*/
    static bool isContained(
        const NaifId target, const nlohmann::json& brieEphUnit)
    {
        if (target == 0) {
            return true;
        } else {
            bool found = false;
            for (auto bodyUnit : brieEphUnit) {
                if (bodyUnit["target"] == target) {
                    found = true;
                    break;
                }
            }
            return found;
        }
    }

    /**
     * @brief Return the NAIF IDs conained in this metadata set
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
            if (!ref.hasTargetBody(iref[i]))
                missingSize++;
        }

        /* Allocate, fill and return (a zero-size feta array is not
         * allocated and must not be referenced) */
        feta::scalar::Array<NaifId> out(missingSize);
        if (missingSize != 0) {
            feta::scalar::Array<NaifId>::GRef outref = out.hostRef();
            j                                        = 0;
            for (i = 0; i < NaifIDs.size(); i++) {
                if (!ref.hasTargetBody(iref[i])) {
                    outref[j] = iref[i];
                    j++;
                }
            }
        }

        return out;
    }

    /**
     * @brief Return the "true" subset arrays, containing the NAIF ids of all
     * the given bodies plus the ones required for all the possible combinations
     * of ephemeris evaluations
     *
     * @param referenceNaifIDs
     * @return trueNaifIDs
     *
     */
    feta::scalar::Array<NaifId> makeTrueNaifIDs(
        const feta::scalar::Array<NaifId>& NaifIDs) const
    {
        return this->makeTrueNaifIDs_<GRef>(this->hostRef(), NaifIDs);
    }

    /**
     * @brief Static method - return the "true" subset arrays, containing the
     * NAIF ids of all the given bodies plus the ones required for all the
     * possible combinations of ephemeris evaluations
     *
     * @param refEphUnit
     * @param referenceNaifIDs
     * @return trueNaifIDs
     *
     */
    static feta::scalar::Array<NaifId> makeTrueNaifIDs(
        const GRef& ref, const feta::scalar::Array<NaifId>& NaifIDs)
    {
        return Self::makeTrueNaifIDs_<GRef>(ref, NaifIDs);
    }

    /**
     * @brief Static method - return the "true" subset arrays, containing the
     * NAIF ids of all the given bodies plus the ones required for all the
     * possible combinations of ephemeris evaluations
     *
     * @param brieEphUnit
     * @param referenceNaifIDs
     * @return trueNaifIDs
     *
     */
    static feta::scalar::Array<NaifId> makeTrueNaifIDs(
        const nlohmann::json& brieEphUnit,
        const feta::scalar::Array<NaifId>& NaifIDs)
    {
        return Self::makeTrueNaifIDs_<nlohmann::json>(brieEphUnit, NaifIDs);
    }

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

private:
    /**
     * @brief Return the "true" subset arrays, containing the
     * NAIF ids of all the given bodies plus the ones required for all the
     * possible combinations of ephemeris evaluations
     *
     * @param brieOrRefEphUnit Brie data or global reference
     * @param referenceNaifIDs
     * @return trueNaifIDs
     *
     */
    template<typename JsonOrHostRef>
    static feta::scalar::Array<NaifId> makeTrueNaifIDs_(
        const JsonOrHostRef& brieEphUnit,
        const feta::scalar::Array<NaifId>& NaifIDs);
};

/* Explicit instantiations */
extern template feta::scalar::Array<NaifId>
EphUnit::makeTrueNaifIDs_<nlohmann::json>(
    const nlohmann::json& brieEphUnit, const feta::scalar::Array<NaifId>&);
extern template feta::scalar::Array<NaifId>
EphUnit::makeTrueNaifIDs_<EphUnit::GRef>(
    const EphUnit::GRef& brieEphUnit, const feta::scalar::Array<NaifId>&);

} // namespace metadata
} // namespace states
} // namespace brie
