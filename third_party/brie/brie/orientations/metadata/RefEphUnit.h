#pragma once

#include "brie/core/Metadata.h"
#include "brie/orientations/metadata/Layout.h"

namespace brie {
namespace orientations {
namespace metadata {

/**
 * @brief Non-owning reference to `brie::orientations::metadata::EphUnit`
 *
 */
template<typename EphUnitT, bool work, bool MaybeVolatile = false>
class RefEphUnit
    : public core::RefMetadata<EphUnitT::IntSize, 1, EphUnitT::RealSize, work,
          MaybeVolatile> {
    using ParentT = core::RefMetadata<EphUnitT::IntSize, 1, EphUnitT::RealSize,
        work, MaybeVolatile>;
    /* Data types aliases */
    using IntArrayT  = typename ParentT::IntArrayT;
    using RealArrayT = typename ParentT::RealArrayT;
    /* Vector type aliases */
    using IntVecT  = typename ParentT::IntVecT;
    using RealVecT = typename ParentT::RealVecT;

public:
    /** @brief Volatile is allowed only with work */
    static constexpr bool Volatile = MaybeVolatile && work;

private:

public:
    /**
     * @brief factory method to construct from the parent type
     *
     */
    DEVICEHOST() static RefEphUnit make(const ParentT& parent)
    {
        RefEphUnit out;
        static_cast<ParentT&>(out) = parent;
        return out;
    }

    /**
     * @brief Return the initial epoch for the given body count
     *
     * @param bodyCount
     * @return initialEpoch
     *
     */
    DEVICEHOST() decltype(auto) getInitialEpoch(const idx_t& bodyCount)
    {
        return this->template getReal<INITIALEPOCH>(bodyCount);
    }
    DEVICEHOST() decltype(auto) getInitialEpoch(const idx_t& bodyCount) const
    {
        return this->template getReal<INITIALEPOCH>(bodyCount);
    }

    /**
     * @brief Return the final epoch for a given body count
     *
     * @param bodyCount
     * @return finalEpoch
     *
     */
    DEVICEHOST() decltype(auto) getFinalEpoch(const idx_t& bodyCount)
    {
        return this->template getReal<FINALEPOCH>(bodyCount);
    }
    DEVICEHOST() decltype(auto) getFinalEpoch(const idx_t& bodyCount) const
    {
        return this->template getReal<FINALEPOCH>(bodyCount);
    }

    /**
     * @brief Return the  unit type for the given body count
     *
     * @param bodyCount
     * @return unit type
     *
     */
    DEVICEHOST() decltype(auto) getUnitType(const idx_t& bodyCount)
    {
        return this->template getInt<common::UNITTYPE>(bodyCount);
    }
    DEVICEHOST() decltype(auto) getUnitType(const idx_t& bodyCount) const
    {
        return this->template getInt<common::UNITTYPE>(bodyCount);
    }

    /**
     * @brief Return the target NaifID for the given body count
     *
     * @param bodyCount
     * @return targetNaifID
     *
     */
    DEVICEHOST() decltype(auto) getTarget(const idx_t& bodyCount)
    {
        return this->template getInt<common::TARGET>(bodyCount);
    }
    DEVICEHOST() decltype(auto) getTarget(const idx_t& bodyCount) const
    {
        return this->template getInt<common::TARGET>(bodyCount);
    }

    /**
     * @brief Return the data offset for the given body count
     *
     * @param bodyCount
     * @return dataOffset
     *
     */
    DEVICEHOST() decltype(auto) getDataOffset(const idx_t& bodyCount)
    {
        return this->template getInt<common::DATAOFFSET>(bodyCount);
    }
    DEVICEHOST() decltype(auto) getDataOffset(const idx_t& bodyCount) const
    {
        return this->template getInt<common::DATAOFFSET>(bodyCount);
    }

    /**
     * @brief Return the data size for the given body count
     *
     * @param bodyCount
     * @return dataSize
     *
     */
    DEVICEHOST() decltype(auto) getDataSize(const idx_t& bodyCount)
    {
        return this->template getInt<common::DATASIZE>(bodyCount);
    }
    DEVICEHOST() decltype(auto) getDataSize(const idx_t& bodyCount) const
    {
        return this->template getInt<common::DATASIZE>(bodyCount);
    }

    /**
     * @brief Return the body count index for a given target
     *
     * @param targetNaifID
     * @return bodyCount
     *
     */
    DEVICEHOST() idx_t getTargetBodyCount(const NaifId& target) const
    {
        idx_t out = this->size();
        for (idx_t i = 0; i < out; i++) {
            if (this->getTarget(i) == target) {
                out = i;
                break;
            }
        }

        return out;
    }

    /**
     * @brief Check if the given input NAIF ID is in this metadata
     *
     * @param NaifId
     * @return bool
     *
     */
    DEVICEHOST() bool hasBody(const NaifId& NaifID) const
    {
        idx_t count = this->getTargetBodyCount(NaifID);
        if (count < this->size())
            return true;
        else
            return false;
    }

    /** @brief Wrapper - check if the given target body count has a barycenter
     * in the current unit */
    DEVICEHOST() bool hasBarycenter(const idx_t& bodyCount) const
    {
        return hasBody(
            this->template getInt<metadata::body::BARYCENTER>(bodyCount));
    }

    /**
     * @brief Static method - Check if the given input NAIF ID is in the given
     * `feta::scalar::Array`
     *
     * @param arrayOfNaifID
     * @param NaifID
     * @return bool
     *
     */
    static bool hasBody(
        const feta::scalar::Array<NaifId>& IDs, const NaifId& testID)
    {
        bool out                               = false;
        feta::scalar::Array<NaifId>::GRef iref = IDs.hostRef();
        for (idx_t i = 0; i < IDs.size(); i++) {
            if (iref[i] == testID) {
                out = true;
                break;
            }
        }
        return out;
    }

    /**
     * @brief Update data offsets based on the data sizes currently stored in
     * this `brie::EphUnitMetaData`
     *
     */
    DEVICEHOST() void updateDataOffsets()
    {
        idx_t offset = 0;
        for (idx_t i = 0; i < this->size(); i++) {
            this->getDataOffset(i) = offset;
            offset += this->getDataSize(i);
        }
    }

    /**
     * @brief Check if the given epoch is in the available data range for the
     * given body count
     *
     * @param epoch
     * @param targetNaifID
     *
     */
    DEVICEHOST()
    bool isEpochInRange(const Real& epoch, const NaifId& target) const
    {
        idx_t bodyCount = this->getTargetBodyCount(target);
        return (epoch >= this->getInitialEpoch(bodyCount)
            && epoch <= this->getFinalEpoch(bodyCount));
    }

    /** @brief Clone these metadata reference */
    DEVICEHOST()
    RefEphUnit clone() const { return *this; }
};

} // namespace metadata
} // namespace orientations
} // namespace brie