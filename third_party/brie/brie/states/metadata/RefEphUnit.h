#pragma once

#include "brie/core/Metadata.h"

namespace brie {
namespace states {
namespace metadata {

/**
 * @brief Enumeration for integer metadata members
 *
 */
enum IntMembers {
    /* Integer members composing the Ephemeris metadata */
    FRAME,
    DTYPE,
    TARGET,
    CENTER,
    NINTERVALS,
    PDEG,
    DATAOFFSET,
    DATASIZE,
    /* Leave the following item as last -- it only acts as enum size */
    INTSIZE
};

/** @brief Enumeration for Handle members */
enum HandleMembers {
    /* Handle members composing the ephemeris metadata */
    CENTERPOS, /* The position of the center relative to*/
    /* Leave the following item as last -- it only acts as enum size */
    HANDLESIZE
};

/**
 * @brief Enumeration for Real metadata members
 *
 */
enum RealMembers {
    /* Real members composing the Ephemeris metadata */
    INITIALEPOCH,
    FINALEPOCH,
    /* Leave the following item as last -- it only act as enum size*/
    REALSIZE
};

/**
 * @brief Non-owning reference to `brie::states::metadata::EphUnit`
 *
 */
template<bool work, bool MaybeVolatile = false>
class RefEphUnit
    : public core::RefMetadata<INTSIZE, HANDLESIZE, REALSIZE, work,
          MaybeVolatile> {
    using ParentT
        = core::RefMetadata<INTSIZE, HANDLESIZE, REALSIZE, work, MaybeVolatile>;
    /* Data types aliases */
    using IntArrayT    = typename ParentT::IntArrayT;
    using HandleArrayT = typename ParentT::HandleArrayT;
    using RealArrayT   = typename ParentT::RealArrayT;
    /* Vector type aliases */
    using IntVecT    = typename ParentT::IntVecT;
    using HandleVecT = typename ParentT::HandleVecT;
    using RealVecT   = typename ParentT::RealVecT;

public:
    /** @brief Volatile is allowed only with work */
    static constexpr bool Volatile = MaybeVolatile && work;

private:

public:
    /**
     * @brief Factory method to construct from parent class
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
    DEVICEHOST()
    decltype(auto) getInitialEpoch(const idx_t& bodyCount)
    {
        return this->template getReal<INITIALEPOCH>(bodyCount);
    }
    DEVICEHOST()
    decltype(auto) getInitialEpoch(const idx_t& bodyCount) const
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
    DEVICEHOST()
    decltype(auto) getFinalEpoch(const idx_t& bodyCount)
    {
        return this->template getReal<FINALEPOCH>(bodyCount);
    }
    DEVICEHOST()
    decltype(auto) getFinalEpoch(const idx_t& bodyCount) const
    {
        return this->template getReal<FINALEPOCH>(bodyCount);
    }

    /**
     * @brief Return the reference frame ID for the given body count
     *
     * @param bodyCount
     * @return frameID
     *
     */
    DEVICEHOST() decltype(auto) getFrame(const idx_t& bodyCount)
    {
        return this->template getInt<FRAME>(bodyCount);
    }
    DEVICEHOST() decltype(auto) getFrame(const idx_t& bodyCount) const
    {
        return this->template getInt<FRAME>(bodyCount);
    }

    /**
     * @brief Return the SPICE kernel type for the given body count
     *
     * @param bodyCount
     * @return spiceKernelID
     *
     */
    DEVICEHOST() decltype(auto) getDtype(const idx_t& bodyCount)
    {
        return this->template getInt<DTYPE>(bodyCount);
    }
    DEVICEHOST() decltype(auto) getDtype(const idx_t& bodyCount) const
    {
        return this->template getInt<DTYPE>(bodyCount);
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
        return this->template getInt<TARGET>(bodyCount);
    }
    DEVICEHOST() decltype(auto) getTarget(const idx_t& bodyCount) const
    {
        return this->template getInt<TARGET>(bodyCount);
    }

    /**
     * @brief Return the center NaifID for the given body count
     *
     * @param bodyCount
     * @return centerNaifID
     *
     */
    DEVICEHOST() decltype(auto) getCenter(const idx_t& bodyCount)
    {
        return this->template getInt<CENTER>(bodyCount);
    }
    DEVICEHOST() decltype(auto) getCenter(const idx_t& bodyCount) const
    {
        return this->template getInt<CENTER>(bodyCount);
    }

    /**
     * @brief Return the number of intervals for the given body count
     *
     * @param bodyCount
     * @return numberOfIntervals
     *
     */
    DEVICEHOST() decltype(auto) getNintervals(const idx_t& bodyCount)
    {
        return this->template getInt<NINTERVALS>(bodyCount);
    }
    DEVICEHOST() decltype(auto) getNintervals(const idx_t& bodyCount) const
    {
        return this->template getInt<NINTERVALS>(bodyCount);
    }

    /**
     * @brief Return the polynomial degree for the given body count
     *
     * @param bodyCount
     * @return polynomialDegree
     *
     */
    DEVICEHOST() decltype(auto) getPdeg(const idx_t& bodyCount)
    {
        return this->template getInt<PDEG>(bodyCount);
    }
    DEVICEHOST() decltype(auto) getPdeg(const idx_t& bodyCount) const
    {
        return this->template getInt<PDEG>(bodyCount);
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
        return this->template getInt<DATAOFFSET>(bodyCount);
    }
    DEVICEHOST() decltype(auto) getDataOffset(const idx_t& bodyCount) const
    {
        return this->template getInt<DATAOFFSET>(bodyCount);
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
        return this->template getInt<DATASIZE>(bodyCount);
    }
    DEVICEHOST() decltype(auto) getDataSize(const idx_t& bodyCount) const
    {
        return this->template getInt<DATASIZE>(bodyCount);
    }

    /**
     * @brief Return the center position for the given body count
     *
     * @param bodyCount
     * @return centerPos
     *
     */
    DEVICEHOST() decltype(auto) getCenterPos(const idx_t& bodyCount)
    {
        return this->template getHandle<CENTERPOS>(bodyCount);
    }
    DEVICEHOST() decltype(auto) getCenterPos(const idx_t& bodyCount) const
    {
        return this->template getHandle<CENTERPOS>(bodyCount);
    }

    /**
     * @brief Return the body count index for a given target
     *
     * @param targetNaifID
     * @return bodyCount
     *
     */
    DEVICEHOST()
    idx_t getTargetBodyCount(const NaifId& target) const
    {
        if (target == 0)
            return this->size();
        else {
            idx_t out = this->size() + 1;
            for (idx_t i = 0; i < this->size(); i++) {
                if (this->getTarget(i) == target) {
                    return i;
                }
            }
            return out;
        }
    }

    /**
     * @brief Return the center body count index for a given target
     *
     * @param CentertNaifID
     * @return bodyCount
     *
     */
    DEVICEHOST() idx_t getCenterBodyCount(const NaifId& target) const
    {
        if (target == 0)
            return this->size();
        else {
            idx_t out = this->size() + 1;
            for (idx_t i = 0; i < this->size(); i++) {
                if (this->getCenter(i) == target) {
                    return i;
                }
            }
            return out;
        }
    }

    /**
     * @brief Check if the given TARGET NAIF ID is in this metadata
     *
     * @param NaifID
     * @return bool
     *
     */
    DEVICEHOST() bool hasTargetBody(const NaifId& NaifID) const
    {
        /* Search among the targets */
        idx_t count = this->getTargetBodyCount(NaifID);
        if (count <= this->size())
            return true;
        else
            return false;
    }


    /**
     * @brief Check if the given CENTER NAIF ID is in this metadata
     *
     * @param NaifID
     * @return bool
     *
     */
    DEVICEHOST() bool hasCenterBody(const NaifId& NaifID) const
    {
        /* Search among the centers */
        idx_t count = this->getCenterBodyCount(NaifID);
        if (count <= this->size())
            return true;
        else
            return false;
    }

    /**
     * @brief Check if the given NAIF ID is in this metadata
     *
     * @param NaifID
     * @return bool
     *
     */
    DEVICEHOST() bool hasBody(const NaifId& NaifID) const
    {
        /* Search among the targets first */
        if (this->hasTargetBody(NaifID))
            return true;
        else /* search among the targets as well */
            return this->hasCenterBody(NaifID);
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
                return true;
            }
        }
        return out;
    }

    /**
     * @brief Update data offsets based on the data sizes currently stored in
     * this `brie::states::metadata::EphUnit`
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
     * @brief Update the center positions based on the data sizes currently
     * stored in this `brie::states::metadata::EphUnit`
     *
     */
    DEVICEHOST() void updateCenterPositions()
    {
        for (idx_t i = 0; i < this->size(); i++) {
            getCenterPos(i) = getTargetBodyCount(getCenter(i));
        }
    }

    /**
     * @brief Find the highest possible common center to be read for the
     * given target-center pair
     *
     * @param targetNaifID
     * @param centerNaifID
     * @return commonCenterNaifID
     *
     */
    DEVICEHOST()
    NaifId commonCenter(const NaifId& target, const NaifId& center) const
    {
        return commonCenter_(target, center);
    }

    /**
     * @brief Find the highest possible common center to be read for the
     * given target-center positions pair
     *
     * @param targetPos
     * @param centerPos
     * @return commonCenterPos
     *
     */
    DEVICEHOST()
    idx_t commonCenterPos(const idx_t& target, const idx_t& center) const
    {
        return commonCenterPos_(target, center);
    }

    /**
     * @brief Check if the given epoch is in the available data range for
     * the given body count
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

    /**
     * @brief Check if the given target and center are contained in this
     * `brie::EphUnit`, as well as if the requested epoch is available for
     * both target and center
     *
     * @param epoch
     * @param targetNaifID
     * @param centerNaifID
     * @return bool True if all the checks are passed
     *
     */
    DEVICEHOST()
    void allChecks([[maybe_unused]] const Real& epoch,
        [[maybe_unused]] const NaifId& target,
        [[maybe_unused]] const NaifId& center) const
    {
#ifdef BRIE_DEBUG_MODE
#ifdef __CUDA_ARCH__
        /* Body must be available */
        if (!this->hasBody(target)) {
            BRIE_GPU_THROW(err::TARGET_BODY_NOT_IN_UNIT);
        }
        if (!this->hasBody(center)) {
            BRIE_GPU_THROW(err::CENTER_BODY_NOT_IN_UNIT);
        }
        /* If the body is available, check that epochs are available for it
        only
         * if it's NOT a pure center*/
        if (this->hasTargetBody(target)) {
            if (!this->isEpochInRange(epoch, target)) {
                BRIE_GPU_THROW(err::OUT_OF_RANGE_TARGET_EPOCH);
            }
        }
        if (this->hasTargetBody(center)) {
            if (!this->isEpochInRange(epoch, center)) {
                BRIE_GPU_THROW(err::OUT_OF_RANGE_CENTER_EPOCH);
            }
        }

#else
        /* Body must be available */
        if (!this->hasBody(target)) {
            std::stringstream ss;
            ss << "TARGET body not in the current unit ";
            ss << "(" << target << ")";
            BRIE_THROW(std::runtime_error, ss.str().c_str());
        }
        if (!this->hasBody(center)) {
            std::stringstream ss;
            ss << "TARGET body not in the current unit ";
            ss << "(" << center << ")";
            BRIE_THROW(std::runtime_error, ss.str().c_str());
        }
        /* If the body is available, check that epochs are available for it
        only
         * if it's NOT a pure center*/
        if (this->hasTargetBody(target)) {
            if (!this->isEpochInRange(epoch, target)) {
                BRIE_THROW(std::runtime_error,
                    "Epoch for TARGET not in the available range");
            }
        }
        if (this->hasTargetBody(center)) {
            if (!this->isEpochInRange(epoch, center)) {
                BRIE_THROW(std::runtime_error,
                    "Epoch for CENTER not in the available range");
            }
        }

#endif
#endif
    }

    /**
     * @brief Check if the given target and center locations are contained in
     * this `brie::EphUnit`, as well as if the requested epoch is available for
     * both target and center
     *
     * @param epoch
     * @param targetNaifID
     * @param centerNaifID
     * @return bool True if all the checks are passed
     *
     */
    DEVICEHOST()
    void allChecksLoc([[maybe_unused]] const Real& epoch,
        [[maybe_unused]] const idx_t& targetPos,
        [[maybe_unused]] const idx_t& centerPos) const
    {
/* Check the validit of epochs and bodies */
#ifdef BRIE_DEBUG_MODE
#ifdef __CUDA_ARCH__
        BRIE_GPU_ASSERT(
            targetPos <= this->size(), err::TARGET_BODY_NOT_IN_UNIT);
        BRIE_GPU_ASSERT(
            centerPos <= this->size(), err::CENTER_BODY_NOT_IN_UNIT);
        if (targetPos != this->size()) {
            if (!this->isEpochInRange(epoch, this->getTarget(targetPos))) {
                BRIE_GPU_THROW(err::OUT_OF_RANGE_TARGET_EPOCH);
            }
        }
        if (centerPos != this->size()) {
            if (!this->isEpochInRange(epoch, this->getTarget(centerPos))) {
                BRIE_GPU_THROW(err::OUT_OF_RANGE_CENTER_EPOCH);
            }
        }
#else
        BRIE_ASSERT(
            targetPos <= this->size(), "TARGET body not in the current unit");
        BRIE_ASSERT(
            centerPos <= this->size(), "CENTER body not in the current unit");
        if (targetPos != this->size()) {
            if (!this->isEpochInRange(epoch, this->getTarget(targetPos))) {
                BRIE_THROW(std::runtime_error,
                    "Epoch for TARGET not in the available range");
            }
        }
        if (centerPos != this->size()) {
            if (!this->isEpochInRange(epoch, this->getTarget(centerPos))) {
                BRIE_THROW(std::runtime_error,
                    "Epoch for CENTER not in the available range");
            }
        }
#endif
#endif
    }

    /** @brief Create new global reference to this metadata set */
    DEVICEHOST() RefEphUnit<work> clone() const { return *this; }

protected:
    /**
     * @brief Find the highest possible common center to be read for the
     * given target-center pair
     *
     * @param targetNaifID
     * @param centerNaifID
     * @return commonCenterNaifID
     *
     */
    DEVICEHOST()
    NaifId commonCenter_(const NaifId& target, const NaifId& center) const
    {
        /* Solar System Barycenter case */
        if (target == 0 || center == 0) {
            return 0;
        }
        /* search the first common center otherwise */
        else {
            NaifId tc = target;
            NaifId cc = center;
            NaifId tcnew, ccnew;
            while (tc != cc && tc != 0 && cc != 0) {
                /* read the center of the target */
                tcnew = this->getCenter(this->getTargetBodyCount(tc));
                /* check if it is the center of the old center */
                if (tcnew == cc)
                    return tcnew;
                /* read the center of the center */
                ccnew = this->getCenter(this->getTargetBodyCount(cc));
                /* check if it is the center of the old target */
                if (ccnew == tc)
                    return ccnew;
                /* update the old values */
                tc = tcnew;
                cc = ccnew;
            }
            /* return the newly found common center */
            if (tc == 0 || cc == 0)
                return 0;
            else
                return tc;
        }
    }

    /**
     * @brief Find the highest possible common center position to be read
     * for the given target-center pair
     *
     * @param targetNaifID
     * @param centerNaifID
     * @return commonCenterNaifID
     *
     */
    DEVICEHOST()
    idx_t commonCenterPos_(const idx_t& targetPos, const idx_t& centerPos) const
    {
        /* Solar System Barycenter case */
        if (targetPos == this->size() || centerPos == this->size()) {
            return this->size();
        }
        /* search the first common center otherwise */
        else {
            idx_t tc = targetPos;
            idx_t cc = centerPos;
            idx_t tcnew, ccnew;
            while (tc != cc && tc != this->size() && cc != this->size()) {
                /* read the center of the target */
                tcnew = this->getCenterPos(tc);
                /* check if it is the center of the old center */
                if (tcnew == cc)
                    return tcnew;
                /* read the center of the center */
                ccnew = this->getCenterPos(cc);
                /* check if it is the center of the old target */
                if (ccnew == tc)
                    return ccnew;
                /* update the old values */
                tc = tcnew;
                cc = ccnew;
            }
            /* return the newly found common center */
            if (tc == this->size() || cc == this->size())
                return this->size();
            else
                return tc;
        }
    }
};

} // namespace metadata
} // namespace states
} // namespace brie