#pragma once

#include "brie/typedefs.h"
#include "brie/util.h"

namespace brie {
namespace core {

/** @brief Non-owning reference to the metadata */
template<idx_t IntSize, idx_t HandleSize, idx_t RealSize, bool work,
    bool MaybeVolatile = false>
class RefMetadata {
    /* No texture for metadata */
public:
    /** @brief Volatile is allowed only with work */
    static constexpr bool Volatile = MaybeVolatile && work;
    /** @brief Epose the data types */
    using IntArrayT    = typename feta::vector::texture::Array<NaifId, IntSize,
           false>::template Ref<work, MaybeVolatile>;
    using HandleArrayT = typename feta::vector::texture::Array<idx_t,
        HandleSize, false>::template Ref<work, MaybeVolatile>;
    using RealArrayT   = typename feta::vector::texture::Array<Real, RealSize,
          false>::template Ref<work, MaybeVolatile>;
    using IntVecT      = typename feta::vector::Item<NaifId, IntSize>;
    using HandleVecT   = typename feta::vector::Item<idx_t, HandleSize>;
    using RealVecT     = typename feta::vector::Item<Real, RealSize>;

    /**
     * @brief Factory method to construct from the references of the data
     * members
     *
     */
    DEVICEHOST()
    static RefMetadata make(const IntArrayT& intMembers,
        const HandleArrayT& handleMembers, const RealArrayT& realMembers)
    {
        return { intMembers, handleMembers, realMembers };
    }

    /**
     * @brief Return one of the integer data members
     *
     * @tparam intItemCount
     * @param bodyCount
     * @return intMember
     *
     */
    template<idx_t item>
    DEVICEHOST()
    decltype(auto) getInt(const idx_t& bodyCount)
    {
        return this->intMembers_.template get<item>(bodyCount);
    }
    template<idx_t item>
    DEVICEHOST()
    decltype(auto) getInt(const idx_t& bodyCount) const
    {
        return this->intMembers_.template get<item>(bodyCount);
    }

    /**
     * @brief Return a `feta::vector::Item<int>` containing all the integer
     * metadata for the given body count
     *
     * @param bodyCount
     * @return intVector
     *
     */
    DEVICEHOST() IntVecT getIntVec(const idx_t& bodyCount) const
    {
        feta::SampleIndex idx = feta::SampleIndex::make(bodyCount);
        IntVecT out           = this->intMembers_[idx];
        return out;
    }


    /**
     * @brief Assign the given integer vector members to the integer metadata
     * corresponding to the given body count
     *
     * @param bodyCount
     * @param intVector
     *
     */
    DEVICEHOST() void assignIntVec(const idx_t& bodyCount, const IntVecT& vec)
    {
        feta::SampleIndex idx  = feta::SampleIndex::make(bodyCount);
        this->intMembers_[idx] = vec;
    }

    /**
     * @brief Return one of the handle data members
     *
     * @tparam handleItemCount
     * @param bodyCount
     * @return handleMember
     *
     */
    template<idx_t item>
    DEVICEHOST()
    decltype(auto) getHandle(const idx_t& bodyCount)
    {
        return this->handleMembers_.template get<item>(bodyCount);
    }
    template<idx_t item>
    DEVICEHOST()
    decltype(auto) getHandle(const idx_t& bodyCount) const
    {
        return this->handleMembers_.template get<item>(bodyCount);
    }

    /**
     * @brief Return a `feta::vector::Item<idx_t>` containing all the handle
     * metadata for the given body count
     *
     * @param bodyCount
     * @return handleVector
     *
     */
    DEVICEHOST() HandleVecT getHandleVec(const idx_t& bodyCount) const
    {
        feta::SampleIndex idx = feta::SampleIndex::make(bodyCount);
        return this->handleMembers_[idx];
    }


    /**
     * @brief Assign the given handle vector members to the handle metadata
     * corresponding to the given body count
     *
     * @param bodyCount
     * @param handleVector
     *
     */
    DEVICEHOST()
    void assignHandleVec(const idx_t& bodyCount, const HandleVecT& vec)
    {
        feta::SampleIndex idx     = feta::SampleIndex::make(bodyCount);
        this->handleMembers_[idx] = vec;
    }


    /**
     * @brief Retrieve a Real metadata member
     *
     * @tparam RealItemCount
     * @param bodyCount
     * @return realMember
     *
     */
    template<idx_t item>
    DEVICEHOST()
    decltype(auto) getReal(const idx_t& bodyCount)
    {
        return this->realMembers_.template get<item>(bodyCount);
    }
    template<idx_t item>
    DEVICEHOST()
    decltype(auto) getReal(const idx_t& bodyCount) const
    {
        return this->realMembers_.template get<item>(bodyCount);
    }

    /**
     * @brief Return a `feta::vector::Item` vector containing the Real metadata
     * for a given body count
     *
     * @param bodyCount
     * @return RealMetadataVector
     *
     */
    DEVICEHOST() RealVecT getRealVec(const idx_t& bodyCount) const
    {
        feta::SampleIndex idx = feta::SampleIndex::make(bodyCount);
        return this->realMembers_[idx];
    }

    /**
     * @brief Assign the Real metadata members belonging to the given
     * `feta::vector::Item<int>` vector
     *
     * @param bodyCount
     * @param RealMetadataVector
     *
     */
    DEVICEHOST() void assignRealVec(const idx_t& bodyCount, const RealVecT& vec)
    {
        feta::SampleIndex idx   = feta::SampleIndex::make(bodyCount);
        this->realMembers_[idx] = vec;
    }

    /**
     * @brief Return the number of bodies in the corresponding
     * `brie::states::metadata::EphUnit`
     *
     * @return Nbodies
     *
     */
    DEVICEHOST() idx_t size() const { return this->intMembers_.size(); }

    /** @brief Create new global reference to this metadata set */
    DEVICEHOST() RefMetadata clone() const { return *this; }

    /** @brief Data members made public for PODification */

    /** @brief reference to the integer metadata members **/
    IntArrayT intMembers_;
    /** @brief reference to the handle metadata members **/
    HandleArrayT handleMembers_;
    /** @brief reference to the double metadata members **/
    RealArrayT realMembers_;
};

} // namespace core
} // namespace brie