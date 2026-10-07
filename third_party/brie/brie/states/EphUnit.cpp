#include "brie/states/EphUnit.h"

namespace brie {
namespace states {

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
template<bool UseTexture>
EphUnit<UseTexture> EphUnit<UseTexture>::merge(const EphUnit<UseTexture>& other)
{

    /* Create actual merge subset and extract the missing metadata */
    feta::scalar::Array<NaifId> otherIDs = other.metadata_.getNaifIDs();
    feta::scalar::Array<NaifId> missingIDs
        = this->metadata_.getMissingNaifIDs(otherIDs);

    /* every target already present: nothing to merge */
    if (missingIDs.size() == 0)
        return this->clone();

    EphUnit<UseTexture> missing = other.makeSubset(missingIDs);

    /* Create new metadata */
    MetadataT metadata = this->metadata_.merge(missing.metadata_);

    /* Create new empty data */
    DataT data(metadata.totalDataSize());

    /* Create new EphUnit */
    EphUnit ephs(std::move(metadata), std::move(data));

    Real* dst        = ephs.data_.getHostData();
    const Real* src  = data_.getHostData();
    const Real* msrc = missing.data_.getHostData();

    std::copy(src, src + this->data_.size(), dst);
    std::copy(msrc, msrc + missing.data_.size(), dst + this->data_.size());

    /* Chains will be built lazily when hostRef() is called */

    /* Finally return */
    return ephs;
}

/**
 * @brief Internal copy from method
 *
 */
template<bool UseTexture>
void EphUnit<UseTexture>::copyFrom_(
    const EphUnit<UseTexture>::MetadataT::GRef& otherRef,
    const EphUnit<UseTexture>::DataT& otherData)
{
    MetadataT::GRef newMref = metadata_.hostRef();
    /* read data */
    for (idx_t i = 0; i < newMref.size(); i++) {
        for (idx_t bodyCount = 0; bodyCount < otherRef.size(); bodyCount++) {
            if (newMref.getTarget(i) == otherRef.getTarget(bodyCount)) {
                /* compute start and end of offset */
                Real* dst = data_.getHostData() + newMref.getDataOffset(i);
                const Real* src = otherData.getHostData()
                    + otherRef.getDataOffset(bodyCount);
                std::copy(src, src + newMref.getDataSize(i), dst);
                break;
            }
        }
    }
}

/** @brief Internal copy to method */
template<bool UseTexture>
void EphUnit<UseTexture>::copyTo_(
    const EphUnit<UseTexture>::MetadataT::GRef& otherRef,
    EphUnit<UseTexture>::DataT& otherData) const
{
    MetadataT::GRef newMref = metadata_.hostRef();

    /* read data */
    for (idx_t bodyCount = 0; bodyCount < otherRef.size(); bodyCount++) {
        for (idx_t i = 0; i < newMref.size(); i++) {
            if (newMref.getTarget(i) == otherRef.getTarget(bodyCount)) {
                /* compute start and end of offset */
                Real* dst = otherData.getHostData()
                    + otherRef.getDataOffset(bodyCount);
                const Real* src
                    = data_.getHostData() + newMref.getDataOffset(i);
                std::copy(src, src + newMref.getDataSize(i), dst);
                break;
            }
        }
    }
}

/* Explicit instantiation */

template class EphUnit<true>;
template class EphUnit<false>;

} // namespace states
} // namespace brie
