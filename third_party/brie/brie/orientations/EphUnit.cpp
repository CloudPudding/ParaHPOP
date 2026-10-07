#include "brie/orientations/EphUnit.h"

namespace brie {
namespace orientations {

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
template<bool UseTexture>
void EphUnit<UseTexture>::copyFrom(const EphUnit<UseTexture>& otherEph)
{

    /* get references to work with */
    MetadataT::GRef otherRef = otherEph.metadata_.hostRef();
    MetadataT::GRef newMref  = metadata_.hostRef();

    /* read data */
    for (idx_t i = 0; i < newMref.size(); i++) {
        for (idx_t bodyCount = 0; bodyCount < otherRef.size(); bodyCount++) {
            if (newMref.getTarget(i) == otherRef.getTarget(bodyCount)) {
                Real* dst = data_.getHostData() + newMref.getDataOffset(i);
                const Real* src = otherEph.data_.getHostData()
                    + otherRef.getDataOffset(bodyCount);
                std::copy(src, src + newMref.getDataSize(i), dst);
                break;
            }
        }
    }
}

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
template<bool UseTexture>
void EphUnit<UseTexture>::copyTo(EphUnit<UseTexture>& otherEph) const
{

    /* get references to work with */
    MetadataT::GRef otherRef = otherEph.metadata_.hostRef();
    MetadataT::GRef newMref  = metadata_.hostRef();

    /* read data */
    for (idx_t bodyCount = 0; bodyCount < otherRef.size(); bodyCount++)
        for (idx_t i = 0; i < newMref.size(); i++) {
            {
                if (newMref.getTarget(i) == otherRef.getTarget(bodyCount)) {
                    /* compute start and end of offset */
                    Real* dst = otherEph.data_.getHostData()
                        + otherRef.getDataOffset(bodyCount);
                    const Real* src
                        = data_.getHostData() + newMref.getDataOffset(i);
                    std::copy(src, src + otherRef.getDataSize(bodyCount), dst);
                    break;
                }
            }
        }
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
template<bool UseTexture>
EphUnit<UseTexture> EphUnit<UseTexture>::merge(
    const EphUnit<UseTexture>& other) const
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

    /* Finally return */
    return ephs;
}

/* Explicit instantiation */
template class EphUnit<true>;
template class EphUnit<false>;


} // namespace orientations
} // namespace brie
