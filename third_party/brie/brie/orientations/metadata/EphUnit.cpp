#include "brie/orientations/metadata/EphUnit.h"

#include <algorithm>
#include <vector>

namespace brie {
namespace orientations {
namespace metadata {

/**
 * @brief Return the "true" subset array, containing the NAIF ids of all the
 * given rotation targets plus the barycenter parents of every requested
 * Type-1 body unit
 *
 * @param referenceNaifIDs
 * @return trueNaifIDs
 *
 */
feta::scalar::Array<NaifId> EphUnit::makeTrueNaifIDs(
    const feta::scalar::Array<NaifId>& NaifIDs) const
{
    GRef ref                               = this->hostRef();
    feta::scalar::Array<NaifId>::GRef iref = NaifIDs.hostRef();

    /* collect the requested ids (deduplicated, order-preserving); unknown
     * ids are kept so that the subsequent subset creation throws with an
     * actionable message instead of silently dropping them */
    std::vector<NaifId> ids;
    auto push = [&ids](const NaifId& id) {
        if (std::find(ids.begin(), ids.end(), id) == ids.end())
            ids.push_back(id);
    };
    for (idx_t i = 0; i < NaifIDs.size(); i++) {
        push(iref[i]);
        /* Type-1 units reference their barycenter nutation-precession unit
         * by NaifId at evaluation time, and the correction silently drops
         * if the parent unit is absent: always carry it along */
        const idx_t bodyCount = ref.getTargetBodyCount(iref[i]);
        if (bodyCount < ref.size() && ref.getUnitType(bodyCount) == 1) {
            const NaifId bary = ref.getInt<body::BARYCENTER>(bodyCount);
            if (ref.hasBody(bary))
                push(bary);
        }
    }

    /* fill and return */
    feta::scalar::Array<NaifId> out(static_cast<idx_t>(ids.size()));
    feta::scalar::Array<NaifId>::GRef oref = out.hostRef();
    for (idx_t i = 0; i < static_cast<idx_t>(ids.size()); i++)
        oref[i] = ids[i];

    return out;
}

/**
 * @brief Read the metadata of a .brie file (Layout 1)
 *
 * @param brieEphemerisData
 *
 */
void EphUnit::readBrie1(const nlohmann::json& brieEphUnit)
{
    /* work with reference */
    GRef ref        = this->hostRef();
    idx_t bodyCount = 0;
    for (auto bodyUnit : brieEphUnit) {

        /* read int members */
        int unit_type              = bodyUnit["metadata"]["unit_type"];
        ref.getUnitType(bodyCount) = unit_type;

        /* data size */
        ref.getDataSize(bodyCount) = bodyUnit["data"].size();

        /* determine data offset for this body unit */
        if (bodyCount == 0) {
            ref.getDataOffset(bodyCount) = 0;
        } else {
            ref.getDataOffset(bodyCount) = ref.getDataOffset(bodyCount - 1)
                + ref.getDataSize(bodyCount - 1);
        }

        if (unit_type == 1) {

            ref.getTarget(bodyCount) = bodyUnit["metadata"]["target"];

            ref.getInt<body::PDEG_RA>(bodyCount)
                = bodyUnit["metadata"]["pdeg_ra"];
            ref.getInt<body::PDEG_DEC>(bodyCount)
                = bodyUnit["metadata"]["pdeg_dec"];
            ref.getInt<body::PDEG_PM>(bodyCount)
                = bodyUnit["metadata"]["pdeg_pm"];

            ref.getInt<body::NUM_DIMS_RADII>(bodyCount)
                = bodyUnit["metadata"]["num_dims_radii"];

            ref.getInt<body::BARYCENTER>(bodyCount)
                = bodyUnit["metadata"]["barycenter"];

            ref.getInt<body::NUM_NP_TERMS_RA>(bodyCount)
                = bodyUnit["metadata"]["num_np_terms_ra"];
            ref.getInt<body::NUM_NP_TERMS_DEC>(bodyCount)
                = bodyUnit["metadata"]["num_np_terms_dec"];
            ref.getInt<body::NUM_NP_TERMS_PM>(bodyCount)
                = bodyUnit["metadata"]["num_np_terms_pm"];

        } else if (unit_type == 2) {

            ref.getTarget(bodyCount) = bodyUnit["metadata"]["target"];

            ref.getInt<bary::PDEG_NP>(bodyCount)
                = bodyUnit["metadata"]["pdeg_np"];
            ref.getInt<bary::NUM_NP_ANGLES>(bodyCount)
                = bodyUnit["metadata"]["num_np_angles"];

        } else if (unit_type == 3) {

            ref.getTarget(bodyCount) = bodyUnit["metadata"]["frameClassID"];

            ref.getInt<binary::INERT_FRAME>(bodyCount)
                = bodyUnit["metadata"]["inertialFrame"];
            ref.getInt<binary::INTERPOL_TYPE>(bodyCount)
                = bodyUnit["metadata"]["dtype"];
            ref.getInt<binary::NINTERVALS>(bodyCount)
                = bodyUnit["metadata"]["nintervals"];
            ref.getInt<binary::PDEG>(bodyCount) = bodyUnit["metadata"]["pdeg"];

            /* read double members */
            ref.getInitialEpoch(bodyCount) = bodyUnit["metadata"]["startEpoch"];
            ref.getFinalEpoch(bodyCount)   = bodyUnit["metadata"]["finalEpoch"];

        } else if (unit_type == 4) {

            /* model-based orientation unit (native IPF → IERS2000) */
            ref.getTarget(bodyCount) = bodyUnit["metadata"]["target"];

            ref.getInt<iers::MODEL_ID>(bodyCount)
                = bodyUnit["metadata"]["model_id"];
            ref.getInt<iers::INTERP_ID>(bodyCount)
                = bodyUnit["metadata"]["interp_id"];
            ref.getInt<iers::N_NODES_NUT>(bodyCount)
                = bodyUnit["metadata"]["n_nodes_nut"];
            ref.getInt<iers::N_NODES_ERP>(bodyCount)
                = bodyUnit["metadata"]["n_nodes_erp"];
            ref.getInt<iers::LAG_ORDER_NUT>(bodyCount)
                = bodyUnit["metadata"]["lag_order_nut"];
            ref.getInt<iers::LAG_ORDER_ERP>(bodyCount)
                = bodyUnit["metadata"]["lag_order_erp"];

            /* read double members (envelope, SPICE seconds) */
            ref.getInitialEpoch(bodyCount) = bodyUnit["metadata"]["startEpoch"];
            ref.getFinalEpoch(bodyCount)   = bodyUnit["metadata"]["finalEpoch"];
        }

        /* increase count */
        bodyCount += 1;
    }
}

/**
 * @brief Read the metadata of a .brie file (Layout 2)
 *
 * @param brieEphemerisData
 *
 */
void EphUnit::readBrie2(const nlohmann::json& brieEphUnitMetaData)
{
    /* read integer members */
    std::copy(brieEphUnitMetaData["intMetadata"].begin(),
        brieEphUnitMetaData["intMetadata"].end(),
        this->intMembers_.hostRef().data());
    /* read double members */

    if (brieEphUnitMetaData.contains(
            "doubleMetadata")) { // create doubleMetadata only if it exists
                                 // in the json - in practice, with
                                 // unit_type 3
        std::copy(brieEphUnitMetaData["doubleMetadata"].begin(),
            brieEphUnitMetaData["doubleMetadata"].end(),
            this->realMembers_.hostRef().data());
    }
}

} // namespace metadata
} // namespace orientations
} // namespace brie