#include "brie/states/metadata/EphUnit.h"
#include "brie/gravity.h"

namespace brie {
namespace states {
namespace metadata {

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
feta::scalar::Array<NaifId> EphUnit::makeTrueNaifIDs_(
    const JsonOrHostRef& brieEphUnit,
    const feta::scalar::Array<NaifId>& NaifIDs)
{
    idx_t i;
    NaifId cc, newcc;

    /* initialize vector */
    std::vector<NaifId> trueIdVec;
    feta::scalar::Array<NaifId>::GRef iref = NaifIDs.hostRef();
    for (i = 0; i < NaifIDs.size(); i++) {
        /* assert that the given body is contained in these ephemeris data */
        if (!Self::isContained(iref[i], brieEphUnit)) {
            /* This body is not contained in this ephemeris unit. The name
             * probe must be non-throwing: the missing id may lie outside
             * the NAIF body catalog (e.g. a spacecraft id requested as an
             * ephemeris target). */
            std::stringstream ss;
            std::string name;
            if (gravity::Parser::tryParsedName(iref[i], name))
                ss << "Body " << name << " (" << iref[i] << ")";
            else
                ss << "Id " << iref[i];
            ss << " could not be found in the loaded file(s)";
            BRIE_THROW(std::runtime_error, ss.str().c_str());
        }
        trueIdVec.push_back(iref[i]);
    }

    /* recursively include all the required centers */
    for (i = 0; i < NaifIDs.size(); i++) {
        /* find center */
        cc = Self::findCenter(iref[i], brieEphUnit);
        /* if center is not barycenter or contained, add it */
        if (std::find(trueIdVec.begin(), trueIdVec.end(), cc) == trueIdVec.end()
            && cc != 0) {
            /* add and advance */
            trueIdVec.push_back(cc);
            /* recursively do the same for the given centers */
            newcc = 1;
            while (newcc != 0) {
                newcc = 0; // work around 1 being a naif id
                newcc = Self::findCenter(cc, brieEphUnit);
                if (std::find(trueIdVec.begin(), trueIdVec.end(), newcc)
                        == trueIdVec.end()
                    && newcc != 0) {
                    trueIdVec.push_back(newcc);
                    cc = newcc;
                } else
                    break;
            }
        }
    }

    /* allocate memory */
    feta::scalar::Array<NaifId> out(trueIdVec.size());
    feta::scalar::Array<NaifId>::GRef outref = out.hostRef();
    i                                        = 0;
    for (NaifId id : trueIdVec) {
        outref[i] = id;
        i++;
    }

    /* finally return */
    return out;
}

/* Explicit instantiations */
template feta::scalar::Array<NaifId> EphUnit::makeTrueNaifIDs_<nlohmann::json>(
    const nlohmann::json& brieEphUnit, const feta::scalar::Array<NaifId>&);
template feta::scalar::Array<NaifId> EphUnit::makeTrueNaifIDs_<EphUnit::GRef>(
    const EphUnit::GRef& brieEphUnit, const feta::scalar::Array<NaifId>&);

} // namespace metadata
} // namespace states
} // namespace brie
