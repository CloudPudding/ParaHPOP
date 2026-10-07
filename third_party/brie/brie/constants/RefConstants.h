#pragma once

#include <sstream>

#include "brie/typedefs.h"
#include "brie/util//DeviceError.h"
#include "brie/util//throw.h"

#include "brie/constants/RefBodyConstants.h"

namespace brie {
namespace detail {
enum UNIVERSAL_CONSTANTS : unsigned int {
    AU,
    CLIGHT,
    /* Leave the following item as last - it only acts as enum size */
    UNI_CONSTANT_COUNT
};

/** @brief Non-owning reference giving access to universal physical constants */
class RefConstants {
public:
    /** @brief factory method to construct from data members */
    DEVICEHOST()
    static RefConstants make(const feta::scalar::Array<NaifId>::GRef& bodyIds,
        const feta::scalar::Array<double>::GRef& values)
    {
        RefBodyConstants::InnerT bodyDataView;
        bodyDataView.data_      = values.data() + detail::UNI_CONSTANT_COUNT;
        bodyDataView.nVecs_     = bodyIds.size();
        bodyDataView.dimOffset_ = bodyIds.size();
        return { bodyIds, values, bodyDataView };
    }

    /** @brief Speed of light in a vacuum [km/s] */
    DEVICEHOST() double cLight() const { return values_[detail::CLIGHT]; }

    /** @brief Astronomical unit [km] */
    DEVICEHOST() double au() const { return values_[detail::AU]; }

    /**
     * @brief Access to a body's constants by NAIF ID
     *
     * NOTE: This performs a linear search over the stored body IDs. If you need
     * to access multiple constants for the same body, it is recommended to keep
     * a copy of this method's return value, so as to avoid re-doing the search.
     *
     * @param naifId NAIF ID of the body of interest.
     * @throws An error is thrown if the NAIF ID is not found. On host,
     * `std::runtime_error` is thrown; on device, `brie::err::INVALID_NAIF` is
     * signalled to the host.
     * @return A non-owning reference to the constants for the requested body.
     */
    DEVICEHOST() RefBodyConstants body(const NaifId& naifId) const
    {
        idx_t thei = bodyIds_.size();
        for (idx_t i = 0; i < bodyIds_.size(); i++) {
            if (naifId == bodyIds_[i]) {
                thei = i;
                break;
            }
        }
        if (thei == bodyIds_.size()) {
#ifdef __CUDA_ARCH__
            BRIE_GPU_THROW(err::INVALID_NAIF);
#else
            auto lazyErrMsg = [&naifId]() {
                std::stringstream ss;
                ss << "Unrecognised NAIF ID: " << naifId;
                return ss.str();
            };
            BRIE_THROW(std::runtime_error, lazyErrMsg());
#endif
        }

        return { bodyDataView_, thei };
    }

    /** @brief create a new global reference to these constants */
    DEVICEHOST() RefConstants clone() const { return *this; }

    /** @brief Data members made public for PODification */
    feta::scalar::Array<NaifId>::GRef bodyIds_;
    feta::scalar::Array<double>::GRef values_;
    RefBodyConstants::InnerT bodyDataView_;
};

} // namespace detail
} // namespace brie
