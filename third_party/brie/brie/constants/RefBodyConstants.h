#pragma once

#include <sstream>

#include "brie/typedefs.h"
#include "brie/util//DeviceError.h"
#include "brie/util//throw.h"

namespace brie {
namespace detail {

enum BODY_CONSTANTS : unsigned int {
    GM,
    R,
    AMEAN,
    J2,
    SOI,

    BODY_CONSTANT_COUNT
};


/** @brief Non-owning reference giving access to per-body physical constants */
class RefBodyConstants {
public:
    /** @brief Short-hand for the internal data type */
    using InnerT
        = feta::vector::Array<double, detail::BODY_CONSTANT_COUNT>::GRef;

    /** @brief Factory method to construct fom the given data */
    DEVICEHOST()
    static RefBodyConstants make(const InnerT& data, const idx_t& idx)
    {
        return { data, idx };
    }

    /** @brief Body radius [km] */
    DEVICEHOST() double r() const { return data_.template get<R>(bodyIdx_); }

    /** @brief Body gravitational parameter [km^3/s^2] */
    DEVICEHOST() double gm() const { return data_.template get<GM>(bodyIdx_); }

    /** @brief Mean semi-major axis of the body's orbit [km] */
    DEVICEHOST() double aMean() const
    {
        return data_.template get<AMEAN>(bodyIdx_);
    }

    /** @brief Body's oblateness parameter [-] */
    DEVICEHOST() double j2() const { return data_.template get<J2>(bodyIdx_); }

    /** @brief Body's Spher of Influence radius [km] */
    DEVICEHOST() double soi() const
    {
        return data_.template get<SOI>(bodyIdx_);
    }

    /** @brief Data members made public for POD-ification */
    InnerT data_;
    const idx_t bodyIdx_;
};

} // namespace detail
} // namespace brie
