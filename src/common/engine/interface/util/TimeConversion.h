#pragma once

#include "interface/typedefs.h"

#define SECONDSPERDAY 86400.0
#define HALFDAY 0.5
#define MHALFSECONDSPERDAY -43200.0

namespace interface {
namespace util {

using brie::Real;

namespace detail {

/** @brief templated FMA */
template<typename T>
DEVICEHOST()
constexpr T timefma(const T& a, const T& b, const T& c)
{
    return a * b + c;
}

} // namespace detail

/** @brief Collection of time conversion routines */
struct TimeConversion {

    /** @brief Convert MJD2000 to SPICE time */
    DEVICEHOST() static constexpr Real mjdToSpice(const Real& mjd)
    {
        return detail::timefma(mjd, SECONDSPERDAY, MHALFSECONDSPERDAY);
    }

    /** @brief Convert SPICE time to MJD2000 */
    DEVICEHOST() static constexpr Real spiceToMjd(const Real& spice)
    {
        constexpr Real dps = 1.0 / SECONDSPERDAY;
        return detail::timefma(spice, dps, HALFDAY);
    }

    /** @brief Convert Days to Seconds */
    DEVICEHOST() static constexpr Real daysToSeconds(const Real& days)
    {
        return days * SECONDSPERDAY;
    }

    /** @brief Convert Seconds to Days */
    DEVICEHOST() static constexpr Real secondsToDays(const Real& seconds)
    {
        constexpr Real dps = 1.0 / SECONDSPERDAY;
        return seconds * dps;
    }
};

/** @brief Days to seconds conversions */
struct DaysToSeconds {
    /** @brief Convert Days to Seconds */
    DEVICEHOST() static constexpr Real forwardEval(const Real& days)
    {
        return TimeConversion::daysToSeconds(days);
    }

    /** @brief Convert Seconds to Days */
    DEVICEHOST() static constexpr Real backwardEval(const Real& seconds)
    {
        return TimeConversion::secondsToDays(seconds);
    }
};

/** @brief MJD2000 to SPICE time conversions */
struct MjdToSPice {
    /** @brief Convert MJD2000 to SPICE time */
    DEVICEHOST() static constexpr Real forwardEval(const Real& mjd)
    {
        return TimeConversion::mjdToSpice(mjd);
    }

    /** @brief Convert SPICE time to MJD2000 */
    DEVICEHOST() static constexpr Real backwardEval(const Real& spice)
    {
        return TimeConversion::spiceToMjd(spice);
    }
};

} // namespace util
} // namespace interface