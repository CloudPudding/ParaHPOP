#pragma once

#include "interface/typedefs.h"

namespace interface {
namespace naming {
namespace bands {

/* Reserved id bands carved out of the high positive NAIF-id namespace.
 * Real NAIF usage tops out around 5.4e7 (asteroid/comet ids); negative ids
 * (spacecraft) are used as-is. All ceilings stay below INT32_MAX so the
 * int-based core representation is collision-safe. */

namespace sample {
/** @brief Samples occupy [BASE, END): `Sample_<idx>` maps to BASE + idx */
constexpr NaifId BASE = 1900000000;
constexpr NaifId END  = 2100000000;
} // namespace sample

namespace custompoint {
/** @brief Registry-assigned custom-point ids occupy [BASE, END) */
constexpr NaifId BASE = 1800000000;
constexpr NaifId END  = 1900000000;
} // namespace custompoint

namespace dynamicorientation {
/** @brief Documented-reserved for future dynamic orientation frames
 * (co-rotating, ITRF). No producer or consumer yet. */
constexpr NaifId BASE = 1700000000;
constexpr NaifId END  = 1800000000;
} // namespace dynamicorientation

/** @brief Whether the given id lies in the reserved sample band */
inline bool inSampleBand(const NaifId& id)
{
    return id >= sample::BASE && id < sample::END;
}

/** @brief Whether the given id lies in the reserved custom-point band */
inline bool inCustomPointBand(const NaifId& id)
{
    return id >= custompoint::BASE && id < custompoint::END;
}

/** @brief Whether the given id lies in any reserved band */
inline bool inReservedBands(const NaifId& id)
{
    return id >= dynamicorientation::BASE && id < sample::END;
}

/** @brief Sample collection index encoded in the given sample-band id */
inline idx_t sampleIndexOf(const NaifId& id)
{
    return static_cast<idx_t>(id - sample::BASE);
}

/** @brief Sample-band id encoding the given sample collection index */
inline NaifId sampleIdOf(const idx_t& idx)
{
    return sample::BASE + static_cast<NaifId>(idx);
}

} // namespace bands
} // namespace naming
} // namespace interface
