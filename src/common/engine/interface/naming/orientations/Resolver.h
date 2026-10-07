#pragma once

#include <string>
#include <vector>

#include "interface/naming/orientations/Registry.h"
#include "interface/util/err.h"

namespace interface {
namespace naming {
namespace orientations {

/** @brief Resolve the rotation-data target a body's force-model rotations
 * (spherical harmonics, J2, atmospheric drag) evaluate under, given an
 * optional explicit per-body frame setting and a predicate reporting
 * whether a candidate target's rotation data is actually loaded.
 *
 * Ranking, highest first:
 *   1. an explicit `bodies.<name>.orientation` setting (validated: known
 *      frame, body-fixed/binary not inertial, describes THIS body, data
 *      loaded) — users always name a frame, never an id;
 *   2. the unique loaded binary (`.brot` Type-3) frame of this body — the
 *      auto-best default (e.g. Earth picks up ITRF93 once its high-precision
 *      data is loaded, the Moon picks up MOON_PA), more than one is
 *      ambiguous and forces an explicit choice;
 *   3. the body's own NAIF id — its IAU body-fixed polynomial, always
 *      available and the historical hardwired behaviour. The auto fallback
 *      returns it WITHOUT consulting `targetLoaded`, so an environment that
 *      loads no binary frame for the body resolves to exactly the id the
 *      launch sites used to hardwire — bit-for-bit unchanged.
 *
 * Auto-registered `FRAME_<id>` frames carry no body association (body 0),
 * so they never enter the auto ranking and cannot be named explicitly for a
 * body until a future extension declares their body — by construction they
 * fail the "describes THIS body" check.
 *
 * @param body          NAIF id of the body whose orientation is selected
 * @param explicitName  user frame name/alias, or "" for auto-best
 * @param targetLoaded  predicate `bool(const OrientationId& target)` — is
 *                      that rotation-data target loaded into the environment?
 * @returns the rotation-data target id to feed the per-body rotation cache
 */
template <class LoadedPred>
inline OrientationId resolveBodyOrientationTarget(const NaifId& body,
    const std::string& explicitName, LoadedPred&& targetLoaded)
{
    auto& registry = Registry::instance();

    if (!explicitName.empty()) {
        const auto entry = registry.find(explicitName);
        PARAHPOP_ASSERT(entry.has_value(),
            "Body orientation '" + explicitName
                + "' is not a known orientation frame; name a builtin "
                  "(ITRF93, MOON_PA, <BODY>_BODY_FIXED) or a frame declared "
                  "in the environment's orientations");
        PARAHPOP_ASSERT(entry->kind != Kind::Inertial,
            "Body orientation '" + explicitName
                + "' is an inertial frame; a body's force-model rotation "
                  "needs a body-fixed or binary frame");
        PARAHPOP_ASSERT(entry->body == body,
            "Body orientation '" + explicitName + "' describes body "
                + std::to_string(entry->body) + ", not body "
                + std::to_string(body)
                + "; set it on the body it belongs to");
        PARAHPOP_ASSERT(targetLoaded(entry->dataTarget()),
            "Body orientation '" + explicitName
                + "' needs its rotation data loaded; add the defining .brot "
                  "to the environment's orientations");
        return entry->dataTarget();
    }

    /* auto-best: the unique loaded binary frame of this body, else the
     * body's IAU polynomial (id itself, always valid, no data required) */
    std::vector<Entry> loaded;
    for (const auto& e : registry.binaryFramesOf(body))
        if (targetLoaded(e.dataTarget()))
            loaded.push_back(e);

    PARAHPOP_ASSERT(loaded.size() <= 1,
        "Body " + std::to_string(body)
            + " has more than one loaded high-precision orientation frame; "
              "set bodies.<name>.orientation explicitly to choose between "
              "them");
    if (loaded.size() == 1)
        return loaded[0].dataTarget();
    return static_cast<OrientationId>(body);
}

} // namespace orientations
} // namespace naming
} // namespace interface
