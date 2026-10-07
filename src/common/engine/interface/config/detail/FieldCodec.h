#pragma once

/* Shared field-name codec for the config JSON parsers.
 *
 * Every config block that parses a JSON object loops over the keys and maps
 * each one to an enum via the same normalise-then-lookup recipe: lower-case
 * the key, strip underscores (so snake_case / camelCase / mixed forms all
 * collapse to one canonical lower-case key), then look the result up in a
 * `std::map<std::string, EnumT>`; a miss returns the block's INVALID
 * sentinel.
 *
 * This is the ONE shared body for that recipe. Each block keeps its own enum
 * and its own name map — the public JSON vocabulary lives there; only the
 * mechanical normalise+lookup is shared. Behaviour is byte-for-byte identical
 * to the hand-written bodies it replaces: hyphens are deliberately preserved
 * (only underscores are stripped) so literal hyphenated keys such as
 * "bi-directional" still match.
 *
 * NOTE: blocks whose recipe differs are intentionally NOT routed here — e.g.
 * `body::parse` (lower-cases only, never strips underscores) and
 * `time::Parser::parse` (lower-cases only and throws on a miss instead of
 * returning a sentinel). Folding those would silently change behaviour. The
 * per-block error-message builders (`ErrorMessage_`) are likewise left in
 * place: their text has drifted across blocks and is public-facing. */

#include "interface/typedefs.h"

#include <algorithm>
#include <map>
#include <string>
#include <utility>

namespace interface {
namespace config {
namespace detail {

/** @brief Normalise a JSON key (lower-case, then strip underscores) and look
 *  it up in @p nameMap, returning the mapped enum or @p invalid on a miss.
 *
 *  Shared body for the config field/direction/guard parsers — see file
 *  header. Hyphens are preserved; only underscores are stripped. */
template<typename EnumT>
inline EnumT parseField(std::string name,
    const std::map<std::string, EnumT>& nameMap, EnumT invalid)
{
    /* make lower case and strip underscores (accept snake_case keys) */
    brie::util::Strings::lowerCase(name);
    name.erase(std::remove(name.begin(), name.end(), '_'), name.end());

    /* find the corresponding iterator */
    auto iterator = nameMap.find(name);
    if (iterator != nameMap.end())
        return iterator->second;

    /* no valid key was parsed */
    return invalid;
}

} // namespace detail
} // namespace config
} // namespace interface
