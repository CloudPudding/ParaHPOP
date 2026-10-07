#pragma once

namespace interface {
namespace config {
namespace model {
namespace environment {
namespace keys {

/* JSON key strings of the environment's orientations surface, centralized
 * so the renaming session is a one-file edit. */
constexpr const char* orientations = "orientations";
constexpr const char* files        = "files";
constexpr const char* only         = "only";          /* load-time subset filter */
constexpr const char* localorbital = "local_orbital"; /* dynamic local-orbital frames */
constexpr const char* synodic      = "synodic";       /* synodic (co-rotating RTN) frames */

/* keys of a single local_orbital entry */
constexpr const char* lo_name       = "name";
constexpr const char* lo_convention = "convention";
constexpr const char* lo_from       = "from";
constexpr const char* lo_to         = "to";
constexpr const char* lo_corotating = "corotating";

/* keys of a single synodic entry (RTN + co-rotating are implied) */
constexpr const char* syn_name = "name";
constexpr const char* syn_from = "from";
constexpr const char* syn_to   = "to";

} // namespace keys
} // namespace environment
} // namespace model
} // namespace config
} // namespace interface
