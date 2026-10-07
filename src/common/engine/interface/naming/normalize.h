#pragma once

#include <algorithm>
#include <string>

#include "interface/typedefs.h"

namespace interface {
namespace naming {

/** @brief Canonical name normalization for all naming lookups: lower case
 * plus stripped whitespace, underscores, and hyphens. "Sun-Earth_L1",
 * "sun earth l1", and "SunEarthL1" all normalize to "sunearthl1". */
inline std::string normalized(std::string name)
{
    brie::util::Strings::lowerCase(name);
    name.erase(std::remove_if(name.begin(), name.end(),
                   [](const char& c) {
                       return c == ' ' || c == '_' || c == '-' || c == '\t';
                   }),
        name.end());
    return name;
}

} // namespace naming
} // namespace interface
