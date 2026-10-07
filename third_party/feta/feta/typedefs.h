#pragma once

#ifdef FETA_CPU_ONLY
#include <cmath>
#endif

#include "feta/macros.h"
#include <algorithm>
#include <vector>

namespace feta {
/** @brief Type used for vector dimensions */
using dims_t = unsigned int;
/** @brief Type used for array indices and sizes */
using idx_t = unsigned int;
} // namespace feta
