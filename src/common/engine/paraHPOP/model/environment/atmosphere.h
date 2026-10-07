#pragma once

/**
 * @brief Umbrella header for atmosphere density models.
 *
 * The drag kernel includes this single header to reach all concrete
 * atmosphere models.  Adding a new model (e.g. NRLMSISE, Mars-GRAM)
 * means dropping a sibling under ``environment/atmosphere/`` and
 * adding the include here.
 */

#include "paraHPOP/model/environment/atmosphere/Atmosphere.h"
#include "paraHPOP/model/environment/atmosphere/PiecewiseExponentialAtmosphere.h"
#include "paraHPOP/model/environment/atmosphere/Nrlmsise00.h"
