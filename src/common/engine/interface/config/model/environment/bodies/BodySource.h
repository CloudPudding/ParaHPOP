#pragma once

/* Body feature configuration.
 *
 * Split for readability into three cohesive units, re-exported here so that
 * #include ".../bodies/BodySource.h" keeps its original public surface:
 *   - FeatureToggles.h : Features enum + name map + leaf toggle/view widgets
 *   - BodyConfig.h     : standalone owned Config
 *   - BodyView.h       : non-owning array views (Gravity/Body/ConstBody)
 */

#include "interface/config/model/environment/bodies/FeatureToggles.h"
#include "interface/config/model/environment/bodies/BodyConfig.h"
#include "interface/config/model/environment/bodies/BodyView.h"
