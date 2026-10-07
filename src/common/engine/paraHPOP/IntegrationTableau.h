#pragma once

#include "parm/integrate/rk/coefs/tableau.h"

namespace paraHPOP {

// Shared thirteen-stage RKF78 coefficients for the CPU and GPU backends.
using IntegrationTableau = parm::integrate::rk::coefs::RKF78;

} // namespace paraHPOP
