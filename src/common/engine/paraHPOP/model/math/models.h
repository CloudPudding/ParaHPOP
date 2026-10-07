#pragma once

#include "paraHPOP/model/math/detail/Model.h"
#include "paraHPOP/model/physics.h"
#include "paraHPOP/model/samples.h"

namespace paraHPOP {
namespace model {
namespace math {
namespace detail {

/* Cartesian dimensional */
extern template class Model<samples::Cartesian, physics::CartesianDim>;

} // namespace detail
} // namespace math

/** @brief Cartesian dimensional */
using CartesianDim
    = math::detail::Model<samples::Cartesian, physics::CartesianDim>;

} // namespace model
} // namespace paraHPOP