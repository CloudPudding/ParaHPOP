#include "paraHPOP/model/math/models.h"

namespace paraHPOP {
namespace model {
namespace math {
namespace detail {

/* Cartesian dimensional */
template class Model<samples::cartesian::Samples, physics::full::Dimensional>;

} // namespace detail
} // namespace math
} // namespace model
} // namespace paraHPOP