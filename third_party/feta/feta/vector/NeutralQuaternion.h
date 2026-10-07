#pragma once

#include "feta/vector/ConstantLeaf.h"

namespace feta {
namespace vector {

/**
 * @brief Represent the quaternion identity element as a lazy vector expression.
 *
 * Returns 1 for component 0 (scalar part) and 0 for all other components.
 *
 * @tparam DataT  Scalar component type.
 */
template<typename DataT>
using NeutralQuaternion = ConstantLeaf<DataT, 4, detail::QuaternionIdentity<DataT>>;

} // namespace vector
} // namespace feta
