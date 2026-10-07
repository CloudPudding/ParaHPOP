#pragma once

#include "feta/vector/ConstantLeaf.h"

namespace feta {
namespace vector {

/** @brief A vector expression where each component is 0. */
template<typename DataT, dims_t VectorDim>
using ZerosNT = ConstantLeaf<DataT, VectorDim, detail::UniformZero<DataT>>;

} // namespace vector
} // namespace feta
