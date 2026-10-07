#pragma once

#include "feta/vector/ConstantLeaf.h"

namespace feta {
namespace vector {

/** @brief A vector expression where each component is 1. */
template<typename DataT, dims_t VectorDim>
using OnesNT = ConstantLeaf<DataT, VectorDim, detail::UniformOne<DataT>>;

} // namespace vector
} // namespace feta
