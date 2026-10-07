#pragma once

#include <cmath>
#include <feta/feta.h>
#include <limits>
#include <nlohmann/json.hpp>

/** @brief Top-level parm namespace. */
namespace parm {

/** @brief Re-export FETA's per-sample index type. */
using feta::SampleIndex;

/** @brief Scalar floating-point type used throughout parm. */
using Real     = double;
/** @brief Index type (signed, from FETA). */
using idx_t    = feta::idx_t;
/** @brief General-purpose signed integer type for metadata fields. */
using mInt_t   = int;
/** @brief Unsigned integer type. */
using uint_t   = unsigned int;
/** @brief Dimension type (from FETA). */
using dims_t   = feta::dims_t;
/** @brief Vector-dimension type (alias for ``dims_t``). */
using vecdim_t = feta::dims_t;

/** @brief FETA vector array with element type ``DT`` and dimension ``VD``. */
template<typename DT, idx_t VD>
using VecNTArray = feta::vector::Array<DT, VD>;
/** @brief FETA vector item with element type ``DT`` and dimension ``VD``. */
template<typename DT, idx_t VD>
using VecNT = feta::vector::Item<DT, VD>;

/** @brief 3-D double-precision vector. */
using Vec3R      = VecNT<Real, 3>;
/** @brief 6-D double-precision vector (position + velocity). */
using Vec6R      = VecNT<Real, 6>;
/** @brief SoA array of 3-D double-precision vectors. */
using Vec3RArray = VecNTArray<Real, 3>;
/** @brief SoA array of 6-D double-precision vectors. */
using Vec6RArray = VecNTArray<Real, 6>;

/** @brief Shorthand for the global reference type of a FETA scalar array. */
template<typename ComponentT>
using GRefArrT = typename feta::scalar::Array<ComponentT>::GRef;

} // namespace parm