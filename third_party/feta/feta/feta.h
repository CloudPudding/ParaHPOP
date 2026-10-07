#pragma once

#include "feta/typedefs.h"

#include "feta/buffer/buffer.h"

#include "feta/core/SampleIndex.h"

#include "feta/err/DeviceError.h"

#include "feta/scalar/Array.h"

#include "feta/vector/Array.h"
#include "feta/vector/expr/ComponentView.h"
#include "feta/vector/expr/cpu/TiledEval.h"
#include "feta/vector/expr/View.h"

#include "feta/core/simd/simd.h"
#include "feta/vector/expr/cpu/PacketOps.h"

#include "feta/math/math.h"

namespace feta {

/** @brief Owning scalar array of `double` values (host + device). */
using DoubleArray = scalar::Array<double>;
/** @brief Owning scalar array of `float` values (host + device). */
using FloatArray  = scalar::Array<float>;
/** @brief Owning scalar array of `bool` values (host + device). */
using BoolArray   = scalar::Array<bool>;
/** @brief Owning scalar array of `int` values (host + device). */
using IntArray    = scalar::Array<int>;

/** @brief Owning 3-component vector array with arbitrary scalar type. */
template<typename DataType>
using Vec3TArray = vector::Array<DataType, 3>;
/** @brief Owning 4-component vector array with arbitrary scalar type. */
template<typename DataType>
using Vec4TArray = vector::Array<DataType, 4>;
/** @brief Owning 6-component vector array with arbitrary scalar type. */
template<typename DataType>
using Vec6TArray = vector::Array<DataType, 6>;
/** @brief Owning 9-component vector array with arbitrary scalar type. */
template<typename DataType>
using Vec9TArray = vector::Array<DataType, 9>;

/** @brief Owning 3-component double vector array (host + device). */
using Vec3dArray = Vec3TArray<double>;
/** @brief Owning 4-component double vector array (host + device). */
using Vec4dArray = Vec4TArray<double>;
/** @brief Owning 6-component double vector array (host + device). */
using Vec6dArray = Vec6TArray<double>;
/** @brief Owning 9-component double vector array (host + device). */
using Vec9dArray = Vec9TArray<double>;

/** @brief Owning 3-component float vector array (host + device). */
using Vec3fArray = Vec3TArray<float>;
/** @brief Owning 4-component float vector array (host + device). */
using Vec4fArray = Vec4TArray<float>;
/** @brief Owning 6-component float vector array (host + device). */
using Vec6fArray = Vec6TArray<float>;
/** @brief Owning 9-component float vector array (host + device). */
using Vec9fArray = Vec9TArray<float>;

/** @brief Per-thread 3-component vector with arbitrary scalar type. */
template<typename DataType>
using Vec3T = vector::Item<DataType, 3>;
/** @brief Per-thread 4-component vector with arbitrary scalar type. */
template<typename DataType>
using Vec4T = vector::Item<DataType, 4>;
/** @brief Per-thread 6-component vector with arbitrary scalar type. */
template<typename DataType>
using Vec6T = vector::Item<DataType, 6>;
/** @brief Per-thread 9-component vector with arbitrary scalar type. */
template<typename DataType>
using Vec9T = vector::Item<DataType, 9>;

/** @brief Per-thread 3-component double vector (register-resident). */
using Vec3d = Vec3T<double>;
/** @brief Per-thread 4-component double vector (register-resident). */
using Vec4d = Vec4T<double>;
/** @brief Per-thread 6-component double vector (register-resident). */
using Vec6d = Vec6T<double>;
/** @brief Per-thread 9-component double vector (register-resident). */
using Vec9d = Vec9T<double>;

/** @brief Per-thread 3-component float vector (register-resident). */
using Vec3f = Vec3T<float>;
/** @brief Per-thread 4-component float vector (register-resident). */
using Vec4f = Vec4T<float>;
/** @brief Per-thread 6-component float vector (register-resident). */
using Vec6f = Vec6T<float>;
/** @brief Per-thread 9-component float vector (register-resident). */
using Vec9f = Vec9T<float>;
} // namespace feta
