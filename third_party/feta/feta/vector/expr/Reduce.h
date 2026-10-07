/**
 * This header contains helpers for reduction operations on vector components.
 * The reductions are unrolled at compile time, so they come at zero runtime
 * overhead.
 */
#pragma once

#include "feta/core/Functors.h"
#include "feta/core/SampleIndex.h"
#include "feta/vector/expr/ReduceDetail.h"

namespace feta {
namespace vector {
namespace expr {
namespace reduce {

using namespace core;

/**
 * @brief Recursive template to compute the dot product between two
 * N-dimensional vectors
 *
 * E.g. this:
 * ```
 * dot<3, Vec3RArray>::eval(a, b, i)
 * ```
 * unrolls to:
 * ```
 * a.get<2>(i) * b.get<2>(i) +
 * a.get<1>(i) * b.get<1>(i) +
 * a.get<0>(i) * b.get<0>(i);
 * ```
 * This comes at zero runtime cost compared to manual unrolling.
 *
 * @tparam dims Number of vector dimensions to reduce over.
 * @tparam VecT Vector array type. Must implement a `get<k>(i)` method which
 * returns the k-th dimension of the i-th vector. VecT::ComponentT should be the
 * type of the vector's components.
 * @tparam ReturnT Desired return type.
 */
template<dims_t dims, typename L, typename R, typename ReturnT>
using dot = detail::GenericBinaryReduce<dims, L, R, ReturnT,
    binaryOps::sum<ReturnT>, binaryOps::times<ReturnT>>;

/**
 * @brief Recursive template to compute the sum of the components of an
 * N-dimensional vector.
 *
 * E.g. this:
 * ```
 * sum<3, Vec3RArray>::eval(vec, i)
 * ```
 * unrolls to:
 * ```
 * vec.get<2>(i) +
 * vec.get<1>(i) +
 * vec.get<0>(i) ;
 * ```
 * This comes at zero runtime cost compared to manual unrolling.
 *
 * @tparam dims Number of vector dimensions to reduce over.
 * @tparam VecT Vector array type. Must implement a `get<k>(i)` method which
 * returns the k-th dimension of the i-th vector. VecT::ComponentT should be the
 * type of the vector's components.
 * @tparam ReturnT Desired return type.
 */
template<dims_t dims, typename VecT, typename ReturnT>
using sum = detail::GenericReduce<dims, VecT, ReturnT, binaryOps::sum<ReturnT>,
    unaryOps::self<ReturnT>>;

/**
 * @brief Recursive template to compute the squared norm of an N-dimensional
 * vector.
 *
 * E.g. this:
 * ```
 * squaredNorm<3, Vec3RArray>::eval(vec, i)
 * ```
 * unrolls to:
 * ```
 * vec.get<2>(i) * vec.get<2>(i) +
 * vec.get<1>(i) * vec.get<1>(i) +
 * vec.get<0>(i) * vec.get<0>(i);
 * ```
 * This comes at zero runtime cost compared to manual unrolling.
 *
 * @tparam dims Number of vector dimensions to reduce over.
 * @tparam VecT Vector array type. Must implement a `get<k>(i)` method which
 * returns the k-th dimension of the i-th vector. VecT::ComponentT should be the
 * type of the vector's components.
 * @tparam ReturnT Desired return type.
 */
template<dims_t dims, typename VecT, typename ReturnT>
using squaredNorm = detail::GenericReduce<dims, VecT, ReturnT,
    binaryOps::sum<ReturnT>, unaryOps::square<ReturnT>>;

/**
 * @brief Recursive template to compute the L1 norm of an N-dimensional
 * vector.
 *
 * E.g. this:
 * ```
 * maxNorm<3, Vec3RArray>::eval(vec, i)
 * ```
 * unrolls to:
 * ```
 * max(abs(vec.get<2>(i)),
       max(abs(vec.get<1>(i)), abs(vec.get<0>(i)))
      );
 * ```
 * This comes at zero runtime cost compared to manual unrolling.
 *
 * @tparam dims Number of vector dimensions to reduce over.
 * @tparam VecT Vector array type. Must implement a `get<k>(i)` method which
 * returns the k-th dimension of the i-th vector. VecT::ComponentT should be the
 * type of the vector's components.
 * @tparam ReturnT Desired return type.
 */
template<dims_t dims, typename VecT, typename ReturnT>
using maxNorm = detail::GenericReduce<dims, VecT, ReturnT,
    binaryOps::max<ReturnT>, unaryOps::abs<ReturnT>>;

/**
 * @brief Recursive template to check finiteness of the components of an
 * N-dimensional vector.
 *
 * E.g. this:
 * ```
 * isFinite<3, Vec3RArray>::eval(vec, i)
 * ```
 * unrolls to
 * ```
 * isfinite(vec.get<2>(i)) &&
 * isfinite(vec.get<1>(i)) &&
 * isfinite(vec.get<0>(i));
 * ```
 * This comes at zero runtime cost compared to manual unrolling.
 *
 * @tparam dims Number of vector dimensions to reduce over.
 * @tparam VecT Vector array type. Must implement a `get<k>(i)` method which
 * returns the k-th dimension of the i-th vector. VecT::ComponentT should be the
 * type of the vector's components.
 */
template<dims_t dims, typename VecT>
using isFinite = detail::GenericReduce<dims, VecT, bool, binaryOps::logicalAnd,
    unaryOps::isFinite<typename VecT::ComponentT>>;

} // namespace reduce
} // namespace expr
} // namespace vector
} // namespace feta
