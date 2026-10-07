/**
 * This header contains implementation details for the expression-templated
 * vector reduction operations exposed in `VectorReduce.cuh`.
 */
#pragma once

#include "feta/core/SampleIndex.h"
#include "feta/typedefs.h"

namespace feta {
namespace vector {
namespace expr {
namespace reduce {
namespace detail {

/**
 * @brief Self-unrolling recursive reduction template.
 * It applies an arbitrary unary transform to each component of a vector, and
 * then combines the components with an arbitrary reduction binary operator.
 * NOTE: Vector type must be at least two-dimensional!
 *
 * @tparam dim Dimension to begin recursion at.
 * @tparam VecT Vector array type. Must implement a `get<k>(i)` method which
 * returns the k-th dimension of the i-th vector.
 * @tparam ReduceT Reduction operator type. Must have a two-argument
 * `operator()` that returns a single value of the same type.
 * @tparam TransfT Transform operator type. Must have a unary
 * `operator()` that returns a value of type `OutputT`.
 * @tparam OutputT Data type returned by the transform operator.
 */
template<dims_t dim, typename VecT, typename ReduceT, typename TransfT,
    typename OutputT>
struct RecursiveReduce {
    DEVICEHOST()
    inline static OutputT eval(const VecT& vec, const SampleIndex& i)
    {
        return ReduceT()(TransfT()(vec.template get<dim>(i)),
            RecursiveReduce<dim - 1, VecT, ReduceT, TransfT, OutputT>::eval(
                vec, i));
    }

    DEVICEHOST()
    inline static OutputT eval(const VecT& vec, const idx_t i)
    {
        return ReduceT()(TransfT()(vec.template get<dim>(i)),
            RecursiveReduce<dim - 1, VecT, ReduceT, TransfT, OutputT>::eval(
                vec, i));
    }

    DEVICEHOST() inline static OutputT eval(const VecT& vec)
    {
        return ReduceT()(TransfT()(vec.template get<dim>()),
            RecursiveReduce<dim - 1, VecT, ReduceT, TransfT, OutputT>::eval(
                vec));
    }
};

/**
 * @brief Termination of the template recursion.
 */
template<typename VecT, typename ReduceT, typename TransfT, typename OutputT>
struct RecursiveReduce<0, VecT, ReduceT, TransfT, OutputT> {
    DEVICEHOST()
    inline static OutputT eval(const VecT& vec, const SampleIndex& i)
    {
        return TransfT()(vec.template get<0>(i));
    }

    DEVICEHOST()
    inline static OutputT eval(const VecT& vec, const idx_t i)
    {
        return TransfT()(vec.template get<0>(i));
    }

    DEVICEHOST() inline static OutputT eval(const VecT& vec)
    {
        return TransfT()(vec.template get<0>());
    }
};

template<dims_t dims, typename VecT, typename OutputT, typename ReduceT,
    typename TransfT>
struct GenericReduce {
    DEVICEHOST()
    inline static OutputT eval(const VecT& vec, const SampleIndex& i)
    {
        return RecursiveReduce<dims - 1, VecT, ReduceT, TransfT, OutputT>::eval(
            vec, i);
    }

    DEVICEHOST()
    inline static OutputT eval(const VecT& vec, const idx_t i)
    {
        return RecursiveReduce<dims - 1, VecT, ReduceT, TransfT, OutputT>::eval(
            vec, i);
    }

    DEVICEHOST() inline static OutputT eval(const VecT& vec)
    {
        return RecursiveReduce<dims - 1, VecT, ReduceT, TransfT, OutputT>::eval(
            vec);
    }
};

/**
 * @brief Binary Self-unrolling recursive reduction template.
 * It applies an arbitrary binary transform to each component of two input
 * vectors.
 *
 * @tparam dim Dimension to begin recursion at.
 * @tparam VecT Vector array type. Must implement a `get<k>(i)` method which
 * returns the k-th dimension of the i-th vector.
 * @tparam ReduceT Reduction operator type. Must have a two-argument
 * `operator()` that returns a single value of the same type.
 * @tparam TransfT Transform operator type. Must have a Binary
 * `operator()` that returns a value of type `OutputT`.
 * @tparam OutputT Data type returned by the transform operator.
 */
template<dims_t dim, typename L, typename R, typename ReduceT, typename TransfT,
    typename OutputT>
struct BinaryRecursiveReduce {
    DEVICEHOST()
    inline static OutputT eval(const L& a, const R& b, const SampleIndex& i)
    {
        return ReduceT()(
            TransfT()(a.template get<dim>(i), b.template get<dim>(i)),
            BinaryRecursiveReduce<dim - 1, L, R, ReduceT, TransfT,
                OutputT>::eval(a, b, i));
    }

    DEVICEHOST()
    inline static OutputT eval(const L& a, const R& b, const idx_t i)
    {
        return ReduceT()(
            TransfT()(a.template get<dim>(i), b.template get<dim>(i)),
            BinaryRecursiveReduce<dim - 1, L, R, ReduceT, TransfT,
                OutputT>::eval(a, b, i));
    }

    DEVICEHOST() inline static OutputT eval(const L& a, const R& b)
    {
        return ReduceT()(
            TransfT()(a.template get<dim>(), b.template get<dim>()),
            BinaryRecursiveReduce<dim - 1, L, R, ReduceT, TransfT,
                OutputT>::eval(a, b));
    }
};

/**
 * @brief Termination of the template recursion.
 */
template<typename L, typename R, typename ReduceT, typename TransfT,
    typename OutputT>
struct BinaryRecursiveReduce<0, L, R, ReduceT, TransfT, OutputT> {
    DEVICEHOST()
    inline static OutputT eval(const L& a, const R& b, const SampleIndex& i)
    {
        return TransfT()(a.template get<0>(i), b.template get<0>(i));
    }

    DEVICEHOST()
    inline static OutputT eval(const L& a, const R& b, const idx_t i)
    {
        return TransfT()(a.template get<0>(i), b.template get<0>(i));
    }

    DEVICEHOST() inline static OutputT eval(const L& a, const R& b)
    {
        return TransfT()(a.template get<0>(), b.template get<0>());
    }
};

/** @brief Generic binary reduction */
template<dims_t dims, typename L, typename R, typename OutputT,
    typename ReduceT, typename TransfT>
struct GenericBinaryReduce {
    DEVICEHOST()
    inline static OutputT eval(const L& a, const R& b, const SampleIndex& i)
    {
        return BinaryRecursiveReduce<dims - 1, L, R, ReduceT, TransfT,
            OutputT>::eval(a, b, i);
    }

    DEVICEHOST()
    inline static OutputT eval(const L& a, const R& b, const idx_t i)
    {
        return BinaryRecursiveReduce<dims - 1, L, R, ReduceT, TransfT,
            OutputT>::eval(a, b, i);
    }

    DEVICEHOST() inline static OutputT eval(const L& a, const R& b)
    {
        return BinaryRecursiveReduce<dims - 1, L, R, ReduceT, TransfT,
            OutputT>::eval(a, b);
    }
};

} // namespace detail
} // namespace reduce
} // namespace expr
} // namespace vector
} // namespace feta
