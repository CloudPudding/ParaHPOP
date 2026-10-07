#pragma once

#include "feta/vector/expr/Expression.h"

namespace feta {
namespace vector {

namespace detail {

/**
 * @brief Compile-time component-value policies for ConstantLeaf.
 *
 * Each policy supplies a single `value<dim>()` returning the constant a leaf
 * exposes at component `dim`. The leaf ignores all run-time indices, so the
 * policy is the only thing that distinguishes OnesNT / ZerosNT /
 * NeutralQuaternion (see the aliases below).
 */
template<typename DataT>
struct UniformOne {
    template<dims_t /* dim */>
    DEVICEHOST() static inline constexpr DataT value() { return static_cast<DataT>(1); }
};

template<typename DataT>
struct UniformZero {
    template<dims_t /* dim */>
    DEVICEHOST() static inline constexpr DataT value() { return static_cast<DataT>(0); }
};

template<typename DataT>
struct QuaternionIdentity {
    template<dims_t dim>
    DEVICEHOST() static inline constexpr DataT value()
    {
        return dim == 0 ? static_cast<DataT>(1) : static_cast<DataT>(0);
    }
};

} // namespace detail

/**
 * @brief A constant-valued leaf vector expression.
 *
 * Generalises the former OnesNT / ZerosNT / NeutralQuaternion leaves: the
 * per-component compile-time value comes from `Policy::template value<dim>()`,
 * so no data is stored and every accessor is a compile-time constant. The three
 * historical names remain as aliases (OnesNT.h / ZerosNT.h /
 * NeutralQuaternion.h), keeping their include paths and types stable.
 *
 * @tparam DataT      Scalar component type.
 * @tparam VectorDim  Number of components.
 * @tparam Policy     Supplies `value<dim>()` for each component.
 */
template<typename DataT, dims_t VectorDim, typename Policy>
class ConstantLeaf
    : public expr::Expression<ConstantLeaf<DataT, VectorDim, Policy>, DataT> {

public:
    /** @brief Datatype of the vector elements */
    using ComponentT = DataT;

    /** @brief Number of components in the vectors */
    static constexpr dims_t VecDims = VectorDim;

    /** @brief This type is a leaf in the expression template trees */
    static constexpr bool isLeaf = false;

    /** @brief Whether this vector represents a work sample */
    static constexpr bool work = false;

    static constexpr expr::ExprKind kind = expr::ExprKind::Leaf;
    static constexpr dims_t depth        = 0;
    static constexpr idx_t arraySize     = 1;

    /** @brief Constant component value (ignores the sample index). */
    template<dims_t dim>
    DEVICEHOST()
    inline constexpr DataT get(const SampleIndex& /* index */) const
    {
        static_assert(dim < VecDims, "Invalid vector dimension access");
        return Policy::template value<dim>();
    }

    /** @brief Constant component value (ignores the flat index). */
    template<dims_t dim>
    DEVICEHOST()
    inline constexpr DataT get(const idx_t /* index */) const
    {
        static_assert(dim < VecDims, "Invalid vector dimension access");
        return Policy::template value<dim>();
    }

    /** @brief Constant component value (index-free overload). */
    template<dims_t dim>
    DEVICEHOST()
    inline constexpr DataT get() const
    {
        static_assert(dim < VecDims, "Invalid vector dimension access");
        return Policy::template value<dim>();
    }
};

} // namespace vector
} // namespace feta
