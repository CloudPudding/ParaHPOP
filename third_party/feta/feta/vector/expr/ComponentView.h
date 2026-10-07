#pragma once

#include "feta/core/SampleIndex.h"
#include "feta/vector/OnesNT.h"
#include "feta/vector/ZerosNT.h"
#include "feta/vector/expr/Expression.h"
#include "feta/vector/expr/Operations.h"

namespace feta {
namespace vector {
namespace expr {

namespace detail {
/**
 * @brief Expression template representing a view onto a subset of the
 * components of a vector array.
 *
 * For example, `ComponentView<VecNTArray<Real, 6>, 3, 3>` acts as a
 * 3-dimensional vector whose .get<0> becomes a .get<3> on the underlying array.
 *
 * This is used for example by `CartesianStates::pos` to give access to the
 * first three components of the state vector, which represent position.
 *
 * @tparam ArrayT Type of the underlying vector array.
 * @tparam dimOffset Offset from the first dimension of the underlying array.
 * @tparam nDims Number of dimensions exposed by the view.
 */
template<typename InnerT, dims_t dimOffset, dims_t nDims>
class BaseComponentView
    : public Expression<BaseComponentView<InnerT, dimOffset, nDims>,
          typename InnerT::ComponentT> {

public:
    static constexpr bool isLeaf      = true;
    static constexpr bool isWritable  = true;
    static constexpr dims_t VecDims   = nDims;
    static constexpr bool work        = InnerT::work;
    static constexpr ExprKind kind    = ExprKind::View;
    static constexpr dims_t depth     = InnerT::depth;
    static constexpr idx_t arraySize  = InnerT::arraySize;

    DEVICEHOST()
    BaseComponentView(InnerT& vecArray)
        : vecArray_{ vecArray }
    {
        static_assert(dimOffset < InnerT::VecDims,
            "Invalid dimension offset for ComponentView!");
        static_assert(nDims <= InnerT::VecDims,
            "ComponentView dimensions cannot exceed underlying vector's!");
        static_assert(dimOffset + nDims <= InnerT::VecDims,
            "ComponentView would exceed dimensions of underlying vector!");
    }

    /**
     * @brief Getter method which maps to the underlying structure with an
     * offset.
     */
    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get(const SampleIndex& i) const
    {
        static_assert(dim < nDims, "Invalid vector dimension access");
        return vecArray_.template get<dimOffset + dim>(i);
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get(const idx_t i) const
    {
        static_assert(dim < nDims, "Invalid vector dimension access");
        return vecArray_.template get<dimOffset + dim>(i);
    }

    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get() const
    {
        static_assert(dim < nDims, "Invalid vector dimension access");
        static_assert(hasDirectAccess<InnerT>::value,
            "Managed type must be direct-accessible! You may need to specify a "
            "sample index with '[i]'.");
        return vecArray_.template get<dimOffset + dim>();
    }

    // Public for packet-based SIMD access
    InnerT& vecArray_;
};

/**
 * @brief Expression template representing an enlarged view of the given array.
 *
 * For example, `ExpandedView<VecNTArray<Real, 3>, 0, 3>` acts as a
 * 6-dimensional vector appending three dimensions at the end, with null values.
 *
 * This is used for example to extend a Cartesian position to a full cartesian
 * Position + velocity vector.
 *
 * @tparam ArrayT Type of the underlying vector array.
 * @tparam frontExtra Offset Number of dimensions to add at the beginning of the
 * underlying array.
 * @tparam backExtra Number of dimensions to add at the end of the underlying
 * array.
 */
template<typename InnerT, dims_t frontExtra, dims_t backExtra>
class ExpandedView
    : public Expression<ExpandedView<InnerT, frontExtra, backExtra>,
          typename InnerT::ComponentT> {
    static constexpr dims_t TrueEnd = frontExtra + InnerT::VecDims;

public:
    static constexpr bool isLeaf      = false;
    static constexpr bool isWritable  = true;
    static constexpr dims_t VecDims   = TrueEnd + backExtra;
    static constexpr bool work        = InnerT::work;
    static constexpr ExprKind kind    = ExprKind::View;
    static constexpr dims_t depth     = InnerT::depth;
    static constexpr idx_t arraySize  = InnerT::arraySize;

    DEVICEHOST()
    ExpandedView(InnerT& vecArray)
        : vecArray_{ vecArray }
    {
    }

    /**
     * @brief Getter method which maps to the underlying structure with an
     * offset.
     */
    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get(const SampleIndex& i) const
    {
        static_assert(dim < VecDims, "Invalid vector dimension access");
        if constexpr (dim < frontExtra || dim >= TrueEnd)
            return 0;
        else
            return vecArray_.template get<dim - frontExtra>(i);
    }
    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get(const idx_t i) const
    {
        static_assert(dim < VecDims, "Invalid vector dimension access");
        if constexpr (dim < frontExtra || dim >= TrueEnd)
            return 0;
        else
            return vecArray_.template get<dim - frontExtra>(i);
    }
    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get() const
    {
        static_assert(hasDirectAccess<InnerT>::value,
            "Managed type must be direct-accessible! You may need to specify a "
            "sample index with '[i]'.");
        static_assert(dim < VecDims, "Invalid vector dimension access");
        if constexpr (dim < frontExtra || dim >= TrueEnd)
            return 0;
        else
            return vecArray_.template get<dim - frontExtra>();
    }
    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get(const SampleIndex& i)
    {
        static_assert(dim < VecDims, "Invalid vector dimension access");
        if constexpr (dim < frontExtra || dim >= TrueEnd)
            return 0;
        else
            return vecArray_.template get<dim - frontExtra>(i);
    }
    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get(const idx_t i)
    {
        static_assert(dim < VecDims, "Invalid vector dimension access");
        if constexpr (dim < frontExtra || dim >= TrueEnd)
            return 0;
        else
            return vecArray_.template get<dim - frontExtra>(i);
    }
    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get()
    {
        static_assert(hasDirectAccess<InnerT>::value,
            "Managed type must be direct-accessible! You may need to specify a "
            "sample index with '[i]'.");
        static_assert(dim < VecDims, "Invalid vector dimension access");
        if constexpr (dim < frontExtra || dim >= TrueEnd)
            return 0;
        else
            return vecArray_.template get<dim - frontExtra>();
    }

    // Public for packet-based SIMD access
    InnerT& vecArray_;
};

} // namespace detail


/**
 * @brief Expression template representing a view onto a subset of the
 * components of a vector array.
 *
 * For example, `ComponentView<VecNTArray<Real, 6>, 3, 3>` acts as a
 * 3-dimensional vector whose .get<0> becomes a .get<3> on the underlying array.
 *
 * This is used for example by `CartesianStates::pos` to give access to the
 * first three components of the state vector, which represent position.
 *
 * @tparam ArrayT Type of the underlying vector array.
 * @tparam dimOffset Offset from the first dimension of the underlying array.
 * @tparam nDims Number of dimensions exposed by the view.
 */
template<typename ArrayT, dims_t dimOffset, dims_t nDims>
class ComponentView
    : public detail::BaseComponentView<ArrayT, dimOffset, nDims> {
    using BaseT = detail::BaseComponentView<ArrayT, dimOffset, nDims>;
    using Self  = ComponentView<ArrayT, dimOffset, nDims>;

public:
    using BaseT::BaseT;

    /**
     * @brief Returns a view onto the i-th vector of this array.
     *
     * This is useful to trigger evaluation of expression templates (see
     * `VecView`).
     */
    DEVICEHOST() inline decltype(auto) operator[](const SampleIndex& i)
    {
        return View<Self>(*this, i);
    }
    DEVICEHOST() inline decltype(auto) operator[](const idx_t& i)
    {
        return View<Self>(*this, SampleIndex::make(i));
    }
    /** @brief Const overload returning a const view */
    DEVICEHOST() inline decltype(auto) operator[](const SampleIndex& i) const
    {
        return ConstView<Self>(const_cast<Self&>(*this), i);
    }
    DEVICEHOST() inline decltype(auto) operator[](const idx_t& i) const
    {
        return ConstView<Self>(const_cast<Self&>(*this), SampleIndex::make(i));
    }

    /**
     * @brief Evaluate a given expression and store its value in the subvector.
     *
     * Only valid if both this and the expression are direct accessible (e.g. a
     * `VecView` on one sample or a `VecNT`).
     */
    template<typename Expr>
    DEVICEHOST()
    Self& eval(const Expression<Expr, typename Expr::ComponentT>& expr)
    {
        static_assert(core::expr::hasDirectAccess<Expr>::value,
            "Can only assign from direct-accessible expressions (such as "
            "VecView)!");
        static_assert(nDims == Expr::VecDims,
            "Cannot assign an expression to a vector with a different "
            "dimension count!");
        expr::assign<nDims, Self, Expr>::eval(
            *this, *static_cast<const Expr*>(&expr));
        return *this;
    }

    template<typename Expr>
    DEVICEHOST()
    Self& operator=(const Expression<Expr, typename Expr::ComponentT>& expr)
    {
        return eval(*static_cast<const Expr*>(&expr));
    }

    template<typename Expr>
    DEVICEHOST()
    Self& operator+=(const Expr& expr)
    {
        return eval((*this) + expr);
    }

    template<typename Expr>
    DEVICEHOST()
    Self& operator-=(const Expr& expr)
    {
        return eval((*this) - expr);
    }

    template<typename OtherT>
    DEVICEHOST()
    Self& operator*=(const OtherT& operand)
    {
        return eval((*this) * operand);
    }

    template<typename OtherT>
    DEVICEHOST()
    Self& operator/=(const OtherT& operand)
    {
        return eval((*this) / operand);
    }
};

template<typename ArrayT, dims_t dimOffset, dims_t nDims>
class ConstComponentView
    : public detail::BaseComponentView<const ArrayT, dimOffset, nDims> {
    using BaseT = detail::BaseComponentView<const ArrayT, dimOffset, nDims>;
    using Self  = ConstComponentView<ArrayT, dimOffset, nDims>;

public:
    using BaseT::BaseT;

    /**
     * @brief Returns a view onto the i-th vector of this array.
     *
     * This is useful to trigger evaluation of expression templates (see
     * `VecView`).
     */
    DEVICEHOST() inline decltype(auto) operator[](const SampleIndex& i) const
    {
        return ConstView<Self>(*this, i);
    }
    DEVICEHOST() inline decltype(auto) operator[](const idx_t& i) const
    {
        return ConstView<Self>(*this, SampleIndex::make(i));
    }
};

/**
 * @brief Expression template representing an enlarged view of the given array.
 *
 * For example, `ExpandedView<VecNTArray<Real, 3>, 0, 3>` acts as a
 * 6-dimensional vector appending three dimensions at the end, with null values.
 *
 * This is used for example to extend a Cartesian position to a full cartesian
 * Position + velocity vector.
 *
 * @tparam ArrayT Type of the underlying vector array.
 * @tparam frontExtra Offset Number of dimensions to add at the beginning of the
 * underlying array.
 * @tparam backExtra Number of dimensions to add at the end of the underlying
 * array.
 */
template<typename ArrayT, dims_t frontExtra, dims_t backExtra>
class ExpandedView
    : public detail::ExpandedView<ArrayT, frontExtra, backExtra> {
    using BaseT = detail::ExpandedView<ArrayT, frontExtra, backExtra>;
    using Self  = ExpandedView<ArrayT, frontExtra, backExtra>;

public:
    using BaseT::BaseT;
    static constexpr dims_t VecDims = BaseT::VecDims;

    /**
     * @brief Returns a view onto the i-th vector of this array.
     *
     * This is useful to trigger evaluation of expression templates (see
     * `VecView`).
     */
    DEVICEHOST() inline decltype(auto) operator[](const SampleIndex& i)
    {
        return View<Self>(*this, i);
    }
    DEVICEHOST() inline decltype(auto) operator[](const idx_t& i)
    {
        return View<Self>(*this, SampleIndex::make(i));
    }
    DEVICEHOST() inline decltype(auto) operator[](const SampleIndex& i) const
    {
        return ConstView<Self>(*this, i);
    }
    DEVICEHOST() inline decltype(auto) operator[](const idx_t& i) const
    {
        return ConstView<Self>(*this, SampleIndex::make(i));
    }

    /**
     * @brief Evaluate a given expression and store its value in the subvector.
     *
     * Only valid if both this and the expression are direct accessible (e.g. a
     * `VecView` on one sample or a `VecNT`).
     */
    template<typename Expr>
    DEVICEHOST()
    Self& eval(const Expression<Expr, typename Expr::ComponentT>& expr)
    {
        static_assert(core::expr::hasDirectAccess<Expr>::value,
            "Can only assign from direct-accessible expressions (such as "
            "VecView)!");
        static_assert(VecDims == Expr::VecDims,
            "Cannot assign an expression to a vector with a different "
            "dimension count!");
        expr::assign<VecDims, Self, Expr>::eval(
            *this, *static_cast<const Expr*>(&expr));
        return *this;
    }

    template<typename Expr>
    DEVICEHOST()
    Self& operator=(const Expression<Expr, typename Expr::ComponentT>& expr)
    {
        return eval(*static_cast<const Expr*>(&expr));
    }

    template<typename Expr>
    DEVICEHOST()
    Self& operator+=(const Expr& expr)
    {
        return eval((*this) + expr);
    }

    template<typename Expr>
    DEVICEHOST()
    Self& operator-=(const Expr& expr)
    {
        return eval((*this) - expr);
    }

    template<typename OtherT>
    DEVICEHOST()
    Self& operator*=(const OtherT& operand)
    {
        return eval((*this) * operand);
    }

    template<typename OtherT>
    DEVICEHOST()
    Self& operator/=(const OtherT& operand)
    {
        return eval((*this) / operand);
    }
};

template<typename ArrayT, dims_t frontExtra, dims_t backExtra>
class ConstExpandedView
    : public detail::ExpandedView<const ArrayT, frontExtra, backExtra> {
    using BaseT = detail::ExpandedView<const ArrayT, frontExtra, backExtra>;
    using Self  = ConstExpandedView<ArrayT, frontExtra, backExtra>;

public:
    using BaseT::BaseT;
    static constexpr dims_t VecDims = BaseT::VecDims;

    /**
     * @brief Returns a view onto the i-th vector of this array.
     *
     * This is useful to trigger evaluation of expression templates (see
     * `VecView`).
     */
    DEVICEHOST() inline decltype(auto) operator[](const SampleIndex& i) const
    {
        return ConstView<Self>(*this, i);
    }
    DEVICEHOST() inline decltype(auto) operator[](const idx_t& i) const
    {
        return ConstView<Self>(*this, SampleIndex::make(i));
    }
};

} // namespace expr
} // namespace vector
} // namespace feta
