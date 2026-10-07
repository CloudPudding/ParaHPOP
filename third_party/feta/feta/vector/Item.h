#pragma once

#include "feta/buffer/vector/Array.h"
#include "feta/core/type_traits.h"
#include "feta/vector/NeutralQuaternion.h"
#include "feta/vector/OnesNT.h"
#include "feta/vector/ZerosNT.h"
#include "feta/vector/expr/Expression.h"
#include "feta/vector/expr/Operations.h"

namespace feta {
namespace vector {

/**
 * @brief Owning N-dimensional vector with components of type T.
 *
 * Unlike `Array`, this type is stack-allocated, and can be used directly
 * in both host and device code. On device code, it will be allocated in
 * registers, or possibly thread-local memory (determined by the compiler).
 */
template<typename DataT, dims_t VectorDim>
class Item : public expr::Expression<Item<DataT, VectorDim>, DataT> {
    using Self          = Item<DataT, VectorDim>;
    using BufViewT      = buffer::vector::View<DataT, VectorDim>;
    using ConstBufViewT = buffer::vector::ConstView<DataT, VectorDim>;

public:
    /** @brief Buffer type used for host-side storage and I/O. */
    using BufT = buffer::vector::Item<DataT, VectorDim>;

    /** @brief Inner expression type (self, since `Item` is a leaf). */
    using InnerT = Self;

    /** @brief Number of scalar components in the vector. */
    static constexpr dims_t VecDims = VectorDim;

    /** @brief Indicate that this type is a leaf node in expression template trees. */
    static constexpr bool isLeaf    = true;
    static constexpr bool isWritable = true;

    /** @brief Indicate that this item does not reside in work (shared) memory. */
    static constexpr bool work = false;

    /** @brief Expression tree annotations. */
    static constexpr expr::ExprKind kind = expr::ExprKind::Leaf;
    static constexpr dims_t depth        = 0;
    static constexpr idx_t arraySize     = 1; ///< Single vector (degenerate case)

    /** @brief A vector expression where each component is 1. */
    using Ones = OnesNT<DataT, VectorDim>;

    /** @brief A vector expression where each component is 0. */
    using Zeros = ZerosNT<DataT, VectorDim>;

    DEVICEHOST() Item();
    DEVICEHOST() Item(const DataT& initialValue);

    /** @brief Construct from Buffer */
    Item(const BufT& buf)
        : Item{}
    {
        *this = buf;
    }
    Item(const BufViewT& buf)
        : Item{}
    {
        *this = buf;
    }
    Item(const ConstBufViewT& buf)
        : Item{}
    {
        *this = buf;
    }

    /** @brief Construct by evaluating an expression */
    template<typename Expr>
    DEVICEHOST()
    Item(const expr::Expression<Expr, typename Expr::ComponentT>& expr);

    /**
     * @brief Element access.
     *
     * The index parameter is ignored and only provided for compatibility with
     * `expr::Expression`.
     *
     * @tparam dim The index of the dimension to return.
     */
    template<dims_t dim>
    DEVICEHOST()
    inline const DataT& get(const SampleIndex& /* i */) const;
    template<dims_t dim>
    DEVICEHOST()
    inline const DataT& get(const idx_t /* i */) const;
    template<dims_t dim>
    DEVICEHOST()
    inline DataT& get(const SampleIndex& /* i */);
    template<dims_t dim>
    DEVICEHOST()
    inline DataT& get(const idx_t /* i */);

    /**
     * @brief Element access.
     *
     * @tparam dim The index of the dimension to return.
     */
    template<dims_t dim>
    DEVICEHOST()
    inline const DataT& get() const;
    template<dims_t dim>
    DEVICEHOST()
    inline DataT& get();

    /**
     * @brief Evaluate a given expression and store its value in the vector.
     *
     * Only valid if the expression is direct accessible (e.g. a `VecView` on
     * one sample).
     */
    template<typename Expr>
    DEVICEHOST()
    Self& eval(const expr::Expression<Expr, typename Expr::ComponentT>& expr)
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

    /** @brief Assign component values from a buffer. */
    Self& operator=(const BufT& buf)
    {
        std::copy(buf.data(), buf.data() + VectorDim, data_);
        return *this;
    }
    Self& operator=(const BufViewT& buf)
    {
        for (int i = 0; i < VectorDim; i++)
            data_[i] = buf[i];
        return *this;
    }

    Self& operator=(const ConstBufViewT& buf)
    {
        for (int i = 0; i < VectorDim; i++)
            data_[i] = buf[i];
        return *this;
    }

    template<typename Expr>
    DEVICEHOST()
    Self& operator=(
        const expr::Expression<Expr, typename Expr::ComponentT>& expr)
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

    /** @brief Set all components to zero. */
    DEVICEHOST()
    void setZero()
    {
        for (dims_t i = 0; i < VectorDim; i++) {
            data_[i] = 0;
        }
    }

    /** @brief Return a unit-length copy of this vector. */
    DEVICEHOST()
    Self normalize() const { return this->rNorm() * (*this); }

    /** @brief Return a `buffer::vector::Item` copy of this vector's data. */
    BufT buffer() const
    {
        BufT out;
        std::copy(data_, data_ + VectorDim, out.data());
        return out;
    }

    /** @brief Return a pointer to the raw component data. */
    DEVICEHOST() DataT* data() { return data_; }
    DEVICEHOST() const DataT* data() const { return data_; }

    /** @brief Return the number of components (`VectorDim`). */
    DEVICEHOST() idx_t size() const { return VectorDim; }

private:
    DataT data_[VectorDim];
};

template<typename DataT, dims_t VD>
DEVICEHOST()
Item<DataT, VD>::Item()
    : data_{ 0 }
{
}

template<typename DataT, dims_t VD>
DEVICEHOST()
Item<DataT, VD>::Item(const DataT& initialValue)
    : Item{}
{
    for (dims_t i = 0; i < VD; i++) {
        data_[i] = initialValue;
    }
}

template<typename DataT, dims_t VD>
template<typename Expr>
DEVICEHOST()
Item<DataT, VD>::Item(
    const expr::Expression<Expr, typename Expr::ComponentT>& expr)
    : Item{}
{
    eval(expr);
}

template<typename DataT, dims_t VD>
template<dims_t dim>
DEVICEHOST()
inline const DataT& Item<DataT, VD>::get(const SampleIndex& /* index */) const
{
    return get<dim>();
}

template<typename DataT, dims_t VD>
template<dims_t dim>
DEVICEHOST()
inline const DataT& Item<DataT, VD>::get(const idx_t /* index */) const
{
    return get<dim>();
}

template<typename DataT, dims_t VD>
template<dims_t dim>
DEVICEHOST()
inline DataT& Item<DataT, VD>::get(const SampleIndex& /* index */)
{
    return get<dim>();
}

template<typename DataT, dims_t VD>
template<dims_t dim>
DEVICEHOST()
inline DataT& Item<DataT, VD>::get(const idx_t /* index */)
{
    return get<dim>();
}

template<typename DataT, dims_t VD>
template<dims_t dim>
DEVICEHOST()
inline const DataT& Item<DataT, VD>::get() const
{
    return data_[dim];
}

template<typename DataT, dims_t VD>
template<dims_t dim>
DEVICEHOST()
inline DataT& Item<DataT, VD>::get()
{
    return data_[dim];
}

} // namespace vector
} // namespace feta
