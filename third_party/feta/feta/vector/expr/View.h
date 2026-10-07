#pragma once

#include "feta/buffer/vector/Array.h"
#include "feta/core/SampleIndex.h"
#include "feta/vector/Item.h"
#include "feta/vector/expr/Expression.h"
#include "feta/vector/expr/Operations.h"

namespace feta {
namespace vector {
namespace expr {

/**
 * @brief Expression template representing a view onto a specific vector in an
 * array of vectors.
 *
 * These are generally produced by calling `VecNTArray::operator[]`, and
 * are used to trigger evaluation of an expression for a specific vector in the
 * array by using an assignment operator.
 * For example, using 3D vectors, the following:
 * ```
 * vecArr1[i] += vecArr2 + .5 * vecArr3;
 * ```
 * Becomes:
 * ```
 * vecArr1.template get<0>(i) += \
 *     vecArr2.template get<0>(i) + .5 * vecArr3.template get<0>(i);
 * vecArr1.template get<1>(i) += \
 *     vecArr2.template get<1>(i) + .5 * vecArr3.template get<1>(i);
 * vecArr1.template get<2>(i) += \
 *     vecArr2.template get<2>(i) + .5 * vecArr3.template get<2>(i);
 * ```
 *
 * @note Same-typed view-to-view assignment (``dst[i] = src[j]`` where
 *       both views share the same expression type ``E``) does **not**
 *       compile. ``View`` holds reference members (``vecArray_`` and
 *       ``i_``), so the compiler generates the implicit
 *       ``View<E>::operator=(const View<E>&)`` and immediately marks
 *       it deleted. The deleted overload is preferred over the
 *       templated ``operator=(const Expression&)``, so the assignment
 *       is rejected even though ``View`` IS an ``Expression``.
 *
 *       Two equivalent workarounds — both register-resident in device
 *       code:
 *
 *       ```
 *       // 1. Item materialization (used by cudajectory's compaction
 *       //    kernel for per-slot vector copies):
 *       typename ArrRefT::ItemT slot(src[i]);
 *       dst[dst_i] = slot;          // Item is an Expression, fine.
 *
 *       // 2. ``View::copyFrom`` (this header):
 *       dst[dst_i].copyFrom(src[i]);
 *       ```
 *
 *       Both paths route through the templated
 *       ``operator=(const Expression&)`` overload internally and
 *       generate identical SASS.
 */
template<typename E>
class View : public Expression<View<E>, typename E::ComponentT> {
    E& vecArray_;
    const SampleIndex i_;

public:
    static constexpr bool isLeaf      = false;
    static constexpr bool isWritable  = true;
    static constexpr dims_t VecDims   = E::VecDims;
    static constexpr bool work        = E::work;
    static constexpr ExprKind kind    = ExprKind::View;
    static constexpr dims_t depth     = E::depth;
    static constexpr idx_t arraySize  = 1; ///< Bound to a single element

    /** @brief Constructing with an rvalue SampleIndex is forbidden. */
    DEVICEHOST()
    View(Expression<E, typename E::ComponentT>& vecArray, SampleIndex&& i)
        = delete;

    /**
     * @brief Constructor. Usually invoked indirectly via
     * `Expression::operator[]`.
     */
    DEVICEHOST()
    View(Expression<E, typename E::ComponentT>& vecArray, const SampleIndex& i)
        : vecArray_{ *static_cast<E*>(&vecArray) }
        , i_{ i }
    {
    }

    DEVICEHOST()
    View(E& vecArray, const SampleIndex& i)
        : vecArray_{ vecArray }
        , i_{ i }
    {
    }

    /** @brief Copy-construct the view */
    DEVICEHOST()
    View(const View& other)
        : vecArray_{ other.vecArray_ }
        , i_{ other.i_ }
    {
    }

    /** @brief Expose a pointer to the first data element of this view */
    DEVICEHOST() inline decltype(auto) data()
    {
        idx_t idx = work ? i_.work() : i_.global();
        return vecArray_.data() + idx;
    }
    DEVICEHOST() inline decltype(auto) data() const
    {
        idx_t idx = work ? i_.work() : i_.global();
        return vecArray_.data() + idx;
    }

    /** @brief Access individual component `i` from this vector view (runtime index). */
    DEVICEHOST() inline decltype(auto) operator[](const idx_t& i)
    {
#ifdef __CUDA_ARCH__
        FETA_GPU_ASSERT(i < VecDims, ::feta::err::OUT_OF_RANGE_COMPONENT);
#else
        FETA_ASSERT(i < VecDims, "VectorView: out of bounds component access");
#endif
        idx_t base = work ? i_.work() : i_.global();
        return data()[vecArray_.size() * i + base];
    }
    DEVICEHOST() inline decltype(auto) operator[](const idx_t& i) const
    {
#ifdef __CUDA_ARCH__
        FETA_GPU_ASSERT(i < VecDims, ::feta::err::OUT_OF_RANGE_COMPONENT);
#else
        FETA_ASSERT(i < VecDims, "VectorView: out of bounds component access");
#endif
        idx_t base = work ? i_.work() : i_.global();
        return data()[vecArray_.size() * i + base];
    }

    /**
     * @brief Evaluate a given expression and store its value in the view's
     * vector.
     *
     * If this is a view on the `i-th` vector, then the expression will be
     * evaluated for the `i-th` vector as well.
     */
    template<typename Expr>
    DEVICEHOST()
    inline View<E>& eval(
        const Expression<Expr, typename Expr::ComponentT>& expr)
    {
        static_assert(E::isWritable, "Can only assign to a writable expression!");
        static_assert(E::VecDims == Expr::VecDims,
            "Cannot assign an expression to a vector with a different "
            "dimension count!");
        assign<E::VecDims, E, Expr>::eval(
            i_, vecArray_, *static_cast<const Expr*>(&expr));
        return *this;
    }

    /** @brief Evaluate a given expression and store its value in the view's
     * vector atomically */
    template<typename Expr>
    DEVICEHOST()
    inline View<E>& atomicEval(
        const Expression<Expr, typename Expr::ComponentT>& expr)
    {
        static_assert(E::isWritable, "Can only assign to a writable expression!");
        static_assert(E::VecDims == Expr::VecDims,
            "Cannot assign an expression to a vector with a different "
            "dimension count!");
        atomicAssign<E::VecDims, E, Expr>::eval(
            i_, vecArray_, *static_cast<const Expr*>(&expr));
        return *this;
    }

    /** @brief Atomically add the given expression to this view (in-place).
     *
     *  Usage in device code: `vec[0].atomicSum(vec[i]);` — this performs
     *  per-component atomicAdd into `vec[0]` using the values from `vec[i]`.
     */
    template<typename Expr>
    DEVICEHOST()
    inline View<E>& atomicSum(
        const Expression<Expr, typename Expr::ComponentT>& expr)
    {
        static_assert(E::isWritable, "Can only operate on a writable expression!");
        static_assert(E::VecDims == Expr::VecDims,
            "Cannot atomic-sum expressions with different dimensions");
        /* Build an AtomicSum expression and evaluate its component-wise
         * get() which performs the atomicAdd side-effect. */
        auto tmp = detail::AtomicSum<View<E>, Expr>(
            *this, *static_cast<const Expr*>(&expr));
        detail::RecursiveAtomicSum<E::VecDims - 1, decltype(tmp)>::eval(
            i_, tmp);
        return *this;
    }

    template<typename Expr>
    DEVICEHOST()
    inline View<E>& operator=(const Expr& expr)
    {
        return eval(expr);
    }

    template<typename Expr>
    DEVICEHOST()
    inline View<E>& operator=(
        const Expression<Expr, typename Expr::ComponentT>& expr)
    {
        return eval(*static_cast<const Expr*>(&expr));
    }

    /** @brief Copy from another same-typed ``View`` or ``ConstView``,
     *  bypassing the implicitly-deleted ``operator=(const View<E>&)``.
     *
     *  See the class-level note for context. The copy goes through a
     *  register-resident ``Item`` materialization, which restores the
     *  templated expression-template assignment path. Codegen matches
     *  what the user would write by hand.
     */
    DEVICEHOST() inline View<E>& copyFrom(const View<E>& other)
    {
        Item<typename E::ComponentT, E::VecDims> slot(other);
        return eval(slot);
    }
    DEVICEHOST() inline View<E>& copyFrom(const ConstView<E>& other)
    {
        Item<typename E::ComponentT, E::VecDims> slot(other);
        return eval(slot);
    }

    template<typename OtherT>
    DEVICEHOST()
    inline View<E>& operator+=(const OtherT& operand)
    {
        return eval((*this) + operand);
    }

    template<typename Expr>
    DEVICEHOST()
    inline View<E>& operator-=(const Expr& expr)
    {
        return eval((*this) - expr);
    }

    template<typename OtherT>
    DEVICEHOST()
    inline View<E>& operator*=(const OtherT& operand)
    {
        return eval((*this) * operand);
    }

    template<typename OtherT>
    DEVICEHOST()
    inline View<E>& operator/=(const OtherT& operand)
    {
        return eval((*this) / operand);
    }

    /**
     * @brief Element access. Ignores the index parameter, since this is a view
     * on a specific index.
     */
    template<dims_t dim>
    DEVICEHOST()
    inline decltype(auto) get(const SampleIndex& /* i */) const
    {
        return get<dim>();
    }
    template<dims_t dim>
    DEVICEHOST()
    inline decltype(auto) get(const idx_t /* i */) const
    {
        return get<dim>();
    }
    /** @brief Direct element access. */
    template<dims_t dim>
    DEVICEHOST()
    inline decltype(auto) get() const
    {
        return vecArray_.template get<dim>(i_);
    }
    template<dims_t dim>
    DEVICEHOST()
    inline decltype(auto) get(const SampleIndex& /* i */)
    {
        return get<dim>();
    }
    template<dims_t dim>
    DEVICEHOST()
    inline decltype(auto) get(const idx_t /* i */)
    {
        return get<dim>();
    }
    /** @brief Direct element access. */
    template<dims_t dim>
    DEVICEHOST()
    inline decltype(auto) get()
    {
        return vecArray_.template get<dim>(i_);
    }

    /** @brief Expose a buffer view of this vector view */
    template<typename DataT>
    buffer::vector::View<DataT, VecDims> interface()
    {
        static_assert(!work, "Cannot take an interface view of a work sample");
        return buffer::vector::View<DataT, VecDims>(
            vecArray_.data() + (work ? i_.work() : i_.global()),
            vecArray_.size());
    }

    /** @brief Expose a const buffer view of this vector view */
    template<typename DataT>
    buffer::vector::ConstView<DataT, VecDims> interface() const
    {
        static_assert(!work, "Cannot take an interface view of a work sample");
        return buffer::vector::ConstView<DataT, VecDims>(
            vecArray_.data() + (work ? i_.work() : i_.global()),
            vecArray_.size());
    }
};

/**
 * @brief Much like `View`, but doesn't allow modification of the contents.
 */
template<typename E>
class ConstView : public Expression<ConstView<E>, typename E::ComponentT> {
    const E& vecArray_;
    const SampleIndex i_;

public:
    static constexpr bool isLeaf      = false;
    static constexpr dims_t VecDims   = E::VecDims;
    static constexpr bool work        = E::work;
    static constexpr ExprKind kind    = ExprKind::View;
    static constexpr dims_t depth     = E::depth;
    static constexpr idx_t arraySize  = 1;

    /** @brief Constructing with an rvalue SampleIndex is forbidden. */
    DEVICEHOST()
    ConstView(Expression<E, typename E::ComponentT>& vecArray, SampleIndex&& i)
        = delete;
    /**
     * @brief Constructor. Usually invoked indirectly via
     * `Expression::operator[]`.
     */
    DEVICEHOST()
    ConstView(const Expression<E, typename E::ComponentT>& vecArray,
        const SampleIndex& i)
        : vecArray_{ *static_cast<const E*>(&vecArray) }
        , i_{ i }
    {
    }

    DEVICEHOST()
    ConstView(const E& vecArray, const SampleIndex& i)
        : vecArray_{ vecArray }
        , i_{ i }
    {
    }

    /** @brief Copy-construct the view */
    DEVICEHOST()
    ConstView(const ConstView& other)
        : vecArray_{ other.vecArray_ }
        , i_{ other.i_ }
    {
    }

    /** @brief Expose a pointer to the first data element of this view */
    DEVICEHOST() inline decltype(auto) data() const
    {
        idx_t idx = work ? i_.work() : i_.global();
        return vecArray_.data() + idx;
    }
    /**
     * @brief Element access. Ignores the index parameter, since this is a view
     * on a specific index.
     */
    template<dims_t dim>
    DEVICEHOST()
    inline decltype(auto) get(const SampleIndex& /* i */) const
    {
        return vecArray_.template get<dim>(i_);
    }
    template<dims_t dim>
    DEVICEHOST()
    inline decltype(auto) get(const idx_t /* i */) const
    {
        return vecArray_.template get<dim>(i_);
    }
    /** @brief Direct element access. */
    template<dims_t dim>
    DEVICEHOST()
    inline decltype(auto) get() const
    {
        return vecArray_.template get<dim>(i_);
    }

    /** @brief Expose a const buffer view of this vector view */
    template<typename DataT>
    buffer::vector::ConstView<DataT, VecDims> interface() const
    {
        static_assert(!work, "Cannot take an interface view of a work sample");
        return buffer::vector::ConstView<DataT, VecDims>(
            vecArray_.data() + (work ? i_.work() : i_.global()),
            vecArray_.size());
    }
};
} // namespace expr
} // namespace vector
} // namespace feta
