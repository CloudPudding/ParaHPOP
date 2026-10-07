#pragma once

#include "feta/core/SampleIndex.h"
#include "feta/core/type_traits.h"
#include "feta/math/math.h"
#include "feta/typedefs.h"
#include "feta/vector/expr/Reduce.h"

namespace feta {
namespace vector {
namespace expr {

using namespace feta::core::expr;

/**
 * @brief Compile-time tag identifying the kind of an expression node.
 *
 * Used so that generic algorithms (e.g. prefetch dispatch, packet
 * evaluation) can inspect node kind without RTTI.
 */
enum class ExprKind : unsigned {
    Leaf,     ///< Stores data (RefArray, Item, OnesNT, ZerosNT)
    Unary,    ///< One child (CWiseAbs, QuatConj, Normalize, …)
    Binary,   ///< Two children (Sum, CWiseMult, Cross, QuatMul, …)
    View,     ///< Bound view onto an array element
    Reduction ///< Scalar reduction (squaredNorm, dot, …)
};

template<typename ArrayT>
class View;
template<typename ArrayT>
class ConstView;

template<typename ArrayT, dims_t dimOffset, dims_t nDims>
class ComponentView;
template<typename ArrayT, dims_t dimOffset, dims_t nDims>
class ConstComponentView;

template<typename ArrayT, dims_t frontExtra, dims_t backExtra>
class ExpandedView;
template<typename ArrayT, dims_t frontExtra, dims_t backExtra>
class ConstExpandedView;

namespace detail {
template<typename E>
class CWiseAbs;
template<typename L, typename R>
class CWiseMax;
template<typename L, typename R>
class Cross;
template<typename E>
class Normalize;
template<typename E>
class QuatConj;
template<typename E>
class QuatAntiInvolute;
template<typename L, typename R>
class QuatMul;
template<typename L, typename R>
class AtomicSum;
template<typename Q, typename V>
class QuatRotate;
} // namespace detail


/**
 * @brief Base class for vector expressions.
 */
template<typename Expr, typename ComponentT_>
class Expression {
    using Self = Expression<Expr, ComponentT_>;

public:
    /**
     * @brief Whether this expression is a leaf of an expression tree.
     *
     * If true, it stores actual values (not just sub-expressions).
     */
    static constexpr bool isLeaf = false;

    /** @brief Whether this expression can be assigned to (written). */
    static constexpr bool isWritable = false;

    /** @brief Number of dimensions of the vector. */
    static constexpr dims_t VecDims = Expr::VecDims;

    /** @brief Mark this (and all types that inherit from it) as a vector
     * expression. */
    static constexpr bool isVectorExpression = true;

    /** @brief Indicate whether this expression operates on work (shared)
     * memory. */
    static constexpr bool work = Expr::work;

    /**
     * @brief Expression tree depth (leaves = 0, each composite adds 1).
     *
     * Default pulls from the derived type; leaves should define `depth = 0`.
     */
    static constexpr dims_t depth = Expr::depth;

    /**
     * @brief Expression node kind (Leaf, Unary, Binary, View, Reduction).
     *
     * Default pulls from the derived type.
     */
    static constexpr ExprKind kind = Expr::kind;

    /**
     * @brief Compile-time known array size, or 0 for runtime-sized.
     *
     * When > 0, the expression is known at compile time to have exactly
     * this many elements. When 0, call `size()` at runtime.
     */
    static constexpr idx_t arraySize = Expr::arraySize;

    /** @brief Expose the inner expression type */
    using InnerT = Expr;

    /** @brief Expose the component type */
    using ComponentT = ComponentT_;

    /**
     * @brief Return a mutable view onto the i-th vector of this array.
     *
     * The returned `View` is an assignable lvalue that triggers expression-tree
     * evaluation on assignment.  See `VecView` for the evaluation semantics.
     *
     * @param i  Sample index selecting the vector.
     * @return Mutable `View<Expr>` bound to element `i`.
     */
    DEVICEHOST() inline decltype(auto) operator[](const SampleIndex& i)
    {
        return View<Expr>(static_cast<Expr&>(*this), i);
    }
    /** @overload operator[](const SampleIndex&) — flat integer index variant.
     */
    DEVICEHOST() inline decltype(auto) operator[](const idx_t& i)
    {
        return View<Expr>(static_cast<Expr&>(*this), SampleIndex::make(i));
    }

    /**
     * @brief Return a read-only view onto the i-th vector of this array.
     *
     * @param i  Sample index selecting the vector.
     * @return Read-only `ConstView<Expr>` bound to element `i`.
     */
    DEVICEHOST() inline decltype(auto) operator[](const SampleIndex& i) const
    {
        return ConstView<Expr>(static_cast<const Expr&>(*this), i);
    }
    /** @overload operator[](const SampleIndex&) const — flat integer index
     * variant. */
    DEVICEHOST() inline decltype(auto) operator[](const idx_t& i) const
    {
        return ConstView<Expr>(
            static_cast<const Expr&>(*this), SampleIndex::make(i));
    }

    /**
     * @brief Prevent calls with rvalue `SampleIndex` arguments.
     *
     * Rvalue indices cause dangling-reference issues when the expression
     * is not evaluated immediately.
     */
    DEVICEHOST() inline decltype(auto) operator[](const SampleIndex&&)
    {
        static_assert(alwaysFalse<Expr>::value,
            "The SampleIndex used in operator[] must be an lvalue reference!");
    }

    /**
     * @brief Return the `dim`-th component of the i-th vector.
     *
     * @tparam dim  Compile-time component index; must be in `[0, VecDims)`.
     * @param i  Sample index selecting the vector.
     * @return Reference to the `dim`-th scalar component.
     */
    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get(const SampleIndex& i) const
    {
        // Delegate to actual type.
        // This avoids runtime polymorphism (virtual functions)
        return static_cast<Expr const&>(*this).template get<dim>(i);
    }

    /** @overload get(const SampleIndex&) const — flat integer index variant. */
    template<dims_t dim>
    DEVICEHOST()
    decltype(auto) get(const idx_t i) const
    {
        // Delegate to actual type.
        // This avoids runtime polymorphism (virtual functions)
        return static_cast<Expr const&>(*this).template get<dim>(i);
    }

    /**
     * @brief Compute the squared L2 norm of the i-th vector (sum of squares of
     * components).
     * @tparam T  Scalar return type; defaults to `double`.
     * @param idx  Sample index selecting the vector.
     * @return Squared L2 norm ‖v‖² as type `T`.
     */
    template<typename T = double, typename IndexT>
    DEVICEHOST()
    inline T squaredNorm(const IndexT& idx) const
    {
        return reduce::squaredNorm<VecDims, Expr, T>::eval(
            static_cast<Expr const&>(*this), idx);
    }

    /**
     * @brief Return the reciprocal of the squared L2 norm of the i-th vector.
     * @tparam T  Scalar return type; defaults to `double`.
     * @param idx  Sample index selecting the vector.
     * @return 1/‖v‖² as type `T`.
     */
    template<typename T = double, typename IndexT>
    DEVICEHOST()
    inline T rSquaredNorm(const IndexT& idx) const
    {
        return T{ 1 } / squaredNorm<T>(idx);
    }

    /**
     * @brief Compute the L2 norm of the i-th vector.
     * @tparam T  Scalar return type; defaults to `double`.
     * @param i  Sample index selecting the vector.
     * @return ‖v‖ as type `T`.
     */
    template<typename T = double, typename IndexT>
    DEVICEHOST()
    inline T norm(const IndexT& i) const
    {
        return math::sqrt(squaredNorm<T>(i));
    }

    /**
     * @brief Return the reciprocal of the L2 norm for the i-th vector.
     * @tparam T  Scalar return type; defaults to `double`.
     * @param i  Sample index selecting the vector.
     * @return 1/‖v‖ as type `T`.
     * @note Uses accurate division rather than hardware `rsqrt` to avoid
     *       the ~1 ulp error introduced by fast reciprocal sqrt.
     */
    template<typename T = double, typename IndexT>
    DEVICEHOST()
    inline T rNorm(const IndexT& i) const
    {
        // Prefer numerically accurate inverse sqrt over fast rsqrt on GPU.
        // Using rsqrt here can increase error (especially for double) by ~1
        // ulp. If fast math is desired, consider adding a macro switch to
        // re-enable rsqrt.
        return T{ 1 } / math::sqrt(squaredNorm<T>(i));
    }

    /**
     * @brief Compute the cubed L2 norm of the i-th vector (‖v‖³).
     * @tparam T  Scalar return type; defaults to `double`.
     * @param i  Sample index selecting the vector.
     * @return ‖v‖³ as type `T`.
     */
    template<typename T = double, typename IndexT>
    DEVICEHOST()
    inline T cubedNorm(const IndexT& i) const
    {
        const T sNorm = squaredNorm<T>(i);
        return sNorm * math::sqrt(sNorm);
    }

    /**
     * @brief Return the reciprocal of the cubed L2 norm of the i-th vector
     * (1/‖v‖³).
     * @tparam T  Scalar return type; defaults to `double`.
     * @param i  Sample index selecting the vector.
     * @return 1/‖v‖³ as type `T`.
     */
    template<typename T = double, typename IndexT>
    DEVICEHOST()
    inline T rCubedNorm(const IndexT& i) const
    {
        return 1 / cubedNorm<T>(i);
    }

    /**
     * @brief Compute the L∞ (max) norm of the i-th vector.
     * @tparam T  Scalar return type; defaults to `double`.
     * @param i  Sample index selecting the vector.
     * @return max(|vₓ|, |vᵧ|, |v_z|, …) as type `T`.
     */
    template<typename T = double, typename IndexT>
    DEVICEHOST()
    inline T maxNorm(const IndexT& i) const
    {
        return reduce::maxNorm<VecDims, Expr, T>::eval(
            static_cast<Expr const&>(*this), i);
    }

    /**
     * @brief Compute the sum of all components of the i-th vector.
     * @tparam T  Scalar return type; defaults to `double`.
     * @param i  Sample index selecting the vector.
     * @return Sum of all components as type `T`.
     */
    template<typename T = double, typename IndexT>
    DEVICEHOST()
    inline T sum(const IndexT& i) const
    {
        return reduce::sum<VecDims, Expr, T>::eval(
            static_cast<Expr const&>(*this), i);
    }

    /**
     * @brief Return the unit vector of the i-th vector (normalised to length
     * 1).
     * @param i  Sample index selecting the vector.
     * @return Normalised vector expression with the same direction.
     */
    template<typename IndexT>
    DEVICEHOST()
    inline decltype(auto) unitVector(const IndexT& i) const
    {
        return static_cast<const Expr*>(this)->normalize(i);
    }

    /**
     * @brief Compute the squared L2 norm of this vector (sum of squares of
     * components).
     * @tparam T  Scalar return type; defaults to `double`.
     * @return ‖v‖² as type `T`.
     */
    template<typename T = double>
    DEVICEHOST()
    inline T squaredNorm() const
    {
        return reduce::squaredNorm<VecDims, Expr, T>::eval(
            static_cast<Expr const&>(*this));
    }

    /**
     * @brief Return the reciprocal of the squared L2 norm of this vector.
     * @tparam T  Scalar return type; defaults to `double`.
     * @return 1/‖v‖² as type `T`.
     */
    template<typename T = double>
    DEVICEHOST()
    inline T rSquaredNorm() const
    {
        return T{ 1 } / squaredNorm<T>();
    }

    /**
     * @brief Compute the L2 norm of this vector.
     * @tparam T  Scalar return type; defaults to `double`.
     * @return ‖v‖ as type `T`.
     */
    template<typename T = double>
    DEVICEHOST()
    inline T norm() const
    {
        return math::sqrt(squaredNorm<T>());
    }

    /**
     * @brief Return the reciprocal of the L2 norm of this vector.
     * @tparam T  Scalar return type; defaults to `double`.
     * @return 1/‖v‖ as type `T`.
     * @note Uses accurate division rather than hardware `rsqrt` to avoid
     *       the ~1 ulp error introduced by fast reciprocal sqrt.
     */
    template<typename T = double>
    DEVICEHOST()
    inline T rNorm() const
    {
        // Prefer numerically accurate inverse sqrt over fast rsqrt on GPU.
        return T{ 1 } / math::sqrt(squaredNorm<T>());
    }

    /**
     * @brief Compute the cubed L2 norm of this vector (‖v‖³).
     * @tparam T  Scalar return type; defaults to `double`.
     * @return ‖v‖³ as type `T`.
     */
    template<typename T = double>
    DEVICEHOST()
    inline T cubedNorm() const
    {
        const T sNorm = squaredNorm<T>();
        return sNorm * math::sqrt(sNorm);
    }

    /**
     * @brief Return the reciprocal of the cubed L2 norm of this vector
     * (1/‖v‖³).
     * @tparam T  Scalar return type; defaults to `double`.
     * @return 1/‖v‖³ as type `T`.
     */
    template<typename T = double>
    DEVICEHOST()
    inline T rCubedNorm() const
    {
        return T{ 1 } / cubedNorm<T>();
    }

    /**
     * @brief Compute the L∞ (max) norm of this vector.
     * @tparam T  Scalar return type; defaults to `double`.
     * @return max(|vₓ|, |vᵧ|, …) as type `T`.
     */
    template<typename T = double>
    DEVICEHOST()
    inline T maxNorm() const
    {
        return reduce::maxNorm<VecDims, Expr, T>::eval(
            static_cast<Expr const&>(*this));
    }

    /**
     * @brief Compute the sum of all components of this vector.
     * @tparam T  Scalar return type; defaults to `double`.
     * @return Sum of all components as type `T`.
     */
    template<typename T = double>
    DEVICEHOST()
    inline T sum() const
    {
        return reduce::sum<VecDims, Expr, T>::eval(
            static_cast<Expr const&>(*this));
    }

    /**
     * @brief Compute the dot product with another vector expression.
     * @tparam E  Type of the right-hand expression; must have the same
     * `VecDims`.
     * @param other  Right-hand vector expression.
     * @return Scalar dot product as a lazy expression evaluated at assignment.
     */
    template<typename E>
    DEVICEHOST()
    inline decltype(auto)
        dot(const Expression<E, typename E::ComponentT>& other) const
    {
        return reduce::dot<VecDims, Expr, E, typename E::ComponentT>::eval(
            *static_cast<const Expr*>(this), *static_cast<const E*>(&other));
    }

    /**
     * @brief Compute the cross product with another 3D vector expression.
     * @tparam E  Type of the right-hand expression; `VecDims` must equal 3.
     * @param other  Right-hand 3D vector expression.
     * @return Lazy cross-product expression; evaluate by assigning to a
     * `Vec3d`.
     */
    template<typename E>
    DEVICEHOST()
    inline decltype(auto)
        cross(const Expression<E, typename E::ComponentT>& other) const
    {
        return detail::Cross<Expr, E>(
            *static_cast<const Expr*>(this), *static_cast<const E*>(&other));
    }

    /**
     * @brief Return the unit vector of this expression (normalised to length
     * 1).
     * @return Lazy normalisation expression; evaluate by assigning to a
     * `Vec*d`.
     */
    DEVICEHOST()
    inline decltype(auto) unitVector() const
    {
        return detail::Normalize<Expr>(*static_cast<const Expr*>(this));
    }

    /**
     * @brief Compute the quaternion product *this ⊗ other.
     * @tparam E  Type of the right-hand quaternion expression; `VecDims` must
     * equal 4.
     * @param other  Right-hand quaternion expression.
     * @return Lazy quaternion product expression; evaluate by assigning to a
     * `Vec4d`.
     */
    template<typename E>
    DEVICEHOST()
    inline decltype(auto)
        quatMul(const Expression<E, typename E::ComponentT>& other) const
    {
        return detail::QuatMul<Expr, E>(
            *static_cast<const Expr*>(this), *static_cast<const E*>(&other));
    }

    /**
     * @brief Return the quaternion reciprocal (q⁻¹ = q* / ‖q‖²).
     * @return Lazy expression for the quaternion inverse; evaluate by assigning
     * to a `Vec4d`.
     */
    DEVICEHOST()
    inline decltype(auto) quatReciprocal() const
    {
        return quatConj() * rSquaredNorm();
    }

    /**
     * @brief Return the quaternion conjugate (negate the vector part).
     * @return Lazy conjugate expression; evaluate by assigning to a `Vec4d`.
     */
    DEVICEHOST()
    inline decltype(auto) quatConj() const
    {
        return detail::QuatConj<Expr>(*static_cast<const Expr*>(this));
    }

    /**
     * @brief Rotate a 3D vector by this unit quaternion using the Rodrigues
     * formula.
     *
     * Given q = (w, x, y, z) and a 3D vector v, computes:
     *   t  = 2 * (u × v),  u = (x, y, z)
     *   v' = v + w*t + u × t
     *
     * 30 FLOPs per sample — roughly half the cost of the naive sandwich
     * product q ⊗ (0,v) ⊗ q*.
     *
     * @tparam E  Type of the 3D vector expression; `VecDims` must equal 3.
     * @param vec  3D vector expression to rotate.
     * @return Lazy 3D expression for the rotated vector.
     */
    template<typename E>
    DEVICEHOST()
    inline decltype(auto)
        quatRotate(const Expression<E, typename E::ComponentT>& vec) const
    {
        return detail::QuatRotate<Expr, E>(
            *static_cast<const Expr*>(this), *static_cast<const E*>(&vec));
    }

    /**
     * @brief Return the quaternion anti-involute (sandwich product) expression.
     *
     * Computes the lazy expression for the sandwich product *this ⊗ *this*,
     * i.e. the rotation of a vector by the quaternion represented by `*this`.
     * Combine with an argument via the returned expression object.
     *
     * @return Lazy anti-involute expression; evaluate by assigning to a
     * `Vec4d`.
     */
    DEVICEHOST()
    inline decltype(auto) quatAntiInvolute() const
    {
        return detail::QuatAntiInvolute<Expr>(*static_cast<const Expr*>(this));
    }

    /**
     * @brief Return the component-wise absolute value of this expression.
     * @return Lazy component-wise absolute value expression.
     */
    DEVICEHOST() inline decltype(auto) cwiseAbs() const
    {
        return detail::CWiseAbs<Expr>(*static_cast<const Expr*>(this));
    }

    /**
     * @brief Return the component-wise maximum between this and another
     * expression.
     * @tparam E  Type of the right-hand expression; must have the same
     * `VecDims`.
     * @param other  Right-hand vector expression.
     * @return Lazy component-wise max expression.
     */
    template<typename E>
    DEVICEHOST()
    inline decltype(auto)
        cwiseMax(const Expression<E, typename E::ComponentT>& other) const
    {
        return detail::CWiseMax<Expr, E>(
            *static_cast<const Expr*>(this), *static_cast<const E*>(&other));
    }

    /**
     * @brief Atomically add another expression into this one, component-wise.
     *
     * Performs a GPU atomic addition for each component. Only valid in device
     * code when `*this` is a reference to global or shared memory.
     *
     * @tparam E  Type of the right-hand expression; must have the same
     * `VecDims`.
     * @param other  Expression whose components are atomically added into
     * `*this`.
     * @return Lazy atomic-add expression; evaluation triggers the atomic
     * operations.
     */
    template<typename E>
    DEVICEHOST()
    inline decltype(auto)
        atomicSum(const Expression<E, typename E::ComponentT>& other)
    {
        return detail::AtomicSum<Expr, E>(
            *static_cast<const Expr*>(this), *static_cast<const E*>(&other));
    }

    /**
     * @brief Return a mutable view on a contiguous subset of this expression's
     * dimensions.
     *
     * @tparam dimOffset  Index of the first component to expose.
     * @tparam nDims      Number of components to expose.
     * @return `ComponentView` assignable expression of dimension `nDims`.
     */
    template<dims_t dimOffset, dims_t nDims>
    DEVICEHOST()
    inline decltype(auto) segment()
    {
        return ComponentView<Expr, dimOffset, nDims>(*static_cast<Expr*>(this));
    }

    /**
     * @brief Return a read-only view on a contiguous subset of this
     * expression's dimensions.
     *
     * @tparam dimOffset  Index of the first component to expose.
     * @tparam nDims      Number of components to expose.
     * @return `ConstComponentView` expression of dimension `nDims`.
     */
    template<dims_t dimOffset, dims_t nDims>
    DEVICEHOST()
    inline decltype(auto) segment() const
    {
        return ConstComponentView<Expr, dimOffset, nDims>(
            *static_cast<const Expr*>(this));
    }

    /**
     * @brief Return a mutable expanded view with null-valued padding
     * dimensions.
     *
     * @tparam frontExtra  Number of zero-valued dimensions to prepend.
     * @tparam backExtra   Number of zero-valued dimensions to append.
     * @return `ExpandedView` of dimension `VecDims + frontExtra + backExtra`.
     */
    template<dims_t frontExtra, dims_t backExtra>
    DEVICEHOST()
    inline decltype(auto) expansion()
    {
        return ExpandedView<Expr, frontExtra, backExtra>(
            *static_cast<Expr*>(this));
    }

    /**
     * @brief Return a read-only expanded view with null-valued padding
     * dimensions.
     *
     * @tparam frontExtra  Number of zero-valued dimensions to prepend.
     * @tparam backExtra   Number of zero-valued dimensions to append.
     * @return `ConstExpandedView` of dimension `VecDims + frontExtra +
     * backExtra`.
     */
    template<dims_t frontExtra, dims_t backExtra>
    DEVICEHOST()
    inline decltype(auto) expansion() const
    {
        return ConstExpandedView<Expr, frontExtra, backExtra>(
            *static_cast<const Expr*>(this));
    }

    /**
     * @brief Return a mutable view on the first `nDims` components.
     *
     * @tparam nDims  Number of leading components to expose.
     * @return `ComponentView` of dimension `nDims` starting at index 0.
     */
    template<dims_t nDims>
    DEVICEHOST()
    inline decltype(auto) head()
    {
        return segment<0, nDims>();
    }

    /**
     * @brief Return a read-only view on the first `nDims` components.
     *
     * @tparam nDims  Number of leading components to expose.
     * @return `ConstComponentView` of dimension `nDims` starting at index 0.
     */
    template<dims_t nDims>
    DEVICEHOST()
    inline decltype(auto) head() const
    {
        return segment<0, nDims>();
    }

    /**
     * @brief Return a mutable view on the last `nDims` components.
     *
     * @tparam nDims  Number of trailing components to expose.
     * @return `ComponentView` of dimension `nDims` starting at index `VecDims -
     * nDims`.
     */
    template<dims_t nDims>
    DEVICEHOST()
    inline decltype(auto) tail()
    {
        return segment<VecDims - nDims, nDims>();
    }

    /**
     * @brief Return a read-only view on the last `nDims` components.
     *
     * @tparam nDims  Number of trailing components to expose.
     * @return `ConstComponentView` of dimension `nDims` starting at index
     * `VecDims - nDims`.
     */
    template<dims_t nDims>
    DEVICEHOST()
    inline decltype(auto) tail() const
    {
        return segment<VecDims - nDims, nDims>();
    }

    /**
     * @brief Return a mutable front-padded view with `extra` null-valued
     * dimensions prepended.
     *
     * @tparam extra  Number of zero-valued dimensions to prepend.
     * @return `ExpandedView` of dimension `VecDims + extra`.
     */
    template<dims_t extra>
    DEVICEHOST()
    inline decltype(auto) headExpansion()
    {
        return expansion<extra, 0>();
    }

    /**
     * @brief Return a read-only front-padded view with `extra` null-valued
     * dimensions prepended.
     *
     * @tparam extra  Number of zero-valued dimensions to prepend.
     * @return `ConstExpandedView` of dimension `VecDims + extra`.
     */
    template<dims_t extra>
    DEVICEHOST()
    inline decltype(auto) headExpansion() const
    {
        return expansion<extra, 0>();
    }

    /**
     * @brief Return a mutable back-padded view with `extra` null-valued
     * dimensions appended.
     *
     * @tparam extra  Number of zero-valued dimensions to append.
     * @return `ExpandedView` of dimension `VecDims + extra`.
     */
    template<dims_t extra>
    DEVICEHOST()
    inline decltype(auto) tailExpansion()
    {
        return expansion<0, extra>();
    }

    /**
     * @brief Return a read-only back-padded view with `extra` null-valued
     * dimensions appended.
     *
     * @tparam extra  Number of zero-valued dimensions to append.
     * @return `ConstExpandedView` of dimension `VecDims + extra`.
     */
    template<dims_t extra>
    DEVICEHOST()
    inline decltype(auto) tailExpansion() const
    {
        return expansion<0, extra>();
    }

    /**
     * @brief Return a 4D view of this 3D vector as a pure quaternion (zero
     * scalar prepended).
     *
     * `[x, y, z]` → `[0, x, y, z]`.  Only valid when `VecDims == 3`.
     *
     * @return Mutable `ExpandedView` of dimension 4.
     */
    DEVICEHOST()
    inline decltype(auto) asPureQuaternion()
    {
        static_assert(
            VecDims == 3, "Only 3d vectors can be viewed as pure quaternions!");
        return headExpansion<1>();
    }
    /** @overload asPureQuaternion() — read-only variant. */
    DEVICEHOST()
    inline decltype(auto) asPureQuaternion() const
    {
        static_assert(
            VecDims == 3, "Only 3d vectors can be viewed as pure quaternions!");
        return headExpansion<1>();
    }

    /**
     * @brief Return a 4D view of this 3D vector as a front quaternion (zero
     * scalar appended).
     *
     * `[x, y, z]` → `[x, y, z, 0]`.  Only valid when `VecDims == 3`.
     *
     * @return Mutable `ExpandedView` of dimension 4.
     */
    DEVICEHOST()
    inline decltype(auto) asFrontQuaternion()
    {
        static_assert(VecDims == 3,
            "Only 3d vectors can be viewed as front quaternions!");
        return tailExpansion<1>();
    }
    /** @overload asFrontQuaternion() — read-only variant. */
    DEVICEHOST()
    inline decltype(auto) asFrontQuaternion() const
    {
        static_assert(VecDims == 3,
            "Only 3d vectors can be viewed as front quaternions!");
        return tailExpansion<1>();
    }

    /**
     * @brief Return a 3D view of this quaternion's vector part (last three
     * components).
     *
     * Discards the scalar (first) component.  Only valid when `VecDims == 4`.
     *
     * @return Mutable `ComponentView` of dimension 3 starting at index 1.
     */
    DEVICEHOST()
    inline decltype(auto) asBack3DVector()
    {
        static_assert(VecDims == 4,
            "Only 4d vectors / Quaternions can be viewed as back 3D vectors!");
        return tail<3>();
    }
    /** @overload asBack3DVector() — read-only variant. */
    DEVICEHOST()
    inline decltype(auto) asBack3DVector() const
    {
        static_assert(VecDims == 4,
            "Only 4d vectors / Quaternions can be viewed as back 3D vectors!");
        return tail<3>();
    }

    /**
     * @brief Return a 3D view of this quaternion's first three components.
     *
     * Discards the last component.  Only valid when `VecDims == 4`.
     *
     * @return Mutable `ComponentView` of dimension 3 starting at index 0.
     */
    DEVICEHOST()
    inline decltype(auto) asFront3DVector()
    {
        static_assert(VecDims == 4,
            "Only 4d vectors / Quaternions can be viewed as front 3D vectors!");
        return head<3>();
    }
    /** @overload asFront3DVector() — read-only variant. */
    DEVICEHOST()
    inline decltype(auto) asFront3DVector() const
    {
        static_assert(VecDims == 4,
            "Only 4d vectors / Quaternions can be viewed as front 3D vectors!");
        return head<3>();
    }
};

} // namespace expr
} // namespace vector
} // namespace feta
