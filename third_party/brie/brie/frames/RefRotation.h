#pragma once

#include "brie/frames/Rotation.h"

namespace brie {
namespace frames {

/** @brief Shmem-backed view of a quaternion rotation.
 *
 * Sources the (w, x, y, z) quaternion data from a per-thread feta
 * vector handle slot rather than holding a register-resident
 * @c Vec4T<Real> copy.  Mirrors the @ref Rotation public surface used
 * by acceleration kernels (``rotate``, ``applyInverse``), so it can
 * be a drop-in substitute at call sites where the rotation data
 * lives in shared memory.
 *
 * Use case: the SH gravity kernel needs to rotate twice (forward to
 * body frame, then inverse back to inertial) across a long
 * register-bound evaluation loop.  Storing the quaternion in shmem
 * and reading it on-demand inside @ref rotate / @ref applyInverse
 * keeps the 32 B of quaternion state out of the register file across
 * the loop body.  Each call materialises the quaternion into a Vec4R
 * local exactly once (4 LDS reads via feta ET), uses it for the
 * @c quatMul chain, and drops it before returning.
 *
 * The view itself is two references — no per-instance footprint.
 *
 * @tparam Direct  Active (true) vs passive (false) rotation, mirroring
 *                 @ref Rotation<Direct>.
 * @tparam HandleT The feta vector handle type for the per-thread
 *                 quaternion slot.  Expected to be
 *                 @c feta::vector::Array<Real, 4>::WRef::HandleT or its
 *                 volatile sibling.
 */
template<bool Direct, typename HandleT>
class RefRotation {
    using QuatT = feta::Vec4T<Real>;
    using Vec3R = feta::Vec3T<Real>;

    HandleT& slot_;
    const SampleIndex& i_;

public:
    FORCEINLINE() DEVICEHOST()
    RefRotation(HandleT& slot, const SampleIndex& i)
        : slot_{ slot }
        , i_{ i }
    {
    }

    /** @brief Initialise the per-thread slot from any 4-D feta
     * expression.  The assignment goes through feta's
     * @c View::operator= and unfolds to four per-component stores
     * without a Vec4R intermediate. */
    template<typename Expr>
    FORCEINLINE() DEVICEHOST() static void init(
        HandleT& slot, const SampleIndex& i, const Expr& expr)
    {
        slot[i] = expr;
    }

    /** @brief Apply the direct rotation to a 3-D or 4-D vector
     * expression.  Behaviour matches @ref Rotation<Direct>::rotate
     * exactly; the only difference is that the quaternion is loaded
     * from the per-thread shmem slot rather than a stored member. */
    template<typename Expr>
    FORCEINLINE() DEVICEHOST()
        std::conditional_t<(Expr::VecDims == 4), QuatT, Vec3R>
        rotate(const Expr& vec) const
    {
        static_assert(feta::core::expr::isExpression<Expr>::value,
            "Input must be a feta vector expression");
        static_assert(Expr::VecDims == 3 || Expr::VecDims == 4,
            "Input axis must be 3D or 4D vector");

        const QuatT q    = slot_[i_];           // single 4-LDS materialisation
        const QuatT qinv = q.quatReciprocal();  // register-only

        if constexpr (Expr::VecDims == 4) {
            return q.quatMul(vec).quatMul(qinv);
        } else {
            return q.quatMul(vec.asPureQuaternion())
                .quatMul(qinv)
                .asBack3DVector();
        }
    }

    /** @brief Apply the inverse rotation.  Same materialisation
     * pattern as @ref rotate. */
    template<typename Expr>
    FORCEINLINE() DEVICEHOST()
        std::conditional_t<(Expr::VecDims == 4), QuatT, Vec3R>
        applyInverse(const Expr& vec) const
    {
        static_assert(feta::core::expr::isExpression<Expr>::value,
            "Input must be a feta vector expression");
        static_assert(Expr::VecDims == 3 || Expr::VecDims == 4,
            "Input axis must be 3D or 4D vector");

        const QuatT q    = slot_[i_];
        const QuatT qinv = q.quatReciprocal();

        if constexpr (Expr::VecDims == 4) {
            return qinv.quatMul(vec).quatMul(q);
        } else {
            return qinv.quatMul(vec.asPureQuaternion())
                .quatMul(q)
                .asBack3DVector();
        }
    }
};

} // namespace frames
} // namespace brie
