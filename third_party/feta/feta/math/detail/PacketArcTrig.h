#pragma once

/**
 * @file PacketArcTrig.h
 * @brief Vectorised atan, atan2, asin, acos for Packet<float,W> / Packet<double,W>.
 *
 * atan: range-reduce to [0,1] then apply degree-9 (float) / degree-15 (double)
 *       odd minimax polynomial.
 * atan2: atan(|y|/|x|) + quadrant correction.
 * asin/acos: identity via atan2 + sqrt.
 */

#include "feta/core/simd/Packet.h"
#include "feta/math/detail/PacketBitOps.h"
#include "feta/macros.h"

namespace feta {
namespace math {
namespace detail {

// ─────────────────────────────────────────────────────────────────────────────
//  atan
// ─────────────────────────────────────────────────────────────────────────────

template<feta::idx_t W>
FORCEINLINE() simd::Packet<float, W> packetAtan(simd::Packet<float, W> x)
{
    using P = simd::Packet<float, W>;

    // Save sign, work on |x|
    auto s  = copysign(P::broadcast(1.f), x);
    auto ax = abs(x);

    // Reduce ax > 1: atan(ax) = pi/2 - atan(1/ax)
    auto one     = P::broadcast(1.f);
    auto flag_gt1 = cmpGt(ax, one);
    auto ax_r     = select(flag_gt1, one / ax, ax);  // ax_r ∈ [0,1]

    // Reduce ax_r > tan(pi/8) ≈ 0.4142: atan(ax_r) = pi/4 + atan((ax_r-1)/(ax_r+1))
    auto tan_pio8  = P::broadcast(0.41421356f);  // sqrt(2)-1
    auto flag_gt8  = cmpGt(ax_r, tan_pio8);
    auto ax_r2     = select(flag_gt8, (ax_r - one) / (ax_r + one), ax_r);

    // degree-9 odd minimax polynomial for atan(u), u ∈ [0, tan(pi/8)]
    auto u  = ax_r2;
    auto u2 = u * u;
    auto p = P::broadcast(-0.0520194f);
    p = fmadd(p, u2, P::broadcast( 0.1110899f));
    p = fmadd(p, u2, P::broadcast(-0.1428571f));
    p = fmadd(p, u2, P::broadcast( 0.1999999f));
    p = fmadd(p, u2, P::broadcast(-0.3333333f));
    p = fmadd(p, u2, P::broadcast( 1.0f));
    p = p * u;

    // Reconstruct pi/4 offset from second reduction
    const auto pio4 = P::broadcast(0.78539816339744830962f);
    p = select(flag_gt8, p + pio4, p);

    // Reconstruct pi/2 offset from first reduction
    const auto pio2 = P::broadcast(1.57079632679489661923f);
    p = select(flag_gt1, pio2 - p, p);

    // Restore sign
    return copysign(p, s);
}

template<feta::idx_t W>
FORCEINLINE() simd::Packet<double, W> packetAtan(simd::Packet<double, W> x)
{
    using P = simd::Packet<double, W>;

    auto s  = copysign(P::broadcast(1.0), x);
    auto ax = abs(x);

    auto one      = P::broadcast(1.0);
    auto flag_gt1 = cmpGt(ax, one);
    auto ax_r     = select(flag_gt1, one / ax, ax);

    auto tan_pio8 = P::broadcast(0.41421356237309504880);
    auto flag_gt8 = cmpGt(ax_r, tan_pio8);
    auto ax_r2    = select(flag_gt8, (ax_r - one) / (ax_r + one), ax_r);

    auto u  = ax_r2;
    auto u2 = u * u;

    // degree-15 odd minimax polynomial
    auto p = P::broadcast( 9.99999999999999999956e-1);
    p = fmadd(p, u2, P::broadcast(-3.33333333333333327430e-1));
    p = fmadd(p, u2, P::broadcast( 1.99999999999999993575e-1));
    p = fmadd(p, u2, P::broadcast(-1.42857142857142840530e-1));
    p = fmadd(p, u2, P::broadcast( 1.11111111111110491699e-1));
    p = fmadd(p, u2, P::broadcast(-9.09090909090906479592e-2));
    p = fmadd(p, u2, P::broadcast( 7.69230769230769120607e-2));
    p = fmadd(p, u2, P::broadcast(-6.66666666666666579930e-2));
    // Rewrite: start from constant term and accumulate
    // (using Horner in increasing-power form via previous accumulation)
    p = p * u;

    const auto pio4 = P::broadcast(0.78539816339744830962);
    const auto pio2 = P::broadcast(1.57079632679489661923);
    p = select(flag_gt8, p + pio4, p);
    p = select(flag_gt1, pio2 - p, p);
    return copysign(p, s);
}

// ─────────────────────────────────────────────────────────────────────────────
//  atan2(y, x)
// ─────────────────────────────────────────────────────────────────────────────

template<typename DataT, feta::idx_t W>
FORCEINLINE() simd::Packet<DataT, W> packetAtan2(simd::Packet<DataT, W> y, simd::Packet<DataT, W> x)
{
    using P = simd::Packet<DataT, W>;
    const auto pi  = P::broadcast(DataT(3.14159265358979323846));
    const auto pio2 = P::broadcast(DataT(1.57079632679489661923));
    const auto zero = P::zero();

    auto ay = abs(y);
    auto ax = abs(x);

    // Base angle: atan(ay/ax), handling ax==0 via select
    auto safe_ax = select(cmpGe(ax, P::broadcast(DataT(1e-300))), ax, P::broadcast(DataT(1)));
    auto a       = packetAtan(ay / safe_ax);
    // If ax is zero: angle should be pi/2
    a = select(cmpLt(ax, P::broadcast(DataT(1e-300))), pio2, a);

    // x < 0: a = pi - a
    a = select(cmpLt(x, zero), pi - a, a);

    // Restore sign of y (atan2 has the sign of y)
    return copysign(a, y);
}

// ─────────────────────────────────────────────────────────────────────────────
//  asin(x) = atan2(x, sqrt(1 - x^2))
// ─────────────────────────────────────────────────────────────────────────────

template<typename DataT, feta::idx_t W>
FORCEINLINE() simd::Packet<DataT, W> packetAsin(simd::Packet<DataT, W> x)
{
    using P = simd::Packet<DataT, W>;
    auto one  = P::broadcast(DataT(1));
    // Clamp for numerical safety
    auto ax   = abs(x);
    ax        = min(ax, one);
    auto cosx = sqrt(fnmadd(ax, ax, one));  // sqrt(1 - |x|^2)
    auto a    = packetAtan2(ax, cosx);
    return copysign(a, x);
}

// ─────────────────────────────────────────────────────────────────────────────
//  acos(x) = atan2(sqrt(1 - x^2), x)
// ─────────────────────────────────────────────────────────────────────────────

template<typename DataT, feta::idx_t W>
FORCEINLINE() simd::Packet<DataT, W> packetAcos(simd::Packet<DataT, W> x)
{
    using P = simd::Packet<DataT, W>;
    auto one  = P::broadcast(DataT(1));
    auto ax   = min(abs(x), one);
    auto sinx = sqrt(fnmadd(ax, ax, one));
    // acos(|x|), then adjust for sign of x:
    //   acos(x) = pi - acos(-x) for x < 0
    // Use atan2(sinx, ax) for |x|, then correct
    auto a    = packetAtan2(sinx, ax);
    const auto pi = P::broadcast(DataT(3.14159265358979323846));
    return select(cmpLt(x, P::zero()), pi - a, a);
}

} // namespace detail
} // namespace math
} // namespace feta
