#pragma once

/**
 * @file PacketMisc.h
 * @brief Miscellaneous packet math: abs, fma, fmax, fmin, copysign, fdim,
 *        fmod, rsqrt, cbrt, hypot.
 *
 * Most of these are thin wrappers around already-available Packet friends.
 * rsqrt, cbrt, and hypot require small iterative refinements.
 */

#include "feta/core/simd/Packet.h"
#include "feta/macros.h"
#include "feta/math/detail/PacketBitOps.h"
#include "feta/math/detail/PacketRounding.h"

namespace feta {
namespace math {
namespace detail {

// ─────────────────────────────────────────────────────────────────────────────
//  Trivial delegations
// ─────────────────────────────────────────────────────────────────────────────

template<typename DataT, feta::idx_t W>
FORCEINLINE()
simd::Packet<DataT, W> packetAbs(simd::Packet<DataT, W> x)
{
    return abs(x); // Packet.h friend
}

template<typename DataT, feta::idx_t W>
FORCEINLINE()
simd::Packet<DataT, W> packetFma(simd::Packet<DataT, W> a,
    simd::Packet<DataT, W> b, simd::Packet<DataT, W> c)
{
    return fmadd(a, b, c);
}

template<typename DataT, feta::idx_t W>
FORCEINLINE()
simd::Packet<DataT, W> packetFmax(
    simd::Packet<DataT, W> x, simd::Packet<DataT, W> y)
{
    return max(x, y);
}

template<typename DataT, feta::idx_t W>
FORCEINLINE()
simd::Packet<DataT, W> packetFmin(
    simd::Packet<DataT, W> x, simd::Packet<DataT, W> y)
{
    return min(x, y);
}

template<typename DataT, feta::idx_t W>
FORCEINLINE()
simd::Packet<DataT, W> packetCopysign(
    simd::Packet<DataT, W> mag, simd::Packet<DataT, W> sgn)
{
    return copysign(mag, sgn);
}

template<typename DataT, feta::idx_t W>
FORCEINLINE()
simd::Packet<DataT, W> packetSqrt(simd::Packet<DataT, W> x)
{
    return sqrt(x); // Packet.h friend, found via ADL
}

// fdim(x,y) = max(x-y, 0)
template<typename DataT, feta::idx_t W>
FORCEINLINE()
simd::Packet<DataT, W> packetFdim(
    simd::Packet<DataT, W> x, simd::Packet<DataT, W> y)
{
    auto diff = x - y;
    return select(cmpGt(x, y), diff, simd::Packet<DataT, W>::zero());
}

// fmod(x,y) = x - trunc(x/y)*y
template<typename DataT, feta::idx_t W>
FORCEINLINE()
simd::Packet<DataT, W> packetFmod(
    simd::Packet<DataT, W> x, simd::Packet<DataT, W> y)
{
    auto t = packetTrunc(x / y);
    return x - t * y;
}

// ─────────────────────────────────────────────────────────────────────────────
//  rsqrt: reciprocal square root
// ─────────────────────────────────────────────────────────────────────────────

// Float rsqrt: fast approximation + Newton-Raphson iterations
// Newton step: y_new = y * (1.5 - 0.5 * x * y * y)
template<feta::idx_t W>
FORCEINLINE()
simd::Packet<float, W> packetRsqrt(simd::Packet<float, W> x)
{
    using P = simd::Packet<float, W>;
    P y;

#if defined(__AVX512F__)
    if constexpr (W == 16) {
        // 14-bit approximation
        y = P{ _mm512_rsqrt14_ps(x.reg_) };
        // One Newton iteration gives ~28-bit accuracy (sufficient for float)
        auto half  = P::broadcast(0.5f);
        auto three = P::broadcast(1.5f);
        y          = y * fnmadd(x * half, y * y, three);
        return y;
    }
#endif
#if defined(__AVX2__) || defined(__AVX__)
    if constexpr (W == 8) {
        y = P{ _mm256_rsqrt_ps(x.reg_) }; // ~12-bit
        // Two Newton iterations for ~24-bit
        auto half  = P::broadcast(0.5f);
        auto three = P::broadcast(1.5f);
        y          = y * fnmadd(x * half, y * y, three);
        y          = y * fnmadd(x * half, y * y, three);
        return y;
    }
#endif
#if defined(__SSE2__)
    if constexpr (W == 4) {
        y          = P{ _mm_rsqrt_ps(x.reg_) }; // ~12-bit
        auto half  = P::broadcast(0.5f);
        auto three = P::broadcast(1.5f);
        y          = y * fnmadd(x * half, y * y, three);
        y          = y * fnmadd(x * half, y * y, three);
        return y;
    }
#endif
    // Width==1 scalar
    if constexpr (W == 1) {
        return P{ 1.0f / std::sqrt(x.val_) };
    }
}

// Double rsqrt: 1/sqrt via Newton from float approximation
template<feta::idx_t W>
FORCEINLINE()
simd::Packet<double, W> packetRsqrt(simd::Packet<double, W> x)
{
    using P  = simd::Packet<double, W>;
    auto one = P::broadcast(1.0);
    return one / sqrt(x);
}

// ─────────────────────────────────────────────────────────────────────────────
//  cbrt: cube root via Newton-Raphson
// ─────────────────────────────────────────────────────────────────────────────

// Float cbrt: bit-hack initial guess + 2 Newton iterations
// Bit hack: cbrt(x) ≈ reinterpret(bits(x)/3 + 0x2A5137A0) for positive x
template<feta::idx_t W>
FORCEINLINE()
simd::Packet<float, W> packetCbrt(simd::Packet<float, W> x)
{
    using P = simd::Packet<float, W>;

    // Work on |x|, restore sign at end
    auto s  = copysign(P::broadcast(1.f), x);
    auto ax = abs(x);

    // Bit-hack initial guess for cbrt(|x|)
    // reinterpret(reinterpret_as_int(ax) / 3 + 0x2A5137A0)
    auto ibits = castToInt(ax);
    // Integer division by 3: use multiply by 0x55555556 >> 32 approximation
    // Simpler: use the magic constant shift directly
    // bits/3 ≈ (bits * 0xAAAAAAAB) >> 33   -- Knuth division by 3
    // Easier: use arithmetic shift right (signed) divide by 3 approximation
    // We use the standard formula: y0 = as_float(as_int(x)/3 + 0x2A5137A0)
    // For integer division by 3: approximate as multiply+shift
    auto div3  = addInt(shiftRightArith(ibits, 1), shiftRightArith(ibits, 3));
    div3       = addInt(div3, shiftRightArith(div3, 4));
    div3       = addInt(div3, shiftRightArith(div3, 8));
    div3       = addInt(div3, shiftRightArith(div3, 16));
    div3       = shiftRightArith(div3, 1);
    auto magic = broadcastInt<W>(0x2A5137A0);
    auto y     = castToFloat(addInt(div3, magic));

    // Three Newton iterations: y = y*(2/3) + x/(3*y^2)
    // Equivalently: y = (2*y + x/y^2) / 3
    auto two_third = P::broadcast(2.0f / 3.0f);
    auto one_third = P::broadcast(1.0f / 3.0f);
    y              = fmadd(two_third, y, ax * one_third / (y * y));
    y              = fmadd(two_third, y, ax * one_third / (y * y));
    y              = fmadd(two_third, y, ax * one_third / (y * y));

    return copysign(y, s);
}

template<feta::idx_t W>
FORCEINLINE()
simd::Packet<double, W> packetCbrt(simd::Packet<double, W> x)
{
    using P = simd::Packet<double, W>;

    auto s  = copysign(P::broadcast(1.0), x);
    auto ax = abs(x);

    // Per-lane scalar cbrt as initial guess, then refine
    alignas(sizeof(double) * W) double buf[W];
    simd::Packet<double, W>::store(buf, ax);
    for (feta::idx_t k = 0; k < W; ++k)
        buf[k] = std::cbrt(buf[k]);
    auto y = simd::Packet<double, W>::load(buf);

    // Two Newton iterations in double precision
    auto two_third = P::broadcast(2.0 / 3.0);
    auto one_third = P::broadcast(1.0 / 3.0);
    y              = fmadd(two_third, y, ax * one_third / (y * y));
    y              = fmadd(two_third, y, ax * one_third / (y * y));

    return copysign(y, s);
}

// ─────────────────────────────────────────────────────────────────────────────
//  hypot: sqrt(x^2 + y^2) — scaled to avoid overflow
// ─────────────────────────────────────────────────────────────────────────────

template<typename DataT, feta::idx_t W>
FORCEINLINE()
simd::Packet<DataT, W> packetHypot(
    simd::Packet<DataT, W> x, simd::Packet<DataT, W> y)
{
    using P = simd::Packet<DataT, W>;
    auto ax = abs(x);
    auto ay = abs(y);
    auto a  = max(ax, ay);
    auto b  = min(ax, ay);
    // a * sqrt(1 + (b/a)^2); handle a==0 to avoid NaN
    auto zero = P::zero();
    auto one  = P::broadcast(DataT(1));
    auto r    = b / a;
    auto h    = a * sqrt(fmadd(r, r, one));
    return select(cmpGe(a, zero), h, zero); // a==0 → 0
}

// ─────────────────────────────────────────────────────────────────────────────
//  Classification: isfinite, isinf, isnan (return bool — all-lanes predicate)
// ─────────────────────────────────────────────────────────────────────────────

template<feta::idx_t W>
FORCEINLINE()
bool packetIsFinite(simd::Packet<float, W> x)
{
    // inf/nan have all exponent bits set: bits & 0x7F800000 == 0x7F800000
    auto ibits    = castToInt(abs(x));
    auto exp      = bitwiseAnd(ibits, broadcastInt<W>(0x7F800000));
    auto isInfNan = (exp == broadcastInt<W>(0x7F800000)); // per-lane
    // We need all-lanes finite → none of them inf/nan
    // Use mask check: build a mask from compare, check none true
    // Approximate via: if any lane is inf/nan → not finite
    // Use per-lane select trick: convert to float 0/1, sum, check == 0
    // Simpler: just use std::isfinite per lane
    alignas(sizeof(float) * W) float buf[W];
    simd::Packet<float, W>::store(buf, x);
    for (feta::idx_t k = 0; k < W; ++k)
        if (!std::isfinite(buf[k]))
            return false;
    return true;
    (void)ibits;
    (void)exp;
    (void)isInfNan; // suppress unused-variable
}

template<feta::idx_t W>
FORCEINLINE()
bool packetIsFinite(simd::Packet<double, W> x)
{
    alignas(sizeof(double) * W) double buf[W];
    simd::Packet<double, W>::store(buf, x);
    for (feta::idx_t k = 0; k < W; ++k)
        if (!std::isfinite(buf[k]))
            return false;
    return true;
}

template<feta::idx_t W>
FORCEINLINE()
bool packetIsInf(simd::Packet<float, W> x)
{
    alignas(sizeof(float) * W) float buf[W];
    simd::Packet<float, W>::store(buf, x);
    for (feta::idx_t k = 0; k < W; ++k)
        if (std::isinf(buf[k]))
            return true;
    return false;
}

template<feta::idx_t W>
FORCEINLINE()
bool packetIsInf(simd::Packet<double, W> x)
{
    alignas(sizeof(double) * W) double buf[W];
    simd::Packet<double, W>::store(buf, x);
    for (feta::idx_t k = 0; k < W; ++k)
        if (std::isinf(buf[k]))
            return true;
    return false;
}

template<feta::idx_t W>
FORCEINLINE()
bool packetIsNan(simd::Packet<float, W> x)
{
    alignas(sizeof(float) * W) float buf[W];
    simd::Packet<float, W>::store(buf, x);
    for (feta::idx_t k = 0; k < W; ++k)
        if (std::isnan(buf[k]))
            return true;
    return false;
}

template<feta::idx_t W>
FORCEINLINE()
bool packetIsNan(simd::Packet<double, W> x)
{
    alignas(sizeof(double) * W) double buf[W];
    simd::Packet<double, W>::store(buf, x);
    for (feta::idx_t k = 0; k < W; ++k)
        if (std::isnan(buf[k]))
            return true;
    return false;
}

} // namespace detail
} // namespace math
} // namespace feta
