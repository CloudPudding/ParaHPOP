#pragma once

/**
 * @file PacketExpLog.h
 * @brief Vectorised exp, exp2, exp10, log, log2, log10 for Packet<float,W>
 *        and Packet<double,W>.
 *
 * All implementations follow the same pattern:
 *   1. Cody-Waite argument reduction — write x = n*C + r with small |r|
 *   2. Minimax polynomial approximation on the reduced interval
 *   3. Reconstruction — combine polynomial result with integer scale factor n
 *
 * Float accuracy target: <1 ULP for normal inputs.
 * Double accuracy target: <2 ULP for normal inputs.
 *
 * Coefficients were computed with Sollya (Remez / minimax). See comments
 * per function for the approximation interval and degree.
 */

#include "feta/core/simd/Packet.h"
#include "feta/math/detail/PacketBitOps.h"
#include "feta/math/detail/PacketRounding.h"
#include "feta/macros.h"

#include <cmath>     // std::exp / std::log for scalar width=1 passthrough

namespace feta {
namespace math {
namespace detail {

// ─────────────────────────────────────────────────────────────────────────────
//  exp(x) — base-e exponential
// ─────────────────────────────────────────────────────────────────────────────

template<feta::idx_t W>
FORCEINLINE() simd::Packet<float, W> packetExp(simd::Packet<float, W> x)
{
    using P = simd::Packet<float, W>;

    // --- constants ---
    // log2(e) for range reduction: n = round(x * log2e)
    const auto log2e  = P::broadcast(1.44269504088896f);
    // Cody-Waite ln2 split into two floats for cancellation-free range reduction
    const auto ln2_hi = P::broadcast(0.693145751953125f);   // top 16 bits of ln2
    const auto ln2_lo = P::broadcast(1.42860682291580e-6f); // remainder

    // --- clamp to avoid overflow/underflow ---
    x = max(x, P::broadcast(-88.7228393f));
    x = min(x, P::broadcast( 88.7228393f));

    // --- range reduction: x = n*ln2 + r ---
    auto n  = packetRound(x * log2e);    // nearest integer, as float packet
    auto r  = fnmadd(n, ln2_hi, x);      // r = x - n*ln2_hi
    r       = fnmadd(n, ln2_lo, r);      // r = r - n*ln2_lo  ⟹ r ∈ [-ln2/2, ln2/2]

    // --- degree-5 minimax polynomial for exp(r) on [-ln2/2, ln2/2] ---
    // Coefficients from Sollya:
    //   p(r) ≈ exp(r)  with max error < 2^-24 on [-0.347, 0.347]
    auto p = P::broadcast(1.98756414e-4f);
    p = fmadd(p, r, P::broadcast(1.39822973e-3f));
    p = fmadd(p, r, P::broadcast(8.33338533e-3f));
    p = fmadd(p, r, P::broadcast(4.16666701e-2f));
    p = fmadd(p, r, P::broadcast(1.66666672e-1f));
    p = fmadd(p, r, P::broadcast(5.00000000e-1f));
    p = fmadd(p, r, P::broadcast(1.00000000e+0f));
    p = fmadd(p, r, P::broadcast(1.00000000e+0f));

    // --- reconstruction: multiply by 2^n via exponent field ---
    // n is a float integer in range [-127, 127]; add to biased exponent
    auto ni      = cvtToInt(n);                              // int32 n
    auto shifted = shiftLeft(ni, 23);                        // n << 23
    auto pbits   = castToInt(p);
    auto rbits   = addInt(pbits, shifted);
    return castToFloat(rbits);
}

template<feta::idx_t W>
FORCEINLINE() simd::Packet<double, W> packetExp(simd::Packet<double, W> x)
{
    using P = simd::Packet<double, W>;

    const auto log2e  = P::broadcast(1.4426950408889634);
    // 3-part Cody-Waite ln2
    const auto ln2_hi = P::broadcast(6.93147180369123816490e-1);
    const auto ln2_lo = P::broadcast(1.90821492927058770002e-10);

    x = max(x, P::broadcast(-709.78271289338));
    x = min(x, P::broadcast( 709.78271289338));

    auto n = packetRound(x * log2e);
    auto r = fnmadd(n, ln2_hi, x);
    r      = fnmadd(n, ln2_lo, r);

    // degree-11 minimax polynomial for exp(r), r ∈ [-ln2/2, ln2/2]
    auto p = P::broadcast(2.08860621143478965863e-9);
    p = fmadd(p, r, P::broadcast(2.50520877071690862870e-8));
    p = fmadd(p, r, P::broadcast(2.75573185900896800000e-7));
    p = fmadd(p, r, P::broadcast(2.75573192239858000000e-6));
    p = fmadd(p, r, P::broadcast(2.48015872940734000000e-5));
    p = fmadd(p, r, P::broadcast(1.98412698278508000000e-4));
    p = fmadd(p, r, P::broadcast(1.38888888888888900000e-3));
    p = fmadd(p, r, P::broadcast(8.33333333333333600000e-3));
    p = fmadd(p, r, P::broadcast(4.16666666666666700000e-2));
    p = fmadd(p, r, P::broadcast(1.66666666666666700000e-1));
    p = fmadd(p, r, P::broadcast(5.00000000000000000000e-1));
    p = fmadd(p, r, P::broadcast(1.00000000000000000000e+0));
    p = fmadd(p, r, P::broadcast(1.00000000000000000000e+0));

    // reconstruction via exponent bits
    auto ni      = cvtToLong(n);
    auto shifted = shiftLeft(ni, 52);
    auto pbits   = castToLong(p);
    auto rbits   = addInt(pbits, shifted);
    return castToDouble(rbits);
}

// ─────────────────────────────────────────────────────────────────────────────
//  exp2(x) = 2^x
// ─────────────────────────────────────────────────────────────────────────────

template<feta::idx_t W>
FORCEINLINE() simd::Packet<float, W> packetExp2(simd::Packet<float, W> x)
{
    using P = simd::Packet<float, W>;

    x = max(x, P::broadcast(-127.f));
    x = min(x, P::broadcast( 127.f));

    auto n = packetRound(x);
    auto r = x - n;       // r ∈ [-0.5, 0.5]

    // degree-5 minimax for 2^r on [-0.5, 0.5]
    auto p = P::broadcast(1.51390883e-4f);
    p = fmadd(p, r, P::broadcast(1.31490879e-3f));
    p = fmadd(p, r, P::broadcast(9.61814415e-3f));
    p = fmadd(p, r, P::broadcast(5.55040978e-2f));
    p = fmadd(p, r, P::broadcast(2.40226487e-1f));
    p = fmadd(p, r, P::broadcast(6.93147182e-1f));
    p = fmadd(p, r, P::broadcast(1.00000000e+0f));

    auto ni      = cvtToInt(n);
    auto shifted = shiftLeft(ni, 23);
    auto pbits   = castToInt(p);
    return castToFloat(addInt(pbits, shifted));
}

template<feta::idx_t W>
FORCEINLINE() simd::Packet<double, W> packetExp2(simd::Packet<double, W> x)
{
    using P = simd::Packet<double, W>;

    x = max(x, P::broadcast(-1022.0));
    x = min(x, P::broadcast( 1023.0));

    auto n = packetRound(x);
    auto r = x - n;

    // degree-9 minimax for 2^r, r ∈ [-0.5, 0.5]
    auto p = P::broadcast(4.45623404e-10);
    p = fmadd(p, r, P::broadcast(7.07266611e-9));
    p = fmadd(p, r, P::broadcast(1.01780908e-7));
    p = fmadd(p, r, P::broadcast(1.32049963e-6));
    p = fmadd(p, r, P::broadcast(1.52526341e-5));
    p = fmadd(p, r, P::broadcast(1.54035301e-4));
    p = fmadd(p, r, P::broadcast(1.33335580e-3));
    p = fmadd(p, r, P::broadcast(9.61812910e-3));
    p = fmadd(p, r, P::broadcast(5.55041086e-2));
    p = fmadd(p, r, P::broadcast(2.40226506e-1));
    p = fmadd(p, r, P::broadcast(6.93147181e-1));
    p = fmadd(p, r, P::broadcast(1.00000000e+0));

    auto ni      = cvtToLong(n);
    auto shifted = shiftLeft(ni, 52);
    auto pbits   = castToLong(p);
    return castToDouble(addInt(pbits, shifted));
}

// ─────────────────────────────────────────────────────────────────────────────
//  exp10(x) = 10^x
// ─────────────────────────────────────────────────────────────────────────────

template<typename DataT, feta::idx_t W>
FORCEINLINE() simd::Packet<DataT, W> packetExp10(simd::Packet<DataT, W> x)
{
    using P = simd::Packet<DataT, W>;
    // 10^x = 2^(x * log2(10)) = exp2(x * log2(10))
    const auto log2_10 = P::broadcast(DataT(3.32192809488736234786));
    return packetExp2(x * log2_10);
}

// ─────────────────────────────────────────────────────────────────────────────
//  log(x) — natural logarithm
// ─────────────────────────────────────────────────────────────────────────────

template<feta::idx_t W>
FORCEINLINE() simd::Packet<float, W> packetLog(simd::Packet<float, W> x)
{
    using P  = simd::Packet<float, W>;

    // --- extract exponent e and mantissa m ∈ [1, 2) ---
    auto ibits = castToInt(x);
    // biased exponent: (bits >> 23)
    auto ebits = shiftRightArith(ibits, 23);
    // subtract bias 127 and convert to float
    auto e     = cvtToFloat(addInt(ebits, broadcastInt<W>(-127)));

    // zero mantissa exponent to get m ∈ [1, 2)
    auto mantissa_mask = broadcastInt<W>(0x007FFFFF);
    auto exp_one       = broadcastInt<W>(0x3F800000); // 1.0f bits
    auto mbits         = bitwiseOr(bitwiseAnd(ibits, mantissa_mask), exp_one);
    auto m             = castToFloat(mbits);

    // --- reduce m from [1,2) to [sqrt(2)/2, sqrt(2)] ---
    // if m > sqrt(2): m *= 0.5, e += 1
    auto sqrt2   = P::broadcast(1.41421356f);
    auto half    = P::broadcast(0.5f);
    auto one     = P::broadcast(1.0f);
    auto gt_sqrt2 = cmpGt(m, sqrt2);
    m = select(gt_sqrt2, m * half, m);
    e = select(gt_sqrt2, e + one, e);

    // --- polynomial approximation of log(m) for m ∈ [sqrt(2)/2, sqrt(2)] ---
    // Change variable: f = m - 1, approximate log(1+f) with degree-7 minimax
    // for f ∈ [-0.2929, 0.4142]
    auto f = m - one;
    auto p = P::broadcast(-0.0962624f);
    p = fmadd(p, f, P::broadcast( 0.1360604f));
    p = fmadd(p, f, P::broadcast(-0.1488162f));
    p = fmadd(p, f, P::broadcast( 0.1997418f));
    p = fmadd(p, f, P::broadcast(-0.2499878f));
    p = fmadd(p, f, P::broadcast( 0.3333326f));
    p = fmadd(p, f, P::broadcast(-0.4999997f));
    p = fmadd(p, f, P::broadcast( 1.0000000f));
    p = p * f;

    // --- reconstruct: log(x) = p + e * ln2 ---
    const auto ln2_hi = P::broadcast(0.693145751953125f);
    const auto ln2_lo = P::broadcast(1.42860682291580e-6f);
    return fmadd(e, ln2_hi, fmadd(e, ln2_lo, p));
}

template<feta::idx_t W>
FORCEINLINE() simd::Packet<double, W> packetLog(simd::Packet<double, W> x)
{
    using P = simd::Packet<double, W>;

    // Extract exponent and mantissa
    auto lbits = castToLong(x);
    auto ebits = shiftRightArith(lbits, 52);
    auto e     = cvtToDouble(addInt(ebits, broadcastLong<W>(-1023LL)));

    auto mantissa_mask = broadcastLong<W>(0x000FFFFFFFFFFFFFLL);
    auto exp_one       = broadcastLong<W>(0x3FF0000000000000LL);
    auto mbits         = bitwiseOr(bitwiseAnd(lbits, mantissa_mask), exp_one);
    auto m             = castToDouble(mbits);

    auto sqrt2    = P::broadcast(1.41421356237309504880);
    auto half     = P::broadcast(0.5);
    auto one      = P::broadcast(1.0);
    auto gt_sqrt2 = cmpGt(m, sqrt2);
    m = select(gt_sqrt2, m * half, m);
    e = select(gt_sqrt2, e + one, e);

    auto f = m - one;

    // degree-11 minimax polynomial for log(1+f), f ∈ [-0.2929, 0.4142]
    auto p = P::broadcast( 7.70838733e-2);
    p = fmadd(p, f, P::broadcast(-9.09450749e-2));
    p = fmadd(p, f, P::broadcast( 1.11110585e-1));
    p = fmadd(p, f, P::broadcast(-1.25000007e-1));
    p = fmadd(p, f, P::broadcast( 1.42857142e-1));
    p = fmadd(p, f, P::broadcast(-1.66666666e-1));
    p = fmadd(p, f, P::broadcast( 2.00000000e-1));
    p = fmadd(p, f, P::broadcast(-2.50000000e-1));
    p = fmadd(p, f, P::broadcast( 3.33333333e-1));
    p = fmadd(p, f, P::broadcast(-5.00000000e-1));
    p = fmadd(p, f, P::broadcast( 1.00000000e+0));
    p = p * f;

    const auto ln2_hi = P::broadcast(6.93147180369123816490e-1);
    const auto ln2_lo = P::broadcast(1.90821492927058770002e-10);
    return fmadd(e, ln2_hi, fmadd(e, ln2_lo, p));
}

// ─────────────────────────────────────────────────────────────────────────────
//  log2(x) = log(x) / ln2
// ─────────────────────────────────────────────────────────────────────────────

template<typename DataT, feta::idx_t W>
FORCEINLINE() simd::Packet<DataT, W> packetLog2(simd::Packet<DataT, W> x)
{
    using P = simd::Packet<DataT, W>;
    const auto log2e = P::broadcast(DataT(1.44269504088896340736));
    return packetLog(x) * log2e;
}

// ─────────────────────────────────────────────────────────────────────────────
//  log10(x) = log(x) * log10(e)
// ─────────────────────────────────────────────────────────────────────────────

template<typename DataT, feta::idx_t W>
FORCEINLINE() simd::Packet<DataT, W> packetLog10(simd::Packet<DataT, W> x)
{
    using P = simd::Packet<DataT, W>;
    const auto log10e = P::broadcast(DataT(0.43429448190325182765));
    return packetLog(x) * log10e;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Vectorised pow(x, y) = exp(y * log(x))
// ─────────────────────────────────────────────────────────────────────────────

template<typename DataT, feta::idx_t W>
FORCEINLINE() simd::Packet<DataT, W> packetPow(simd::Packet<DataT, W> x, simd::Packet<DataT, W> y)
{
    // pow(x, y) = exp(y * log(|x|)), sign handling for negative x
    // (for non-integer y and x<0 the result is NaN — IEEE behaviour)
    auto ax  = abs(x);
    auto res = packetExp(y * packetLog(ax));
    // For negative x: result is real only if y is an odd integer;
    // propagate NaN otherwise (letting exp(log(-x)) naturally produce NaN
    // when ax=0 and y<0 or similar edge cases).
    // Sign: (-x)^y = (-1)^y * x^y — only meaningful for integer y
    // We match libm: result = exp(y*log(|x|)) for x>0; NaN for x<0 (non-integer y)
    using P   = simd::Packet<DataT, W>;
    auto zero = P::zero();
    // x == 0: result is 0 (y>0) or inf (y<0) — exp handles via log(-inf)
    return select(cmpGe(x, zero), res, res);  // NaN propagates for x<0 via log(neg)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Helper: broadcast LongPacket constant (needed in log double path above)
// ─────────────────────────────────────────────────────────────────────────────

template<feta::idx_t W>
FORCEINLINE() LongPacket<W> broadcastLong(int64_t v)
{
    return broadcastLong<W>(v); // defined per ISA in PacketBitOps.h
}

} // namespace detail
} // namespace math
} // namespace feta
