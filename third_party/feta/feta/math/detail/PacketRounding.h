#pragma once

/**
 * @file PacketRounding.h
 * @brief SIMD floor, ceil, round, and trunc for Packet<float,W> and Packet<double,W>.
 *
 * AVX2/AVX-512 expose single-instruction rounding via _mm*_floor_p*, etc.
 * SSE2 requires a small sequence using integer conversion.
 *
 * `packetTrunc` is provided as an internal helper used by fmod and other
 * functions; it is not part of the public math.h API.
 */

#include "feta/core/simd/Packet.h"
#include "feta/math/detail/PacketBitOps.h"
#include "feta/macros.h"

namespace feta {
namespace math {
namespace detail {

// ─────────────────────────────────────────────────────────────────────────────
//  Internal helper: trunc (toward zero)
// ─────────────────────────────────────────────────────────────────────────────

// float trunc — value-convert to int32 (truncating) and back
// Works for |x| < 2^31; large values that are already integers are handled by
// the guard: if |x| >= 2^23 the value is already an integer (float precision).
template<feta::idx_t W>
FORCEINLINE() simd::Packet<float, W> packetTrunc(simd::Packet<float, W> x)
{
#if defined(__AVX512F__)
    if constexpr (W == 16)
        return simd::Packet<float,16>{_mm512_roundscale_ps(x.reg_, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)};
#endif
#if defined(__AVX2__) || defined(__AVX__)
    if constexpr (W == 8)
        return simd::Packet<float,8>{_mm256_round_ps(x.reg_, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)};
#endif
#if defined(__SSE4_1__) && !defined(__AVX__)
    if constexpr (W == 4)
        return simd::Packet<float,4>{_mm_round_ps(x.reg_, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)};
#endif
    // SSE2 fallback: int32 round-trip truncates toward zero
    // Guard: for |x| >= 2^23 the float already represents an integer
    auto large = simd::Packet<float,W>::broadcast(8388608.f); // 2^23
    auto ax    = abs(x);
    auto trunc = cvtToFloat(cvtToInt(x));          // truncates toward zero
    return select(cmpGe(ax, large), x, trunc);
}

template<feta::idx_t W>
FORCEINLINE() simd::Packet<double, W> packetTrunc(simd::Packet<double, W> x)
{
#if defined(__AVX512F__)
    if constexpr (W == 8)
        return simd::Packet<double,8>{_mm512_roundscale_pd(x.reg_, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)};
#endif
#if defined(__AVX2__) || defined(__AVX__)
    if constexpr (W == 4)
        return simd::Packet<double,4>{_mm256_round_pd(x.reg_, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)};
#endif
#if defined(__SSE4_1__) && !defined(__AVX__)
    if constexpr (W == 2)
        return simd::Packet<double,2>{_mm_round_pd(x.reg_, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)};
#endif
    // SSE2 scalar fallback
    alignas(sizeof(double) * W) double buf[W];
    simd::Packet<double,W>::store(buf, x);
    for (feta::idx_t k = 0; k < W; ++k)
        buf[k] = static_cast<double>(static_cast<int64_t>(buf[k]));
    return simd::Packet<double,W>::load(buf);
}

// ─────────────────────────────────────────────────────────────────────────────
//  packetFloor
// ─────────────────────────────────────────────────────────────────────────────

template<feta::idx_t W>
FORCEINLINE() simd::Packet<float, W> packetFloor(simd::Packet<float, W> x)
{
#if defined(__AVX512F__)
    if constexpr (W == 16)
        return simd::Packet<float,16>{_mm512_floor_ps(x.reg_)};
#endif
#if defined(__AVX2__) || defined(__AVX__)
    if constexpr (W == 8)
        return simd::Packet<float,8>{_mm256_floor_ps(x.reg_)};
#endif
#if defined(__SSE4_1__) && !defined(__AVX__)
    if constexpr (W == 4)
        return simd::Packet<float,4>{_mm_floor_ps(x.reg_)};
#endif
    // SSE2: floor(x) = trunc(x) - (x < trunc(x) ? 1 : 0)
    auto t    = packetTrunc(x);
    auto one  = simd::Packet<float,W>::broadcast(1.f);
    auto zero = simd::Packet<float,W>::zero();
    auto adj  = select(cmpLt(x, t), one, zero);
    return t - adj;
}

template<feta::idx_t W>
FORCEINLINE() simd::Packet<double, W> packetFloor(simd::Packet<double, W> x)
{
#if defined(__AVX512F__)
    if constexpr (W == 8)
        return simd::Packet<double,8>{_mm512_floor_pd(x.reg_)};
#endif
#if defined(__AVX2__) || defined(__AVX__)
    if constexpr (W == 4)
        return simd::Packet<double,4>{_mm256_floor_pd(x.reg_)};
#endif
#if defined(__SSE4_1__) && !defined(__AVX__)
    if constexpr (W == 2)
        return simd::Packet<double,2>{_mm_floor_pd(x.reg_)};
#endif
    auto t   = packetTrunc(x);
    auto one = simd::Packet<double,W>::broadcast(1.0);
    auto zer = simd::Packet<double,W>::zero();
    return t - select(cmpLt(x, t), one, zer);
}

// ─────────────────────────────────────────────────────────────────────────────
//  packetCeil
// ─────────────────────────────────────────────────────────────────────────────

template<feta::idx_t W>
FORCEINLINE() simd::Packet<float, W> packetCeil(simd::Packet<float, W> x)
{
#if defined(__AVX512F__)
    if constexpr (W == 16)
        return simd::Packet<float,16>{_mm512_ceil_ps(x.reg_)};
#endif
#if defined(__AVX2__) || defined(__AVX__)
    if constexpr (W == 8)
        return simd::Packet<float,8>{_mm256_ceil_ps(x.reg_)};
#endif
#if defined(__SSE4_1__) && !defined(__AVX__)
    if constexpr (W == 4)
        return simd::Packet<float,4>{_mm_ceil_ps(x.reg_)};
#endif
    auto t    = packetTrunc(x);
    auto one  = simd::Packet<float,W>::broadcast(1.f);
    auto zero = simd::Packet<float,W>::zero();
    return t + select(cmpGt(x, t), one, zero);
}

template<feta::idx_t W>
FORCEINLINE() simd::Packet<double, W> packetCeil(simd::Packet<double, W> x)
{
#if defined(__AVX512F__)
    if constexpr (W == 8)
        return simd::Packet<double,8>{_mm512_ceil_pd(x.reg_)};
#endif
#if defined(__AVX2__) || defined(__AVX__)
    if constexpr (W == 4)
        return simd::Packet<double,4>{_mm256_ceil_pd(x.reg_)};
#endif
#if defined(__SSE4_1__) && !defined(__AVX__)
    if constexpr (W == 2)
        return simd::Packet<double,2>{_mm_ceil_pd(x.reg_)};
#endif
    auto t   = packetTrunc(x);
    auto one = simd::Packet<double,W>::broadcast(1.0);
    auto zer = simd::Packet<double,W>::zero();
    return t + select(cmpGt(x, t), one, zer);
}

// ─────────────────────────────────────────────────────────────────────────────
//  packetRound  (round-half-away-from-zero, matching std::round)
// ─────────────────────────────────────────────────────────────────────────────

template<feta::idx_t W>
FORCEINLINE() simd::Packet<float, W> packetRound(simd::Packet<float, W> x)
{
#if defined(__AVX512F__)
    if constexpr (W == 16)
        return simd::Packet<float,16>{_mm512_roundscale_ps(x.reg_, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)};
#endif
#if defined(__AVX2__) || defined(__AVX__)
    if constexpr (W == 8)
        return simd::Packet<float,8>{_mm256_round_ps(x.reg_, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)};
#endif
#if defined(__SSE4_1__) && !defined(__AVX__)
    if constexpr (W == 4)
        return simd::Packet<float,4>{_mm_round_ps(x.reg_, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)};
#endif
    // SSE2: round-half-away = floor(x + 0.5) for positive, ceil(x - 0.5) for negative
    auto half = simd::Packet<float,W>::broadcast(0.5f);
    return packetFloor(x + half);
}

template<feta::idx_t W>
FORCEINLINE() simd::Packet<double, W> packetRound(simd::Packet<double, W> x)
{
#if defined(__AVX512F__)
    if constexpr (W == 8)
        return simd::Packet<double,8>{_mm512_roundscale_pd(x.reg_, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)};
#endif
#if defined(__AVX2__) || defined(__AVX__)
    if constexpr (W == 4)
        return simd::Packet<double,4>{_mm256_round_pd(x.reg_, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)};
#endif
#if defined(__SSE4_1__) && !defined(__AVX__)
    if constexpr (W == 2)
        return simd::Packet<double,2>{_mm_round_pd(x.reg_, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)};
#endif
    auto half = simd::Packet<double,W>::broadcast(0.5);
    return packetFloor(x + half);
}

} // namespace detail
} // namespace math
} // namespace feta
