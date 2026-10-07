#pragma once

/**
 * @file PacketBitOps.h
 * @brief Integer-level reinterpretation, shift, and conversion operations on
 *        SIMD Packet types.
 *
 * Transcendental function implementations (exp, log, sin, cos, …) require
 * direct manipulation of the IEEE-754 bit representation — exponent
 * extraction, mantissa masking, int↔float conversion for range reduction,
 * etc.  These operations are not exposed by `Packet.h` because they are not
 * needed for ordinary vector arithmetic.
 *
 * This header defines:
 *   - `IntPacket<W>` and `LongPacket<W>` — 32- and 64-bit integer SIMD lanes
 *   - `castToInt` / `castToFloat`  — bit-level float ↔ int32 reinterpret
 *   - `castToLong` / `castToDouble` — bit-level double ↔ int64 reinterpret
 *   - `shiftRightArith` / `shiftLeft` — per-lane integer shifts
 *   - `bitwiseAnd` / `bitwiseOr` / `bitwiseAndNot`
 *   - `cvtToFloat` / `cvtToInt` — value-converting int↔float (truncating)
 *   - `cvtToDouble` / `cvtToLong` — value-converting int64↔double (AVX-512)
 *   - `addInt` — integer lane addition
 *
 * All operations are specialized for every ISA width that `Packet.h` covers:
 * AVX-512, AVX2, SSE2, and scalar (Width==1).
 *
 * Usage:
 * @code
 *   // Extract biased exponent of a float packet
 *   auto ibits = castToInt(x);                        // reinterpret bits
 *   auto exp   = shiftRightArith(ibits, 23);          // shift right 23
 *   auto biased = bitwiseAnd(exp, IntPacket<W>{127}); // mask lower 8 bits
 * @endcode
 */

#include "feta/core/simd/Packet.h"
#include "feta/core/simd/PacketTraits.h"

#include <cstdint>
#include <cstring>  // memcpy for bit reinterpretation (nvcc lacks std::bit_cast)

namespace feta {
namespace math {
namespace detail {

// ─────────────────────────────────────────────────────────────────────────────
//  IntPacket<W>  — width-W lanes of int32_t
//  LongPacket<W> — width-W lanes of int64_t
// ─────────────────────────────────────────────────────────────────────────────

template<feta::idx_t W>
struct IntPacket;

template<feta::idx_t W>
struct LongPacket;

// Primary templates for broadcast — specialized per ISA below
template<feta::idx_t W> inline IntPacket<W>  broadcastInt (int32_t v);
template<feta::idx_t W> inline LongPacket<W> broadcastLong(int64_t v);

// ═══════════════════════════════════════════════════════════════════════════
//  Scalar Width == 1
// ═══════════════════════════════════════════════════════════════════════════

template<>
struct IntPacket<1> {
    int32_t val;
    explicit IntPacket(int32_t v) : val(v) {}
    IntPacket() = default;
};

template<>
struct LongPacket<1> {
    int64_t val;
    explicit LongPacket(int64_t v) : val(v) {}
    LongPacket() = default;
};

// --- scalar bit ops ---

inline IntPacket<1> castToInt(simd::Packet<float,1> p) {
    int32_t r; std::memcpy(&r, &p.val_, 4); return IntPacket<1>{r};
}
inline simd::Packet<float,1> castToFloat(IntPacket<1> i) {
    float r; std::memcpy(&r, &i.val, 4); return simd::Packet<float,1>{r};
}
inline LongPacket<1> castToLong(simd::Packet<double,1> p) {
    int64_t r; std::memcpy(&r, &p.val_, 8); return LongPacket<1>{r};
}
inline simd::Packet<double,1> castToDouble(LongPacket<1> l) {
    double r; std::memcpy(&r, &l.val, 8); return simd::Packet<double,1>{r};
}

inline IntPacket<1>  shiftRightArith(IntPacket<1>  a, int n) { return IntPacket<1> {a.val >> n}; }
inline IntPacket<1>  shiftLeft      (IntPacket<1>  a, int n) { return IntPacket<1> {a.val << n}; }
inline LongPacket<1> shiftRightArith(LongPacket<1> a, int n) { return LongPacket<1>{a.val >> n}; }
inline LongPacket<1> shiftLeft      (LongPacket<1> a, int n) { return LongPacket<1>{a.val << n}; }

inline IntPacket<1>  bitwiseAnd   (IntPacket<1>  a, IntPacket<1>  b) { return IntPacket<1> {a.val & b.val}; }
inline IntPacket<1>  bitwiseOr    (IntPacket<1>  a, IntPacket<1>  b) { return IntPacket<1> {a.val | b.val}; }
inline IntPacket<1>  bitwiseAndNot(IntPacket<1>  a, IntPacket<1>  b) { return IntPacket<1> {(~a.val) & b.val}; }
inline LongPacket<1> bitwiseAnd   (LongPacket<1> a, LongPacket<1> b) { return LongPacket<1>{a.val & b.val}; }
inline LongPacket<1> bitwiseOr    (LongPacket<1> a, LongPacket<1> b) { return LongPacket<1>{a.val | b.val}; }

inline IntPacket<1>  addInt(IntPacket<1>  a, IntPacket<1>  b) { return IntPacket<1> {a.val + b.val}; }
inline LongPacket<1> addInt(LongPacket<1> a, LongPacket<1> b) { return LongPacket<1>{a.val + b.val}; }

inline simd::Packet<float, 1>  cvtToFloat (IntPacket<1>  a) { return simd::Packet<float, 1>{static_cast<float>(a.val)}; }
inline IntPacket<1>             cvtToInt   (simd::Packet<float, 1> p) { return IntPacket<1>{static_cast<int32_t>(p.val_)}; }
inline simd::Packet<double,1>  cvtToDouble(LongPacket<1>  a) { return simd::Packet<double,1>{static_cast<double>(a.val)}; }
inline LongPacket<1>            cvtToLong  (simd::Packet<double,1> p) { return LongPacket<1>{static_cast<int64_t>(p.val_)}; }

// Broadcast helpers
template<> inline IntPacket<1>  broadcastInt<1> (int32_t v) { return IntPacket<1>{v}; }
template<> inline LongPacket<1> broadcastLong<1>(int64_t v) { return LongPacket<1>{v}; }

// ═══════════════════════════════════════════════════════════════════════════
//  AVX-512  (float×16, double×8)
// ═══════════════════════════════════════════════════════════════════════════

#if defined(__AVX512F__)

template<>
struct IntPacket<16> {
    __m512i reg;
    explicit IntPacket(__m512i r) : reg(r) {}
    IntPacket() = default;
};

template<>
struct LongPacket<8> {
    __m512i reg;
    explicit LongPacket(__m512i r) : reg(r) {}
    LongPacket() = default;
};

// float×16 ↔ int×16
inline IntPacket<16>              castToInt  (simd::Packet<float,16>  p) { return IntPacket<16> {_mm512_castps_si512(p.reg_)}; }
inline simd::Packet<float,16>     castToFloat(IntPacket<16>           i) { return simd::Packet<float,16>{_mm512_castsi512_ps(i.reg)}; }
// double×8 ↔ int64×8
inline LongPacket<8>              castToLong  (simd::Packet<double,8> p) { return LongPacket<8>{_mm512_castpd_si512(p.reg_)}; }
inline simd::Packet<double,8>     castToDouble(LongPacket<8>          l) { return simd::Packet<double,8>{_mm512_castsi512_pd(l.reg)}; }

inline IntPacket<16>  shiftRightArith(IntPacket<16>  a, int n) { return IntPacket<16> {_mm512_srai_epi32(a.reg, n)}; }
inline IntPacket<16>  shiftLeft      (IntPacket<16>  a, int n) { return IntPacket<16> {_mm512_slli_epi32(a.reg, n)}; }
inline LongPacket<8>  shiftRightArith(LongPacket<8>  a, int n) { return LongPacket<8> {_mm512_srai_epi64(a.reg, n)}; }
inline LongPacket<8>  shiftLeft      (LongPacket<8>  a, int n) { return LongPacket<8> {_mm512_slli_epi64(a.reg, n)}; }

inline IntPacket<16>  bitwiseAnd   (IntPacket<16>  a, IntPacket<16>  b) { return IntPacket<16> {_mm512_and_si512(a.reg, b.reg)}; }
inline IntPacket<16>  bitwiseOr    (IntPacket<16>  a, IntPacket<16>  b) { return IntPacket<16> {_mm512_or_si512 (a.reg, b.reg)}; }
inline IntPacket<16>  bitwiseAndNot(IntPacket<16>  a, IntPacket<16>  b) { return IntPacket<16> {_mm512_andnot_si512(a.reg, b.reg)}; }
inline LongPacket<8>  bitwiseAnd   (LongPacket<8>  a, LongPacket<8>  b) { return LongPacket<8> {_mm512_and_si512(a.reg, b.reg)}; }
inline LongPacket<8>  bitwiseOr    (LongPacket<8>  a, LongPacket<8>  b) { return LongPacket<8> {_mm512_or_si512 (a.reg, b.reg)}; }

inline IntPacket<16>  addInt(IntPacket<16>  a, IntPacket<16>  b) { return IntPacket<16> {_mm512_add_epi32(a.reg, b.reg)}; }
inline LongPacket<8>  addInt(LongPacket<8>  a, LongPacket<8>  b) { return LongPacket<8> {_mm512_add_epi64(a.reg, b.reg)}; }

inline simd::Packet<float,16>   cvtToFloat (IntPacket<16>  a) { return simd::Packet<float,16>{_mm512_cvtepi32_ps(a.reg)}; }
inline IntPacket<16>             cvtToInt   (simd::Packet<float,16>  p) { return IntPacket<16>{_mm512_cvttps_epi32(p.reg_)}; }
inline simd::Packet<double,8>   cvtToDouble(LongPacket<8>   a) { return simd::Packet<double,8>{_mm512_cvtepi64_pd(a.reg)}; }
inline LongPacket<8>             cvtToLong  (simd::Packet<double,8>  p) { return LongPacket<8>{_mm512_cvttpd_epi64(p.reg_)}; }

template<> inline IntPacket<16>  broadcastInt<16> (int32_t v) { return IntPacket<16>{_mm512_set1_epi32(v)}; }
template<> inline LongPacket<8>  broadcastLong<8> (int64_t v) { return LongPacket<8>{_mm512_set1_epi64(v)}; }

#endif // __AVX512F__

// ═══════════════════════════════════════════════════════════════════════════
//  AVX2  (float×8, double×4)
// ═══════════════════════════════════════════════════════════════════════════

#if defined(__AVX2__) || defined(__AVX__)

template<>
struct IntPacket<8> {
    __m256i reg;
    explicit IntPacket(__m256i r) : reg(r) {}
    IntPacket() = default;
};

template<>
struct LongPacket<4> {
    __m256i reg;
    explicit LongPacket(__m256i r) : reg(r) {}
    LongPacket() = default;
};

// float×8 ↔ int×8
inline IntPacket<8>            castToInt  (simd::Packet<float,8>  p) { return IntPacket<8> {_mm256_castps_si256(p.reg_)}; }
inline simd::Packet<float,8>   castToFloat(IntPacket<8>           i) { return simd::Packet<float,8>{_mm256_castsi256_ps(i.reg)}; }
// double×4 ↔ int64×4
inline LongPacket<4>            castToLong  (simd::Packet<double,4> p) { return LongPacket<4>{_mm256_castpd_si256(p.reg_)}; }
inline simd::Packet<double,4>   castToDouble(LongPacket<4>          l) { return simd::Packet<double,4>{_mm256_castsi256_pd(l.reg)}; }

inline IntPacket<8>  shiftRightArith(IntPacket<8>  a, int n) { return IntPacket<8> {_mm256_srai_epi32(a.reg, n)}; }
inline IntPacket<8>  shiftLeft      (IntPacket<8>  a, int n) { return IntPacket<8> {_mm256_slli_epi32(a.reg, n)}; }
// AVX2 has no 64-bit shift — use per-lane workaround via two 128-bit halves
inline LongPacket<4> shiftRightArith(LongPacket<4> a, int n)
{
    // Arithmetic shift on int64: sign-extend using sign bit propagation
    __m128i lo = _mm256_extracti128_si256(a.reg, 0);
    __m128i hi = _mm256_extracti128_si256(a.reg, 1);
    // Shift each 64-bit lane using two 32-bit lanes trick
    __m128i lo_s = _mm_or_si128(_mm_srli_epi64(lo, n),
        _mm_and_si128(_mm_srai_epi32(lo, 31), _mm_set1_epi64x(-(1LL << (64 - n)))));
    __m128i hi_s = _mm_or_si128(_mm_srli_epi64(hi, n),
        _mm_and_si128(_mm_srai_epi32(hi, 31), _mm_set1_epi64x(-(1LL << (64 - n)))));
    return LongPacket<4>{_mm256_set_m128i(hi_s, lo_s)};
}
inline LongPacket<4> shiftLeft(LongPacket<4> a, int n) { return LongPacket<4>{_mm256_slli_epi64(a.reg, n)}; }

inline IntPacket<8>  bitwiseAnd   (IntPacket<8>  a, IntPacket<8>  b) { return IntPacket<8> {_mm256_and_si256(a.reg, b.reg)}; }
inline IntPacket<8>  bitwiseOr    (IntPacket<8>  a, IntPacket<8>  b) { return IntPacket<8> {_mm256_or_si256 (a.reg, b.reg)}; }
inline IntPacket<8>  bitwiseAndNot(IntPacket<8>  a, IntPacket<8>  b) { return IntPacket<8> {_mm256_andnot_si256(a.reg, b.reg)}; }
inline LongPacket<4> bitwiseAnd   (LongPacket<4> a, LongPacket<4> b) { return LongPacket<4>{_mm256_and_si256(a.reg, b.reg)}; }
inline LongPacket<4> bitwiseOr    (LongPacket<4> a, LongPacket<4> b) { return LongPacket<4>{_mm256_or_si256 (a.reg, b.reg)}; }

inline IntPacket<8>  addInt(IntPacket<8>  a, IntPacket<8>  b) { return IntPacket<8> {_mm256_add_epi32(a.reg, b.reg)}; }
inline LongPacket<4> addInt(LongPacket<4> a, LongPacket<4> b) { return LongPacket<4>{_mm256_add_epi64(a.reg, b.reg)}; }

inline simd::Packet<float,8>  cvtToFloat(IntPacket<8>  a) { return simd::Packet<float,8>{_mm256_cvtepi32_ps(a.reg)}; }
inline IntPacket<8>            cvtToInt  (simd::Packet<float,8>  p) { return IntPacket<8>{_mm256_cvttps_epi32(p.reg_)}; }

// AVX2 has no direct cvt int64→double; use per-lane scalar extraction
inline simd::Packet<double,4> cvtToDouble(LongPacket<4> a)
{
    alignas(32) int64_t tmp[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), a.reg);
    return simd::Packet<double,4>{_mm256_set_pd(
        static_cast<double>(tmp[3]), static_cast<double>(tmp[2]),
        static_cast<double>(tmp[1]), static_cast<double>(tmp[0]))};
}
inline LongPacket<4> cvtToLong(simd::Packet<double,4> p)
{
    alignas(32) double tmp[4];
    _mm256_store_pd(tmp, p.reg_);
    alignas(32) int64_t out[4] = {
        static_cast<int64_t>(tmp[0]), static_cast<int64_t>(tmp[1]),
        static_cast<int64_t>(tmp[2]), static_cast<int64_t>(tmp[3])};
    return LongPacket<4>{_mm256_load_si256(reinterpret_cast<const __m256i*>(out))};
}

template<> inline IntPacket<8>  broadcastInt<8> (int32_t v) { return IntPacket<8> {_mm256_set1_epi32(v)}; }
template<> inline LongPacket<4> broadcastLong<4>(int64_t v) { return LongPacket<4>{_mm256_set1_epi64x(v)}; }

#endif // __AVX2__ || __AVX__

// ═══════════════════════════════════════════════════════════════════════════
//  SSE2  (float×4, double×2)
// ═══════════════════════════════════════════════════════════════════════════

#if defined(__SSE2__) && !defined(__AVX2__) && !defined(__AVX__)

template<>
struct IntPacket<4> {
    __m128i reg;
    explicit IntPacket(__m128i r) : reg(r) {}
    IntPacket() = default;
};

template<>
struct LongPacket<2> {
    __m128i reg;
    explicit LongPacket(__m128i r) : reg(r) {}
    LongPacket() = default;
};

inline IntPacket<4>            castToInt  (simd::Packet<float,4>  p) { return IntPacket<4> {_mm_castps_si128(p.reg_)}; }
inline simd::Packet<float,4>   castToFloat(IntPacket<4>           i) { return simd::Packet<float,4>{_mm_castsi128_ps(i.reg)}; }
inline LongPacket<2>            castToLong  (simd::Packet<double,2> p) { return LongPacket<2>{_mm_castpd_si128(p.reg_)}; }
inline simd::Packet<double,2>   castToDouble(LongPacket<2>          l) { return simd::Packet<double,2>{_mm_castsi128_pd(l.reg)}; }

inline IntPacket<4>  shiftRightArith(IntPacket<4>  a, int n) { return IntPacket<4> {_mm_srai_epi32(a.reg, n)}; }
inline IntPacket<4>  shiftLeft      (IntPacket<4>  a, int n) { return IntPacket<4> {_mm_slli_epi32(a.reg, n)}; }
inline LongPacket<2> shiftLeft      (LongPacket<2> a, int n) { return LongPacket<2>{_mm_slli_epi64(a.reg, n)}; }
inline LongPacket<2> shiftRightArith(LongPacket<2> a, int n)
{
    // SSE2 lacks 64-bit arithmetic shift; emulate
    __m128i sign = _mm_srai_epi32(a.reg, 31);       // propagate sign from high 32 bits
    sign = _mm_shuffle_epi32(sign, _MM_SHUFFLE(3,3,1,1)); // broadcast sign bit to paired lane
    __m128i shifted = _mm_srli_epi64(a.reg, n);
    // blend in sign bits for bits above n
    __m128i mask = _mm_set1_epi64x(n >= 64 ? -1LL : (-(1LL << (64 - n))));
    return LongPacket<2>{_mm_or_si128(shifted, _mm_and_si128(sign, mask))};
}

inline IntPacket<4>  bitwiseAnd   (IntPacket<4>  a, IntPacket<4>  b) { return IntPacket<4> {_mm_and_si128(a.reg, b.reg)}; }
inline IntPacket<4>  bitwiseOr    (IntPacket<4>  a, IntPacket<4>  b) { return IntPacket<4> {_mm_or_si128 (a.reg, b.reg)}; }
inline IntPacket<4>  bitwiseAndNot(IntPacket<4>  a, IntPacket<4>  b) { return IntPacket<4> {_mm_andnot_si128(a.reg, b.reg)}; }
inline LongPacket<2> bitwiseAnd   (LongPacket<2> a, LongPacket<2> b) { return LongPacket<2>{_mm_and_si128(a.reg, b.reg)}; }
inline LongPacket<2> bitwiseOr    (LongPacket<2> a, LongPacket<2> b) { return LongPacket<2>{_mm_or_si128 (a.reg, b.reg)}; }

inline IntPacket<4>  addInt(IntPacket<4>  a, IntPacket<4>  b) { return IntPacket<4> {_mm_add_epi32(a.reg, b.reg)}; }
inline LongPacket<2> addInt(LongPacket<2> a, LongPacket<2> b) { return LongPacket<2>{_mm_add_epi64(a.reg, b.reg)}; }

inline simd::Packet<float,4>  cvtToFloat(IntPacket<4>  a) { return simd::Packet<float,4>{_mm_cvtepi32_ps(a.reg)}; }
inline IntPacket<4>            cvtToInt  (simd::Packet<float,4>  p) { return IntPacket<4>{_mm_cvttps_epi32(p.reg_)}; }

// SSE2: scalar int64↔double conversions
inline simd::Packet<double,2> cvtToDouble(LongPacket<2> a)
{
    alignas(16) int64_t tmp[2];
    _mm_store_si128(reinterpret_cast<__m128i*>(tmp), a.reg);
    return simd::Packet<double,2>{_mm_set_pd(static_cast<double>(tmp[1]), static_cast<double>(tmp[0]))};
}
inline LongPacket<2> cvtToLong(simd::Packet<double,2> p)
{
    alignas(16) double tmp[2];
    _mm_store_pd(tmp, p.reg_);
    alignas(16) int64_t out[2] = {static_cast<int64_t>(tmp[0]), static_cast<int64_t>(tmp[1])};
    return LongPacket<2>{_mm_load_si128(reinterpret_cast<const __m128i*>(out))};
}

template<> inline IntPacket<4>  broadcastInt<4> (int32_t v) { return IntPacket<4> {_mm_set1_epi32(v)}; }
template<> inline LongPacket<2> broadcastLong<2>(int64_t v) { return LongPacket<2>{_mm_set1_epi64x(v)}; }

#endif // SSE2 only

// ─────────────────────────────────────────────────────────────────────────────
//  Convenience: infer IntPacket / LongPacket width from Packet<DataT, W>
// ─────────────────────────────────────────────────────────────────────────────

// For float packets → IntPacket of same width
template<feta::idx_t W>
using FloatIntPacket = IntPacket<W>;

// For double packets → LongPacket of same width
template<feta::idx_t W>
using DoubleIntPacket = LongPacket<W>;

} // namespace detail
} // namespace math
} // namespace feta
