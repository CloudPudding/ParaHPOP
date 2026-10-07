/**
 * @file Packet.h
 *
 * SIMD packet type wrapping platform intrinsics for CPU vectorisation.
 *
 * `Packet<DataT, Width>` represents `Width` lanes of `DataT` processed
 * in a single SIMD instruction.  Full specialisations are provided for:
 *
 *   | DataT  | Width | Register     | ISA        |
 *   |--------|-------|------------- |------------|
 *   | double | 8     | `__m512d`    | AVX-512    |
 *   | float  | 16    | `__m512`     | AVX-512    |
 *   | double | 4     | `__m256d`    | AVX2       |
 *   | float  | 8     | `__m256`     | AVX2       |
 *   | double | 2     | `__m128d`    | SSE2       |
 *   | float  | 4     | `__m128`     | SSE2       |
 *   | any    | 1     | plain scalar | (fallback) |
 *
 * Every specialisation provides: `load`, `store`, `maskLoad`, `maskStore`,
 * `broadcast`, and arithmetic operators `+`, `-`, `*`, `/`, unary `-`,
 * plus `abs`, `sqrt`, `max`, `min`, `fmadd`, `fmsub`, `fnmadd`,
 * `select`, `copysign`, `pow`, and comparison-to-mask (`cmpGe`, `cmpLt`).
 *
 * The Width==1 specialisation compiles down to plain scalar operations
 * with zero overhead (no SIMD headers, no register moves).
 */
#pragma once

#include "feta/core/simd/PacketMask.h"
#include "feta/core/simd/PacketTraits.h"

#include <cmath>

namespace feta {
namespace simd {

// ═════════════════════════════════════════════════════════════════════════════
//  Primary template — should never be instantiated directly; only
//  the specialisations below are used.
// ═════════════════════════════════════════════════════════════════════════════

template<typename DataT, idx_t Width>
struct Packet;

// ═════════════════════════════════════════════════════════════════════════════
//  Width == 1  (scalar fallback — zero overhead)
// ═════════════════════════════════════════════════════════════════════════════

template<typename DataT>
struct Packet<DataT, 1> {
    using MaskT = PacketMask<DataT, 1>;
    static constexpr idx_t width = 1;

    DataT val_;

    Packet() = default;
    explicit Packet(DataT v) : val_(v) {}

    static Packet broadcast(DataT v) { return Packet{v}; }
    static Packet zero()             { return Packet{DataT{0}}; }

    static Packet load(const DataT* p)                     { return Packet{*p}; }
    static void   store(DataT* p, Packet a)                { *p = a.val_; }
    static Packet maskLoad(const DataT* p, MaskT m)        { return m.mask_ ? Packet{*p} : zero(); }
    static void   maskStore(DataT* p, MaskT m, Packet a)   { if (m.mask_) *p = a.val_; }

    friend Packet operator+(Packet a, Packet b) { return Packet{a.val_ + b.val_}; }
    friend Packet operator-(Packet a, Packet b) { return Packet{a.val_ - b.val_}; }
    friend Packet operator*(Packet a, Packet b) { return Packet{a.val_ * b.val_}; }
    friend Packet operator/(Packet a, Packet b) { return Packet{a.val_ / b.val_}; }
    Packet operator-() const { return Packet{-val_}; }

    friend Packet abs(Packet a)                    { return Packet{std::abs(a.val_)}; }
    friend Packet sqrt(Packet a)                   { return Packet{std::sqrt(a.val_)}; }
    friend Packet max(Packet a, Packet b)          { return Packet{std::max(a.val_, b.val_)}; }
    friend Packet min(Packet a, Packet b)          { return Packet{std::min(a.val_, b.val_)}; }
    friend Packet fmadd(Packet a, Packet b, Packet c) { return Packet{a.val_ * b.val_ + c.val_}; }
    friend Packet fmsub(Packet a, Packet b, Packet c) { return Packet{a.val_ * b.val_ - c.val_}; }
    friend Packet fnmadd(Packet a, Packet b, Packet c) { return Packet{-(a.val_ * b.val_) + c.val_}; }

    friend Packet select(MaskT m, Packet a, Packet b) { return m.mask_ ? a : b; }
    friend Packet copysign(Packet mag, Packet sgn) { return Packet{std::copysign(mag.val_, sgn.val_)}; }
    friend Packet pow(Packet a, Packet b) { return Packet{std::pow(a.val_, b.val_)}; }

    friend MaskT cmpGe(Packet a, Packet b) { return MaskT{a.val_ >= b.val_}; }
    friend MaskT cmpLt(Packet a, Packet b) { return MaskT{a.val_ < b.val_}; }
    friend MaskT cmpGt(Packet a, Packet b) { return MaskT{a.val_ > b.val_}; }

    friend bool isFinite(Packet a) { return std::isfinite(a.val_); }
};

// ═════════════════════════════════════════════════════════════════════════════
//  AVX-512  double×8
// ═════════════════════════════════════════════════════════════════════════════

#if defined(__AVX512F__)

template<>
struct Packet<double, 8> {
    using MaskT = PacketMask<double, 8>;
    static constexpr idx_t width = 8;

    __m512d reg_;

    Packet() = default;
    explicit Packet(__m512d r) : reg_(r) {}

    static Packet broadcast(double v) { return Packet{_mm512_set1_pd(v)}; }
    static Packet zero()              { return Packet{_mm512_setzero_pd()}; }

    static Packet load(const double* p)                    { return Packet{_mm512_loadu_pd(p)}; }
    static void   store(double* p, Packet a)               { _mm512_storeu_pd(p, a.reg_); }
    static Packet maskLoad(const double* p, MaskT m)       { return Packet{_mm512_maskz_loadu_pd(m.mask_, p)}; }
    static void   maskStore(double* p, MaskT m, Packet a)  { _mm512_mask_storeu_pd(p, m.mask_, a.reg_); }

    friend Packet operator+(Packet a, Packet b) { return Packet{_mm512_add_pd(a.reg_, b.reg_)}; }
    friend Packet operator-(Packet a, Packet b) { return Packet{_mm512_sub_pd(a.reg_, b.reg_)}; }
    friend Packet operator*(Packet a, Packet b) { return Packet{_mm512_mul_pd(a.reg_, b.reg_)}; }
    friend Packet operator/(Packet a, Packet b) { return Packet{_mm512_div_pd(a.reg_, b.reg_)}; }
    Packet operator-() const { return Packet{_mm512_sub_pd(_mm512_setzero_pd(), reg_)}; }

    friend Packet abs(Packet a)
    {
        return Packet{_mm512_abs_pd(a.reg_)};
    }
    friend Packet sqrt(Packet a)                   { return Packet{_mm512_sqrt_pd(a.reg_)}; }
    friend Packet max(Packet a, Packet b)          { return Packet{_mm512_max_pd(a.reg_, b.reg_)}; }
    friend Packet min(Packet a, Packet b)          { return Packet{_mm512_min_pd(a.reg_, b.reg_)}; }
    friend Packet fmadd(Packet a, Packet b, Packet c) { return Packet{_mm512_fmadd_pd(a.reg_, b.reg_, c.reg_)}; }
    friend Packet fmsub(Packet a, Packet b, Packet c) { return Packet{_mm512_fmsub_pd(a.reg_, b.reg_, c.reg_)}; }
    friend Packet fnmadd(Packet a, Packet b, Packet c) { return Packet{_mm512_fnmadd_pd(a.reg_, b.reg_, c.reg_)}; }

    friend Packet select(MaskT m, Packet a, Packet b)
    {
        return Packet{_mm512_mask_blend_pd(m.mask_, b.reg_, a.reg_)};
    }
    friend Packet copysign(Packet mag, Packet sgn)
    {
        // Extract sign from sgn, magnitude from mag
        __m512i signBit = _mm512_set1_epi64(static_cast<int64_t>(0x8000000000000000ULL));
        __m512d sign = _mm512_castsi512_pd(_mm512_and_si512(
            _mm512_castpd_si512(sgn.reg_), signBit));
        __m512d absMag = _mm512_abs_pd(mag.reg_);
        return Packet{_mm512_castsi512_pd(_mm512_or_si512(
            _mm512_castpd_si512(absMag), _mm512_castpd_si512(sign)))};
    }
    friend Packet pow(Packet a, Packet b)
    {
        // No SIMD pow; per-lane fallback
        alignas(64) double va[8], vb[8], vr[8];
        _mm512_store_pd(va, a.reg_);
        _mm512_store_pd(vb, b.reg_);
        for (int k = 0; k < 8; ++k) vr[k] = std::pow(va[k], vb[k]);
        return Packet{_mm512_load_pd(vr)};
    }

    friend MaskT cmpGe(Packet a, Packet b)
    {
        return MaskT{_mm512_cmp_pd_mask(a.reg_, b.reg_, _CMP_GE_OQ)};
    }
    friend MaskT cmpLt(Packet a, Packet b)
    {
        return MaskT{_mm512_cmp_pd_mask(a.reg_, b.reg_, _CMP_LT_OQ)};
    }
    friend MaskT cmpGt(Packet a, Packet b)
    {
        return MaskT{_mm512_cmp_pd_mask(a.reg_, b.reg_, _CMP_GT_OQ)};
    }
};

template<>
struct Packet<float, 16> {
    using MaskT = PacketMask<float, 16>;
    static constexpr idx_t width = 16;

    __m512 reg_;

    Packet() = default;
    explicit Packet(__m512 r) : reg_(r) {}

    static Packet broadcast(float v)  { return Packet{_mm512_set1_ps(v)}; }
    static Packet zero()              { return Packet{_mm512_setzero_ps()}; }

    static Packet load(const float* p)                    { return Packet{_mm512_loadu_ps(p)}; }
    static void   store(float* p, Packet a)               { _mm512_storeu_ps(p, a.reg_); }
    static Packet maskLoad(const float* p, MaskT m)       { return Packet{_mm512_maskz_loadu_ps(m.mask_, p)}; }
    static void   maskStore(float* p, MaskT m, Packet a)  { _mm512_mask_storeu_ps(p, m.mask_, a.reg_); }

    friend Packet operator+(Packet a, Packet b) { return Packet{_mm512_add_ps(a.reg_, b.reg_)}; }
    friend Packet operator-(Packet a, Packet b) { return Packet{_mm512_sub_ps(a.reg_, b.reg_)}; }
    friend Packet operator*(Packet a, Packet b) { return Packet{_mm512_mul_ps(a.reg_, b.reg_)}; }
    friend Packet operator/(Packet a, Packet b) { return Packet{_mm512_div_ps(a.reg_, b.reg_)}; }
    Packet operator-() const { return Packet{_mm512_sub_ps(_mm512_setzero_ps(), reg_)}; }

    friend Packet abs(Packet a)
    {
        return Packet{_mm512_abs_ps(a.reg_)};
    }
    friend Packet sqrt(Packet a)                   { return Packet{_mm512_sqrt_ps(a.reg_)}; }
    friend Packet max(Packet a, Packet b)          { return Packet{_mm512_max_ps(a.reg_, b.reg_)}; }
    friend Packet min(Packet a, Packet b)          { return Packet{_mm512_min_ps(a.reg_, b.reg_)}; }
    friend Packet fmadd(Packet a, Packet b, Packet c) { return Packet{_mm512_fmadd_ps(a.reg_, b.reg_, c.reg_)}; }
    friend Packet fmsub(Packet a, Packet b, Packet c) { return Packet{_mm512_fmsub_ps(a.reg_, b.reg_, c.reg_)}; }
    friend Packet fnmadd(Packet a, Packet b, Packet c) { return Packet{_mm512_fnmadd_ps(a.reg_, b.reg_, c.reg_)}; }

    friend Packet select(MaskT m, Packet a, Packet b)
    {
        return Packet{_mm512_mask_blend_ps(m.mask_, b.reg_, a.reg_)};
    }
    friend Packet copysign(Packet mag, Packet sgn)
    {
        __m512i signBit = _mm512_set1_epi32(static_cast<int32_t>(0x80000000U));
        __m512 sign = _mm512_castsi512_ps(_mm512_and_si512(
            _mm512_castps_si512(sgn.reg_), signBit));
        __m512 absMag = _mm512_abs_ps(mag.reg_);
        return Packet{_mm512_castsi512_ps(_mm512_or_si512(
            _mm512_castps_si512(absMag), _mm512_castps_si512(sign)))};
    }
    friend Packet pow(Packet a, Packet b)
    {
        alignas(64) float va[16], vb[16], vr[16];
        _mm512_store_ps(va, a.reg_);
        _mm512_store_ps(vb, b.reg_);
        for (int k = 0; k < 16; ++k) vr[k] = std::pow(va[k], vb[k]);
        return Packet{_mm512_load_ps(vr)};
    }

    friend MaskT cmpGe(Packet a, Packet b)
    {
        return MaskT{_mm512_cmp_ps_mask(a.reg_, b.reg_, _CMP_GE_OQ)};
    }
    friend MaskT cmpLt(Packet a, Packet b)
    {
        return MaskT{_mm512_cmp_ps_mask(a.reg_, b.reg_, _CMP_LT_OQ)};
    }
    friend MaskT cmpGt(Packet a, Packet b)
    {
        return MaskT{_mm512_cmp_ps_mask(a.reg_, b.reg_, _CMP_GT_OQ)};
    }
};

#endif // __AVX512F__

// ═════════════════════════════════════════════════════════════════════════════
//  AVX2  double×4, float×8
// ═════════════════════════════════════════════════════════════════════════════

#if defined(__AVX2__) || defined(__AVX__)

namespace detail {
/** @brief Convert a comparison-result __m256d to a mask storage type. */
template<typename MaskT>
inline typename MaskT::StorageT avx2MaskFromCmp256d(__m256d cmp)
{
    return static_cast<typename MaskT::StorageT>(_mm256_movemask_pd(cmp));
}
/** @brief Convert a comparison-result __m256 to a mask storage type. */
template<typename MaskT>
inline typename MaskT::StorageT avx2MaskFromCmp256(__m256 cmp)
{
    return static_cast<typename MaskT::StorageT>(_mm256_movemask_ps(cmp));
}
/** @brief Build an AVX2 blend mask (__m256d) from a PacketMask bitmask. */
inline __m256d avx2BlendMask256d(unsigned mask)
{
    alignas(32) int64_t bits[4];
    for (idx_t k = 0; k < 4; ++k)
        bits[k] = (mask >> k) & 1u ? int64_t(-1) : int64_t(0);
    return _mm256_castsi256_pd(
        _mm256_load_si256(reinterpret_cast<const __m256i*>(bits)));
}
/** @brief Build an AVX2 blend mask (__m256) from a PacketMask bitmask. */
inline __m256 avx2BlendMask256(unsigned mask)
{
    alignas(32) int32_t bits[8];
    for (idx_t k = 0; k < 8; ++k)
        bits[k] = (mask >> k) & 1u ? int32_t(-1) : int32_t(0);
    return _mm256_castsi256_ps(
        _mm256_load_si256(reinterpret_cast<const __m256i*>(bits)));
}
} // namespace detail

template<>
struct Packet<double, 4> {
    using MaskT = PacketMask<double, 4>;
    static constexpr idx_t width = 4;

    __m256d reg_;

    Packet() = default;
    explicit Packet(__m256d r) : reg_(r) {}

    static Packet broadcast(double v) { return Packet{_mm256_set1_pd(v)}; }
    static Packet zero()              { return Packet{_mm256_setzero_pd()}; }

    static Packet load(const double* p)  { return Packet{_mm256_loadu_pd(p)}; }
    static void   store(double* p, Packet a) { _mm256_storeu_pd(p, a.reg_); }

    static Packet maskLoad(const double* p, MaskT m)
    {
        alignas(32) int64_t bits[4];
        for (idx_t k = 0; k < 4; ++k)
            bits[k] = m.lane(k) ? int64_t(-1) : int64_t(0);
        __m256i imask = _mm256_load_si256(reinterpret_cast<const __m256i*>(bits));
        return Packet{_mm256_maskload_pd(p, imask)};
    }

    static void maskStore(double* p, MaskT m, Packet a)
    {
        alignas(32) int64_t bits[4];
        for (idx_t k = 0; k < 4; ++k)
            bits[k] = m.lane(k) ? int64_t(-1) : int64_t(0);
        __m256i imask = _mm256_load_si256(reinterpret_cast<const __m256i*>(bits));
        _mm256_maskstore_pd(p, imask, a.reg_);
    }

    friend Packet operator+(Packet a, Packet b) { return Packet{_mm256_add_pd(a.reg_, b.reg_)}; }
    friend Packet operator-(Packet a, Packet b) { return Packet{_mm256_sub_pd(a.reg_, b.reg_)}; }
    friend Packet operator*(Packet a, Packet b) { return Packet{_mm256_mul_pd(a.reg_, b.reg_)}; }
    friend Packet operator/(Packet a, Packet b) { return Packet{_mm256_div_pd(a.reg_, b.reg_)}; }
    Packet operator-() const { return Packet{_mm256_sub_pd(_mm256_setzero_pd(), reg_)}; }

    friend Packet abs(Packet a)
    {
        __m256d signMask = _mm256_castsi256_pd(
            _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
        return Packet{_mm256_and_pd(a.reg_, signMask)};
    }
    friend Packet sqrt(Packet a)                   { return Packet{_mm256_sqrt_pd(a.reg_)}; }
    friend Packet max(Packet a, Packet b)          { return Packet{_mm256_max_pd(a.reg_, b.reg_)}; }
    friend Packet min(Packet a, Packet b)          { return Packet{_mm256_min_pd(a.reg_, b.reg_)}; }
#if defined(__FMA__)
    friend Packet fmadd(Packet a, Packet b, Packet c) { return Packet{_mm256_fmadd_pd(a.reg_, b.reg_, c.reg_)}; }
    friend Packet fmsub(Packet a, Packet b, Packet c) { return Packet{_mm256_fmsub_pd(a.reg_, b.reg_, c.reg_)}; }
    friend Packet fnmadd(Packet a, Packet b, Packet c) { return Packet{_mm256_fnmadd_pd(a.reg_, b.reg_, c.reg_)}; }
#else
    friend Packet fmadd(Packet a, Packet b, Packet c) { return a * b + c; }
    friend Packet fmsub(Packet a, Packet b, Packet c) { return a * b - c; }
    friend Packet fnmadd(Packet a, Packet b, Packet c) { return c - a * b; }
#endif

    friend Packet select(MaskT m, Packet a, Packet b)
    {
        __m256d bmask = detail::avx2BlendMask256d(m.mask_);
        return Packet{_mm256_blendv_pd(b.reg_, a.reg_, bmask)};
    }
    friend Packet copysign(Packet mag, Packet sgn)
    {
        __m256d signBit = _mm256_castsi256_pd(
            _mm256_set1_epi64x(static_cast<int64_t>(0x8000000000000000ULL)));
        __m256d sign = _mm256_and_pd(sgn.reg_, signBit);
        __m256d absMask = _mm256_castsi256_pd(
            _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
        __m256d absMag = _mm256_and_pd(mag.reg_, absMask);
        return Packet{_mm256_or_pd(absMag, sign)};
    }
    friend Packet pow(Packet a, Packet b)
    {
        alignas(32) double va[4], vb[4], vr[4];
        _mm256_store_pd(va, a.reg_);
        _mm256_store_pd(vb, b.reg_);
        for (int k = 0; k < 4; ++k) vr[k] = std::pow(va[k], vb[k]);
        return Packet{_mm256_load_pd(vr)};
    }

    friend MaskT cmpGe(Packet a, Packet b)
    {
        return MaskT{detail::avx2MaskFromCmp256d<MaskT>(
            _mm256_cmp_pd(a.reg_, b.reg_, _CMP_GE_OQ))};
    }
    friend MaskT cmpLt(Packet a, Packet b)
    {
        return MaskT{detail::avx2MaskFromCmp256d<MaskT>(
            _mm256_cmp_pd(a.reg_, b.reg_, _CMP_LT_OQ))};
    }
    friend MaskT cmpGt(Packet a, Packet b)
    {
        return MaskT{detail::avx2MaskFromCmp256d<MaskT>(
            _mm256_cmp_pd(a.reg_, b.reg_, _CMP_GT_OQ))};
    }
};

template<>
struct Packet<float, 8> {
    using MaskT = PacketMask<float, 8>;
    static constexpr idx_t width = 8;

    __m256 reg_;

    Packet() = default;
    explicit Packet(__m256 r) : reg_(r) {}

    static Packet broadcast(float v) { return Packet{_mm256_set1_ps(v)}; }
    static Packet zero()             { return Packet{_mm256_setzero_ps()}; }

    static Packet load(const float* p)  { return Packet{_mm256_loadu_ps(p)}; }
    static void   store(float* p, Packet a) { _mm256_storeu_ps(p, a.reg_); }

    static Packet maskLoad(const float* p, MaskT m)
    {
        alignas(32) int32_t bits[8];
        for (idx_t k = 0; k < 8; ++k)
            bits[k] = m.lane(k) ? int32_t(-1) : int32_t(0);
        __m256i imask = _mm256_load_si256(reinterpret_cast<const __m256i*>(bits));
        return Packet{_mm256_maskload_ps(p, imask)};
    }

    static void maskStore(float* p, MaskT m, Packet a)
    {
        alignas(32) int32_t bits[8];
        for (idx_t k = 0; k < 8; ++k)
            bits[k] = m.lane(k) ? int32_t(-1) : int32_t(0);
        __m256i imask = _mm256_load_si256(reinterpret_cast<const __m256i*>(bits));
        _mm256_maskstore_ps(p, imask, a.reg_);
    }

    friend Packet operator+(Packet a, Packet b) { return Packet{_mm256_add_ps(a.reg_, b.reg_)}; }
    friend Packet operator-(Packet a, Packet b) { return Packet{_mm256_sub_ps(a.reg_, b.reg_)}; }
    friend Packet operator*(Packet a, Packet b) { return Packet{_mm256_mul_ps(a.reg_, b.reg_)}; }
    friend Packet operator/(Packet a, Packet b) { return Packet{_mm256_div_ps(a.reg_, b.reg_)}; }
    Packet operator-() const { return Packet{_mm256_sub_ps(_mm256_setzero_ps(), reg_)}; }

    friend Packet abs(Packet a)
    {
        __m256 signMask = _mm256_castsi256_ps(
            _mm256_set1_epi32(0x7FFFFFFF));
        return Packet{_mm256_and_ps(a.reg_, signMask)};
    }
    friend Packet sqrt(Packet a)                   { return Packet{_mm256_sqrt_ps(a.reg_)}; }
    friend Packet max(Packet a, Packet b)          { return Packet{_mm256_max_ps(a.reg_, b.reg_)}; }
    friend Packet min(Packet a, Packet b)          { return Packet{_mm256_min_ps(a.reg_, b.reg_)}; }
#if defined(__FMA__)
    friend Packet fmadd(Packet a, Packet b, Packet c) { return Packet{_mm256_fmadd_ps(a.reg_, b.reg_, c.reg_)}; }
    friend Packet fmsub(Packet a, Packet b, Packet c) { return Packet{_mm256_fmsub_ps(a.reg_, b.reg_, c.reg_)}; }
    friend Packet fnmadd(Packet a, Packet b, Packet c) { return Packet{_mm256_fnmadd_ps(a.reg_, b.reg_, c.reg_)}; }
#else
    friend Packet fmadd(Packet a, Packet b, Packet c) { return a * b + c; }
    friend Packet fmsub(Packet a, Packet b, Packet c) { return a * b - c; }
    friend Packet fnmadd(Packet a, Packet b, Packet c) { return c - a * b; }
#endif

    friend Packet select(MaskT m, Packet a, Packet b)
    {
        __m256 bmask = detail::avx2BlendMask256(m.mask_);
        return Packet{_mm256_blendv_ps(b.reg_, a.reg_, bmask)};
    }
    friend Packet copysign(Packet mag, Packet sgn)
    {
        __m256 signBit = _mm256_castsi256_ps(
            _mm256_set1_epi32(static_cast<int32_t>(0x80000000U)));
        __m256 sign = _mm256_and_ps(sgn.reg_, signBit);
        __m256 absMask = _mm256_castsi256_ps(
            _mm256_set1_epi32(0x7FFFFFFF));
        __m256 absMag = _mm256_and_ps(mag.reg_, absMask);
        return Packet{_mm256_or_ps(absMag, sign)};
    }
    friend Packet pow(Packet a, Packet b)
    {
        alignas(32) float va[8], vb[8], vr[8];
        _mm256_store_ps(va, a.reg_);
        _mm256_store_ps(vb, b.reg_);
        for (int k = 0; k < 8; ++k) vr[k] = std::pow(va[k], vb[k]);
        return Packet{_mm256_load_ps(vr)};
    }

    friend MaskT cmpGe(Packet a, Packet b)
    {
        return MaskT{detail::avx2MaskFromCmp256<MaskT>(
            _mm256_cmp_ps(a.reg_, b.reg_, _CMP_GE_OQ))};
    }
    friend MaskT cmpLt(Packet a, Packet b)
    {
        return MaskT{detail::avx2MaskFromCmp256<MaskT>(
            _mm256_cmp_ps(a.reg_, b.reg_, _CMP_LT_OQ))};
    }
    friend MaskT cmpGt(Packet a, Packet b)
    {
        return MaskT{detail::avx2MaskFromCmp256<MaskT>(
            _mm256_cmp_ps(a.reg_, b.reg_, _CMP_GT_OQ))};
    }
};

#endif // __AVX2__ || __AVX__

// ═════════════════════════════════════════════════════════════════════════════
//  SSE2  double×2, float×4
// ═════════════════════════════════════════════════════════════════════════════

#if defined(__SSE2__) && !defined(__AVX2__) && !defined(__AVX__)

namespace detail {
inline unsigned sse2MaskFromCmp128d(__m128d cmp)
{
    return static_cast<unsigned>(_mm_movemask_pd(cmp));
}
inline unsigned sse2MaskFromCmp128(__m128 cmp)
{
    return static_cast<unsigned>(_mm_movemask_ps(cmp));
}
} // namespace detail

template<>
struct Packet<double, 2> {
    using MaskT = PacketMask<double, 2>;
    static constexpr idx_t width = 2;

    __m128d reg_;

    Packet() = default;
    explicit Packet(__m128d r) : reg_(r) {}

    static Packet broadcast(double v) { return Packet{_mm_set1_pd(v)}; }
    static Packet zero()              { return Packet{_mm_setzero_pd()}; }

    static Packet load(const double* p)  { return Packet{_mm_loadu_pd(p)}; }
    static void   store(double* p, Packet a) { _mm_storeu_pd(p, a.reg_); }

    static Packet maskLoad(const double* p, MaskT m)
    {
        alignas(16) double tmp[2] = {0.0, 0.0};
        if (m.lane(0)) tmp[0] = p[0];
        if (m.lane(1)) tmp[1] = p[1];
        return Packet{_mm_load_pd(tmp)};
    }

    static void maskStore(double* p, MaskT m, Packet a)
    {
        alignas(16) double tmp[2];
        _mm_store_pd(tmp, a.reg_);
        if (m.lane(0)) p[0] = tmp[0];
        if (m.lane(1)) p[1] = tmp[1];
    }

    friend Packet operator+(Packet a, Packet b) { return Packet{_mm_add_pd(a.reg_, b.reg_)}; }
    friend Packet operator-(Packet a, Packet b) { return Packet{_mm_sub_pd(a.reg_, b.reg_)}; }
    friend Packet operator*(Packet a, Packet b) { return Packet{_mm_mul_pd(a.reg_, b.reg_)}; }
    friend Packet operator/(Packet a, Packet b) { return Packet{_mm_div_pd(a.reg_, b.reg_)}; }
    Packet operator-() const { return Packet{_mm_sub_pd(_mm_setzero_pd(), reg_)}; }

    friend Packet abs(Packet a)
    {
        __m128d signMask = _mm_castsi128_pd(
            _mm_set1_epi64x(0x7FFFFFFFFFFFFFFF));
        return Packet{_mm_and_pd(a.reg_, signMask)};
    }
    friend Packet sqrt(Packet a)          { return Packet{_mm_sqrt_pd(a.reg_)}; }
    friend Packet max(Packet a, Packet b) { return Packet{_mm_max_pd(a.reg_, b.reg_)}; }
    friend Packet min(Packet a, Packet b) { return Packet{_mm_min_pd(a.reg_, b.reg_)}; }
    friend Packet fmadd(Packet a, Packet b, Packet c) { return a * b + c; }
    friend Packet fmsub(Packet a, Packet b, Packet c) { return a * b - c; }
    friend Packet fnmadd(Packet a, Packet b, Packet c) { return c - a * b; }

    friend Packet select(MaskT m, Packet a, Packet b)
    {
        // SSE2: per-lane scalar select
        alignas(16) double va[2], vb[2], vr[2];
        _mm_store_pd(va, a.reg_);
        _mm_store_pd(vb, b.reg_);
        vr[0] = m.lane(0) ? va[0] : vb[0];
        vr[1] = m.lane(1) ? va[1] : vb[1];
        return Packet{_mm_load_pd(vr)};
    }
    friend Packet copysign(Packet mag, Packet sgn)
    {
        __m128d signBit = _mm_castsi128_pd(
            _mm_set1_epi64x(static_cast<int64_t>(0x8000000000000000ULL)));
        __m128d sign = _mm_and_pd(sgn.reg_, signBit);
        __m128d absMask = _mm_castsi128_pd(
            _mm_set1_epi64x(0x7FFFFFFFFFFFFFFF));
        __m128d absMag = _mm_and_pd(mag.reg_, absMask);
        return Packet{_mm_or_pd(absMag, sign)};
    }
    friend Packet pow(Packet a, Packet b)
    {
        alignas(16) double va[2], vb[2], vr[2];
        _mm_store_pd(va, a.reg_);
        _mm_store_pd(vb, b.reg_);
        for (int k = 0; k < 2; ++k) vr[k] = std::pow(va[k], vb[k]);
        return Packet{_mm_load_pd(vr)};
    }

    friend MaskT cmpGe(Packet a, Packet b)
    {
        return MaskT{detail::sse2MaskFromCmp128d(
            _mm_cmpge_pd(a.reg_, b.reg_))};
    }
    friend MaskT cmpLt(Packet a, Packet b)
    {
        return MaskT{detail::sse2MaskFromCmp128d(
            _mm_cmplt_pd(a.reg_, b.reg_))};
    }
    friend MaskT cmpGt(Packet a, Packet b)
    {
        return MaskT{detail::sse2MaskFromCmp128d(
            _mm_cmpgt_pd(a.reg_, b.reg_))};
    }
};

template<>
struct Packet<float, 4> {
    using MaskT = PacketMask<float, 4>;
    static constexpr idx_t width = 4;

    __m128 reg_;

    Packet() = default;
    explicit Packet(__m128 r) : reg_(r) {}

    static Packet broadcast(float v) { return Packet{_mm_set1_ps(v)}; }
    static Packet zero()             { return Packet{_mm_setzero_ps()}; }

    static Packet load(const float* p)  { return Packet{_mm_loadu_ps(p)}; }
    static void   store(float* p, Packet a) { _mm_storeu_ps(p, a.reg_); }

    static Packet maskLoad(const float* p, MaskT m)
    {
        alignas(16) float tmp[4] = {0.f, 0.f, 0.f, 0.f};
        for (idx_t k = 0; k < 4; ++k)
            if (m.lane(k)) tmp[k] = p[k];
        return Packet{_mm_load_ps(tmp)};
    }

    static void maskStore(float* p, MaskT m, Packet a)
    {
        alignas(16) float tmp[4];
        _mm_store_ps(tmp, a.reg_);
        for (idx_t k = 0; k < 4; ++k)
            if (m.lane(k)) p[k] = tmp[k];
    }

    friend Packet operator+(Packet a, Packet b) { return Packet{_mm_add_ps(a.reg_, b.reg_)}; }
    friend Packet operator-(Packet a, Packet b) { return Packet{_mm_sub_ps(a.reg_, b.reg_)}; }
    friend Packet operator*(Packet a, Packet b) { return Packet{_mm_mul_ps(a.reg_, b.reg_)}; }
    friend Packet operator/(Packet a, Packet b) { return Packet{_mm_div_ps(a.reg_, b.reg_)}; }
    Packet operator-() const { return Packet{_mm_sub_ps(_mm_setzero_ps(), reg_)}; }

    friend Packet abs(Packet a)
    {
        __m128 signMask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
        return Packet{_mm_and_ps(a.reg_, signMask)};
    }
    friend Packet sqrt(Packet a)          { return Packet{_mm_sqrt_ps(a.reg_)}; }
    friend Packet max(Packet a, Packet b) { return Packet{_mm_max_ps(a.reg_, b.reg_)}; }
    friend Packet min(Packet a, Packet b) { return Packet{_mm_min_ps(a.reg_, b.reg_)}; }
    friend Packet fmadd(Packet a, Packet b, Packet c) { return a * b + c; }
    friend Packet fmsub(Packet a, Packet b, Packet c) { return a * b - c; }
    friend Packet fnmadd(Packet a, Packet b, Packet c) { return c - a * b; }

    friend Packet select(MaskT m, Packet a, Packet b)
    {
        alignas(16) float va[4], vb[4], vr[4];
        _mm_store_ps(va, a.reg_);
        _mm_store_ps(vb, b.reg_);
        for (idx_t k = 0; k < 4; ++k)
            vr[k] = m.lane(k) ? va[k] : vb[k];
        return Packet{_mm_load_ps(vr)};
    }
    friend Packet copysign(Packet mag, Packet sgn)
    {
        __m128 signBit = _mm_castsi128_ps(
            _mm_set1_epi32(static_cast<int32_t>(0x80000000U)));
        __m128 sign = _mm_and_ps(sgn.reg_, signBit);
        __m128 absMask = _mm_castsi128_ps(
            _mm_set1_epi32(0x7FFFFFFF));
        __m128 absMag = _mm_and_ps(mag.reg_, absMask);
        return Packet{_mm_or_ps(absMag, sign)};
    }
    friend Packet pow(Packet a, Packet b)
    {
        alignas(16) float va[4], vb[4], vr[4];
        _mm_store_ps(va, a.reg_);
        _mm_store_ps(vb, b.reg_);
        for (int k = 0; k < 4; ++k) vr[k] = std::pow(va[k], vb[k]);
        return Packet{_mm_load_ps(vr)};
    }

    friend MaskT cmpGe(Packet a, Packet b)
    {
        return MaskT{detail::sse2MaskFromCmp128(
            _mm_cmpge_ps(a.reg_, b.reg_))};
    }
    friend MaskT cmpLt(Packet a, Packet b)
    {
        return MaskT{detail::sse2MaskFromCmp128(
            _mm_cmplt_ps(a.reg_, b.reg_))};
    }
    friend MaskT cmpGt(Packet a, Packet b)
    {
        return MaskT{detail::sse2MaskFromCmp128(
            _mm_cmpgt_ps(a.reg_, b.reg_))};
    }
};

#endif // __SSE2__ only

// ═════════════════════════════════════════════════════════════════════════════
//  Compound assignment operators
//
//  Defined as non-member templates dispatching to the immutable
//  ``operator+`` etc. on the matching Packet specialisation, so all
//  backends pick them up uniformly without duplicating per-spec
//  bodies.  Lets generic code (e.g. ``parm/integrate/rk/host/Step.h``'s
//  RK accumulators) use ``packet += other`` symmetric with the scalar
//  ``Real += Real`` path.
// ═════════════════════════════════════════════════════════════════════════════

template<typename DataT, idx_t W>
inline Packet<DataT, W>& operator+=(
    Packet<DataT, W>& a, const Packet<DataT, W>& b)
{
    a = a + b;
    return a;
}

template<typename DataT, idx_t W>
inline Packet<DataT, W>& operator-=(
    Packet<DataT, W>& a, const Packet<DataT, W>& b)
{
    a = a - b;
    return a;
}

template<typename DataT, idx_t W>
inline Packet<DataT, W>& operator*=(
    Packet<DataT, W>& a, const Packet<DataT, W>& b)
{
    a = a * b;
    return a;
}

template<typename DataT, idx_t W>
inline Packet<DataT, W>& operator/=(
    Packet<DataT, W>& a, const Packet<DataT, W>& b)
{
    a = a / b;
    return a;
}

// ═════════════════════════════════════════════════════════════════════════════
//  Convenience alias: the "native" packet for a given scalar type
// ═════════════════════════════════════════════════════════════════════════════

template<typename DataT>
using NativePacket = Packet<DataT, ::feta::simd::PreferredWidth<DataT> >;

template<typename DataT>
using NativeMask = PacketMask<DataT, ::feta::simd::PreferredWidth<DataT> >;

// ═════════════════════════════════════════════════════════════════════════════
//  Readable comparison aliases
// ═════════════════════════════════════════════════════════════════════════════

template<typename DataT, idx_t W>
inline PacketMask<DataT, W> greaterOrEqual(Packet<DataT, W> a, Packet<DataT, W> b) { return cmpGe(a, b); }

template<typename DataT, idx_t W>
inline PacketMask<DataT, W> lessThan(Packet<DataT, W> a, Packet<DataT, W> b) { return cmpLt(a, b); }

template<typename DataT, idx_t W>
inline PacketMask<DataT, W> greaterThan(Packet<DataT, W> a, Packet<DataT, W> b) { return cmpGt(a, b); }

} // namespace simd
} // namespace feta
