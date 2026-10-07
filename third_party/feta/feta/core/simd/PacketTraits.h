/**
 * @file PacketTraits.h
 *
 * Compile-time SIMD ISA detection and packet width selection.
 *
 * Detects the widest available SIMD instruction set at compile time
 * and exposes `PreferredWidth<DataT>` — the number of elements that
 * fit in a single SIMD register for a given scalar type.
 */
#pragma once

#include "feta/typedefs.h"

// ISA detection headers (only on CPU builds)
#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace feta {
namespace simd {

// ─── ISA level enum ─────────────────────────────────────────────────────────

enum class ISA : unsigned {
    Scalar  = 0,
    SSE2    = 1,
    AVX2    = 2,
    AVX512  = 3
};

/** @brief The highest ISA level available at compile time. */
inline constexpr ISA DetectedISA =
#if defined(__AVX512F__)
    ISA::AVX512;
#elif defined(__AVX2__)
    ISA::AVX2;
#elif defined(__SSE2__)
    ISA::SSE2;
#else
    ISA::Scalar;
#endif

// ─── Register width in bytes ────────────────────────────────────────────────

inline constexpr idx_t RegisterBytes =
    (DetectedISA == ISA::AVX512) ? 64 :
    (DetectedISA == ISA::AVX2)   ? 32 :
    (DetectedISA == ISA::SSE2)   ? 16 : 0;

// ─── Preferred width per scalar type ────────────────────────────────────────

/**
 * @brief Number of elements of type `DataT` that fit in one SIMD register.
 *
 * Falls back to 1 (scalar) when no SIMD is available or the type is
 * not a supported floating-point type.
 */
template<typename DataT>
inline constexpr idx_t PreferredWidth =
    (RegisterBytes > 0 && (std::is_same_v<DataT, double> ||
                           std::is_same_v<DataT, float>))
    ? static_cast<idx_t>(RegisterBytes / sizeof(DataT))
    : 1;

// ─── PacketTraits struct ────────────────────────────────────────────────────

/**
 * @brief Compile-time traits for SIMD packets of a given scalar type.
 *
 * @tparam DataT  Scalar element type (float or double).
 */
template<typename DataT>
struct PacketTraits {
    static constexpr idx_t Width = PreferredWidth<DataT>;
    static constexpr ISA   Level = DetectedISA;

    /** @brief True when SIMD is actually active (Width > 1). */
    static constexpr bool HasSIMD = (Width > 1);
};

} // namespace simd
} // namespace feta
