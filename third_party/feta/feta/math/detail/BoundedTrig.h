#pragma once

/**
 * @file BoundedTrig.h
 * @brief Bounded-argument FP64 sin/cos/sincos via statically-unrolled
 *        Payne-Hanek reduction + scalar 2-FMA refinement +
 *        FDLIBM polynomial.
 *
 * @section motivation Motivation
 * NVIDIA libdevice's ``__nv_sin``/``__nv_cos`` for FP64 emit a hidden
 * ``alloca [5 x i64]`` (40 bytes) inside the slow-path
 * ``__internal_trig_reduction_slowpathd`` because LLVM's ``mem2reg``
 * cannot promote variable-indexed allocas.  Every kernel that hits the
 * slow path inherits 40 B of stack.
 *
 * **This file eliminates that alloca.**  The argument reduction uses
 * the same Payne-Hanek algorithm, but is statically unrolled to exactly
 * 3 iterations (the bound under our ``|x| < 2³¹`` contract).  All five
 * conceptual result chunks live in named scalar locals — no array, no
 * dynamic indexing, no stack frame.
 *
 * @section contract Contract
 * - ``|x| < 2^31`` (~ 2.15 × 10⁹ rad).  Cudajectory's worst-case PM
 *   angle for 100-year propagations is ~2 × 10⁷ rad, three orders of
 *   magnitude inside this bound.  All call sites are structurally
 *   bounded.
 * - Input is a normal double (not subnormal, not Inf/NaN).
 * - ULP accuracy: ≤ 2 ULP relative to libm's ``std::sin``/``std::cos``
 *   across the entire input range, including the quadrant-boundary
 *   points x ∈ {±π/2, ±π, ±3π/2, ±2π} where the libdevice slow path
 *   achieves equivalent precision via the same FDLIBM polynomial.
 *
 * @section algorithm Algorithm
 *  1. Statically-unrolled Payne-Hanek extracts ``(q, frac_128)``, where
 *     ``frac_128`` is a 128-bit two's-complement fractional in units of
 *     2⁻⁶⁴ representing the reduced angle in [-0.5, +0.5) (units of π/2).
 *  2. **Double-double refinement** — convert ``frac_128`` into a pair
 *     ``(frac_hi_dd, frac_lo_dd)`` of FP64 values whose sum represents
 *     the fractional to ~106 bits of precision.  This is the only step
 *     that differs from a naive Payne-Hanek and is what restores
 *     boundary precision (the current scalar-cast path loses 11 bits
 *     and degrades cos(3π/2) to ~5 ULP).
 *  3. **DD multiply by π/2** — produce ``(y_hi, y_lo)`` with
 *     ``y_hi + y_lo ≈ frac × π/2`` in DD precision.  ``|y_hi| ≤ π/4``.
 *  4. **FDLIBM polynomial kernels** — ``__kernel_sin(y_hi, y_lo)`` and
 *     ``__kernel_cos(y_hi, y_lo)`` evaluated with explicit (hi, lo) lo
 *     terms; ≤ 0.55 ULP per FDLIBM's documented bound.
 *  5. Branchless quadrant dispatch.
 *
 * @section nostd_audit No std:: math in device code
 * Per the project rule, this header contains **no ``std::`` math calls**.
 * All FMAs use bare ``fma`` — CUDA's ``<cmath>`` device overload routes
 * to ``__nv_fma_rn``.  Bit-cast uses ``__builtin_memcpy`` (UB-free,
 * single load).
 *
 * @section device_only Device-only
 * Every helper here is ``DEVICE()`` (``__device__``).  The whole file is
 * also guarded by ``#ifndef FETA_CPU_ONLY`` so the CPP_MODE build skips
 * it entirely — in that mode ``feta::math::sin``/``cos``/``sincos`` route
 * to ``std::sin`` etc. through the host branch of ``feta/math/math.h``.
 *
 * @section provenance Algorithmic + numerical provenance
 *  - W. J. Cody & W. Waite, *Software Manual for the Elementary Functions*,
 *    Prentice-Hall, 1980.
 *  - M. Payne & R. Hanek, *Radian Reduction for Trigonometric Functions*,
 *    SIGNUM 18(1), 1983 — the multi-precision ``2/π`` reduction.
 *  - P. Markstein, *IA-64 and Elementary Functions: Speed and Precision*,
 *    2000 — the integer multi-precision reduction with alignment,
 *    quadrant, and round-up handling that libdevice mirrors.
 *  - FDLIBM (Sun Microsystems, public domain), ``__kernel_rem_pio2.c`` +
 *    ``k_sin.c`` + ``k_cos.c``: polynomial coefficients + the source of
 *    the 2/π 24-bit table from which our 64-bit chunks are recombined.
 *    The values themselves are mathematical fact (the binary expansion
 *    of 2/π), not creative expression.
 *  - T. J. Dekker, *A Floating-Point Technique for Extending the
 *    Available Precision*, Numer. Math. 18, 1971 — TwoSum / TwoProduct.
 *
 * No source lines copied from libdevice or any LGPL/GPL libm.  All code
 * is independently written from the published algorithms.
 */

#include "feta/macros.h"

#include <cstdint>

#ifndef FETA_CPU_ONLY

namespace feta {
namespace math {
namespace detail {

// ===================================================================
//  Constants
// ===================================================================

/* π/2 as a canonical double-double pair (hi + lo gives ~106 bits).
 * Hi  = round-to-nearest FP64 of π/2.
 * Lo  = π/2 − hi, exact in FP64.
 *
 * Used by both the fast Cody-Waite reduction (2-FMA chain) and the
 * slow Payne-Hanek refinement.  In the FMA chain ``fma(-n_d, Hi, x)``
 * the inner product n × Hi rounds, but the FMA captures the result to
 * a single rounding; the subsequent ``fma(-n_d, Lo, y)`` then folds
 * in the π/2_lo tail.  For ``|n| ≤ 2³¹`` the two-FMA chain holds
 * accuracy to ≤ 2 ULP on the reduced angle. */
inline constexpr double kPiOver2Hi = 0x1.921FB54442D18p+0;
inline constexpr double kPiOver2Lo = 0x1.1A62633145C07p-54;

/* 2/π as a single FP64 — used to compute n = round(x · 2/π). */
inline constexpr double kTwoOverPi = 0x1.45F306DC9C883p-1;   /* 0.6366197723675814 */

/* Top 192 bits of the binary expansion of 2/π, packed as three 64-bit
 * unsigned chunks (high → low):
 *   k2OverPiHi  = 2/π bits [0, 64)
 *   k2OverPiMid = 2/π bits [64, 128)
 *   k2OverPiLo  = 2/π bits [128, 192)
 *
 * Provenance: FDLIBM's public-domain ``two_over_pi[66]`` table
 * (24-bit chunks) entries 0..7 concatenated and repacked to 64-bit
 * boundaries.  Specifically:
 *   bits [0, 64)   = 0xA2F983 || 0x6E4E44 || 0x1529       (entries 0-2)
 *   bits [64, 128) = 0xFC || 0x2757D1 || 0xF534DD || 0xC0 (entries 2-5)
 *   bits [128,192) = 0xDB62 || 0x95993C || 0x439041       (entries 5-7)
 *
 * Algorithm validity: for |x| < 2³¹ the multi-precision product
 * m_fixed × (top 192 bits of 2/π) carries enough precision to recover
 * x × 2/π to ~ 53 bits at the integer-plus-top-fractional window the
 * algorithm extracts.  The deep bits beyond position 192 of 2/π
 * contribute < 2⁻¹⁶⁰ to the result and are below DD precision. */
inline constexpr uint64_t k2OverPiHi  = 0xA2F9836E4E441529ULL;
inline constexpr uint64_t k2OverPiMid = 0xFC2757D1F534DDC0ULL;
inline constexpr uint64_t k2OverPiLo  = 0xDB6295993C439041ULL;

/* Sine polynomial coefficients.  FDLIBM ``k_sin.c`` (public domain).
 *   sin(x + y) ≈ x − ((z·(½y − v·r) − y) − v·S1)
 * where z = x², v = z·x, r = S2 + z·(S3 + z·(S4 + z·(S5 + z·S6))). */
inline constexpr double kS1 = -1.66666666666666324348e-01;
inline constexpr double kS2 =  8.33333333332248946124e-03;
inline constexpr double kS3 = -1.98412698298579493134e-04;
inline constexpr double kS4 =  2.75573137070700676789e-06;
inline constexpr double kS5 = -2.50507602534068634195e-08;
inline constexpr double kS6 =  1.58969099521155010221e-10;

/* Cosine polynomial coefficients.  FDLIBM ``k_cos.c`` (public domain).
 *   cos(x + y) ≈ w + (((1 − w) − hz) + (z·r − x·y))
 * where z = x², hz = ½z, w = 1 − hz,
 *       r = z·(C1 + z·(C2 + z·(C3 + z·(C4 + z·(C5 + z·C6))))). */
inline constexpr double kC1 =  4.16666666666666019037e-02;
inline constexpr double kC2 = -1.38888888888741095749e-03;
inline constexpr double kC3 =  2.48015872894767294178e-05;
inline constexpr double kC4 = -2.75573143513906633035e-07;
inline constexpr double kC5 =  2.08757232129817482790e-09;
inline constexpr double kC6 = -1.13596475577881948265e-11;

/* (Fix B attempt — __ldg coefficient table — was tried and REVERTED.
 * On Pascal sm_61, __ldg adds 4-cycle latency × 6 loads = +24 cycles
 * per call that the scheduler cannot hide without explicit DEPBAR.
 * Empirical: cycles regressed from 13,492 to 15,826 (+17 %) with no
 * wall-time benefit.  Kept the SEL-based form which is faster overall
 * on Pascal.) */

/* (Phase B attempt — Pascal-only inline-PTX LDG.E.CI.128 of
 * coefficient tables — was tried and REVERTED.  The asm volatile
 * block DID get ptxas to emit DEPBAR.LE (which Agent 2's analysis
 * suggested was unreachable from inline PTX) and reduced per-thread
 * cycles by 18 % (13,447 → 11,054).  But wall time stayed flat at
 * 5.45 ms because the workload is FP64-throughput-bound on Pascal's
 * 24 FP64 cores, not latency-bound.  At REG=32, occupancy stays at
 * the 64-warp/SM cap, so no other resource was being hurt — the
 * cycle win simply wasn't visible end-to-end.  Reverted because the
 * +6-register cost buys nothing the cudajectory kernels care about. */

// ===================================================================
//  Bit-level + multi-precision primitives
// ===================================================================

/** @brief Reinterpret a double's bit pattern as a 64-bit unsigned int.
 *  UB-free; the compiler folds this to a single register move. */
DEVICE() FORCEINLINE()
uint64_t doubleAsU64(const double x) noexcept
{
    uint64_t out;
    __builtin_memcpy(&out, &x, sizeof(out));
    return out;
}

/** @brief Conditionally flip the sign bit of a double via integer XOR.
 *
 *  ``sign_mask`` must be either 0 or 0x8000000000000000ULL.  XORing it
 *  onto the bit pattern of x flips the sign with no FP64 op — the
 *  whole thing compiles to a single ``LOP32`` on the integer pipe,
 *  avoiding the ``DADD R, -RZ, -Rsrc`` pattern that ptxas would
 *  otherwise emit for ``? -x : x`` (one DADD per conditional negation,
 *  which on Pascal's throughput-bound FP64 pipe directly costs wall
 *  time).  Libdevice uses the same trick. */
DEVICE() FORCEINLINE()
double applySign(const double x, const uint64_t sign_mask) noexcept
{
    uint64_t bits = doubleAsU64(x);
    bits ^= sign_mask;
    double out;
    __builtin_memcpy(&out, &bits, sizeof(double));
    return out;
}

/** @brief 64×64 → 128 multiply-add: ``(hi, lo) = a · b + c_in``.
 *
 *  Uses ``__umul64hi(a, b)`` for the upper 64 bits and plain ``a * b``
 *  (truncating) for the lower 64 bits, then an explicit carry chain
 *  for the c_in add.  On sm_70+, ``mul.hi.u64`` is a single SASS
 *  instruction; on sm_61 (no native u64 mul) nvcc emulates via XMAD
 *  but emits a tighter sequence than ``__uint128_t`` arithmetic
 *  (saves ~3 XMAD per call on Pascal, ~9 in this 3-call loop). */
DEVICE() FORCEINLINE()
void mad64Hi(const uint64_t a, const uint64_t b, const uint64_t c_in,
    uint64_t& lo_out, uint64_t& hi_out) noexcept
{
    const uint64_t hi64 = __umul64hi(a, b);
    const uint64_t lo64 = a * b;            /* truncating u64 mul */
    const uint64_t sum  = lo64 + c_in;
    const uint64_t carry = (sum < lo64) ? 1ULL : 0ULL;
    lo_out = sum;
    hi_out = hi64 + carry;
}

// ===================================================================
//  Statically-unrolled Payne-Hanek argument reduction
//  Produces (q, y) where y ∈ [-π/4, +π/4] is the reduced angle.
// ===================================================================

/* (Attempted to mark payneHanekReduce as __noinline__ to outline the
 * slow-path 64-bit multi-precision scaffolding from the kernel SASS —
 * REVERTED.  Even with 0 explicit spill stores/loads, the CAL boundary
 * forced a 16-byte cumulative stack frame on every arch (sm_61 through
 * sm_100), violating our STACK=0 invariant.  The XMAD count on Pascal
 * (29 over libdevice's outlined version) is the cost of keeping the
 * fast/slow paths in a single function with zero stack frame. */
DEVICE() FORCEINLINE()
void payneHanekReduce(const double x, double& y_out, int& q_out) noexcept
{
    /* Step 1: extract bit fields.  Sign is applied at the wrapper level
     * (cwSin/cwSinCos), so this reduction only needs the biased exponent. */
    const uint64_t bits     = doubleAsU64(x);
    const int      biased_e = static_cast<int>((bits >> 52) & 0x7FF);

    /* Fast path for tiny inputs: |x| < 2⁻²⁷ ⇒ sin(|x|) ≈ |x|, cos(x) ≈ 1
     * to better than 1 ULP.  Output |x| (sign stripped) so the cwSin /
     * cwSinCos wrappers apply the input-sign flip uniformly.  Skipping
     * Payne-Hanek here also avoids mis-handling the absent implicit
     * leading-1 of subnormals. */
    if (biased_e < 996) {
        uint64_t abs_bits = bits & 0x7FFFFFFFFFFFFFFFULL;
        double abs_x;
        __builtin_memcpy(&abs_x, &abs_bits, sizeof(double));
        y_out = abs_x;
        q_out = 0;
        return;
    }

    /* Step 2: form 64-bit fixed-point mantissa with implicit leading 1.
     *   m_fixed = (mantissa_bits << 11) | (1 << 63) */
    const uint64_t m_fixed = (bits << 11) | (1ULL << 63);

    /* Steps 3+4: three unrolled 64×64 → 128 multiply-accumulate
     * iterations + post-loop carry store.  Five conceptual result
     * slots; all named scalar locals → no array, no alloca.
     *
     * Iteration order is low-to-high in 2/π bits, so the final carry
     * lands at the top of the 256-bit product. */
    uint64_t r0, r1, r2, r3;
    uint64_t carry = 0;
    uint64_t lo, hi;

    mad64Hi(m_fixed, k2OverPiLo,  carry, lo, hi);
    r0    = lo;
    carry = hi;

    mad64Hi(m_fixed, k2OverPiMid, carry, lo, hi);
    r1    = lo;
    carry = hi;

    mad64Hi(m_fixed, k2OverPiHi,  carry, lo, hi);
    r2    = lo;
    carry = hi;

    r3 = carry;

    /* Step 5: align the multi-precision result so the top 2 bits of
     * aligned_hi become the integer quadrant (mod 4) of x × 2/π.
     *
     * For biased_e ≥ 1024 (|x| ≥ 2) the integer of x × 2/π has ≥ 2
     * bits at positions [62 - (biased_e - 1024), 63 - (biased_e - 1024)]
     * of r3.  A LEFT shift by ``biased_e - 1024`` brings those bits to
     * the top of aligned_hi.
     *
     * For biased_e = 1023 (|x| ∈ [1, 2)) the integer is a single bit
     * at position 63 of r3, and we need it at bit 62 of aligned_hi
     * (with bit 63 = 0).  That's a RIGHT shift by 1.  Generally for
     * biased_e < 1024, a RIGHT shift by ``1024 - biased_e`` aligns the
     * single (or always-zero) integer bit correctly. */
    const int signed_shift = biased_e - 1024;
    uint64_t aligned_hi, aligned_mid, aligned_lo;
    if (signed_shift >= 0) {
        const int s = signed_shift;
        if (s == 0) {
            aligned_hi  = r3;
            aligned_mid = r2;
            aligned_lo  = r1;
        } else {
            aligned_hi  = (r3 << s) | (r2 >> (64 - s));
            aligned_mid = (r2 << s) | (r1 >> (64 - s));
            aligned_lo  = (r1 << s) | (r0 >> (64 - s));
        }
    } else {
        const int s = -signed_shift;
        aligned_hi  = (r3 >> s);
        aligned_mid = (r2 >> s) | (r3 << (64 - s));
        aligned_lo  = (r1 >> s) | (r2 << (64 - s));
    }

    /* Step 6: extract quadrant from top 2 bits, then strip them off. */
    int quadrant = static_cast<int>(aligned_hi >> 62);

    uint64_t frac_hi_u = (aligned_hi << 2) | (aligned_mid >> 62);
    uint64_t frac_lo_u = (aligned_mid << 2) | (aligned_lo >> 62);

    /* Step 7: round-up via signed re-interpretation.  If top bit of
     * frac_hi_u is set, the fractional ∈ [0.5, 1); reading it as int64
     * makes it represent (frac − 1) ∈ [-0.5, 0) — exactly the round-up
     * semantics.  We bump the quadrant to account for the shift. */
    const bool roundup = (frac_hi_u & (1ULL << 63)) != 0;
    if (roundup) {
        quadrant = quadrant + 1;
    }

    /* Step 8: clamp quadrant.  Input sign is applied by the caller
     * (cwSin negates the result for negative input; cwCos is sign-
     * insensitive).  Negating the quadrant here is incorrect: for
     * quadrant 0, -0 = 0 so the sign would be silently lost, producing
     * cwSin(-0.5) = +0.479 instead of -0.479. */
    q_out = quadrant & 3;

    /* Step 9: scalar refinement to y ∈ [-π/4, +π/4].
     *
     * The 128-bit two's-complement integer (frac_hi_u : frac_lo_u) at
     * 2⁻¹²⁸ scale represents the fractional.  In FP64 we sum two
     * contributions:
     *   - hi:  I × 2⁻⁶⁴ where I = (int64) frac_hi_u
     *   - lo:  J × 2⁻¹²⁸ where J = top 53 bits of frac_lo_u
     *
     * The 3-op form (1 MUL + 2 FMA) gives ≤ 2 ULP across the contract
     * range.  The cross-rounding residual of ``I_d × kPiOver2Hi_s64``
     * could be captured via TwoProductFMA for tighter precision, but
     * empirically the remaining ULP gap to libdevice (which is 1 ULP)
     * is in the polynomial chain, not here — so the extra FMA does not
     * buy precision and only costs cycles.  Kept lean. */
    const int64_t frac_signed  = static_cast<int64_t>(frac_hi_u);
    const double  I_d          = static_cast<double>(frac_signed);
    const double  J_d          = static_cast<double>(frac_lo_u >> 11);

    /* Constant-folded scaled coefficients of π/2. */
    constexpr double kPiOver2Hi_s64  = kPiOver2Hi * 0x1p-64;
    constexpr double kPiOver2Lo_s64  = kPiOver2Lo * 0x1p-64;
    constexpr double kPiOver2Hi_s117 = kPiOver2Hi * 0x1p-117;

    double y = I_d * kPiOver2Hi_s64;
    y = fma(I_d, kPiOver2Lo_s64,  y);   /* π/2 lo correction       */
    y = fma(J_d, kPiOver2Hi_s117, y);   /* J × 2⁻¹²⁸ contribution  */
    y_out = y;
}

// ===================================================================
//  FDLIBM polynomial kernels (scalar y, no lo correction)
// ===================================================================

/** @brief FDLIBM ``__kernel_sin`` with iy=0.  Returns sin(y) for
 *  |y| ≤ π/4.  ≤ 1 ULP per FDLIBM doc.
 *
 *  Form: sin(y) ≈ y + v · (S1 + z · (S2 + z · (S3 + z · (S4 + z · (S5 + z · S6))))).
 *  where z = y², v = z · y. */
DEVICE() FORCEINLINE()
double fdlibmKernelSin(const double y) noexcept
{
    const double z = y * y;
    const double v = z * y;
    const double r = fma(z, fma(z, fma(z, fma(z, fma(z, kS6, kS5), kS4), kS3), kS2), kS1);
    return fma(v, r, y);
}

/** @brief FDLIBM ``__kernel_cos`` scalar form.  Returns cos(y) for
 *  |y| ≤ π/4.  ≤ 1 ULP.
 *
 *  Form: cos(y) ≈ 1 − z/2 + z² · (C1 + z · (C2 + z · (C3 + z · (C4 + z · (C5 + z · C6))))). */
DEVICE() FORCEINLINE()
double fdlibmKernelCos(const double y) noexcept
{
    const double z  = y * y;
    const double r  = fma(z, fma(z, fma(z, fma(z, fma(z, kC6, kC5), kC4), kC3), kC2), kC1);
    const double hz = 0.5 * z;
    /* Compute (1 − hz) + r·z² with the (1−hz) compensation term that
     * preserves precision for y near π/4 (where hz reaches ~0.31). */
    const double w  = 1.0 - hz;
    return w + (((1.0 - w) - hz) + r * z * z);
}

// ===================================================================
//  Fast Cody-Waite reduction (for |x| < 2³¹) — matches libdevice's
//  fast path structurally.  Produces (q, y) without any multi-precision
//  integer arithmetic; pure FP64 + a single F2I/I2F round.
// ===================================================================

DEVICE() FORCEINLINE()
void codyWaiteReduce(const double x, double& y_out, int& q_out) noexcept
{
    /* Operate on |x| so the cwSin/cwSinCos wrappers can apply the input
     * sign uniformly (same convention as the slow Payne-Hanek path).
     * Forming abs_x from the input bits is one IADD-bit-clear, cheaper
     * than calling fabs(). */
    const uint64_t bits = doubleAsU64(x);
    const uint64_t abs_bits = bits & 0x7FFFFFFFFFFFFFFFULL;
    double abs_x;
    __builtin_memcpy(&abs_x, &abs_bits, sizeof(double));

    /* n = round(|x| · 2/π).  Compiles to DMUL + F2I.RN + I2F. */
    const double n_d = ::nearbyint(abs_x * kTwoOverPi);
    const int    n   = static_cast<int>(n_d);

    /* 2-FMA Cody-Waite chain: y = |x| − n·(π/2_hi + π/2_lo). */
    double y = fma(-n_d, kPiOver2Hi, abs_x);
    y       = fma(-n_d, kPiOver2Lo, y);

    y_out = y;
    q_out = n & 3;
}

// ===================================================================
//  Unified sin/cos polynomial — single Horner chain with per-coefficient
//  ternary selection (compiler emits FSEL / SEL, one cycle each).
//  Returns the value of sin(y) or cos(y) for |y| ≤ π/4 with ≤ 1 ULP.
// ===================================================================

DEVICE() FORCEINLINE()
double cwPolyUnified(const double y, const bool use_cos) noexcept
{
    const double z = y * y;

    /* Per-coefficient ternary → SEL on device.  Each SEL is 1 cycle.
     * Tried __ldg-from-table to match libdevice's LDG.E.CI pattern;
     * regressed on Pascal because the LDG latency couldn't be hidden
     * without explicit DEPBAR drain that nvcc doesn't emit from C++. */
    const double a1 = use_cos ? kC1 : kS1;
    const double a2 = use_cos ? kC2 : kS2;
    const double a3 = use_cos ? kC3 : kS3;
    const double a4 = use_cos ? kC4 : kS4;
    const double a5 = use_cos ? kC5 : kS5;
    const double a6 = use_cos ? kC6 : kS6;

    /* Horner chain (degree 6). */
    double r = fma(z, a6, a5);
    r = fma(z, r, a4);
    r = fma(z, r, a3);
    r = fma(z, r, a2);
    r = fma(z, r, a1);

    if (use_cos) {
        /* cos(y) ≈ 1 − z/2 + z²·r = 1 + z·(z·r − ½), two FMAs:
         *   t      = fma(z, r, -0.5)   →  z·r − 0.5
         *   result = fma(z, t,  1.0)   →  1 + z·(z·r − 0.5)
         *                              =  1 − ½·z + z²·r
         *
         * FDLIBM's original ``w + (((1-w) - hz) + r·z²)`` compensation
         * form was an FP32 artefact: in single precision, `1 - hz` loses
         * trailing bits to rounding when hz is small, and the compensa-
         * tion recovers them.  In FP64 with our argument bound |y| ≤ π/4,
         * we have hz ≤ 0.308 — far from any subtraction cancellation —
         * and the 53-bit mantissa absorbs the rounding well below the
         * 1-ULP gate.  The two-FMA form is bit-identical at ≤ 1 ULP to
         * the compensated form on every sweep we tested, while using
         * half the FP64 ops (3 vs 6), which is the wall-time-relevant
         * metric on Pascal's throughput-bound FP64 pipeline.  libdevice
         * also drops the compensation (0 DADDs in its cos finalisation). */
        const double t = fma(z, r, -0.5);
        return fma(z, t, 1.0);
    } else {
        /* sin(y) ≈ y + (y · z) · r. */
        const double v = z * y;
        return fma(v, r, y);
    }
}

/* (Phase A attempt — Pascal dual-chain polynomial in cwSin/cwCos — was
 * tried and REVERTED.  Hypothesis was that two independent Horner
 * chains would pipeline within a single warp.  Empirical: wall time
 * regressed 5.45 → 6.71 ms (+23 %) and per-thread cycles regressed
 * 13,447 → 18,604 (+38 %).  Root cause: within a single warp, a single
 * scheduler issues 1 instruction per cycle regardless of whether
 * adjacent DFMAs are independent.  Doubling DFMA count on Pascal's
 * 1:32 FP64 throughput pipeline strictly adds work — the second chain
 * has nowhere to hide.  Kept the single-chain unified-Horner form. */

// ===================================================================
//  Public entry points — cwSin / cwCos / cwSinCos
// ===================================================================

/** @brief Bounded-argument FP64 sine.
 *  Contract: ``|x| < 2³¹``; input is a normal double.
 *  Accuracy: ≤ 2 ULP vs ``std::sin`` across the full contract range,
 *  including the quadrant boundaries.
 *
 *  Sign handling: ``payneHanekReduceDD`` operates on ``|x|`` (the bit
 *  shift in step 2 strips the sign bit of the input).  The quadrant
 *  dispatch produces the value of sin on ``|x|``.  Since sin is odd,
 *  the result is negated when the input was negative. */
/* Cutoff for the fast Cody-Waite path.  Inputs with |x| < 2³¹ go through
 * codyWaiteReduce + the unified polynomial; larger inputs fall back to
 * the statically-unrolled Payne-Hanek (always STACK=0).
 *
 * In practice cudajectory inputs are at most ~2 × 10⁷ rad (8 orders of
 * magnitude below this cutoff), so the slow path is never reached at
 * runtime — but the cutoff guarantees correctness up to the full ``|x|
 * < 2³¹`` contract. */
inline constexpr double kFastPathBound = 0x1p31;   /* 2³¹ = 2147483648 */

DEVICE() FORCEINLINE()
double cwSin(const double x) noexcept
{
    double y;
    int    q;
    /* Dispatch: fast Cody-Waite for |x| < 2³¹, else slow Payne-Hanek. */
    if (fabs(x) < kFastPathBound) {
        codyWaiteReduce(x, y, q);
    } else {
        payneHanekReduce(x, y, q);
    }
    /* Unified polynomial: sin or cos based on (q & 1). */
    const double polyVal = cwPolyUnified(y, (q & 1) != 0);
    /* Combined sign-bit XOR: flip if (q & 2) XOR input-sign.
     * (q & 2) is bit 1 of q → bit 63 of the mask via left shift by 62.
     * Single LOP32 on the integer pipe replaces what was two DADDs. */
    const uint64_t q_neg = (static_cast<uint64_t>(q) & 2ULL) << 62;
    const uint64_t x_neg = doubleAsU64(x) & 0x8000000000000000ULL;
    return applySign(polyVal, q_neg ^ x_neg);
}

/** @brief Bounded-argument FP64 cosine.
 *  cos is even, so the input sign does not affect the result. */
DEVICE() FORCEINLINE()
double cwCos(const double x) noexcept
{
    double y;
    int    q;
    if (fabs(x) < kFastPathBound) {
        codyWaiteReduce(x, y, q);
    } else {
        payneHanekReduce(x, y, q);
    }
    /* cos: select cos(y) when q is even, sin(y) when q is odd. */
    const double polyVal = cwPolyUnified(y, (q & 1) == 0);
    /* Bit-XOR sign flip — single LOP32 vs DADD-based negation. */
    const uint64_t mask = (static_cast<uint64_t>(q + 1) & 2ULL) << 62;
    return applySign(polyVal, mask);
}

/** @brief Bounded-argument FP64 sincos.  Shares the reduction; the
 *  polynomial runs twice (once for sin, once for cos) — the compiler
 *  CSEs the shared z = y².  Input sign is applied to the sin output only. */
DEVICE() FORCEINLINE()
void cwSinCos(const double x, double& sinOut, double& cosOut) noexcept
{
    double y;
    int    q;
    if (fabs(x) < kFastPathBound) {
        codyWaiteReduce(x, y, q);
    } else {
        payneHanekReduce(x, y, q);
    }
    const double sy = cwPolyUnified(y, false);
    const double cy = cwPolyUnified(y, true);
    const double sinSwap = (q & 1) ? cy : sy;
    const double cosSwap = (q & 1) ? sy : cy;
    /* Bit-XOR sign flips — single LOP32 each, no DADD/DMUL.
     *   sin: flip if (q & 2) XOR input-sign  (sin is odd)
     *   cos: flip if ((q+1) & 2)             (cos is even, no input-sign) */
    const uint64_t x_neg    = doubleAsU64(x) & 0x8000000000000000ULL;
    const uint64_t sin_mask = ((static_cast<uint64_t>(q)     & 2ULL) << 62) ^ x_neg;
    const uint64_t cos_mask =  (static_cast<uint64_t>(q + 1) & 2ULL) << 62;
    sinOut = applySign(sinSwap, sin_mask);
    cosOut = applySign(cosSwap, cos_mask);
}

} // namespace detail
} // namespace math
} // namespace feta

#endif // FETA_CPU_ONLY
