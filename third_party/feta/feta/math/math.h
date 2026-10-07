#pragma once

/**
 * @file math.h
 * @brief FETA math module — portable scalar and SIMD arithmetic
 * types (assumes built-in basic arithmetic operators are available for DataT).
 *
 * This module provides device/host/SIMD packet portable math primitives
 * for native floating-point types. All functions are `DEVICEHOST` so they can
 * be used in CUDA kernels, CPU SIMD loops, and plain host code alike.
 *
 * The wrappers are organised into per-group facade headers (this file simply
 * pulls them all in); the shared float/double/SIMD-packet/std
 * dispatch is single-sourced in `detail/MathDispatch.h`:
 *  - @ref trig.h        — circular sin/cos/tan + inverses + sincos
 *  - @ref explog.h      — exp/exp2/exp10, log/log2/log10, pow
 *  - @ref rounding.h    — ceil/floor/round
 *  - @ref misc.h        — abs/cbrt/sqrt/rsqrt, min/max/fmod/hypot, fma, predicates
 */

#include "feta/math/explog.h"
#include "feta/math/misc.h"
#include "feta/math/rounding.h"
#include "feta/math/trig.h"
