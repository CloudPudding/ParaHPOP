/**
 * @file MathDispatchUndef.h
 * @brief Companion to `MathDispatch.h` — undefines the dispatch macros.
 *
 * Included at the end of each per-group math facade header so the
 * `FETA_MATH_UNARY` / `FETA_MATH_BINARY` macros stay scoped to that header and
 * never leak to consumers. Like `MathDispatch.h`, it intentionally has no
 * `#pragma once` so the define/undef pair can be cycled once per group.
 */

#undef FETA_MATH_UNARY
#undef FETA_MATH_BINARY
