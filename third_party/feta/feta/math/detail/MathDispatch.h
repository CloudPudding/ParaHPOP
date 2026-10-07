/**
 * @file MathDispatch.h
 * @brief Single-source dispatch macros for the `feta::math` scalar wrappers.
 *
 * The dozens of `feta::math` entry points share one of two identical dispatch
 * scaffolds: a unary `T f(T)` shape and a binary `T f(T, T)` shape. Each
 * scaffold selects a float / double / SIMD-packet / std backend
 * by compile-time type traits. These two macros generate those scaffolds so
 * the wrappers read as one-liners in the per-group facade headers.
 *
 * A macro body cannot contain `#ifdef`, so the *definition* of each macro is
 * selected here by the compilation pass (device vs host). Every expansion is
 * therefore token-identical to the former hand-written per-function `#ifdef`
 * scaffolds — the generated code (and device SASS) is unchanged.
 *
 * @note This header intentionally has **no** `#pragma once`. It is meant to be
 *       included by each per-group math header and paired with
 *       `MathDispatchUndef.h` at the end of that header, so the macros stay
 *       scoped to the group and never leak to consumers. The prerequisite
 *       detail headers it pulls in are individually `#pragma once`-guarded, so
 *       re-inclusion across groups is cheap.
 */

#include <cmath>

#include "feta/core/type_traits.h"
#include "feta/math/detail/BoundedTrig.h"
#include "feta/math/type_traits.h"

// Packet detail headers are x86-only; skip during the device compilation pass.
#ifndef __CUDA_ARCH__
#include "feta/math/detail/PacketArcTrig.h"
#include "feta/math/detail/PacketBitOps.h"
#include "feta/math/detail/PacketExpLog.h"
#include "feta/math/detail/PacketMisc.h"
#include "feta/math/detail/PacketRounding.h"
#include "feta/math/detail/PacketSinCos.h"
#endif

// --- Unary float-math dispatch ----------------------------------------------
// STDEXPR is the host std-fallback expression written in terms of `x`.
#ifdef __CUDA_ARCH__
#define FETA_MATH_UNARY(NAME, FLOATFN, DOUBLEFN, PACKETFN, STDEXPR)    \
    template<typename T>                                                       \
    DEVICEHOST()                                                               \
    FORCEINLINE() T NAME(T x)                                                  \
    {                                                                          \
        if constexpr (IsBasic<T>) {                                            \
            if constexpr (std::same_as<T, float>) {                           \
                return FLOATFN(x);                                             \
            } else /* double */ {                                             \
                return DOUBLEFN(x);                                            \
            }                                                                  \
        } else {                                                               \
            static_assert(core::alwaysFalse<T>::value,                         \
                "Unsupported type for math::" #NAME);                          \
        }                                                                      \
    }
#else
#define FETA_MATH_UNARY(NAME, FLOATFN, DOUBLEFN, PACKETFN, STDEXPR)    \
    template<typename T>                                                       \
    DEVICEHOST()                                                               \
    FORCEINLINE() T NAME(T x)                                                  \
    {                                                                          \
        if constexpr (IsPacket<T>) {                                           \
            return PACKETFN(x);                                                \
        } else {                                                               \
            return STDEXPR;                                                    \
        }                                                                      \
    }
#endif

// --- Binary float-math dispatch ---------------------------------------------
// Params standardised to (a, b); STDEXPR is the host std-fallback in terms of
// `a` and `b`. Same #ifdef-wraps-the-definition technique as the unary macro.
#ifdef __CUDA_ARCH__
#define FETA_MATH_BINARY(NAME, FLOATFN, DOUBLEFN, PACKETFN, STDEXPR)   \
    template<typename T>                                                       \
    DEVICEHOST()                                                               \
    FORCEINLINE() T NAME(T a, T b)                                             \
    {                                                                          \
        if constexpr (IsBasic<T>) {                                            \
            if constexpr (std::same_as<T, float>) {                           \
                return FLOATFN(a, b);                                          \
            } else /* double */ {                                             \
                return DOUBLEFN(a, b);                                         \
            }                                                                  \
        } else {                                                               \
            static_assert(core::alwaysFalse<T>::value,                         \
                "Unsupported type for math::" #NAME);                          \
        }                                                                      \
    }
#else
#define FETA_MATH_BINARY(NAME, FLOATFN, DOUBLEFN, PACKETFN, STDEXPR)   \
    template<typename T>                                                       \
    DEVICEHOST()                                                               \
    FORCEINLINE() T NAME(T a, T b)                                             \
    {                                                                          \
        if constexpr (IsPacket<T>) {                                           \
            return PACKETFN(a, b);                                             \
        } else {                                                               \
            return STDEXPR;                                                    \
        }                                                                      \
    }
#endif
