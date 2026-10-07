/* Module with the macros required to the pre-processing time definition of the
 * Runge-Kutta constexpr coefficients */
#pragma once

#include "parm/typedefs.h"

namespace parm {
namespace integrate {
namespace rk {
namespace coefs {

/**
 * Oh ye brave souls who venture forth in this file, avert your children's eyes,
 * for the horrors below know no name.
 *
 * The original intent was to have a structure for each Butcher tableau. Each
 * structure would contain the same set of members, with different values, and
 * these would all be `static constexpr`. This way, by passing the type of the
 * tableau to the RK integrator as a template parameter, it could "unpack" the
 * RK coefficients at compile time.
 *
 * This works great for CPU code, and allows for a high-performance RK
 * integrator that is also usable for any Butcher tableau.
 *
 * Unfortunately, NVCC doesn't allow class/struct members to be defined as
 * `__device__`. The only way to have `static constexpr` values usable from
 * device code is to declare them global (or in a namespace).
 *
 * Hence the monstrosity below, which is a workaround. Macros are defined to
 * fulfill this purpose, automatically creating the corresponding struct
 * according to the interface we want. This struct will automatically switch
 * between ` static constexpr`, defined as a constexpr static class
 * members, and `__device__ static constexpr`, defined as a global variable in a
 * given namespace, and then referenced to by the corresponing struct method.
 *
 * After all the values have been defined, the macro `CREATE_TABLEAU` can be
 * used to create the wrapper structure.
 *
 * IMPORTANT: scalar values can be defined as normal macros, whereas arrays and
 * qualifiers (e.g. __device__ or __constant__) MUST be defined as "lazy
 * evaluable" ones. This module provides all the means to create such evaluable
 * macros, as:
 *  - ARRAY(a,b,c,...) -> Creates a 1D array. Arrays can be used as arguments of
 *    this macro to create multi-dimensional arrays.
 *  - DEVICE(...)   -> inserts the `__device__` qualifier
 *  - HOST(...)     -> inserts the `` qualifier
 *  - CONSTANT(...) -> inserts the `__constant__` qualifier
 *
 * The base class `BaseRK` uses the *curiously recurring template*
 * pattern to provide a common interface, and some compile-time checks.
 *
 * All these crimes against common sense are committed in the name of NVCC. I
 * claim no responsibility for its sins.
 */

/* Helper macro for the "lazy" pre-processor expansion of arrays */
#define EXPAND(...) __VA_ARGS__
#define ARRAY(...) { EXPAND(__VA_ARGS__) }

/* Definitions */
/* Static constexpr scalar */
#define STATIC_CONSTEXPR_SCALAR(hd_qualifier, type, name, value)               \
    hd_qualifier() static constexpr type name = value

/* Host constexpr scalar */
#define CONSTEXPR_SCALAR(type, name, value) static constexpr type name = value

/* Static constexpr 1D array */
#define STATIC_CONSTEXPR_1D_ARRAY(hd_qualifier, type, name, size, value)       \
    hd_qualifier() static constexpr type name[size] = value()

/* host constexpr 1D array */
#define CONSTEXPR_1D_ARRAY(type, name, size, value)                            \
    static constexpr type name[size] = value()

/* Static constexpr 2D array */
#define STATIC_CONSTEXPR_2D_ARRAY(hd_qualifier, type, name, rows, cols, value) \
    hd_qualifier() static constexpr type name[rows][cols] = value()

/* Host constexpr 2D array */
#define CONSTEXPR_2D_ARRAY(type, name, rows, cols, value)                      \
    static constexpr type name[rows][cols] = value()

/* Populate the namespace with the given values */
#define POPULATE_NAMESPACE(OO, N, O, E, C, A, B, BE)                           \
    STATIC_CONSTEXPR_SCALAR(DEVICE, idx_t, d_oo_, OO);                         \
    STATIC_CONSTEXPR_SCALAR(DEVICE, idx_t, d_n_, N);                           \
    STATIC_CONSTEXPR_SCALAR(DEVICE, idx_t, d_o_, O);                           \
    STATIC_CONSTEXPR_SCALAR(DEVICE, bool, d_e_, E);                            \
    STATIC_CONSTEXPR_1D_ARRAY(DEVICE, Real, d_c_, N - 1, C);                   \
    STATIC_CONSTEXPR_2D_ARRAY(DEVICE, Real, d_a_, N - 1, N - 1, A);            \
    STATIC_CONSTEXPR_1D_ARRAY(DEVICE, Real, d_b_, N, B);                       \
    STATIC_CONSTEXPR_1D_ARRAY(DEVICE, Real, d_be_, N, BE);

namespace detail {
/** @brief Structure type to perform device/host switch constexpr calls */
template<typename Child>
struct BaseRK {

    DEVICEHOST() static constexpr idx_t ODEorder()
    {
#ifdef __CUDA_ARCH__
        return Child::d_ODEorder();
#else
        return Child::h_ODEorder();
#endif
    }

    DEVICEHOST() static constexpr idx_t nStages()
    {
#ifdef __CUDA_ARCH__
        return Child::d_nStages();
#else
        return Child::h_nStages();
#endif
    }

    DEVICEHOST() static constexpr idx_t order()
    {
#ifdef __CUDA_ARCH__
        return Child::d_order();
#else
        return Child::h_order();
#endif
    }

    DEVICEHOST() static constexpr bool embedded()
    {
#ifdef __CUDA_ARCH__
        return Child::d_embedded();
#else
        return Child::h_embedded();
#endif
    }

    template<idx_t i>
    DEVICEHOST()
    static constexpr const Real& c()
    {
        static_assert(i < Child::nStages() - 1, "invalid coefficient index");
#ifdef __CUDA_ARCH__
        return Child::template d_c<i>();
#else
        return Child::template h_c<i>();
#endif
    }

    template<idx_t i, idx_t j>
    DEVICEHOST()
    static constexpr const Real& a()
    {
        static_assert(j <= i);
        static_assert(i < Child::nStages() - 1, "invalid coefficient index");
#ifdef __CUDA_ARCH__
        return Child::template d_a<i, j>();
#else
        return Child::template h_a<i, j>();
#endif
    }

    template<idx_t i>
    DEVICEHOST()
    static constexpr const Real& b()
    {
        static_assert(i < Child::nStages(), "invalid coefficient index");
#ifdef __CUDA_ARCH__
        return Child::template d_b<i>();
#else
        return Child::template h_b<i>();
#endif
    }

    template<idx_t i>
    DEVICEHOST()
    static constexpr const Real& be()
    {
        static_assert(
            Child::embedded(), "be() is only valid for embedded methods");
        static_assert(i < Child::nStages(), "invalid coefficient index");
#ifdef __CUDA_ARCH__
        return Child::template d_be<i>();
#else
        return Child::template h_be<i>();
#endif
    }
};
} // namespace detail

#ifndef PARM_CPU_ONLY
/** @brief Macro that creates a struct called `name`, that wraps the access
 * around the Runge Kutta constexpr items. It first populates the namespace for
 * the device values, and then creates the struct. The struct MUST be created in
 * the same namespace where the variables are initialized. */
#define CREATE_TABLEAU(name, OO, N, O, E, C, A, B, BE)                         \
    POPULATE_NAMESPACE(OO, N, O, E, C, A, B, BE)                               \
    struct name : public BaseRK<T> {                                           \
        DEVICE() static constexpr idx_t d_ODEorder()                           \
        {                                                                      \
            return d_oo_;                                                      \
        }                                                                      \
        static constexpr idx_t h_ODEorder()                                    \
        {                                                                      \
            return h_oo_;                                                      \
        }                                                                      \
        DEVICE() static constexpr idx_t d_nStages()                            \
        {                                                                      \
            return d_n_;                                                       \
        }                                                                      \
        static constexpr idx_t h_nStages()                                     \
        {                                                                      \
            return h_n_;                                                       \
        }                                                                      \
        DEVICE() static constexpr idx_t d_order()                              \
        {                                                                      \
            return d_o_;                                                       \
        }                                                                      \
        static constexpr idx_t h_order()                                       \
        {                                                                      \
            return h_o_;                                                       \
        }                                                                      \
        DEVICE() static constexpr idx_t d_embedded()                           \
        {                                                                      \
            return d_e_;                                                       \
        }                                                                      \
        static constexpr idx_t h_embedded()                                    \
        {                                                                      \
            return h_e_;                                                       \
        }                                                                      \
        template<idx_t i>                                                      \
        DEVICE()                                                               \
        static constexpr const Real& d_c()                                     \
        {                                                                      \
            return d_c_[i];                                                    \
        }                                                                      \
        template<idx_t i>                                                      \
        static constexpr const Real& h_c()                                     \
        {                                                                      \
            return h_c_[i];                                                    \
        }                                                                      \
        template<idx_t i, idx_t j>                                             \
        DEVICE()                                                               \
        static constexpr const Real& d_a()                                     \
        {                                                                      \
            return d_a_[i][j];                                                 \
        }                                                                      \
        template<idx_t i, idx_t j>                                             \
        static constexpr const Real& h_a()                                     \
        {                                                                      \
            return h_a_[i][j];                                                 \
        }                                                                      \
        template<idx_t i>                                                      \
        DEVICE()                                                               \
        static constexpr const Real& d_b()                                     \
        {                                                                      \
            return d_b_[i];                                                    \
        }                                                                      \
        template<idx_t i>                                                      \
        static constexpr const Real& h_b()                                     \
        {                                                                      \
            return h_b_[i];                                                    \
        }                                                                      \
        template<idx_t i>                                                      \
        DEVICE()                                                               \
        static constexpr const Real& d_be()                                    \
        {                                                                      \
            return d_be_[i];                                                   \
        }                                                                      \
        template<idx_t i>                                                      \
        static constexpr const Real& h_be()                                    \
        {                                                                      \
            return h_be_[i];                                                   \
        }                                                                      \
                                                                               \
    protected:                                                                 \
        CONSTEXPR_SCALAR(idx_t, h_oo_, OO);                                    \
        CONSTEXPR_SCALAR(idx_t, h_n_, N);                                      \
        CONSTEXPR_SCALAR(idx_t, h_o_, O);                                      \
        CONSTEXPR_SCALAR(bool, h_e_, E);                                       \
        CONSTEXPR_1D_ARRAY(Real, h_c_, N - 1, C);                              \
        CONSTEXPR_2D_ARRAY(Real, h_a_, N - 1, N - 1, A);                       \
        CONSTEXPR_1D_ARRAY(Real, h_b_, N, B);                                  \
        CONSTEXPR_1D_ARRAY(Real, h_be_, N, BE);                                \
    };
#else
/** @brief Macro that creates a struct called `name`, that wraps the access
 * around the Runge Kutta constexpr items. It first populates the namespace for
 * the device values, and then creates the struct. The struct MUST be created in
 * the same namespace where the variables are initialized. */
#define CREATE_TABLEAU(name, OO, N, O, E, C, A, B, BE)                         \
    struct name : public BaseRK<T> {                                           \
        static constexpr idx_t h_ODEorder()                                    \
        {                                                                      \
            return h_oo_;                                                      \
        }                                                                      \
        static constexpr idx_t h_nStages()                                     \
        {                                                                      \
            return h_n_;                                                       \
        }                                                                      \
        static constexpr idx_t h_order()                                       \
        {                                                                      \
            return h_o_;                                                       \
        }                                                                      \
        static constexpr idx_t h_embedded()                                    \
        {                                                                      \
            return h_e_;                                                       \
        }                                                                      \
        template<idx_t i>                                                      \
        static constexpr const Real& h_c()                                     \
        {                                                                      \
            return h_c_[i];                                                    \
        }                                                                      \
        template<idx_t i, idx_t j>                                             \
        static constexpr const Real& h_a()                                     \
        {                                                                      \
            return h_a_[i][j];                                                 \
        }                                                                      \
        template<idx_t i>                                                      \
        static constexpr const Real& h_b()                                     \
        {                                                                      \
            return h_b_[i];                                                    \
        }                                                                      \
        template<idx_t i>                                                      \
        static constexpr const Real& h_be()                                    \
        {                                                                      \
            return h_be_[i];                                                   \
        }                                                                      \
                                                                               \
    protected:                                                                 \
        CONSTEXPR_SCALAR(idx_t, h_oo_, OO);                                    \
        CONSTEXPR_SCALAR(idx_t, h_n_, N);                                      \
        CONSTEXPR_SCALAR(idx_t, h_o_, O);                                      \
        CONSTEXPR_SCALAR(bool, h_e_, E);                                       \
        CONSTEXPR_1D_ARRAY(Real, h_c_, N - 1, C);                              \
        CONSTEXPR_2D_ARRAY(Real, h_a_, N - 1, N - 1, A);                       \
        CONSTEXPR_1D_ARRAY(Real, h_b_, N, B);                                  \
        CONSTEXPR_1D_ARRAY(Real, h_be_, N, BE);                                \
    };
#endif
} // namespace coefs
} // namespace rk
} // namespace integrate
} // namespace parm
