
#pragma once

#include <cstddef> // idx_t

#include "parm/integrate/rk/coefs/macros.h"

#include "parm/typedefs.h"
#include "parm/util.h"

namespace parm {
namespace integrate {
namespace rk {
namespace coefs {

namespace detail {


namespace RKF78 {
#define _ODEORDER 1
#define _NSTAGES 13
#define _ORDER 8
#define _EMBEDDED true
#define _C(...)                                                                \
    ARRAY(2. / 27, 1. / 9, 1. / 6, 5. / 12, 1. / 2, 5. / 6, 1. / 6, 2. / 3,    \
        1. / 3, 1.0, 0, 1.0)
#define _A(...)                                                                \
    ARRAY({ 2. / 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },                        \
        { 1. / 36, 1. / 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },                    \
        { 1. / 24, 0, 1. / 8, 0, 0, 0, 0, 0, 0, 0, 0, 0 },                     \
        { 5. / 12, 0, -25. / 16, 25. / 16, 0, 0, 0, 0, 0, 0, 0, 0 },           \
        { 1. / 20, 0, 0, 1. / 4, 1. / 5, 0, 0, 0, 0, 0, 0, 0 },                \
        { -25. / 108, 0, 0, 125. / 108, -65. / 27, 125. / 54, 0, 0, 0, 0, 0,   \
            0 },                                                               \
        { 31. / 300, 0, 0, 0, 61. / 225, -2. / 9, 13. / 900, 0, 0, 0, 0, 0 },  \
        { 2.0, 0, 0, -53. / 6, 704. / 45, -107. / 9, 67. / 90, 3.0, 0, 0, 0,   \
            0 },                                                               \
        { -91. / 108, 0, 0, 23. / 108, -976. / 135, 311. / 54, -19. / 60,      \
            17. / 6, -1. / 12, 0, 0, 0 },                                      \
        { 2383. / 4100, 0, 0, -341. / 164, 4496. / 1025, -301. / 82,           \
            2133. / 4100, 45. / 82, 45. / 164, 18. / 41, 0, 0 },               \
        { 3. / 205, 0, 0, 0, 0, -6. / 41, -3. / 205, -3. / 41, 3. / 41,        \
            6. / 41, 0, 0 },                                                   \
        { -1777. / 4100, 0, 0, -341. / 164, 4496. / 1025, -289. / 82,          \
            2193. / 4100, 51. / 82, 33. / 164, 12. / 41, 0, 1.0 })
#define _B(...)                                                                \
    ARRAY(0, 0, 0, 0, 0, 34. / 105, 9. / 35, 9. / 35, 9. / 280, 9. / 280, 0,   \
        41. / 840, 41. / 840)
#define _BE(...)                                                               \
    ARRAY(41. / 840, 0, 0, 0, 0, 0, 0, 0, 0, 0, 41. / 840, -41. / 840,         \
        -41. / 840)

CREATE_TABLEAU(T, _ODEORDER, _NSTAGES, _ORDER, _EMBEDDED, _C, _A, _B, _BE)

#undef _ODEORDER
#undef _NSTAGES
#undef _ORDER
#undef _EMBEDDED
#undef _C
#undef _A
#undef _B
#undef _BE

} // namespace RKF78

} // namespace detail



// Only RKF78 is supplied in this project.
using RKF78 = detail::RKF78::T;

} // namespace coefs
} // namespace rk
} // namespace integrate
} // namespace parm