#pragma once

#include "parm/typedefs.h"
#include "parm/util/DeviceError.h"
#include "parm/util/log.h"
#include "parm/util/throw.h"
#include <deque>
#include <list>

#ifndef PARM_CPU_ONLY

namespace parm {
namespace util {
namespace graph {

/** @brief Manage multiple vs single dependencies */
template<typename IdxT>
struct IsMultipleDependency {
    static constexpr bool value = false;
};
template<typename T>
struct IsMultipleDependency<std::vector<T>> {
    static constexpr bool value = true;
};
template<typename T>
struct IsMultipleDependency<std::initializer_list<T>> {
    static constexpr bool value = true;
};
template<typename T>
struct IsMultipleDependency<std::list<T>> {
    static constexpr bool value = true;
};
template<typename T>
struct IsMultipleDependency<std::deque<T>> {
    static constexpr bool value = true;
};

} // namespace graph
} // namespace util
} // namespace parm

#endif