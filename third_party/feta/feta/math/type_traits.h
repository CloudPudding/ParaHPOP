#pragma once

#include "feta/core/simd/Packet.h"
#include "feta/typedefs.h"

/* Math type traits */

namespace feta {
namespace math {

/* Basic data type --> double precision */
template<typename T>
concept IsBasic = std::same_as<T, double> || std::same_as<T, float>
    || std::same_as<T, int> || std::same_as<T, unsigned int>;

/* SIMD packet types */
template<typename T>
concept IsPacket = std::same_as<T, simd::Packet<float, 1>>
    || std::same_as<T, simd::Packet<double, 1>>
    || std::same_as<T, simd::Packet<float, 4>>
    || std::same_as<T, simd::Packet<double, 2>>
    || std::same_as<T, simd::Packet<float, 8>>
    || std::same_as<T, simd::Packet<double, 4>>
    || std::same_as<T, simd::Packet<float, 16>>
    || std::same_as<T, simd::Packet<double, 8>>;

} // namespace math
} // namespace feta