#pragma once

#include <feta/feta.h>
#include <parm/interpolate.h>

namespace brie {
using feta::SampleIndex;

using Real     = double;
using NaifId   = int;
using idx_t    = parm::idx_t;
using uint_t   = parm::uint_t;
using vecdim_t = uint_t;

using Vec3RArray = feta::Vec3TArray<Real>;
using Vec4RArray = feta::Vec4TArray<Real>;
using Vec6RArray = feta::Vec6TArray<Real>;
using Vec9RArray = feta::Vec9TArray<Real>;

using Vec3R = feta::Vec3T<Real>;
using Vec4R = feta::Vec4T<Real>;
using Vec6R = feta::Vec6T<Real>;
using Vec9R = feta::Vec9T<Real>;

} // namespace brie
