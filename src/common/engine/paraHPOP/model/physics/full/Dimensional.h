#pragma once

#include "paraHPOP/model/physics/full/detail/Base.h"

namespace paraHPOP {
namespace model {
namespace physics {
namespace full {
namespace detail {

/* Explicit instantiation */
extern template class Base<reduced::Dimensional>;

} // namespace detail

/* Create alias */
using Dimensional = detail::Base<reduced::Dimensional>;

} // namespace full
} // namespace physics
} // namespace model
} // namespace paraHPOP