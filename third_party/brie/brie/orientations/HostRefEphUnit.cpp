#include "brie/orientations/RefEphUnit.h"

namespace brie {
namespace orientations {

/* Explicit instantiation */
template class RefEphUnit<false, false>;
template class RefEphUnit<true, false>;
template class RefEphUnit<true, true>;
template class RefEphUnit<false, true>;

} // namespace orientations
} // namespace brie
