#include "brie/orientations/RefEphUnit.h"

#ifndef BRIE_CPU_ONLY

namespace brie {
namespace orientations {

/* Explicit instantiation */
template class RefEphUnit<false, false>;
template class RefEphUnit<true, false>;
template class RefEphUnit<true, true>;
template class RefEphUnit<false, true>;

} // namespace orientations
} // namespace brie

#endif
