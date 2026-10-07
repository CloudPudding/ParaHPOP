#include "brie/states/RefEphUnit.h"

namespace brie {
namespace states {

/* Dispatcher definitions */
template<bool work, bool UseTexture, bool MaybeVolatile>
Vec3R RefEphUnit<work, UseTexture, MaybeVolatile>::hostGetPosition_(
    const Real& epoch, const NaifId& target, const NaifId& center) const
{
    return getPosition_(epoch, target, center);
}

template<bool work, bool UseTexture, bool MaybeVolatile>
Vec3R RefEphUnit<work, UseTexture, MaybeVolatile>::hostGetVelocity_(
    const Real& epoch, const NaifId& target, const NaifId& center) const
{
    return getVelocity_(epoch, target, center);
}

template<bool work, bool UseTexture, bool MaybeVolatile>
Vec6R RefEphUnit<work, UseTexture, MaybeVolatile>::hostGetPositionAndVelocity_(
    const Real& epoch, const NaifId& target, const NaifId& center) const
{
    return getPositionAndVelocity_(epoch, target, center);
}

template<bool work, bool UseTexture, bool MaybeVolatile>
void RefEphUnit<work, UseTexture, MaybeVolatile>::hostiGetPosition_(Vec3R& out,
    const Real& epoch, const NaifId& target, const NaifId& center) const
{
    iGetPosition_(out, epoch, target, center);
}

template<bool work, bool UseTexture, bool MaybeVolatile>
void RefEphUnit<work, UseTexture, MaybeVolatile>::hostiGetVelocity_(Vec3R& out,
    const Real& epoch, const NaifId& target, const NaifId& center) const
{
    iGetVelocity_(out, epoch, target, center);
}

template<bool work, bool UseTexture, bool MaybeVolatile>
void RefEphUnit<work, UseTexture, MaybeVolatile>::hostiGetPositionAndVelocity_(Vec6R& out,
    const Real& epoch, const NaifId& target, const NaifId& center) const
{
    iGetPositionAndVelocity_(out, epoch, target, center);
}

/* Explicit instantiations */
template class RefEphUnit<true, true>;
template class RefEphUnit<true, false>;
template class RefEphUnit<false, true>;
template class RefEphUnit<false, false>;
} // namespace states
} // namespace brie