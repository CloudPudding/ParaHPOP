#include "paraHPOP/model/accelerations/PrivateAccelerations.h"
#include "paraHPOP/model/physics/reduced/RefDimensional.h"

namespace paraHPOP {
namespace model {
namespace physics {
namespace reduced {

template<bool work, bool MaybeVolatile>
Vec3R RefDimensional<work, MaybeVolatile>::hostEval_(const SampleIndex& idx,
    const Vec3R& pos, const Vec3R& vel, const Real& epoch, const Real& mass,
    const Real& area, const Real& cr, const Real& cd,
    const brie::NaifId& COI) const
{
    return eval_(idx, pos, vel, epoch, mass, area, cr, cd, COI);
}

/* Explicit instantiation */
template class RefDimensional<true>;
template class RefDimensional<false>;

} // namespace reduced
} // namespace physics
} // namespace model
} // namespace paraHPOP
