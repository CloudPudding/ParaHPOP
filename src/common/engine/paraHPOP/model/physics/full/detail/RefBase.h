#pragma once

#include "paraHPOP/model/physics/reduced.h"

namespace paraHPOP {
namespace model {
namespace physics {
namespace full {
namespace detail {

template<typename ReducedT_, bool work, bool MaybeVolatile = false>
class RefBase {
    using RefReducedT =
        typename ReducedT_::template Ref<work, MaybeVolatile>;

public:
    
    static constexpr bool Volatile = MaybeVolatile && work;
    
    using ReducedT = RefReducedT;
    
    using AccT = typename accelerations::Accelerations::template Ref<work,
        MaybeVolatile>;
    
    using EnvT
        = typename environment::Env::template Ref<work, MaybeVolatile>;
    
    using StatesT = typename RefReducedT::StatesT;
    using StateT  = typename StatesT::StateT;

    DEVICEHOST()
    static RefBase make(RefReducedT&& reduced)
    {
        return RefBase{ std::move(reduced) };
    }

    DEVICEHOST()
    Vec3R eval(const SampleIndex& idx, const Vec3R& pos, const Vec3R& vel,
        const Real& epoch, const Real& mass, const Real& area, const Real& cr,
        const Real& cd, const brie::NaifId& COI) const
    {
        return reduced_.eval(idx, pos, vel, epoch, mass, area, cr, cd, COI);
    }

    DEVICEHOST() const EnvT& env() const { return reduced_.env(); }

    DEVICEHOST() const AccT& accs() const { return reduced_.accs(); }

    DEVICEHOST() const RefReducedT& reduced() const { return reduced_; }

    DEVICEHOST() RefBase<ReducedT_, work> clone() const { return *this; }

    RefReducedT reduced_;
};

} // namespace detail
} // namespace full
} // namespace physics
} // namespace model
} // namespace paraHPOP