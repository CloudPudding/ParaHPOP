#pragma once

#include "paraHPOP/typedefs.h"
#include "paraHPOP/util.h"

#include "paraHPOP/model/accelerations/PrivateAccelerations.h"
#include "paraHPOP/model/math/ode/ODE.h"
#include "paraHPOP/model/physics/reduced/PrivateDimensional.h"

namespace paraHPOP {
namespace model {
namespace math {
namespace detail {

template<typename SamplesT_, typename PhysicsT_, bool work,
    bool MaybeVolatile = false>
class RefModel {
    using Self = RefModel<SamplesT_, PhysicsT_, work, MaybeVolatile>;

public:
    
    static constexpr bool Volatile = MaybeVolatile && work;
    
    using SamplesT = typename SamplesT_::template Ref<work, MaybeVolatile>;
    using PhysicsT = typename PhysicsT_::template Ref<work, MaybeVolatile>;

    DEVICEHOST()
    static RefModel make(const SamplesT samples, const PhysicsT physics)
    {
        return { samples, physics };
    }

    DEVICEHOST() const SamplesT& samples() const { return samples_; }

    DEVICEHOST() const PhysicsT& physics() const { return physics_; }

    DEVICEHOST() RefModel clone() const { return *this; }

    SamplesT samples_;
    PhysicsT physics_;
};

} // namespace detail
} // namespace math
} // namespace model
} // namespace paraHPOP