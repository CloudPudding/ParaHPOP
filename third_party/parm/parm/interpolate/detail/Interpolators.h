#pragma once

#include "parm/interpolate/detail/Coefficients.h"

namespace parm {
namespace interpolate {
namespace detail {

/** @brief Local (Index-private) Scalar interpolator */
template<typename CoeffsT, typename IntervalsT>
class LocalInterpolator {
    using Self = LocalInterpolator;

public:
    /** @brief the radius type */
    using RadiusT = typename CoeffsT::ComponentT;
    /** @brief the return type */
    using ItemT = typename CoeffsT::ItemT;
    /** @brief the data type */
    using ComponentT = typename CoeffsT::ComponentT;

    /** @brief Expose the coefficients */
    DEVICEHOST() const CoeffsT& coeffs() const { return coeffs_; }

    /** @brief Expose the Intervals */
    DEVICEHOST() const IntervalsT& intervals() const { return intervals_; }

    /** @brief Expose the Radius */
    DEVICEHOST() const RadiusT& radius() const { return radius_; }

    /** @brief Making data public for PODification */
    CoeffsT coeffs_       = CoeffsT{};
    IntervalsT intervals_ = IntervalsT{};
    RadiusT radius_       = RadiusT{};
};

/** @brief Global interpolator - adds the interval search step */
template<typename CoeffsT, typename IntervalsT>
class GlobalInterpolator : public LocalInterpolator<CoeffsT, IntervalsT> {
    using ParentT = LocalInterpolator<CoeffsT, IntervalsT>;
    using Self    = GlobalInterpolator;

public:
    /** @brief The radius type */
    using RadiusT = typename CoeffsT::ComponentT;
    /** @brief the return type */
    using ItemT = typename CoeffsT::ItemT;
    /** @brief the data type */
    using ComponentT = typename CoeffsT::ComponentT;

    /** @brief Factory method to construct from parent object */
    DEVICEHOST() static GlobalInterpolator make(const ParentT& parent)
    {
        return GlobalInterpolator{ parent };
    }

    DEVICEHOST()
    static GlobalInterpolator make(const CoeffsT& coeffs,
        const IntervalsT& intervals, const RadiusT& radius)
    {
        return GlobalInterpolator{ ParentT{ coeffs, intervals, radius } };
    }

    /** @brief Find the interval for the given independent variable */
    DEVICEHOST() inline idx_t findInterval(const ComponentT& t) const
    {
        idx_t idx = (idx_t)((t - this->intervals()[0]) / (2 * this->radius()));
#ifdef __CUDA_ARCH__
        PARM_GPU_ASSERT(
            idx < this->intervals().size(), err::INTERP_OUTOFBOUNDS);
#else
        PARM_ASSERT(idx < this->intervals().size(), "Out of interval bounds.");
#endif
        return idx;
    }
};

} // namespace detail
} // namespace interpolate
} // namespace parm