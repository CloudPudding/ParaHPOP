/** @brief Indexed coefficients management - handle an array of coefficients
 * that can have a further "index" dimension */
#pragma once

#include "parm/typedefs.h"
#include "parm/util.h"

namespace parm {
namespace interpolate {
namespace detail {

template<typename CoeffsT_>
class IndexedCoefficients {
    using Self = IndexedCoefficients;

public:
    /** @brief Expose the coefficients type */
    using CoeffsT = CoeffsT_;
    /** @brief the return type */
    using ItemT = typename CoeffsT::ItemT;
    /** @brief the data type */
    using ComponentT = typename CoeffsT::ComponentT;
    /** @brief The vector dimension of the coefficients */
    static constexpr idx_t VecDims
        = feta::core::expr::isExpression<CoeffsT>::value ? CoeffsT::VecDims : 1;
    /** @brief Factory method to construct from data members */
    DEVICEHOST()
    static IndexedCoefficients<CoeffsT> make(
        const CoeffsT& data, const idx_t& numIndexes)
    {
        return IndexedCoefficients<CoeffsT>{ data, numIndexes,
            data.size() / numIndexes };
    }

    /** @brief Return the total size of the coefficients */
    DEVICEHOST() idx_t size() const { return this->data_.size(); }

    /** @brief Return the size of each index block */
    DEVICEHOST() idx_t iOffset(const idx_t& i = 0) const
    {
        return i * this->iOffset_;
    }

    /** @brief Return the number of indexes */
    DEVICEHOST() const idx_t& numIndexes() const { return this->numIndexes_; }

    /** @brief Maximum index */
    DEVICEHOST() const idx_t& maxIndex() const { return this->numIndexes_ - 1; }

    /** @brief Expose the coefficients */
    DEVICEHOST() const CoeffsT& coeffs() const { return data_; }

    /** @brief Expose the data */
    DEVICEHOST() const ComponentT* data() const { return data_.data(); }

    /** @brief Data made public for PODification */
    CoeffsT data_     = CoeffsT{};
    idx_t numIndexes_ = 0;
    idx_t iOffset_    = 0;
};

/** @brief Polynomial coefficients */
template<typename CoeffsT>
class PolynomialCoefficients : public IndexedCoefficients<CoeffsT> {
    using ParentT = IndexedCoefficients<CoeffsT>;
    using Self    = PolynomialCoefficients;

public:
    /** @brief the return type */
    using ItemT = typename CoeffsT::ItemT;
    /** @brief the data type */
    using ComponentT = typename CoeffsT::ComponentT;
    /** @brief The vector dimension of the coefficients */
    static constexpr idx_t VecDims
        = feta::core::expr::isExpression<CoeffsT>::value ? CoeffsT::VecDims : 1;

    /** @brief Factory method to construct from parent type */
    DEVICEHOST()
    static PolynomialCoefficients make(const ParentT& other)
    {
        return PolynomialCoefficients{ other };
    }

    /** @brief Factory method to construct from coefficient and degrees */
    DEVICEHOST()
    static PolynomialCoefficients make(const CoeffsT& data, const idx_t& degree)
    {
        return PolynomialCoefficients{ ParentT::make(data, degree + 1) };
    }

    /** @brief Return the size of each index block */
    DEVICEHOST() idx_t degOffset(const idx_t& i = 0) const
    {
        return i * this->iOffset_;
    }

    /** @brief Reverse register offset */
    DEVICEHOST() idx_t revOffset(const idx_t& i = 0) const
    {
        return (this->numIndexes_ - 1 - i) * this->iOffset_;
    }

    /** @brief Extract the coefficient for the given index and degree */
    template<idx_t dim>
    DEVICEHOST()
    ComponentT revGet(const idx_t& i, const idx_t& deg = 0) const
    {
        return this->data_.template get<dim>(i + this->revOffset(deg));
    }

    /** @brief Segment forwarder */
    template<idx_t From, idx_t HowMany>
    DEVICEHOST()
    inline decltype(auto) segment() const
    {
        return this->data_.template segment<From, HowMany>();
    }

    /** @brief Return the total number of degrees */
    DEVICEHOST() const idx_t& numDegrees() const { return this->numIndexes_; }

    /** @brief Return the highest degree */
    DEVICEHOST() idx_t highestDegree() const { return this->numIndexes_ - 1; }
};

} // namespace detail
} // namespace interpolate
} // namespace parm