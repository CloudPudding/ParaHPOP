#pragma once

#include "interface/typedefs.h"
#include "interface/util.h"
#include <iostream>

namespace interface {
namespace config {
namespace model {
namespace sphericalharmonics {


/** @brief Enum to mark which coefficient to access */
enum class Coefficient : idx_t {
    C = 0, /**< Cosine coefficient */
    S = 1  /**< Sine coefficient */
};

/** @brief Pair of corresponding n-m factor */
struct FactorPair {
    Real anm; /**< Cnm factor */
    Real bnm; /**< Snm factor */
};

/** @brief Reference to the spherical harmonics coefficients */
template<typename CoefficientsT, bool work>
class RefCoefficients {
    using ArrT       = typename CoefficientsT::ArrT::template Ref<work>;
    using HeaderArrT = typename CoefficientsT::HeaderArrT::template Ref<work>;

public:
    /** @brief Expose the factor pair */
    using FactorPairT = FactorPair;

    /** @brief Handle function for the computation of factors */
    DEVICEHOST() static inline Real factor(const idx_t n) { return 1.0 / n; }

    /** @brief Factory method to construct from the data members */
    DEVICEHOST()
    static RefCoefficients make(const ArrT coeffs, const ArrT realHeader,
        const HeaderArrT intHeader, const ArrT factors, const NaifId body)
    {
        return { coeffs, realHeader, intHeader, factors, body };
    }

    /** @brief Return the body radius (stored at realHeader_[0]) */
    DEVICEHOST()
    Real bodyRadius() const { return realHeader_[0]; }

    /** @brief Return the gravitational parameter GM (stored at realHeader_[1])
     */
    DEVICEHOST()
    Real gm() const { return realHeader_[1]; }

    /** @brief Return the maximum degree of these coefficients */
    DEVICEHOST()
    idx_t maxDegree() const { return intHeader_[0]; }

    /** @brief Return the maximum order of these coefficients */
    DEVICEHOST()
    idx_t maxOrder() const { return intHeader_[1]; }

    /** @brief Return whether coefficients are fully normalized (1) or
     * unnormalized (0) */
    DEVICEHOST()
    idx_t normalized() const { return intHeader_[2]; }

    /** @brief Access the coefficients for the given degree and order */
    template<Coefficient coeff>
    DEVICEHOST()
    Real& at(const idx_t degree, const idx_t order)
    {
        idx_t index = (degree * (degree + 1)) / 2 + order;
        if constexpr (coeff == Coefficient::C)
            return coeffs_[index];
        else
            return coeffs_[index + coeffs_.size() / 2];
    }
    template<Coefficient coeff>
    DEVICEHOST()
    Real at(const idx_t degree, const idx_t order) const
    {
        idx_t index = (degree * (degree + 1)) / 2 + order;
        if constexpr (coeff == Coefficient::C)
            return coeffs_[index];
        else
            return coeffs_[index + coeffs_.size() / 2];
    }

    /** @brief Access the precomputed recursion factors */
    DEVICEHOST()
    Real factor(const idx_t n, const idx_t m, const idx_t type) const
    {
        // Triangular index: n * (n + 1) / 2 + m
        // Interleaved: 2 * index + type
        // type 0: anm (or cm), type 1: bnm
        idx_t index = 2 * (n * (n + 1) / 2 + m) + type;
        return factors_[index];
    }

    /** @brief Retrieve both the precomputed recursion factors as a pair */
    DEVICEHOST()
    FactorPair factorPair(const idx_t n, const idx_t m) const
    {
        const idx_t baseIndex = 2 * (n * (n + 1) / 2 + m);
        return FactorPair{ factors_[baseIndex], factors_[baseIndex + 1] };
    }

    /** @brief Analytical limit of P_{n,1}(sinφ)/cosφ as cosφ → 0, at the
     * north pole (sin φ = +1).  South-pole value is this multiplied by
     * (−1)^(n+1) — i.e. positive for odd n, negative for even n.
     *
     * Computed on-demand (cold path: only called when cosφ is exactly zero,
     * i.e. for inputs on the body z-axis).  Closed form:
     *   unnormalized: n(n+1)/2 ≡ dP_n/dx|_{x=+1}
     *   4π-normalized: × √(2(2n+1)/(n(n+1)))  (matches the normalisation
     *                                          convention used by
     *                                          @ref Coefficients::initializeFactors_,
     *                                          where c_1 = √3 and
     *                                          c_m = √((2m+1)/(2m)) for m ≥ 2).
     */
    DEVICEHOST()
    Real polarLimitM1(const idx_t& n) const
    {
        Real v = 0.5 * static_cast<Real>(n) * (n + 1.0);
        if (normalized()) {
            v *= feta::math::sqrt(
                2.0 * (2.0 * n + 1.0)
                / (static_cast<Real>(n) * (n + 1.0)));
        }
        return v;
    }

    /** @brief Expose the raw coefficients */
    DEVICEHOST() ArrT rawCoefficients() const { return coeffs_; }

    /** @brief Expose the real header items */
    DEVICEHOST() ArrT realHeader() const { return realHeader_; }

    /** @brief Expose the integer header items */
    DEVICEHOST() HeaderArrT intHeader() const { return intHeader_; }

    /** @brief Expose the factors array */
    DEVICEHOST() ArrT factors() const { return factors_; }

    /** @brief Expose the body */
    DEVICEHOST() NaifId body() const { return body_; }

    /** @brief Clone this reference coefficients object */
    DEVICEHOST() RefCoefficients clone() const { return *this; }

    /** @brief Equality comparison (for feta::scalar::Array storage) */
    DEVICEHOST() bool operator==(const RefCoefficients& o) const
    {
        return coeffs_.data() == o.coeffs_.data() && body_ == o.body_;
    }
    DEVICEHOST() bool operator!=(const RefCoefficients& o) const
    {
        return !(*this == o);
    }

    /** @brief Data members made public for PODification */
    ArrT coeffs_;          /**< Coefficients array reference */
    ArrT realHeader_;      /**< Real header array reference */
    HeaderArrT intHeader_; /**< Header array reference */
    ArrT factors_;         /**< Precomputed recursion factors */
    NaifId body_; /**< NAIF ID of the body these coefficients refer to */
};

/** @brief Actual spherical harmonics storage and load from file */
class Coefficients {
    using Self = Coefficients;

public:
    using ArrT          = feta::scalar::Array<Real>;
    using HeaderArrT    = feta::scalar::Array<idx_t>;
    using TabulatedData = std::vector<std::vector<Real>>;

    /** @brief Expose Reference types */
    template<bool work>
    using Ref = RefCoefficients<Self, work>;
    /** @brief Expose Global reference type */
    using GRef = Ref<false>;
    /** @brief Expose Work reference type */
    using WRef = Ref<true>;

    /** @brief Load coefficients from the given file path */
    static TabulatedData loadFromFile(const std::string filepath,
        char delimiter, idx_t headerLines,
        const util::Paths paths = util::paths::paraHPOPDefaultPath())
    {
        brie::util::path filePath = paths.requireFile(filepath);

        /* Open the file */
        std::ifstream ifs(filePath, std::ios::in);
        PARAHPOP_ASSERT(
            ifs.is_open(), "Failed to open file: " + filePath.string());
        PARAHPOP_ASSERT(ifs.good(), "Failed to read file: " + filePath.string());

        /* Read coefficients */
        TabulatedData coeffs;
        std::string line;
        while (std::getline(ifs, line)) {
            /* Skip header lines */
            if (headerLines > 0) {
                --headerLines;
                continue;
            }

            /* Parse the line */
            line.erase(std::find(line.begin(), line.end(), '#'), line.end());
            if (line.empty())
                continue;

            /* End of line parsing */
            const auto func
                = [](char const c) { return c == '\n' || c == '\r'; };
            line.erase(
                std::remove_if(line.begin(), line.end(), func), line.end());

            /* Add the empty row */
            coeffs.push_back(std::vector<Real>());

            /* Tokenize the line */
            const auto tokens = parm::util::Strings::split(line, delimiter);
            for (const auto& t : tokens) {
                if (t.empty())
                    continue;
                coeffs.back().push_back(
                    parm::util::Strings::parse<Real>(t.c_str(), nullptr));
            }
        }
        return coeffs;
    }

    /** @brief Default constructor */
    Coefficients() = default;

    /** @brief Construct From the given file path, number of header lines,
     * and number of degrees and orders to keep */
    Coefficients(const NaifId body, const std::string filepath,
        char delimiter = ',', idx_t headerLines = 0, int maxDegree = -1,
        int maxOrder            = -1,
        const util::Paths paths = util::paths::paraHPOPDefaultPath())
    {
        /* Read the file and make sure that it exists */
        TabulatedData rawCoeffs
            = loadFromFile(filepath, delimiter, headerLines, paths);
        PARAHPOP_ASSERT(
            (!rawCoeffs.empty()), "No coefficients found in file: " + filepath);

        /* Make sure that the header is a vaòod PDS-SHA with exactly 8 values */
        std::vector<Real> header = rawCoeffs.front();
        PARAHPOP_ASSERT(header.size() == 8,
            "Invalid PDS-SHA header in file: " + filepath
                + " - Expected 8 values, found "
                + std::to_string(header.size()));

        /* Extract max degree and order from the header */
        const idx_t maxDegHeader     = static_cast<idx_t>(header[3]);
        const idx_t maxOrdHeader     = static_cast<idx_t>(header[4]);
        const idx_t normalizedHeader = static_cast<idx_t>(header[5]);

        /* If different from -1, check that the requested degree is <= than the
         * available one */
        if (maxDegree != -1) {
            PARAHPOP_ASSERT(static_cast<idx_t>(maxDegree) <= maxDegHeader,
                "Requested maximum degree " + std::to_string(maxDegree)
                    + " exceeds available maximum degree "
                    + std::to_string(maxDegHeader) + " in file: " + filepath);
        }
        /* If different from -1, check that the requested order is <= than the
         * available one */
        if (maxOrder != -1) {
            PARAHPOP_ASSERT(static_cast<idx_t>(maxOrder) <= maxOrdHeader,
                "Requested maximum order " + std::to_string(maxOrder)
                    + " exceeds available maximum order "
                    + std::to_string(maxOrdHeader) + " in file: " + filepath);
        }
        /* Validate the normalization flag in the file (either 0 or 1) */
        PARAHPOP_ASSERT((normalizedHeader == 0) || (normalizedHeader == 1),
            "Unsupported normalization flag " + std::to_string(normalizedHeader)
                + " in file: " + filepath
                + " - Only 0 (unnormalized) and 1 (fully normalized) are "
                  "supported.");

        /* Determine the maximum degree and order. If input values are -1, then
         * retain the maximum available from the header */
        idx_t maxDeg
            = (maxDegree == -1) ? maxDegHeader : static_cast<idx_t>(maxDegree);
        idx_t maxOrd
            = (maxOrder == -1) ? maxOrdHeader : static_cast<idx_t>(maxOrder);

        /* Assert that longitude and latitude offsets (item 6 and 7 in the
         * header) are 0 */
        PARAHPOP_ASSERT(header[6] == 0.0 && header[7] == 0.0,
            "Non-zero latitude/longitude offsets are not supported.");

        /* All checks were fine. Store all the data */
        body_ = body;

        /* Now store the integer header items */
        intHeader_                    = std::move(HeaderArrT(3, 0));
        HeaderArrT::GRef intHeaderRef = intHeader_.hostRef();
        intHeaderRef[0]               = maxDeg;
        intHeaderRef[1]               = maxOrd;
        intHeaderRef[2]               = normalizedHeader;

        /* Now store the real header items */
        realHeader_              = std::move(ArrT(2, 0.0));
        ArrT::GRef realHeaderRef = realHeader_.hostRef();
        // The test suite expects header[0] = reference radius, header[1] = GM.
        realHeaderRef[0] = header[0]; // Reference radius
        realHeaderRef[1] = header[1]; // GM

        /* Initialize the recursion factors */
        initializeFactors_(maxDeg, maxOrd, normalizedHeader);

        /* Allocate the necessary memory in coeffs_ */
        initializeCoefficients_(maxDeg + 1);
        /* use an unlocked reference to fill-in the data */
        bool unlocked = true;
        GRef ref      = hostRef(unlocked);

        /* Now store the coefficients */
        for (idx_t i = 0; i < rawCoeffs.size(); ++i) {
            const std::vector<Real>& row = rawCoeffs[i];
            /* Skip header row */
            if (i == 0)
                continue;
            /* Each row must have exactly 4 items: degree, order, Cnm, Snm
             * (maybe 6 if null values for lat and lon offset are present but
             * can be ignored)*/
            PARAHPOP_ASSERT(row.size() == 4 || row.size() == 6,
                "Invalid coefficient row in file: " + filepath
                    + " - Expected 6 values, found "
                    + std::to_string(row.size()));
            const idx_t n = static_cast<idx_t>(row[0]);
            const idx_t m = static_cast<idx_t>(row[1]);
            /* Assert that the degree is always higher than 0 */
            PARAHPOP_ASSERT(n > 0,
                "Invalid degree 0 coefficient found in file: " + filepath);
            /* Assert that order is <= degree */
            PARAHPOP_ASSERT(m <= n,
                "Invalid order " + std::to_string(m) + " for degree "
                    + std::to_string(n) + " found in file: " + filepath);
            /* Assert that the S coefficient for order 0 is always 0 */
            PARAHPOP_ASSERT(m > 0 || row[3] == 0.0,
                "Invalid non-zero Sn0 coefficient found in file: " + filepath);
            /* Skip coefficients exceeding the requested max degree/order */
            if (n > maxDeg || m > maxOrd)
                continue;
            /* Compute the index and store Cnm and Snm */
            /* Follow the followint shape [2 (C first, S then), maxDeg + 1
             * (degree index), maxDeg + 1 (order index)] in row-major style */
            ref.at<Coefficient::C>(n, m) = row[2]; // Cnm
            // Initialize s if the order is higher than 0
            if (m > 0)
                ref.at<Coefficient::S>(n, m) = row[3]; // Snm
        }
        loaded_ = true;
    }

    /** @brief Copy constructor is forbidden */
    Coefficients(const Self& other) = delete;

    /** @brief Move constructor */
    Coefficients(Coefficients&& other) noexcept
        : loaded_{ std::exchange(other.loaded_, false) }
        , coeffs_{ std::exchange(other.coeffs_, ArrT::flexible()) }
        , realHeader_{ std::exchange(other.realHeader_, ArrT::flexible()) }
        , intHeader_{ std::exchange(other.intHeader_, HeaderArrT::flexible()) }
        , factors_{ std::exchange(other.factors_, ArrT::flexible()) }
        , body_{ std::exchange(other.body_, 0) }
    {
    }

    /** @brief Copy assignment operator is forbidden */
    Coefficients& operator=(const Self& other) = delete;

    /** @brief Move assignment operator */
    Coefficients& operator=(Coefficients&& other) noexcept
    {
        if (this != &other) {
            loaded_     = std::exchange(other.loaded_, false);
            coeffs_     = std::exchange(other.coeffs_, ArrT::flexible());
            realHeader_ = std::exchange(other.realHeader_, ArrT::flexible());
            intHeader_
                = std::exchange(other.intHeader_, HeaderArrT::flexible());
            factors_ = std::exchange(other.factors_, ArrT::flexible());
            body_    = std::exchange(other.body_, 0);
        }
        return *this;
    }

#ifndef BRIE_CPU_ONLY

    /** @brief Upload the data to the device */
    void upload(const cudaStream_t stream = 0)
    {
        PARAHPOP_ASSERT(loaded_ || coeffs_.isFlexible(),
            "Cannot upload uninitialized coefficients");
        coeffs_.upload(stream);
        realHeader_.upload(stream);
        intHeader_.upload(stream);
        factors_.upload(stream);
    }

    /** @brief Download the data from the device */
    void download(const cudaStream_t stream = 0)
    {
        PARAHPOP_ASSERT(loaded_ || coeffs_.isFlexible(),
            "Cannot download uninitialized coefficients");
        coeffs_.download(stream);
        realHeader_.download(stream);
        intHeader_.download(stream);
        factors_.download(stream);
    }

    /** @brief Clear the device data */
    void clearDevice()
    {
        coeffs_.clearDevice();
        realHeader_.clearDevice();
        intHeader_.clearDevice();
        factors_.clearDevice();
    }

#endif

    /** @brief Get a global host reference to the coefficients */
    GRef hostRef(const bool unlocked = false) const
    {
        if (!unlocked && !coeffs_.isFlexible())
            PARAHPOP_ASSERT(
                loaded_, "Cannot get reference to uninitialized coefficients");
        return GRef::make(coeffs_.hostRef(), realHeader_.hostRef(),
            intHeader_.hostRef(), factors_.hostRef(), body_);
    }
    GRef ref(const bool unlocked = false) const { return hostRef(unlocked); }

#ifndef BRIE_CPU_ONLY

    /** @brief Get a device reference to the coefficients */
    GRef deviceRef(const bool unlocked = false) const
    {
        if (!unlocked && !coeffs_.isFlexible())
            PARAHPOP_ASSERT(
                loaded_, "Cannot get reference to uninitialized coefficients");
        return GRef::make(coeffs_.deviceRef(), realHeader_.deviceRef(),
            intHeader_.deviceRef(), factors_.deviceRef(), body_);
    }

#endif

    /** @brief Clone this coefficients object */
    Coefficients clone() const
    {
        Coefficients cfs;
        cfs.loaded_     = loaded_;
        cfs.coeffs_     = std::move(coeffs_.clone());
        cfs.realHeader_ = std::move(realHeader_.clone());
        cfs.intHeader_  = std::move(intHeader_.clone());
        cfs.factors_    = std::move(factors_.clone());
        cfs.body_       = body_;
        return cfs;
    }

protected:
    /** @brief Initialize the memory for the coefficients according to the
     * given degree */
    void initializeCoefficients_(const idx_t maxDegree)
    {
        PARAHPOP_ASSERT(maxDegree > 0, "Maximum degree must be > 0");
        const idx_t nCoeffsToStore = (maxDegree + 1) * (maxDegree + 2) / 2;
        coeffs_ = std::move(ArrT(2 * nCoeffsToStore, 0.0));
    }

    /** @brief Initialize the memory for the recursion factors */
    void initializeFactors_(
        const idx_t maxDegree, const idx_t maxOrder, const idx_t normalized)
    {
        // Allocate factors array: (maxDegree + 1) * (maxDegree + 2)
        // Layout: Triangular packed (n * (n + 1) / 2 + m)
        // Interleaved: anm/cm at 2*idx, bnm at 2*idx+1
        idx_t size = (maxDegree + 1) * (maxDegree + 2);

        // Compute factors on host
        factors_       = std::move(ArrT(size, 0.0));
        ArrT::GRef ref = factors_.hostRef();

        for (idx_t m = 0; m <= maxOrder; ++m) {
            for (idx_t n = m; n <= maxDegree; ++n) {
                idx_t index = 2 * (n * (n + 1) / 2 + m);

                if (n == m) {
                    // Diagonal step
                    Real cm;
                    if (normalized) {
                        if (m == 1)
                            cm = sqrt(3.0);
                        else if (m == 0)
                            cm = 1.0; // P00 = 1
                        else
                            cm = sqrt((2.0 * m + 1.0) / (2.0 * m));
                    } else {
                        cm = (m == 1) ? 1.0 : (2.0 * m - 1.0);
                        if (m == 0)
                            cm = 1.0;
                    }
                    ref[index]     = cm;
                    ref[index + 1] = 0.0; // Unused
                } else {
                    // Vertical step
                    Real anm, bnm;
                    if (normalized) {
                        Real n_ = (Real)n;
                        Real m_ = (Real)m;
                        anm     = sqrt(((2.0 * n_ - 1.0) * (2.0 * n_ + 1.0))
                                / ((n_ - m_) * (n_ + m_)));
                        bnm     = sqrt(((2.0 * n_ + 1.0) * (n_ + m_ - 1.0)
                                       * (n_ - m_ - 1.0))
                                / ((2.0 * n_ - 3.0) * (n_ + m_) * (n_ - m_)));
                    } else {
                        Real n_ = (Real)n;
                        Real m_ = (Real)m;
                        anm     = (2.0 * n_ - 1.0) / (n_ - m_);
                        bnm     = (n_ + m_ - 1.0) / (n_ - m_);
                    }
                    ref[index]     = anm;
                    ref[index + 1] = bnm;
                }
            }
        }
    }

    /** @brief Data members */
    bool loaded_          = false; /**< Whether coefficients have been loaded */
    ArrT coeffs_          = ArrT::flexible();       /**< Coefficients array */
    ArrT realHeader_      = ArrT::flexible();       /**< Real Header items */
    HeaderArrT intHeader_ = HeaderArrT::flexible(); /**< Integer Header items */
    ArrT factors_ = ArrT::flexible(); /**< Precomputed recursion factors */
    NaifId body_  = 0; /**< NAIF ID of the body these coefficients refer to */
};

} // namespace sphericalharmonics
} // namespace model
} // namespace config
} // namespace interface