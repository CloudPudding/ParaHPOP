#pragma once

#include <algorithm>
#include "interface/config/detail/FieldCodec.h"
#include "interface/config/model/sphericalharmonics/Coefficients.h"

namespace interface {
namespace config {
namespace model {
namespace environment {

/* name map and valid fields */
namespace sphericalharmonics {

/** @brief Map of valid fields */
enum Fields {
    BODY,
    DELIMITER,
    HEADERLINES,
    FILE,
    MAXDEGREE,
    MAXORDER,
    /* Leave the following item as last, as it only acts as enum size/validity
       checker */
    INVALID

};

using Map = std::map<std::string, Fields>;

/** @brief `std::map` to map the spherical harmonics fields to the
 * corresponding enum
 */
const Map NameMap = {
    /* All lower case */
    { "body", BODY }, { "delimiter", DELIMITER },
    { "headerlines", HEADERLINES }, { "file", FILE }, { "degree", MAXDEGREE },
    { "order", MAXORDER }
};

/** @brief Spherical harmonics fields parser */
inline Fields parse(std::string name)
{
    return interface::config::detail::parseField(
        std::move(name), NameMap, INVALID);
}

} // namespace sphericalharmonics

/** @brief SphericalHarmonics file manager */
class SphericalHarmonics {
    using Self    = SphericalHarmonics;
    using PathsT  = util::Paths;
    using CoeffsT = model::sphericalharmonics::Coefficients;

public:
    using FilesT = std::vector<std::string>;

    /** @brief Default constructor creates an empty list */
    SphericalHarmonics() { }

    /** @brief Construct from Json */
    SphericalHarmonics(const json& j)
    {
        /* not default anymore */
        isDefault_ = false;

        using namespace sphericalharmonics;
        using JIT = json::const_iterator;

        /* Spherical harmonics has a minimum set of valid fields */
        bool foundFile   = false;
        bool foundBody   = false;
        bool foundDegree = false;
        bool foundOrder  = false;

        /* loop over the fields */
        for (JIT jit = j.begin(); jit != j.end(); jit++) {
            std::string field                = jit.key();
            sphericalharmonics::Fields FIELD = sphericalharmonics::parse(field);
            PARAHPOP_ASSERT(FIELD != sphericalharmonics::INVALID,
                Self::ErrorMessage_(field));
            if (FIELD == FILE) {
                foundFile = true;
                /* Process the file and add it to the file list */
                util::FileAdder::add(files_, jit.value());
            } else if (FIELD == sphericalharmonics::BODY) {
                foundBody = true;
                if (jit.value().is_string()) {
                    body_ = brie::gravity::Parser::parsedNaifId(
                        jit.value().get<std::string>());
                } else if (jit.value().is_number_integer()) {
                    body_ = brie::gravity::Parser::parsedNaifId(
                        jit.value().get<NaifId>());
                } else {
                    PARAHPOP_THROW(std::runtime_error,
                        "Spherical harmonics body must be a string (name) or "
                        "an integer (Naif ID).");
                }
            } else if (FIELD == sphericalharmonics::MAXDEGREE) {
                foundDegree = true;
                maxDegree_  = jit.value().get<int>();
            } else if (FIELD == sphericalharmonics::MAXORDER) {
                foundOrder = true;
                maxOrder_  = jit.value().get<int>();
            } else if (FIELD == sphericalharmonics::DELIMITER) {
                delimiter_ = jit.value().get<std::string>()[0];
            } else if (FIELD == sphericalharmonics::HEADERLINES) {
                headerLines_ = jit.value().get<idx_t>();
            }
        }

        /* Assert that all the required fields were found */
        PARAHPOP_ASSERT(foundFile, "Spherical harmonics file not specified.");
        PARAHPOP_ASSERT(foundBody, "Spherical harmonics body not specified.");
        PARAHPOP_ASSERT(foundDegree, "Spherical harmonics degree not specified.");
        PARAHPOP_ASSERT(foundOrder, "Spherical harmonics order not specified.");
    }

    /** @brief Copy constructor */
    SphericalHarmonics(const SphericalHarmonics& other) { *this = other; }

    /** @brief Move constructor */
    SphericalHarmonics(SphericalHarmonics&& other) { *this = std::move(other); }

    /** @brief Copy assignment operators */
    Self& operator=(const SphericalHarmonics& other)
    {
        body_         = other.body_;
        delimiter_    = other.delimiter_;
        headerLines_  = other.headerLines_;
        maxDegree_    = other.maxDegree_;
        maxOrder_     = other.maxOrder_;
        isDefault_    = other.isDefault_;
        coeffs_       = std::move(other.coeffs_.clone());
        coeffsLoaded_ = other.coeffsLoaded_;
        files_        = other.files_;
        paths_        = other.paths_;
        showInfo_     = other.showInfo_;

        return *this;
    }

    /** @brief Move assignment operator */
    Self& operator=(SphericalHarmonics&& other)
    {
        body_         = std::exchange(other.body_, 0);
        delimiter_    = std::exchange(other.delimiter_, ',');
        headerLines_  = std::exchange(other.headerLines_, 0);
        maxDegree_    = std::exchange(other.maxDegree_, -1);
        maxOrder_     = std::exchange(other.maxOrder_, -1);
        isDefault_    = std::exchange(other.isDefault_, true);
        coeffs_       = std::move(other.coeffs_);
        coeffsLoaded_ = std::exchange(other.coeffsLoaded_, false);
        files_        = std::move(other.files_);
        paths_        = std::move(other.paths_);
        showInfo_     = std::exchange(other.showInfo_, false);

        return *this;
    }

    /** @brief Show info mode */
    inline Self& showInfo()
    {
        showInfo_ = true;
        return *this;
    }

    /** @brief No show info mode */
    inline Self& noInfo()
    {
        showInfo_ = false;
        return *this;
    }

    /** @brief Expose the file list */
    FilesT& files() { return files_; }
    const FilesT& files() const { return files_; }

    /** @brief Expose paths */
    PathsT& paths() { return paths_; }
    const PathsT& paths() const { return paths_; }

    /** @brief Update the body */
    Self& body(const NaifId& body)
    {
        using ParserT = brie::gravity::Parser;
        body_         = ParserT::parsedNaifId(body);
        return *this;
    }
    Self& body(const std::string& body)
    {
        using ParserT = brie::gravity::Parser;
        body_         = ParserT::parsedNaifId(body);
        return *this;
    }
    /** @brief Get the body */
    const NaifId& body() const { return body_; }
    /** @brief Get the body name */
    std::string bodyName() const
    {
        return brie::gravity::Parser::parsedName(body_);
    }

    /** @brief Update the delimiter */
    Self& delimiter(const char& delimiter)
    {
        delimiter_ = delimiter;
        return *this;
    }
    /** @brief Get the delimiter */
    const char& delimiter() const { return delimiter_; }

    /** @brief Update the number of header lines */
    Self& headerLines(const idx_t& headerLines)
    {
        headerLines_ = headerLines;
        return *this;
    }
    /** @brief Get the number of header lines */
    const idx_t& headerLines() const { return headerLines_; }

    /** @brief Update the degree */
    Self& degree(const int& deg)
    {
        maxDegree_ = deg;
        return *this;
    }
    /** @brief Get the degree */
    const int& degree() const { return maxDegree_; }

    /** @brief Update the order */
    Self& order(const int& ord)
    {
        maxOrder_ = ord;
        return *this;
    }
    /** @brief Get the order */
    const int& order() const { return maxOrder_; }

    /** @brief Create a spherical harmonics coefficients object */
    Self& make()
    {
        if (!coeffsLoaded_)
            reload();
        return *this;
    }

    /** @brief Reload the spherical harmonics coefficients, regardless the
     * current loaded flag */
    Self& reload()
    {
        adjustDefaultFlag();
        /* Do nothing if default */
        if (isDefault_)
            return *this;

        PARAHPOP_ASSERT(!files_.empty(),
            "Spherical Harmonics Coefficients cannot be created without "
            "spherical harmonics files.");

        if (showInfo_)
            PARAHPOP_INFO("Loading spherical harmonics data from: %s...",
                filestring_(files_).c_str());

        load(std::move(CoeffsT(body_, files_[0], delimiter_, headerLines_,
            maxDegree_, maxOrder_)));

        /* mark as loaded */
        coeffsLoaded_ = true;
        return *this;
    }


    /** @brief Whether this is a default (no-op) SH config — true iff none
     *  of the parse-relevant fields have been set.  Used by
     *  ``Bodies::to_json()`` to decide whether to emit an empty
     *  ``"shape": true`` activation marker or the full SH config block. */
    bool isDefault() const
    {
        return body_ == 0 && delimiter_ == ',' && headerLines_ == 0
            && maxDegree_ == -1 && maxOrder_ == -1 && files_.empty();
    }

    /** @brief Serialise to JSON — symmetric with ``SphericalHarmonics(const json&)``.
     *
     *  Emits the full parse-key set: ``body``, ``delimiter``, ``headerLines``,
     *  ``file`` (always as an array, even for a single file), ``degree``,
     *  ``order``.  The ``body`` key is kept here (unlike Atmosphere) because
     *  the SH file parsing also needs it independently of the outer Bodies
     *  dict — the SH parser will set the body from this field if present
     *  rather than relying solely on the outer dict key. */
    json to_json() const
    {
        json out             = json::object();
        out["body"]          = brie::gravity::Parser::parsedName(body_);
        out["delimiter"]     = std::string(1, delimiter_);
        out["headerLines"]   = headerLines_;
        out["file"]          = files_;
        out["degree"]        = maxDegree_;
        out["order"]         = maxOrder_;
        return out;
    }

    /** @brief move-load the given spherical harmonics coefficients object */
    void load(CoeffsT&& cfs)
    {
        coeffs_       = std::move(cfs);
        coeffsLoaded_ = true;
    }

    /** @brief Move-dump the given spherical harmonics coefficients object */
    CoeffsT dump()
    {
        CoeffsT out   = std::move(coeffs_);
        coeffsLoaded_ = false;
        return out;
    }

protected:
    static inline std::string filestring_(const json& j)
    {
        if (j.is_string())
            return j.dump();
        else {
            /* build a string containing all the files */
            std::stringstream ss;
            ss << "[";
            for (idx_t i = 0; i < j.size(); i++) {
                ss << j[i].dump();
                if (i < j.size() - 1)
                    ss << ", ";
            }
            ss << "]";
            return ss.str();
        }
    }

    /** @brief Adjust default flag. If any of the fields required to build the
     * coefficients is different from the default value, set the flag to false
     */
    void adjustDefaultFlag()
    {
        if (body_ != 0) {
            isDefault_ = false;
            return;
        }
        if (delimiter_ != ',') {
            isDefault_ = false;
            return;
        }
        if (headerLines_ != 0) {
            isDefault_ = false;
            return;
        }
        if (maxDegree_ != -1) {
            isDefault_ = false;
            return;
        }
        if (maxOrder_ != -1) {
            isDefault_ = false;
            return;
        }
        if (!files_.empty()) {
            isDefault_ = false;
            return;
        }
    }

    /** @brief Create a string with the error message */
    static std::string ErrorMessage_(std::string field)
    {
        using namespace environment;
        std::stringstream ss;
        ss << "Invalid spherical harmonics field: '" << field;
        ss << "'. Valid fields are: [";
        idx_t it      = 0;
        auto nameiter = sphericalharmonics::NameMap.begin();
        while (it < sphericalharmonics::INVALID) {
            ss << nameiter->first;
            if (it < sphericalharmonics::INVALID - 1)
                ss << ", ";
            it++;
            nameiter++;
        }
        ss << "] (case insensitive)";
        return ss.str();
    }

    NaifId body_       = 0;
    char delimiter_    = ',';
    idx_t headerLines_ = 0;
    int maxDegree_     = -1;
    int maxOrder_      = -1;
    bool isDefault_    = true;
    CoeffsT coeffs_    = CoeffsT();
    bool coeffsLoaded_ = false;
    FilesT files_;
    PathsT paths_  = util::paths::paraHPOPDefaultPath();
    bool showInfo_ = false;
};

} // namespace environment
} // namespace model
} // namespace config
} // namespace interface