#pragma once

#include "interface/typedefs.h"
#include "interface/util.h"

#include <optional>

namespace interface {
namespace config {
namespace model {
namespace environment {

/** @brief Constants file manager */
class Constants {
    using Self   = Constants;
    using PathsT = util::Paths;

public:
    using FilesT = std::vector<std::string>;

    /** @brief Default constructor creates an empty list */
    Constants() = default;

    /** @brief Construct from json.
     *
     *  Accepts the historical file-list form (a string or array of strings)
     *  unchanged, and an object form ``{"files": [...], "AU": ..., "CLIGHT":
     *  ...}`` that additionally carries inline universal constants — letting a
     *  user edit AU/CLIGHT, or declare a constants set without a file at all
     *  (see ``make()``). */
    Constants(const json& j) { fromJson_(j); }

    /** @brief Construct from files */
    Constants(FilesT& files)
        : files_{ files }
    {
    }
    Constants(const FilesT& files)
        : files_{ files }
    {
    }

    /** @brief Copy constructor */
    Constants(const Constants& other) { *this = other; }

    /** @brief Move constructor */
    Constants(Constants&& other) { *this = std::move(other); }

    /** @brief Copy assignment operators */
    Self& operator=(const Constants& other)
    {
        files_    = other.files_;
        paths_    = other.paths_;
        cdata_    = other.cdata_.clone();
        showInfo_ = other.showInfo_;
        cloaded_  = other.cloaded_;
        au_       = other.au_;
        clight_   = other.clight_;

        return *this;
    }

    /** @brief Move assignment operator */
    Self& operator=(Constants&& other)
    {
        files_    = std::move(other.files_);
        paths_    = std::move(other.paths_);
        cdata_    = std::move(other.cdata_);
        showInfo_ = std::exchange(other.showInfo_, false);
        cloaded_  = std::exchange(other.cloaded_, false);
        au_       = std::move(other.au_);
        clight_   = std::move(other.clight_);

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

    /** @brief Set the astronomical unit (AU) universal constant. When set it
     *  overrides the value read from the constants files, or supplies it for a
     *  fileless declaration. Chainable. */
    inline Self& au(double v)
    {
        au_ = v;
        return *this;
    }

    /** @brief Set the speed of light (CLIGHT) universal constant. When set it
     *  overrides the value read from the constants files, or supplies it for a
     *  fileless declaration. Chainable. */
    inline Self& clight(double v)
    {
        clight_ = v;
        return *this;
    }

    /** @brief Set both universal constants (AU and CLIGHT) at once. */
    inline Self& setUniversals(double au, double clight)
    {
        au_     = au;
        clight_ = clight;
        return *this;
    }

    /** @brief Inline AU override, or `std::nullopt` when taken from files. */
    const std::optional<double>& au() const { return au_; }

    /** @brief Inline CLIGHT override, or `std::nullopt` when taken from files. */
    const std::optional<double>& clight() const { return clight_; }

    /** @brief Whether AU or CLIGHT is supplied inline (drives the constants
     *  build path: an override is applied over the loaded files, and a fileless
     *  set requires both to be present). */
    bool hasInlineUniversals() const
    {
        return au_.has_value() || clight_.has_value();
    }

    /** @brief Expose a host-callable non-owning reference to the loaded
     *  constants data. Requires `loaded()` to be true. */
    brie::Constants::GRef hostRef() const { return cdata_.hostRef(); }

    /** @brief Alias for `hostRef()`. */
    brie::Constants::GRef ref() const { return cdata_.hostRef(); }

    /** @brief Whether the underlying constants have been loaded. */
    bool loaded() const { return cloaded_; }

    /** @brief Create an Constants unit.
     *
     *  With files and no inline universals this is the historical path
     *  (`fromFiles`), unchanged byte-for-byte. Inline universals (`au`/
     *  `clight`) are applied on top of the loaded files. With no files at all
     *  the set is built from the inline universals alone — both AU and CLIGHT
     *  must then be present (inline per-body constants, if any, are merged in
     *  afterwards by the Environment). */
    Self& make()
    {
        if (cloaded_)
            return *this;

        if (files_.empty()) {
            PARAHPOP_ASSERT(au_.has_value() && clight_.has_value(),
                "brie::Constants cannot be created without constants files "
                "unless both AU and CLIGHT are supplied inline.");
            load(brie::Constants::merged(brie::Constants::empty(), {},
                brie::detail::UniversalConstants{ *au_, *clight_ }));
            return *this;
        }

        load(std::move(brie::Constants::fromFiles(files_, paths_)));
        if (hasInlineUniversals()) {
            /* A single-field override (only AU or only CLIGHT) keeps the other
             * universal from the loaded files: read the un-overridden value
             * from the loaded base rather than requiring both to be supplied. */
            const auto   base      = cdata_.hostRef();
            const double auVal     = au_.value_or(base.au());
            const double clightVal = clight_.value_or(base.cLight());
            cdata_                 = brie::Constants::merged(cdata_, {},
                                brie::detail::UniversalConstants{ auVal, clightVal });
        }
        return *this;
    }

    /** @brief move-load the given ephemeris unit */
    void load(brie::Constants&& cdata)
    {
        cdata_   = std::move(cdata);
        cloaded_ = true;
    }

    /** @brief Merge extra per-body constants into the loaded set (e.g.
     *  massless points carrying a radius, or inline-declared bodies). Requires
     *  the constants to be `loaded()`. Identical re-declarations are no-ops; a
     *  conflicting redefinition of an existing id throws (see
     *  `brie::Constants::merged`). */
    Self& mergeBodies(const std::vector<brie::detail::BodyConstants>& extra)
    {
        PARAHPOP_ASSERT(cloaded_,
            "Constants must be loaded before merging extra per-body "
            "constants.");
        if (!extra.empty())
            cdata_ = brie::Constants::merged(cdata_, extra);
        return *this;
    }

    /** @brief Move-dump the given ephemeris unit */
    brie::Constants dump()
    {
        brie::Constants out = std::move(cdata_);
        cloaded_            = false;
        return out;
    }

    /** @brief Serialise to JSON (inverse of Constants(json)). Emits the
     *  historical array-of-files form unless inline universals are set, in
     *  which case the object form carrying ``files``/``AU``/``CLIGHT`` is used
     *  so the override round-trips. */
    json to_json() const
    {
        if (!hasInlineUniversals())
            return json(files_);
        json j     = json::object();
        j["files"] = files_;
        if (au_.has_value())
            j["AU"] = *au_;
        if (clight_.has_value())
            j["CLIGHT"] = *clight_;
        return j;
    }

protected:
    /** @brief Parse the constructor JSON: the array/string form is a plain
     *  file list; the object form additionally carries inline universals. */
    void fromJson_(const json& j)
    {
        if (j.is_object()) {
            if (j.contains("files"))
                util::FileAdder::add(files_, j.at("files"));
            if (j.contains("AU"))
                au_ = j.at("AU").get<double>();
            if (j.contains("CLIGHT"))
                clight_ = j.at("CLIGHT").get<double>();
        } else {
            util::FileAdder::add(files_, j);
        }
    }

    brie::Constants cdata_ = brie::Constants::empty();
    bool cloaded_          = false;
    FilesT files_;
    PathsT paths_  = util::paths::paraHPOPDefaultPath();
    bool showInfo_ = false;
    std::optional<double> au_;
    std::optional<double> clight_;
};

} // namespace environment
} // namespace model
} // namespace config
} // namespace interface