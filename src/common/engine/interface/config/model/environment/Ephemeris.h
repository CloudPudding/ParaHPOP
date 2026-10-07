#pragma once

#include "interface/naming/orientations/Registry.h"
#include "interface/typedefs.h"
#include "interface/util.h"

namespace interface {
namespace config {
namespace model {
namespace environment {

/** @brief Ephemeris file manager */
class Ephemeris {
    using Self   = Ephemeris;
    using PathsT = util::Paths;
    using EphT   = brie::states::EphUnit<true>;

public:
    using FilesT = std::vector<std::string>;

    /** @brief Default constructor creates an empty list */
    Ephemeris() = default;

    /** @brief Construct from Json */
    Ephemeris(const json& j) { util::FileAdder::add(files_, j); }

    /** @brief Construct from files */
    Ephemeris(const FilesT& files)
        : files_{ files }
    {
    }

    /** @brief Copy constructor */
    Ephemeris(const Ephemeris& other) { *this = other; }

    /** @brief Move constructor */
    Ephemeris(Ephemeris&& other) { *this = std::move(other); }

    /** @brief Copy assignment operators */
    Self& operator=(const Ephemeris& other)
    {
        files_     = other.files_;
        paths_     = other.paths_;
        showInfo_  = other.showInfo_;
        ephdata_   = std::move(other.ephdata_.clone());
        ephloaded_ = other.ephloaded_;

        return *this;
    }

    /** @brief Move assignment operator */
    Self& operator=(Ephemeris&& other)
    {
        files_     = std::move(other.files_);
        paths_     = std::move(other.paths_);
        ephdata_   = std::move(other.ephdata_);
        ephloaded_ = std::exchange(other.ephloaded_, false);
        showInfo_  = std::exchange(other.showInfo_, false);

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

    /** @brief Host-callable non-owning reference type to the loaded data */
    using RefT = typename EphT::GRef;

    /** @brief Expose a host-callable non-owning reference to the loaded
     *  ephemeris data. Requires `loaded()` to be true. */
    RefT hostRef() const { return ephdata_.hostRef(); }

    /** @brief Alias for `hostRef()`. */
    RefT ref() const { return ephdata_.hostRef(); }

    /** @brief Whether the underlying ephemeris unit has been loaded. */
    bool loaded() const { return ephloaded_; }

    /** @brief Whether the given id is a target body of the loaded
     *  ephemeris unit, i.e. directly queryable as an ephemeris target.
     *  Requires `loaded()` to be true. */
    bool hasTargetBody(const NaifId& id) const
    {
        return ephdata_.metadata().hostRef().hasTargetBody(id);
    }

    /** @brief Throw when any loaded ephemeris target does not cover the
     *  [loMjd, hiMjd] epoch window (MJD2000 days). brie's own epoch
     *  checks are BRIE_DEBUG_MODE-only: a release-build query outside a
     *  segment's coverage silently extrapolates the Chebyshev
     *  polynomials instead of failing. One O(N segments) host loop per
     *  call, run before any evaluation work. Requires `loaded()`.
     *  Only the first segment per target is checked — brie resolves
     *  every query through the first matching metadata row. */
    void assertEpochCoverage(const Real& loMjd, const Real& hiMjd,
        const std::string& context) const
    {
        const Real lo   = util::TimeConversion::mjdToSpice(loMjd);
        const Real hi   = util::TimeConversion::mjdToSpice(hiMjd);
        const auto mref = ephdata_.metadata().hostRef();
        for (idx_t i = 0; i < mref.size(); i++) {
            const NaifId target = mref.getTarget(i);
            if (mref.getTargetBodyCount(target) != i)
                continue;
            const Real ini = mref.getInitialEpoch(i);
            const Real fin = mref.getFinalEpoch(i);
            if (ini <= lo && hi <= fin)
                continue;

            std::string targetName;
            if (!brie::gravity::Parser::tryParsedName(target, targetName))
                targetName = "Id " + std::to_string(target);
            PARAHPOP_THROW(std::runtime_error,
                context + " reaches epochs [" + std::to_string(loMjd) + ", "
                    + std::to_string(hiMjd)
                    + "] (MJD2000 days), but the loaded ephemeris covers "
                      "target '"
                    + targetName + "' only over ["
                    + std::to_string(util::TimeConversion::spiceToMjd(ini))
                    + ", "
                    + std::to_string(util::TimeConversion::spiceToMjd(fin))
                    + "]. Out-of-coverage queries silently extrapolate the "
                      "Chebyshev data (brie bounds checks are debug-only); "
                      "load ephemeris files spanning the full epoch "
                      "window.");
        }
    }

    /** @brief Create an ephemeris unit */
    Self& make(
        const brie::core::NaifIdArray& bodies = brie::core::NaifIdArray(0))
    {
        /* an empty file list resolves to the default (unspecified) item: leave
         * the unit unloaded and return instead of throwing. Building the
         * environment then succeeds; a body's state is rejected only when that
         * body is actually requested (the loaded() guards on the query paths
         * fire lazily, and the backend's allocateEphCache covers active bodies).
         * The default ephdata_ stays EphT::empty() — we deliberately do NOT
         * load() it, because validating/reading a zero-size unit trips the
         * "container not allocated" check. (Never pass an empty list to fromBrie
         * either: it dereferences the first file unconditionally.) */
        if (files_.empty())
            return *this;

        if (showInfo_)
            PARAHPOP_INFO("Loading ephemeris data from: %s...",
                filestring_(files_).c_str());

        if (!ephloaded_) {
            load(std::move(EphT::fromBrie(files_, bodies, paths_)));
        }
        return *this;
    }

    /** @brief move-load the given ephemeris unit. Every segment must be
     * expressed in ICRF (J2000) — paraHPOP propagates in ICRF only, and
     * a non-ICRF segment would silently corrupt the physics. */
    void load(EphT&& ephdata)
    {
        ephdata_   = std::move(ephdata);
        ephloaded_ = true;
        this->assertICRF_();
    }

    /** @brief Move-dump the given ephemeris unit */
    EphT dump()
    {
        EphT out   = std::move(ephdata_);
        ephloaded_ = false;
        return out;
    }

    /** @brief Serialise to JSON (inverse of Ephemeris(json)). */
    json to_json() const { return json(files_); }

protected:
    /** @brief Throw when any loaded segment is expressed in a non-ICRF
     * frame. SPK/.brie segments carry their SPICE frame id (1 = J2000);
     * nothing else in the pipeline validates it, so a wrong frame would
     * silently corrupt the propagation physics. One O(N segments) host
     * loop at load time. */
    void assertICRF_() const
    {
        constexpr int J2000 = 1;
        const auto mref     = ephdata_.metadata().hostRef();
        for (idx_t i = 0; i < mref.size(); i++) {
            const int frame = mref.getFrame(i);
            if (frame == J2000)
                continue;

            const NaifId target = mref.getTarget(i);
            std::string targetName;
            if (!brie::gravity::Parser::tryParsedName(target, targetName))
                targetName = "Id " + std::to_string(target);

            /* Name the frame through the orientation registry when its
             * SPICE id is in the compat table */
            std::string frameName = "SPICE frame " + std::to_string(frame);
            const auto known
                = naming::orientations::Registry::instance().fromSpice(frame);
            if (known)
                frameName = naming::orientations::Registry::instance().nameOf(
                                *known)
                    + " (" + frameName + ")";

            PARAHPOP_THROW(std::runtime_error,
                "paraHPOP propagates in ICRF (J2000) only, but the "
                "loaded ephemeris carries a segment for target '"
                    + targetName + "' expressed in " + frameName
                    + ". Re-export the ephemeris in J2000 (bspToBrie "
                      "rejects non-ICRF segments unless --allow-non-icrf "
                      "is given).");
        }
    }

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

    EphT ephdata_   = EphT::empty();
    bool ephloaded_ = false;
    FilesT files_;
    PathsT paths_  = util::paths::paraHPOPDefaultPath();
    bool showInfo_ = false;
};

} // namespace environment
} // namespace model
} // namespace config
} // namespace interface