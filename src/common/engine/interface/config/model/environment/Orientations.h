#pragma once

#include <algorithm>
#include <set>

#include "interface/config/model/environment/detail/keys.h"
#include "interface/naming/orientations/Registry.h"
#include "interface/typedefs.h"
#include "interface/util.h"

namespace interface {
namespace config {
namespace model {
namespace environment {

/** @brief Body orientation data file manager and orientation frame
 * declaration proxy. Loaded `.brot` targets auto-register in the
 * process-global orientation registry (invariant: whatever orientation
 * data ends up in the environment is registered and namable); frame
 * declarations double as the load-time subset filter and write through to
 * the same registry. */
class Orientations {
    using Self       = Orientations;
    using PathsT     = util::Paths;
    using RotationsT = brie::orientations::EphUnit<false>;
    using RegistryT  = naming::orientations::Registry;
    using EntryT     = naming::orientations::Entry;

public:
    using FilesT = std::vector<std::string>;
    using OrientationId = naming::orientations::OrientationId;

    /** @brief A declared orientation frame: either a plain name/alias/id
     * entry resolved eagerly through the registry, or a file-reference
     * naming declaration `{"NAME": "file.brot"}` whose id is read from
     * the file's metadata at make() — users never specify ids. */
    struct FrameDecl {
        /** @brief Canonical name (plain form) or user-chosen name (file
         * form) */
        std::string name;
        /** @brief Defining `.brot` file; empty for plain entries */
        std::string file;
        /** @brief Resolved frame id (file form: written at make()) */
        OrientationId id = naming::orientations::ICRF;

        /** @brief Whether this is a file-reference naming declaration */
        bool named() const { return !file.empty(); }
    };

    /** @brief A declared dynamic local-orbital frame: its axes are computed
     * at evaluation time from the ephemeris state of @c to relative to
     * @c from under @c convention. Carries no `.brot` data; its id is
     * auto-assigned (from the reserved dynamicorientation band) and the
     * declaration registers eagerly in the process-global registry. */
    struct LocalOrbitalDecl {
        std::string name;
        brie::frames::AxesSelection convention
            = brie::frames::AxesSelection::RTN;
        NaifId from      = 0; /* reference ("from") body */
        NaifId to        = 0; /* target ("to") body */
        bool corotating  = false; /* co-rotate (omega) vs instantaneous axes */
        OrientationId id = naming::orientations::ICRF; /* assigned at declare */
    };

    /** @brief A declared SYNODIC (rotating two-body) frame: the co-rotating
     * RTN frame of @c to about @c from. RTN and co-rotation are implied, so
     * the declaration carries neither a convention nor a corotating flag.
     * Registers (eagerly) as a co-rotating RTN dynamic frame. */
    struct SynodicDecl {
        std::string name;
        NaifId from      = 0; /* primary ("from") body */
        NaifId to        = 0; /* secondary ("to") body */
        OrientationId id = naming::orientations::ICRF; /* assigned at declare */
    };

    /** @brief Default constructor creates an empty list */
    Orientations() = default;

    /** @brief Construct from Json. Accepts the plain file-array form
     * (``["brot/pck00010.brot"]``) and the dict form with the optional
     * ``only`` load-time subset filter, ``local_orbital`` dynamic
     * frame list, and ``synodic`` frame list:
     * ``{"files": [...], "only": ["Earth_Body_Fixed", "moon pa",
     * {"MY_FRAME": "brot/my_frames.brot"}], "local_orbital": [...],
     * "synodic": [...]}``. */
    Orientations(const json& j)
    {
        if (j.is_object()) {
            for (auto jit = j.begin(); jit != j.end(); ++jit) {
                std::string key = jit.key();
                brie::util::Strings::lowerCase(key);
                if (key == keys::files)
                    util::FileAdder::add(files_, jit.value());
                else if (key == keys::only)
                    this->setOnly(jit.value());
                else if (key == keys::localorbital)
                    this->setLocalOrbital(jit.value());
                else if (key == keys::synodic)
                    this->setSynodic(jit.value());
                else
                    PARAHPOP_THROW(std::runtime_error,
                        "Invalid orientations field: '" + jit.key()
                            + "'. Valid fields are: [files, only, "
                              "local_orbital, synodic] (case insensitive)");
            }
        } else {
            util::FileAdder::add(files_, j);
        }
        isDefault_ = false;
    }

    /** @brief Construct from files */
    Orientations(const FilesT& files)
        : files_{ files }
    {
        isDefault_ = false;
    }

    /** @brief Copy constructor */
    Orientations(const Orientations& other) { *this = other; }

    /** @brief Move constructor */
    Orientations(Orientations&& other) { *this = std::move(other); }

    /** @brief Copy assignment operators */
    Self& operator=(const Orientations& other)
    {
        files_        = other.files_;
        paths_        = other.paths_;
        showInfo_     = other.showInfo_;
        rotationdata_ = std::move(other.rotationdata_.clone());
        rotloaded_    = other.rotloaded_;
        isDefault_    = other.isDefault_;
        only_         = other.only_;
        localOrbital_ = other.localOrbital_;
        synodic_      = other.synodic_;
        return *this;
    }

    /** @brief Move assignment operator */
    Self& operator=(Orientations&& other)
    {
        files_        = std::move(other.files_);
        paths_        = std::move(other.paths_);
        rotationdata_ = std::move(other.rotationdata_);
        rotloaded_    = std::exchange(other.rotloaded_, false);
        showInfo_     = std::exchange(other.showInfo_, false);
        isDefault_    = std::exchange(other.isDefault_, true);
        only_         = std::move(other.only_);
        localOrbital_ = std::move(other.localOrbital_);
        synodic_      = std::move(other.synodic_);

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
    using RefT = typename RotationsT::GRef;

    /** @brief Expose a host-callable non-owning reference to the loaded
     *  rotation data. Requires `loaded()` to be true. */
    RefT hostRef() const { return rotationdata_.hostRef(); }

    /** @brief Whether the underlying rotation unit has been loaded. */
    bool loaded() const { return rotloaded_; }

    /** @brief Whether the given orientation frame's rotation data is
     *  among the loaded `.brot` targets, i.e. its body axes are
     *  evaluable. The frame id maps onto its registry entry's data
     *  target (an ITRF-style frame evaluates under the binary-PCK
     *  frame-class id, not its own). Requires `loaded()` to be true. */
    bool hasTarget(const OrientationId& id) const
    {
        const auto entry      = RegistryT::instance().find(id);
        const OrientationId t = entry ? entry->dataTarget() : id;
        return rotationdata_.hostRef().metadata().hasBody(t);
    }

    /** @brief Throw when a loaded binary-frame rotation unit does not
     *  cover the [loMjd, hiMjd] epoch window (MJD2000 days). Only
     *  Type-3 (binary-PCK, Chebyshev-angle) AND Type-4 (native
     *  IPF/IERS2000-model) units carry real time coverage — Type-1/2
     *  text-PCK polynomial models are valid at any epoch and are skipped.
     *  brie's epoch checks are BRIE_DEBUG_MODE-only: a release-build query
     *  outside coverage silently extrapolates (Chebyshev polynomials, or
     *  the native Lagrange node grid saturating at its edge stencil)
     *  instead of failing. Requires `loaded()`.
     *
     *  `only` optionally restricts the check to the given rotation-data
     *  targets (frame-class ids): the conversion APIs pass the targets
     *  they actually evaluate, while the pre-run guard checks
     *  everything loaded. */
    void assertEpochCoverage(const Real& loMjd, const Real& hiMjd,
        const std::string& context,
        const std::set<OrientationId>* only = nullptr) const
    {
        const Real lo   = util::TimeConversion::mjdToSpice(loMjd);
        const Real hi   = util::TimeConversion::mjdToSpice(hiMjd);
        const auto mref = rotationdata_.metadata().hostRef();
        for (idx_t i = 0; i < mref.size(); i++) {
            const int unitType = mref.getUnitType(i);
            if (unitType != 3 && unitType != 4)
                continue;
            const OrientationId target = mref.getTarget(i);
            /* brie resolves queries through the FIRST matching row */
            if (mref.getTargetBodyCount(target) != i)
                continue;
            if (only && only->count(target) == 0)
                continue;
            const Real ini = mref.getInitialEpoch(i);
            const Real fin = mref.getFinalEpoch(i);
            if (ini <= lo && hi <= fin)
                continue;

            /* name the frame through the registry when the target is
             * claimed by a registered frame */
            std::string frameName
                = "frame-class id " + std::to_string(target);
            const auto entry = RegistryT::instance().findByTarget(target);
            if (entry)
                frameName = "'" + entry->name + "' (rotation target "
                    + std::to_string(target) + ")";
            PARAHPOP_THROW(std::runtime_error,
                context + " reaches epochs [" + std::to_string(loMjd) + ", "
                    + std::to_string(hiMjd)
                    + "] (MJD2000 days), but the loaded rotation data "
                      "covers "
                    + frameName + " only over ["
                    + std::to_string(util::TimeConversion::spiceToMjd(ini))
                    + ", "
                    + std::to_string(util::TimeConversion::spiceToMjd(fin))
                    + "]. Out-of-coverage queries silently extrapolate the "
                      "rotation data (brie bounds checks are debug-only); "
                      "load a rotation kernel spanning the full epoch "
                      "window (e.g. a long-horizon predict file, or the "
                      "native IPF/IERS2000 ITRF frame which spans 2050).");
        }
    }

    /** @brief Create an orientations unit */
    Self& make()
    {
        if (!rotloaded_)
            reload();
        return *this;
    }

    /** @brief Reload the rotations data, regardless the currently loaded flag
     */
    Self& reload()
    {
        adjustDefaultFlag();
        if (isDefault_)
            return *this;

        /* file-reference naming declarations imply their defining file */
        for (const FrameDecl& decl : only_)
            if (decl.named() && !this->hasFile_(decl.file))
                files_.push_back(decl.file);

        /* an empty file list carries no rotation data to load, so resolve to
         * the default (unspecified) item: leave the unit unloaded and return.
         * This covers both an only-list of builtin/derived names or a purely
         * dynamic declaration (local-orbital / synodic frames, whose axes are
         * computed from ephemeris at evaluation time and register eagerly when
         * declared), and a plain empty/unspecified orientations object. In every
         * case the orientation targets fall back lazily to each body's own IAU
         * frame, and a specific body's missing rotation data is only reported
         * when that body is actually requested. */
        if (files_.empty())
            return *this;
        if (showInfo_)
            PARAHPOP_INFO("Loading orientation data from: %s...",
                filestring_(files_).c_str());

        /* load each declared file once: per-file metadata feeds the
         * naming declarations, the fold reproduces fromBrie's
         * first-file-wins multi-file merge */
        RotationsT full = RotationsT::empty();
        for (idx_t i = 0; i < static_cast<idx_t>(files_.size()); i++) {
            RotationsT one = RotationsT::fromBrie(
                files_[i], brie::core::NaifIdArray(0), paths_);
            this->resolveNamedFramesIn_(files_[i], one);
            if (i == 0)
                full = std::move(one);
            else
                full = full.merge(one);
        }

        /* the frames list doubles as the load-time subset filter */
        const auto targets = this->filterTargets_(full);
        if (targets.size() != 0)
            full = full.makeFullSubset(targets);

        load(std::move(full));
        return *this;
    }

    /** @brief Move-load the given orientations unit. Every target of the
     * loaded unit auto-registers in the orientation registry, so whatever
     * the environment can rotate is also namable. */
    void load(RotationsT&& ephdata)
    {
        rotationdata_ = std::exchange(ephdata, RotationsT::empty());
        rotloaded_    = true;
        this->registerLoadedTargets_();
    }

    /** @brief Move-dump the given orientations unit */
    RotationsT dump()
    {
        RotationsT out = std::exchange(rotationdata_, RotationsT::empty());
        rotloaded_     = false;
        return out;
    }

    /** @brief Declare a frame by name, alias, or pre-defined id: resolved
     * eagerly through the registry (any body's `<BODY>_BODY_FIXED` /
     * `IAU_<BODY>` resolves without loading), throwing on unknown names.
     * The declaration doubles as the load-time subset filter; duplicate
     * declarations collapse. */
    Self& addFrame(const std::string& nameOrAlias)
    {
        if (this->hasDecl_(nameOrAlias))
            return *this;
        const auto entry = RegistryT::instance().find(nameOrAlias);
        PARAHPOP_ASSERT(entry.has_value(),
            "'" + nameOrAlias
                + "' is not a known orientation frame. Builtin frames and "
                  "any body's <BODY>_BODY_FIXED / IAU_<BODY> resolve "
                  "without loading; a custom frame is declared by "
                  "referencing its defining file: {\"" + nameOrAlias
                + "\": \"<file.brot>\"} in the 'only' list "
                  "(add(name, file=...) in Python)");
        this->storeDecl_({ entry->name, "", entry->id });
        isDefault_ = false;
        return *this;
    }

    /** @brief Declare a frame by its pre-defined id (builtin or
     * SPICE-originated; custom ids are never user-specified) */
    Self& addFrame(const OrientationId& id)
    {
        const auto entry = RegistryT::instance().find(id);
        PARAHPOP_ASSERT(entry.has_value(),
            "Id " + std::to_string(id)
                + " is not a known orientation frame. Builtin frames and "
                  "any body's NAIF id resolve without loading; a custom "
                  "frame is declared by referencing its defining file in "
                  "the 'only' list");
        this->storeDecl_({ entry->name, "", entry->id });
        isDefault_ = false;
        return *this;
    }

    /** @brief Name the orientation frame defined by the given `.brot`
     * file. The frame-class id is read from the file's metadata at
     * make() — users never specify ids — and the declaration registers
     * the name in the process-global registry. The defining file is
     * appended to the declared files when not already among them; the
     * frame joins the load-time subset filter. Throws at make() when the
     * file defines several not-yet-registered frames (split the file) or
     * none (the frame is already registered — reference it by its
     * existing name instead). */
    Self& addFrame(const std::string& name, const std::string& file)
    {
        PARAHPOP_ASSERT(!name.empty(),
            "Orientation frame names must not be empty");
        PARAHPOP_ASSERT(!file.empty(),
            "Orientation frame '" + name
                + "': the defining .brot file must not be empty");
        if (this->hasDecl_(name))
            return *this;
        this->storeDecl_({ name, file, naming::orientations::ICRF });
        isDefault_ = false;
        return *this;
    }

    /** @brief Expose the frame declarations (the load-time subset filter) */
    const std::vector<FrameDecl>& only() const { return only_; }

    /** @brief Replace the frame declarations from their flat-list json
     * form: name/alias/id entries plus one-key `{"NAME": "file.brot"}`
     * naming declarations */
    Self& setOnly(const json& frames)
    {
        PARAHPOP_ASSERT(frames.is_array(),
            "Orientation 'only' must be a flat json array of frame "
            "names/ids and one-key {\"NAME\": \"file.brot\"} naming "
            "entries");
        only_.clear();
        for (const json& entry : frames) {
            if (entry.is_string())
                this->addFrame(entry.get<std::string>());
            else if (entry.is_number_integer())
                this->addFrame(entry.get<OrientationId>());
            else if (entry.is_object()) {
                PARAHPOP_ASSERT(entry.size() == 1,
                    "Orientation frame naming entries must be one-key "
                    "objects ({\"NAME\": \"file.brot\"}); got: "
                        + entry.dump());
                const auto it = entry.begin();
                PARAHPOP_ASSERT(it.value().is_string(),
                    "Orientation frame '" + it.key()
                        + "': the naming entry value must be the defining "
                          ".brot file (users never specify frame ids)");
                this->addFrame(it.key(), it.value().get<std::string>());
            } else
                PARAHPOP_THROW(std::runtime_error,
                    "Invalid orientation 'only' entry: " + entry.dump()
                        + ". Valid entries are frame names/ids and "
                          "one-key {\"NAME\": \"file.brot\"} naming "
                          "objects");
        }
        isDefault_ = false;
        return *this;
    }

    /** @brief The frame declarations in their flat-list json form
     * (canonical names; naming declarations re-emitted) */
    json onlyJson() const
    {
        json arr = json::array();
        for (const FrameDecl& decl : only_) {
            if (decl.named())
                arr.push_back(json { { decl.name, decl.file } });
            else
                arr.push_back(decl.name);
        }
        return arr;
    }

    /** @brief Declare a dynamic local-orbital frame from body names and an
     * axis-convention token (e.g. ``add_local_orbital("EM_RTN", "RTN",
     * "Earth", "Moon")``). The frame's axes are computed at evaluation time
     * from the ephemeris state of @p to relative to @p from; the
     * declaration registers eagerly in the process-global registry and
     * auto-assigns the id. @p corotating selects co-rotating velocity
     * semantics (default false = instantaneous local-orbital axes); see
     * naming::orientations::Entry::corotating. Duplicate names (with a
     * matching definition) collapse. */
    Self& addLocalOrbital(const std::string& name,
        const std::string& convention, const std::string& from,
        const std::string& to, bool corotating = false)
    {
        return this->addLocalOrbital(name,
            naming::orientations::axesSelectionOf(convention),
            brie::gravity::Parser::parsedNaifId(from),
            brie::gravity::Parser::parsedNaifId(to), corotating);
    }

    /** @brief Declare a dynamic local-orbital frame from a resolved
     * convention and from/to NAIF ids. */
    Self& addLocalOrbital(const std::string& name,
        const brie::frames::AxesSelection& convention, const NaifId& from,
        const NaifId& to, bool corotating = false)
    {
        PARAHPOP_ASSERT(!name.empty(),
            "Local-orbital frame names must not be empty");
        PARAHPOP_ASSERT(from != to,
            "Local-orbital frame '" + name
                + "': the 'from' and 'to' bodies must differ");
        if (this->hasLocalOrbitalDecl_(name))
            return *this;
        const OrientationId id = RegistryT::instance().addDynamic(
            name, convention, from, to, corotating);
        localOrbital_.push_back({ name, convention, from, to, corotating, id });
        isDefault_ = false;
        return *this;
    }

    /** @brief Declare a SYNODIC (rotating two-body) frame from body names:
     * the co-rotating RTN frame of @p to about @p from. A synodic frame
     * rotates with the from&rarr;to line (x along from&rarr;to, z along the
     * orbit normal), so a converted velocity is the velocity relative to the
     * rotating frame. RTN and co-rotation are implied — the synodic
     * convention takes no convention/corotating argument. */
    Self& addSynodic(
        const std::string& name, const std::string& from, const std::string& to)
    {
        return this->addSynodic(name,
            brie::gravity::Parser::parsedNaifId(from),
            brie::gravity::Parser::parsedNaifId(to));
    }

    /** @brief Declare a SYNODIC frame from from/to NAIF ids. Registers (and
     * round-trips) as a co-rotating RTN dynamic frame, stored in the
     * dedicated synodic declaration list. */
    Self& addSynodic(const std::string& name, const NaifId& from,
        const NaifId& to)
    {
        PARAHPOP_ASSERT(!name.empty(), "Synodic frame names must not be empty");
        PARAHPOP_ASSERT(from != to,
            "Synodic frame '" + name
                + "': the 'from' and 'to' bodies must differ");
        if (this->hasSynodicDecl_(name))
            return *this;
        const OrientationId id = RegistryT::instance().addDynamic(
            name, brie::frames::AxesSelection::RTN, from, to,
            /*corotating=*/true);
        synodic_.push_back({ name, from, to, id });
        isDefault_ = false;
        return *this;
    }

    /** @brief Expose the local-orbital frame declarations */
    const std::vector<LocalOrbitalDecl>& localOrbital() const
    {
        return localOrbital_;
    }

    /** @brief Expose the synodic frame declarations */
    const std::vector<SynodicDecl>& synodic() const { return synodic_; }

    /** @brief Replace the local-orbital frame declarations from their json
     * form: an array of ``{"name", "convention", "from", "to"}`` objects
     * (from/to are body names or NAIF ids; an optional ``corotating`` flag
     * defaults to false). */
    Self& setLocalOrbital(const json& arr)
    {
        PARAHPOP_ASSERT(arr.is_array(),
            "Orientation 'local_orbital' must be a json array of "
            "{\"name\", \"convention\", \"from\", \"to\"} objects");
        localOrbital_.clear();
        for (const json& e : arr) {
            PARAHPOP_ASSERT(e.is_object() && e.contains(keys::lo_name)
                    && e.contains(keys::lo_convention)
                    && e.contains(keys::lo_from) && e.contains(keys::lo_to),
                "Each 'local_orbital' entry must be an object with keys "
                "[name, convention, from, to]; got: " + e.dump());
            const bool corotating = e.contains(keys::lo_corotating)
                ? e[keys::lo_corotating].get<bool>()
                : false;
            this->addLocalOrbital(e[keys::lo_name].get<std::string>(),
                naming::orientations::axesSelectionOf(
                    e[keys::lo_convention].get<std::string>()),
                Self::parsedBody_(e[keys::lo_from]),
                Self::parsedBody_(e[keys::lo_to]), corotating);
        }
        isDefault_ = false;
        return *this;
    }

    /** @brief Replace the synodic frame declarations from their json form:
     * an array of ``{"name", "from", "to"}`` objects (from/to are body
     * names or NAIF ids; RTN + co-rotation are implied). */
    Self& setSynodic(const json& arr)
    {
        PARAHPOP_ASSERT(arr.is_array(),
            "Orientation 'synodic' must be a json array of "
            "{\"name\", \"from\", \"to\"} objects");
        synodic_.clear();
        for (const json& e : arr) {
            PARAHPOP_ASSERT(e.is_object() && e.contains(keys::syn_name)
                    && e.contains(keys::syn_from) && e.contains(keys::syn_to),
                "Each 'synodic' entry must be an object with keys "
                "[name, from, to]; got: " + e.dump());
            this->addSynodic(e[keys::syn_name].get<std::string>(),
                Self::parsedBody_(e[keys::syn_from]),
                Self::parsedBody_(e[keys::syn_to]));
        }
        isDefault_ = false;
        return *this;
    }

    /** @brief The local-orbital frame declarations in their json form
     * (from/to re-emitted as canonical body names when known). */
    json localOrbitalJson() const
    {
        json arr = json::array();
        for (const LocalOrbitalDecl& d : localOrbital_) {
            json e;
            e[keys::lo_name]       = d.name;
            e[keys::lo_convention] = naming::orientations::nameOfAxesSelection(
                d.convention);
            e[keys::lo_from] = Self::bodyName_(d.from);
            e[keys::lo_to]   = Self::bodyName_(d.to);
            if (d.corotating)
                e[keys::lo_corotating] = true;
            arr.push_back(e);
        }
        return arr;
    }

    /** @brief The synodic frame declarations in their json form (from/to
     * re-emitted as canonical body names when known). */
    json synodicJson() const
    {
        json arr = json::array();
        for (const SynodicDecl& d : synodic_) {
            json e;
            e[keys::syn_name] = d.name;
            e[keys::syn_from] = Self::bodyName_(d.from);
            e[keys::syn_to]   = Self::bodyName_(d.to);
            arr.push_back(e);
        }
        return arr;
    }

    /** @brief Serialise to JSON (inverse of Orientations(json)). The
     * plain file-array form is kept whenever no frames are declared, so
     * default output stays unchanged. */
    json to_json() const
    {
        if (only_.empty() && localOrbital_.empty() && synodic_.empty())
            return json(files_);
        json j;
        j[keys::files] = files_;
        if (!only_.empty())
            j[keys::only] = this->onlyJson();
        if (!localOrbital_.empty())
            j[keys::localorbital] = this->localOrbitalJson();
        if (!synodic_.empty())
            j[keys::synodic] = this->synodicJson();
        return j;
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
        if (!files_.empty()) {
            isDefault_ = false;
            return;
        }
    }

    /** @brief Whether the given file is among the declared ones */
    bool hasFile_(const std::string& file) const
    {
        return std::find(files_.begin(), files_.end(), file) != files_.end();
    }

    /** @brief Whether a frame of the given (normalized) name is already
     * declared */
    bool hasDecl_(const std::string& name) const
    {
        for (const FrameDecl& decl : only_)
            if (naming::normalized(decl.name) == naming::normalized(name))
                return true;
        return false;
    }

    /** @brief Store a frame declaration, collapsing duplicates */
    void storeDecl_(const FrameDecl& decl)
    {
        if (!this->hasDecl_(decl.name))
            only_.push_back(decl);
    }

    /** @brief Whether a local-orbital frame of the given (normalized) name
     * is already declared */
    bool hasLocalOrbitalDecl_(const std::string& name) const
    {
        for (const LocalOrbitalDecl& d : localOrbital_)
            if (naming::normalized(d.name) == naming::normalized(name))
                return true;
        return false;
    }

    /** @brief Whether a synodic frame of the given (normalized) name is
     * already declared */
    bool hasSynodicDecl_(const std::string& name) const
    {
        for (const SynodicDecl& d : synodic_)
            if (naming::normalized(d.name) == naming::normalized(name))
                return true;
        return false;
    }

    /** @brief Resolve a json body spec — a body name (string) or a NAIF id
     * (integer) — to its NAIF id. Used by the local_orbital / synodic JSON
     * parsers so `from`/`to` accept names or ids interchangeably. */
    static NaifId parsedBody_(const json& j)
    {
        if (j.is_number_integer())
            return static_cast<NaifId>(j.get<NaifId>());
        if (j.is_string())
            return brie::gravity::Parser::parsedNaifId(j.get<std::string>());
        PARAHPOP_THROW(std::runtime_error,
            "Local-orbital / synodic 'from'/'to' must be a body name "
            "(string) or a NAIF id (integer); got: "
                + j.dump());
    }

    /** @brief Canonical body name of a NAIF id (the integer, as a string,
     * when the gravity parser does not know it) */
    static std::string bodyName_(const NaifId& id)
    {
        std::string name;
        if (brie::gravity::Parser::tryParsedName(id, name))
            return name;
        return std::to_string(id);
    }

    /** @brief Resolve the file-reference naming declarations pointing at
     * the given file against its (individually loaded) rotation unit: the
     * declared name takes the file's single not-yet-registered frame
     * target as its id and registers in the process-global registry.
     * Re-declaring an already-registered name+target is a no-op, so
     * configs round-trip and environments rebuild idempotently. */
    void resolveNamedFramesIn_(const std::string& file, const RotationsT& unit)
    {
        for (FrameDecl& decl : only_) {
            if (!decl.named() || decl.file != file)
                continue;

            auto& registry  = RegistryT::instance();
            const auto mref = unit.metadata().hostRef();

            /* candidate frame targets: Type-1/Type-3 units (Type-2
             * barycenter NP units are correction providers, not frames)
             * whose target no registered or derivable frame claims */
            std::vector<idx_t> candidates;
            idx_t chosen    = mref.size();
            bool idempotent = false;
            std::string takenBy;
            for (idx_t i = 0; i < mref.size(); i++) {
                const int type  = mref.getUnitType(i);
                const NaifId id = mref.getTarget(i);
                if (type != 1 && type != 3)
                    continue;
                const auto existing = registry.findByTarget(
                    static_cast<OrientationId>(id));
                if (existing.has_value()) {
                    if (naming::normalized(existing->name)
                        == naming::normalized(decl.name)) {
                        /* re-declaration of an earlier make()/run */
                        chosen     = i;
                        idempotent = true;
                        break;
                    }
                    takenBy += (takenBy.empty() ? "" : ", ")
                        + existing->name;
                    continue;
                }
                candidates.push_back(i);
            }

            if (!idempotent) {
                PARAHPOP_ASSERT(!candidates.empty(),
                    "Orientation frame '" + decl.name + "': every frame "
                    "defined by '" + file + "' is already registered ("
                        + takenBy + "); reference it by its existing name "
                          "instead of declaring a new one");
                PARAHPOP_ASSERT(candidates.size() == 1,
                    "Orientation frame '" + decl.name + "': '" + file
                        + "' defines " + std::to_string(candidates.size())
                        + " not-yet-registered frames; split the file, or "
                          "load it plainly and use the auto-synthesized "
                          "FRAME_<id> names");
                chosen = candidates[0];
            }

            const int type  = mref.getUnitType(chosen);
            const NaifId id = mref.getTarget(chosen);
            EntryT e;
            e.id   = static_cast<OrientationId>(id);
            e.name = decl.name;
            if (type == 3) {
                e.spiceId = id; /* Type-3 targets ARE SPICE frame ids */
                e.kind    = naming::orientations::Kind::BinaryFrame;
                /* body unknown from binary data alone -> body stays 0 */
            } else {
                e.kind = naming::orientations::Kind::BodyFixed;
                e.body = id; /* a Type-1 unit IS its body's IAU frame */
            }
            e.convertible = true;
            registry.add(e); /* identical re-add is a no-op */
            decl.id = e.id;
        }
    }

    /** @brief Resolve the declared frames into the rotation-data targets
     * to filter the loaded unit by. Empty declaration → empty result →
     * load everything (current behaviour). Inertial frames carry no
     * rotation data by construction and are skipped; a declared frame
     * whose target the loaded files do not cover throws (the barycenter
     * nutation-precession parents are auto-included downstream by
     * `makeFullSubset`). */
    feta::scalar::Array<NaifId> filterTargets_(const RotationsT& full) const
    {
        std::vector<NaifId> targets;
        const auto mref = full.metadata().hostRef();
        for (const FrameDecl& decl : only_) {
            const auto entry = RegistryT::instance().find(decl.id);
            PARAHPOP_ASSERT(entry.has_value(),
                "Orientation frame '" + decl.name
                    + "' is not a registered orientation frame");
            if (entry->kind == naming::orientations::Kind::Inertial)
                continue;
            const NaifId target = static_cast<NaifId>(entry->dataTarget());
            PARAHPOP_ASSERT(mref.hasBody(target),
                "Orientation frame '" + decl.name + "' (rotation target "
                    + std::to_string(target)
                    + ") has no rotation data in the declared files "
                    + filestring_(json(files_)));
            if (std::find(targets.begin(), targets.end(), target)
                == targets.end())
                targets.push_back(target);
        }

        feta::scalar::Array<NaifId> out(
            static_cast<idx_t>(targets.size()));
        if (targets.size() != 0) {
            auto oref = out.hostRef();
            for (idx_t i = 0; i < static_cast<idx_t>(targets.size()); i++)
                oref[i] = targets[i];
        }
        return out;
    }

    /** @brief Register every target of the loaded rotation unit in the
     * process-global orientation registry: Type-1 (IAU) body units as
     * `<BODY>_BODY_FIXED` keyed by the body NAIF id (alias `IAU_<BODY>`),
     * Type-3 binary frame units keyed by their SPICE frame class id.
     * Type-2 barycenter units are internal nutation-precession providers,
     * never directly queryable frames, and are skipped. Targets already
     * claimed by a registered or derivable frame (builtins, derived
     * body-fixed names, ITRF-style target mappings, or a re-load of the
     * same files) are left untouched, so only parser-unknown targets get
     * the auto-synthesized FRAME_<id> name. Configuration-time host loop
     * only. */
    void registerLoadedTargets_() const
    {
        const auto mref = rotationdata_.metadata().hostRef();
        auto& registry  = RegistryT::instance();
        for (idx_t i = 0; i < mref.size(); i++) {
            const int type  = mref.getUnitType(i);
            const NaifId id = mref.getTarget(i);
            if (type != 1 && type != 3)
                continue; /* barycenter NP units are not frames */
            if (registry.findByTarget(static_cast<OrientationId>(id))
                    .has_value())
                continue;

            EntryT e;
            e.id   = static_cast<OrientationId>(id);
            e.name = "FRAME_" + std::to_string(id);
            if (type == 1) {
                e.kind = naming::orientations::Kind::BodyFixed;
                e.body = id; /* a Type-1 unit IS its body's IAU frame */
            } else {
                e.spiceId = id; /* Type-3 targets ARE SPICE frame ids */
                e.kind    = naming::orientations::Kind::BinaryFrame;
                /* body unknown from binary data alone -> body stays 0, so
                 * auto-synthesized FRAME_<id> frames are never auto-best */
            }
            e.convertible = true;
            registry.add(e);
        }
    }

    RotationsT rotationdata_ = RotationsT::empty();
    bool rotloaded_          = false;
    bool isDefault_          = true;
    FilesT files_;
    PathsT paths_  = util::paths::paraHPOPDefaultPath();
    bool showInfo_ = false;
    /* Frame declarations (the load-time subset filter): registry
     * write-through happens eagerly for plain entries, at make() for
     * file-reference naming entries */
    std::vector<FrameDecl> only_;
    /* Dynamic local-orbital frame declarations: registered eagerly (they
     * need no `.brot` data; their id is auto-assigned) */
    std::vector<LocalOrbitalDecl> localOrbital_;
    /* Synodic (co-rotating RTN) frame declarations: registered eagerly as
     * co-rotating RTN dynamic frames; kept in a dedicated list so they
     * round-trip under the first-class `synodic` key */
    std::vector<SynodicDecl> synodic_;
};

} // namespace environment
} // namespace model
} // namespace config
} // namespace interface
