#pragma once

#include <algorithm>
#include <map>
#include <optional>
#include <mutex>
#include <string>
#include <vector>

#include "brie/frames/Rotation.h"

#include "interface/naming/bands.h"
#include "interface/naming/normalize.h"
#include "interface/util/err.h"

namespace interface {
namespace naming {
namespace orientations {

/** @brief Orientation frame identifier. A namespace of its own, separate
 * from NAIF body/point ids: 0 is ICRF by hard requirement, body-fixed
 * frames carry their body's NAIF id (EARTH_BODY_FIXED = 399), and binary
 * (.brot Type-3) frames carry their SPICE frame class id (MOON_PA =
 * 31008). Users never invent ids: builtin and derived frames carry
 * pre-defined ones, custom frames take theirs from the `.brot` file that
 * defines them, and any future non-file-backed custom frame will be
 * auto-assigned from the reserved `bands::dynamicorientation` pool. */
using OrientationId = int;

/** @brief The ICRF identity frame: every orientation-aware API treats it
 * as the no-op default */
constexpr OrientationId ICRF = 0;

/** @brief The ECLIPJ2000 inertial frame: a constant rotational offset
 * (ecliptic bias) from ICRF, needing no rotation data to convert */
constexpr OrientationId ECLIPJ2000 = 17;

/** @brief Orientation frame kinds. `Dynamic` frames are local-orbital
 * frames whose axes are computed at evaluation time from the ephemeris
 * state of a `body` (target) relative to a `referenceBody` (centre) under
 * an axis `convention`; they carry no `.brot` rotation data and take their
 * ids from the reserved `bands::dynamicorientation` pool. */
enum class Kind { Inertial, BodyFixed, BinaryFrame, Dynamic };

/** @brief Canonical frame-name token of a body name: upper case with
 * non-alphanumerics turned into underscores ("Jupiter Barycenter" ->
 * "JUPITER_BARYCENTER") */
inline std::string frameToken(const std::string& body)
{
    std::string out = body;
    for (auto& c : out) {
        if (c >= 'a' && c <= 'z')
            c = static_cast<char>(c - 'a' + 'A');
        else if (!((c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')))
            c = '_';
    }
    return out;
}

/** @brief Canonical token of a local-orbital axis convention. */
inline std::string nameOfAxesSelection(const brie::frames::AxesSelection& s)
{
    using brie::frames::AxesSelection;
    switch (s) {
        case AxesSelection::RTN: return "RTN";
        case AxesSelection::RSW: return "RSW";
        case AxesSelection::TNW: return "TNW";
        case AxesSelection::VNB: return "VNB";
        case AxesSelection::LVLH: return "LVLH";
    }
    return "RTN";
}

/** @brief Parse a local-orbital axis-convention token into the brie
 * AxesSelection enum; throws on an unknown token. Both the short
 * abbreviations (RTN, RSW, TNW, VNB, LVLH and the RIC/VNC synonyms) and
 * the fully-spelled long forms (radial_transverse_normal, ...) are
 * accepted; matching is case / whitespace / underscore insensitive (the
 * token is upper-cased and every separator is squashed out before the
 * compare). The canonical emit (`nameOfAxesSelection`) is always the
 * short abbreviation. */
inline brie::frames::AxesSelection axesSelectionOf(const std::string& name)
{
    using brie::frames::AxesSelection;
    /* squash to a separator-free upper-cased token so "radial transverse
     * normal", "Radial_Transverse_Normal" and "RTN" all collapse alike */
    std::string k = frameToken(name); /* upper-cased, separators -> '_' */
    k.erase(std::remove(k.begin(), k.end(), '_'), k.end());
    if (k == "RTN" || k == "RIC" || k == "RADIALTRANSVERSENORMAL"
        || k == "RADIALINTRACKCROSSTRACK")
        return AxesSelection::RTN;
    if (k == "RSW" || k == "RADIALALONGTRACKCROSSTRACK")
        return AxesSelection::RSW;
    if (k == "TNW" || k == "TANGENTIALNORMALCROSSTRACK")
        return AxesSelection::TNW;
    if (k == "VNB" || k == "VNC" || k == "VELOCITYNORMALBINORMAL"
        || k == "VELOCITYNORMALCROSSTRACK")
        return AxesSelection::VNB;
    if (k == "LVLH" || k == "LOCALVERTICALLOCALHORIZONTAL")
        return AxesSelection::LVLH;
    PARAHPOP_THROW(std::runtime_error,
        "'" + name
            + "' is not a known local-orbital axis convention. Valid "
              "conventions are: RTN (=RIC = radial_transverse_normal), RSW "
              "(= radial_alongtrack_crosstrack), TNW (= tangential_normal_"
              "crosstrack), VNB (=VNC = velocity_normal_binormal), LVLH (= "
              "local_vertical_local_horizontal) — case / whitespace / "
              "underscore insensitive");
}

/** @brief A registered orientation frame: a unique id <-> canonical name
 * mapping plus aliases, the bidirectional SPICE frame-id anchor (0 = no
 * SPICE equivalent declared), whether the frame is convertible by the
 * kinematic conversion path (translate-only frames are namable everywhere
 * but rejected at conversion), and the rotation-data target — the `.brot`
 * frame-class id the frame's axes are evaluated under (0 = same as the
 * frame id, the overwhelmingly common case; ITRF93 maps id 13000 onto
 * the Earth high-precision binary-PCK target 3000). */
struct Entry {
    OrientationId id = ICRF;
    std::string name;
    std::vector<std::string> aliases;
    int spiceId          = 0;
    Kind kind            = Kind::BodyFixed;
    bool convertible     = true;
    OrientationId target = 0;

    /** @brief The NAIF body whose orientation this frame describes (0 =
     * none / inertial). Body-fixed and binary frames carry it so the
     * per-body auto-best selector can rank the frames available for a body;
     * inertial frames (ICRF, ECLIPJ2000) leave it 0. Auto-registered
     * `FRAME_<id>` entries from loaded `.brot` data also leave it 0 — no
     * body association is knowable from the data alone, so they are
     * selectable only by explicit name, never picked automatically. */
    NaifId body = 0;

    /** @brief Dynamic (local-orbital) frames only: the reference ("from"
     * / centre) body whose state relative to `body` (the "to" / target)
     * defines the frame at each epoch, and the axis convention. Other
     * kinds leave these at their defaults (referenceBody 0, RTN). */
    NaifId referenceBody                  = 0;
    brie::frames::AxesSelection convention = brie::frames::AxesSelection::RTN;

    /** @brief Dynamic (local-orbital) frames only: when true the frame
     * CO-ROTATES with its reference orbit, so a velocity converted into it
     * is the velocity relative to the rotating frame (v' = R(v - omega x r),
     * omega = (r x v)/|r|^2). When false (the default) the frame provides the
     * INSTANTANEOUS local-orbital axes at the epoch and a converted velocity
     * is the inertial velocity merely re-expressed on those axes (v' = R v).
     * Other kinds leave this false. */
    bool corotating = false;

    /** @brief The rotation-data target this frame evaluates under */
    OrientationId dataTarget() const { return target == 0 ? id : target; }
};

/** @brief Process-global, mutex-guarded orientation frame registry,
 * builtin-seeded with the ICRF/body-fixed/binary compat table. Fed by
 * `environment::Orientations` (custom frame declarations + auto-registered
 * `.brot` targets): identical re-registration is a no-op, conflicting
 * redefinition throws, the registry never shrinks. All lookups happen at
 * configuration time, never on the propagation hot path.
 *
 * On top of the explicit entries, the registry resolves DERIVED body-fixed
 * frames: `<BODY>_BODY_FIXED` / `IAU_<BODY>` (and the matching id) answers
 * for every body the gravity parser knows, without any data loaded.
 * Derived lookups are pure — the synthesized entry is returned by value
 * and never stored, so `size()` keeps counting explicit registrations
 * only, and an explicit entry always wins over a derived one.
 *
 * Invariant: every orientation target loaded into an environment is
 * registered or derivable — `Orientations::load()` derives registrations
 * from the loaded `.brot` content. */
class Registry {
    using Self = Registry;

public:
    /** @brief Access the process-global instance.
     *
     * Defined OUT-OF-LINE in `paraHPOP/paraHPOP/util/naming.cu` (compiled
     * into `libparaHPOP_models.so`; Stage-1 test binaries compile the TU
     * directly) for the same cross-DSO reason as the point registry: a
     * header-local static would be duplicated per Python extension
     * module. */
    static Registry& instance();

    /* Process-global singleton: no copies, no moves */
    Registry(const Registry&)            = delete;
    Registry& operator=(const Registry&) = delete;

    /** @brief Register the given frame, returning its id. Identical
     * re-registration is a no-op; new aliases for an existing entry are
     * merged in; any conflicting mapping throws. */
    OrientationId add(const Entry& entry)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return this->add_(entry);
    }

    /** @brief Register (idempotently, by name) a dynamic local-orbital
     * orientation frame: its axes are computed at evaluation time from the
     * ephemeris state of @p target relative to @p referenceBody under the
     * given axis @p convention (see brie::frames::AxesSelection). The id is
     * auto-assigned from the reserved `bands::dynamicorientation` pool.
     * @p corotating selects co-rotating (velocity relative to the rotating
     * frame) vs instantaneous-axes (inertial velocity re-expressed) velocity
     * semantics — see Entry::corotating. Re-declaring the same name with a
     * matching (convention, referenceBody, target, corotating) tuple returns
     * the existing id; a mismatch throws. */
    OrientationId addDynamic(const std::string& name,
        const brie::frames::AxesSelection& convention,
        const NaifId& referenceBody, const NaifId& target,
        bool corotating = false)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        PARAHPOP_ASSERT(!name.empty(),
            "Dynamic orientation frame names must not be empty");

        auto nit = byName_.find(normalized(name));
        if (nit != byName_.end()) {
            const Entry& e = byId_.at(nit->second);
            PARAHPOP_ASSERT(e.kind == Kind::Dynamic && e.convention == convention
                    && e.referenceBody == referenceBody && e.body == target
                    && e.corotating == corotating,
                "Dynamic orientation frame '" + name
                    + "' is already registered with a different definition");
            return e.id;
        }

        PARAHPOP_ASSERT(nextDynamicId_ < bands::dynamicorientation::END,
            "Dynamic-orientation id band exhausted");
        Entry e;
        e.id            = static_cast<OrientationId>(nextDynamicId_++);
        e.name          = name;
        e.kind          = Kind::Dynamic;
        e.convertible   = true;
        e.body          = target;
        e.referenceBody = referenceBody;
        e.convention    = convention;
        e.corotating    = corotating;
        return this->add_(e);
    }

    /** @brief Non-throwing name lookup (canonical name or alias,
     * normalized comparison; derived body-fixed names resolve for every
     * parser-known body). Writes `id` only on success. */
    bool tryResolve(const std::string& name, OrientationId& id) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto entry = this->findByName_(name);
        if (!entry)
            return false;
        id = entry->id;
        return true;
    }

    /** @brief Whether the given id is a registered or derivable frame */
    bool contains(const OrientationId& id) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return this->findById_(id).has_value();
    }

    /** @brief Resolved id of the given name, throwing when it matches no
     * registered or derivable frame */
    OrientationId resolvedId(const std::string& name) const
    {
        OrientationId id = ICRF;
        PARAHPOP_ASSERT(this->tryResolve(name, id),
            "'" + name
                + "' is not a known orientation frame. Builtin frames "
                  "(ICRF, ECLIPJ2000, MOON_PA, ITRF93, ...) and any body's "
                  "<BODY>_BODY_FIXED / IAU_<BODY> resolve without loading; "
                  "custom frames are named via the environment's "
                  "orientations 'frames' declaration");
        return id;
    }

    /** @brief Validating pass-through of the given id (throws when
     * neither registered nor derivable) */
    OrientationId resolvedId(const OrientationId& id) const
    {
        PARAHPOP_ASSERT(this->contains(id),
            "Id " + std::to_string(id)
                + " is not a known orientation frame. Builtin frames "
                  "(ICRF, ECLIPJ2000, MOON_PA, ITRF93, ...) and any body's "
                  "NAIF id resolve without loading; custom frames are "
                  "named via the environment's orientations 'frames' "
                  "declaration");
        return id;
    }

    /** @brief Canonical name of the given registered or derivable id
     * (throws if unknown) */
    std::string nameOf(const OrientationId& id) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto entry = this->findById_(id);
        PARAHPOP_ASSERT(entry.has_value(),
            "Id " + std::to_string(id)
                + " is not a known orientation frame");
        return entry->name;
    }

    /** @brief SPICE frame id of the given registered frame (throws when
     * the frame is unknown or declares no SPICE equivalent) */
    int toSpice(const OrientationId& id) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto entry = this->findById_(id);
        PARAHPOP_ASSERT(entry.has_value(),
            "Id " + std::to_string(id)
                + " is not a known orientation frame");
        PARAHPOP_ASSERT(entry->spiceId != 0,
            "Orientation frame '" + entry->name
                + "' declares no SPICE frame equivalent");
        return entry->spiceId;
    }

    /** @brief Orientation id of the given SPICE frame id, when a
     * registered frame declares it (derived frames carry no SPICE
     * anchor) */
    std::optional<OrientationId> fromSpice(const int& spiceId) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& kv : byId_)
            if (kv.second.spiceId == spiceId)
                return kv.first;
        return std::nullopt;
    }

    /** @brief Copy of the entry registered or derivable under the given
     * id, if any */
    std::optional<Entry> find(const OrientationId& id) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return this->findById_(id);
    }

    /** @brief Copy of the entry matching the given name/alias, if any */
    std::optional<Entry> find(const std::string& name) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return this->findByName_(name);
    }

    /** @brief Copy of the entry whose rotation-data target is the given
     * `.brot` frame-class id, if any. Explicit entries are scanned first
     * (an ITRF-style frame can claim a target differing from its id),
     * then the derived body-fixed table (where target == id). */
    std::optional<Entry> findByTarget(const OrientationId& target) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& kv : byId_)
            if (kv.second.dataTarget() == target)
                return kv.second;
        Entry derived;
        if (this->makeDerived_(target, derived))
            return derived;
        return std::nullopt;
    }

    /** @brief Every explicitly-registered binary (`.brot` Type-3) frame
     * associated with the given NAIF body. The per-body auto-best selector
     * ranks these above the IAU body-fixed default: zero matches keeps the
     * body on its IAU polynomial, exactly one is auto-selected, more than
     * one is ambiguous and forces an explicit `bodies.<name>.orientation`.
     * Derived body-fixed entries (never binary) and auto-registered
     * `FRAME_<id>` entries (body 0) are intentionally excluded. */
    std::vector<Entry> binaryFramesOf(const NaifId& body) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<Entry> out;
        if (body == 0)
            return out;
        for (const auto& kv : byId_)
            if (kv.second.kind == Kind::BinaryFrame
                && kv.second.body == body)
                out.push_back(kv.second);
        return out;
    }

    /** @brief Number of registered frames (builtins included; derived
     * body-fixed frames are synthesized per lookup and never counted) */
    idx_t size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<idx_t>(byId_.size());
    }

    /** @brief Reset to the freshly-seeded builtin state. Test isolation
     * only — production code never shrinks the registry. */
    void clearForTesting()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        byId_.clear();
        byName_.clear();
        nextDynamicId_ = bands::dynamicorientation::BASE;
        this->seedBuiltins_();
    }

private:
    Registry() { this->seedBuiltins_(); }

    /** @brief Unlocked implementation of add() (also used by seeding) */
    OrientationId add_(const Entry& entry)
    {
        PARAHPOP_ASSERT(!entry.name.empty(),
            "Orientation frame names must not be empty");

        const std::string key = normalized(entry.name);
        auto nit              = byName_.find(key);
        if (nit != byName_.end()) {
            PARAHPOP_ASSERT(nit->second == entry.id,
                "Orientation frame '" + entry.name
                    + "' is already registered with id "
                    + std::to_string(nit->second)
                    + "; conflicting redefinition with id "
                    + std::to_string(entry.id));
        }

        auto iit = byId_.find(entry.id);
        if (iit != byId_.end()) {
            /* Existing id: the canonical mapping must agree; aliases merge */
            Entry& existing = iit->second;
            PARAHPOP_ASSERT(normalized(existing.name) == key,
                "Orientation id " + std::to_string(entry.id)
                    + " is registered as '" + existing.name
                    + "'; conflicting redefinition as '" + entry.name + "'");
            PARAHPOP_ASSERT(existing.kind == entry.kind
                    && existing.convertible == entry.convertible
                    && (entry.spiceId == 0
                        || existing.spiceId == entry.spiceId)
                    && (entry.target == 0
                        || existing.dataTarget() == entry.dataTarget())
                    && (entry.body == 0 || existing.body == entry.body),
                "Orientation frame '" + existing.name
                    + "': conflicting redefinition of kind/convertible/"
                      "SPICE-id/target/body");
            /* A later registration may carry the body association a bare
             * earlier one lacked (e.g. data-load stamps a Type-1 unit) */
            if (existing.body == 0)
                existing.body = entry.body;
            for (const auto& alias : entry.aliases)
                this->addAlias_(existing, alias);
        } else {
            Entry fresh = entry;
            byId_[entry.id] = std::move(fresh);
            this->mapName_(entry.name, entry.id);
            for (const auto& alias : entry.aliases)
                this->mapName_(alias, entry.id);
        }
        return entry.id;
    }

    /** @brief Map a normalized name onto an id, throwing on cross-id
     * clash */
    void mapName_(const std::string& name, const OrientationId& id)
    {
        const std::string key = normalized(name);
        auto it               = byName_.find(key);
        if (it != byName_.end()) {
            PARAHPOP_ASSERT(it->second == id,
                "Orientation name/alias '" + name
                    + "' is already registered for '"
                    + byId_.at(it->second).name + "' (id "
                    + std::to_string(it->second) + ")");
            return;
        }
        byName_[key] = id;
    }

    /** @brief Merge a new alias into an existing entry */
    void addAlias_(Entry& existing, const std::string& alias)
    {
        const std::string key = normalized(alias);
        if (byName_.find(key) != byName_.end()) {
            this->mapName_(alias, existing.id); /* validates the clash */
            return;
        }
        byName_[key] = existing.id;
        existing.aliases.push_back(alias);
    }

    /** @brief Unlocked name lookup: explicit entry first, else the
     * derived body-fixed patterns. A pattern hit whose body id is
     * explicitly registered resolves to the explicit entry when (and only
     * when) that entry is itself body-fixed — "TERRA_BODY_FIXED" finds
     * the curated EARTH_BODY_FIXED, while "SSB_BODY_FIXED" must not
     * resolve to ICRF. */
    std::optional<Entry> findByName_(const std::string& name) const
    {
        auto nit = byName_.find(normalized(name));
        if (nit != byName_.end())
            return byId_.at(nit->second);

        OrientationId bodyId = ICRF;
        if (!Self::parseBodyFixedPattern_(name, bodyId))
            return std::nullopt;
        auto iit = byId_.find(bodyId);
        if (iit != byId_.end()) {
            if (iit->second.kind == Kind::BodyFixed)
                return iit->second;
            return std::nullopt;
        }
        Entry derived;
        if (this->makeDerived_(bodyId, derived))
            return derived;
        return std::nullopt;
    }

    /** @brief Unlocked id lookup: explicit entry first, else derived */
    std::optional<Entry> findById_(const OrientationId& id) const
    {
        auto it = byId_.find(id);
        if (it != byId_.end())
            return it->second;
        Entry derived;
        if (this->makeDerived_(id, derived))
            return derived;
        return std::nullopt;
    }

    /** @brief Match `<BODY>_BODY_FIXED` / `IAU_<BODY>` (normalized) and
     * resolve `<BODY>` through the gravity parser */
    static bool parseBodyFixedPattern_(
        const std::string& name, OrientationId& bodyId)
    {
        static const std::string suffix = "bodyfixed";
        static const std::string prefix = "iau";
        const std::string key           = normalized(name);
        std::string token;
        if (key.size() > suffix.size()
            && key.compare(
                   key.size() - suffix.size(), suffix.size(), suffix)
                == 0)
            token = key.substr(0, key.size() - suffix.size());
        else if (key.size() > prefix.size()
            && key.compare(0, prefix.size(), prefix) == 0)
            token = key.substr(prefix.size());
        else
            return false;

        NaifId id = 0;
        if (!brie::gravity::Parser::tryParsedNaifId(token, id))
            return false;
        bodyId = static_cast<OrientationId>(id);
        return true;
    }

    /** @brief Synthesize the derived body-fixed entry of the given body
     * NAIF id: parser-known bodies only, never shadowing an explicit
     * entry. The SSB and the barycenters (ids 0..9) are excluded — their
     * rotation units are nutation-precession correction providers, not
     * frames. */
    bool makeDerived_(const OrientationId& id, Entry& out) const
    {
        if (id >= 0 && id <= 9)
            return false;
        if (byId_.find(id) != byId_.end())
            return false;
        std::string body;
        if (!brie::gravity::Parser::tryParsedName(
                static_cast<NaifId>(id), body))
            return false;
        out             = Entry{};
        out.id          = id;
        out.name        = frameToken(body) + "_BODY_FIXED";
        out.aliases     = { "IAU_" + frameToken(body) };
        out.kind        = Kind::BodyFixed;
        out.convertible = true;
        out.body        = static_cast<NaifId>(id); /* IAU frame of body id */
        return true;
    }

    /** @brief Seed the builtin frames and the SPICE compat table.
     * ITRF93 evaluates under the Earth high-precision binary-PCK target
     * 3000 (`earth_latest_high_prec.brot` and dated/predict variants) —
     * the one builtin whose rotation-data target differs from its id. */
    void seedBuiltins_()
    {
        auto seed = [this](const OrientationId& id, const std::string& name,
                        const std::vector<std::string>& aliases,
                        const int& spiceId, const Kind& kind,
                        const bool& convertible, const NaifId& body) {
            Entry e;
            e.id          = id;
            e.name        = name;
            e.aliases     = aliases;
            e.spiceId     = spiceId;
            e.kind        = kind;
            e.convertible = convertible;
            e.body        = body;
            this->add_(e);
        };
        seed(0, "ICRF", { "J2000", "EME2000" }, 1, Kind::Inertial, true, 0);
        seed(17, "ECLIPJ2000", { "ECLIPTIC", "ECLIPTICJ2000" }, 17,
            Kind::Inertial, true, 0);
        seed(399, "EARTH_BODY_FIXED", { "IAU_EARTH" }, 10013,
            Kind::BodyFixed, true, 399);
        seed(301, "MOON_BODY_FIXED", { "IAU_MOON" }, 10020, Kind::BodyFixed,
            true, 301);
        seed(31008, "MOON_PA", {}, 31008, Kind::BinaryFrame, true, 301);
        {
            Entry e;
            e.id          = 13000;
            e.name        = "ITRF93";
            e.aliases     = { "ITRF", "EARTH_FIXED" };
            e.spiceId     = 13000;
            e.kind        = Kind::BinaryFrame;
            e.convertible = true;
            e.target      = 3000;
            e.body        = 399;
            this->add_(e);
        }
        /* The native IPF→ITRF Earth frame: the same physical ICRF→ITRF
         * rotation GODOT composes from the operational IERS nodes, evaluated
         * on-device from a brie unit_type=4 (IERS2000-model) unit covering
         * 1992–2050 — strictly wider than the Chebyshev ITRF93 kernel (ends
         * Sept 2024). A second Earth binary frame at rotation-data target
         * 3001 (sibling of ITRF93's 3000): loading both the Chebyshev and the
         * native .brot makes Earth's binary-frame set ambiguous, so the
         * resolver forces an explicit `bodies.earth.orientation` choice — the
         * "keep both, select explicitly" contract. No SPICE frame anchor:
         * it is produced from IPF, not a NAIF kernel (spiceId 0). */
        {
            Entry e;
            e.id          = 13001;
            e.name        = "ITRF_IERS2000";
            e.aliases     = { "ITRF_IPF", "ITRF_NATIVE" };
            e.spiceId     = 0;
            e.kind        = Kind::BinaryFrame;
            e.convertible = true;
            e.target      = 3001;
            e.body        = 399;
            this->add_(e);
        }
    }

    mutable std::mutex mutex_;
    std::map<OrientationId, Entry> byId_;
    std::map<std::string, OrientationId> byName_;
    /** @brief Next id handed out to a dynamic local-orbital frame, walking
     * up the reserved `bands::dynamicorientation` pool. */
    NaifId nextDynamicId_ = bands::dynamicorientation::BASE;
};

} // namespace orientations
} // namespace naming
} // namespace interface
