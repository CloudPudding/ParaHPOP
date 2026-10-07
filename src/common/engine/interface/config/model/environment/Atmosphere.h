#pragma once

#include <algorithm>
#include <string>
#include <vector>
#include "interface/typedefs.h"
#include "interface/util.h"
#include "interface/config/detail/FieldCodec.h"

namespace interface {
namespace config {
namespace model {
namespace environment {

/* name map and valid fields for atmosphere parsing */
namespace atmosphere {

/** @brief Available exponential-segment fields.
 *
 *  These are the fields valid INSIDE one exponential segment block (e.g.
 *  inside ``{"exponential": {...}}`` or each element of
 *  ``{"exponential": [{...}, ...]}``).  The outer ``atmosphere`` block is a
 *  single-key dispatch dict whose key names the model; the per-body
 *  identity is carried by the surrounding ``bodies.<name>`` scope. */
enum Fields {
    RHO0,        /* reference density [kg/km^3] */
    H0,          /* base/reference altitude [km] (also the band selector) */
    SCALEHEIGHT, /* scale height [km] */
    HCUTOFF,     /* drag cutoff altitude [km] */
    /* Leave the following item as last, as it only acts as enum size/validity
       checker */
    INVALID
};

using Map = std::map<std::string, Fields>;

/** @brief `std::map` to map the atmosphere fields to the corresponding enum
 */
const Map NameMap = {
    /* All lower case */
    { "rho0", RHO0 }, { "h0", H0 },
    { "scaleheight", SCALEHEIGHT }, { "hcutoff", HCUTOFF }
};

/** @brief Atmosphere fields parser */
inline Fields parse(std::string name)
{
    return interface::config::detail::parseField(
        std::move(name), NameMap, INVALID);
}

/** @brief One exponential segment: a self-contained
 *  ``{rho0, h0, scaleHeight, hCutoff}`` block.  Within the band that this
 *  segment owns, density is ``rho0 * exp(-(h - h0) / scaleHeight)``,
 *  suppressed to 0 above ``hCutoff``.  ``h0`` doubles as the band's base
 *  altitude — segments are sorted ascending by ``h0`` at config time and
 *  the band for altitude ``h`` is the one with the largest ``h0 <= h``. */
struct Segment {
    double rho0        = 0.0;
    double h0          = 0.0;
    double scaleHeight = 1.0;
    double hCutoff     = 0.0;
};

} // namespace atmosphere

/** @brief Per-body atmosphere config — mirrors `SphericalHarmonics` for
 *  the drag feature.
 *
 *  Holds a *piecewise* exponential profile: an ordered list of
 *  ``{rho0, h0, scaleHeight, hCutoff}`` segments.  A single-block
 *  atmosphere is the degenerate 1-segment case.  The JSON accepts either a
 *  single object (one block) or a list of objects (piecewise); the backend
 *  always stores the list form, sorted ascending by ``h0``.
 *
 *  Architectural note: this header lives in the ``interface::config`` layer
 *  and **must not** include any paraHPOP header.  It stores the segments as
 *  plain ``double`` fields and exposes them through getters.  The paraHPOP
 *  side (``Environment::extractAtmospheres_``) reads these and builds the
 *  device ``PiecewiseExponentialAtmosphere`` itself; ``double → Real``
 *  happens on the paraHPOP side.  Same direction as ``SphericalHarmonics``.
 *
 *  The oblate-ellipsoid ``flattening`` used for geodetic altitude is a
 *  *body* property (``bodies.<name>.flattening``), not an atmosphere
 *  parameter — see ``body::Config::flattening``.
 *
 *  Unit convention: ``rho0`` is in ``kg/km^3`` (= ρ_SI × 1e9); the
 *  altitudes (``h0``, ``scaleHeight``, ``hCutoff``) are in km. */
class Atmosphere {
    using Self = Atmosphere;

public:
    /** @brief Atmosphere implementation selected by the dispatch key. */
    enum class Model { EXPONENTIAL, NRLMSISE00 };

    /** @brief One exponential segment (re-exported). */
    using Segment = atmosphere::Segment;

    /** @brief Default constructor leaves the model in the default (no
     *  segments, zero-density) state. */
    Atmosphere() = default;

    /** @brief Construct from JSON.
     *
     *  Schema (single-key model-dispatch dict).  The ``exponential`` value
     *  is EITHER a single block object OR a list of block objects:
     *  @code
     *  {"exponential": {"rho0": ..., "h0": ..., "scaleHeight": ...,
     *                   "hCutoff": ...}}
     *  {"exponential": [{...}, {...}, ...]}   // piecewise
     *  @endcode
     *
     *  Each block carries the four density parameters (``rho0`` in
     *  ``kg/km^3``, ``h0``/``scaleHeight``/``hCutoff`` in ``km``).  The
     *  oblate-ellipsoid ``flattening`` is a *body* property
     *  (``bodies.<name>.flattening``), not an atmosphere parameter.
     *
     *  The per-body identity is carried by the surrounding
     *  ``bodies.<name>`` scope, not by a ``body`` field inside the block. */
    Atmosphere(const json& j) { *this = j; }

    /** @brief Copy constructor */
    Atmosphere(const Self& other) = default;

    /** @brief Move constructor */
    Atmosphere(Self&& other) = default;

    /** @brief Copy assignment */
    Self& operator=(const Self& other) = default;

    /** @brief Move assignment */
    Self& operator=(Self&& other) noexcept = default;

    /** @brief JSON assignment.  Expects single-key model-dispatch dict
     *  (e.g. ``{"exponential": <object-or-list>}``).  The inner value
     *  carries one block or a list of blocks; the outer key selects the
     *  parser. */
    Self& operator=(const json& j)
    {
        /* not default anymore */
        isDefault_ = false;

        PARAHPOP_ASSERT(j.is_object() && j.size() == 1,
            "Atmosphere block must be a single-key dispatch dict naming "
            "the model.  Example: "
            "{\"exponential\": {\"rho0\": ..., \"h0\": ..., "
            "\"scaleHeight\": ..., \"hCutoff\": ...}} (or a list of such "
            "blocks for a piecewise profile).  Got: " + j.dump());

        const auto modelIt   = j.begin();
        std::string modelKey = modelIt.key();
        brie::util::Strings::lowerCase(modelKey);

        if (modelKey == "exponential") {
            model_ = Model::EXPONENTIAL;
            parseExponential_(modelIt.value());
        } else if (modelKey == "nrlmsise00" || modelKey == "nrlmsise-00") {
            model_ = Model::NRLMSISE00;
            parseNrlmsise00_(modelIt.value());
        } else {
            PARAHPOP_THROW(std::runtime_error,
                "Unknown atmosphere model: '" + modelIt.key() +
                "'.  Supported: exponential, nrlmsise00.");
        }

        return *this;
    }

    /** @brief Body setter */
    Self& body(const NaifId& b)
    {
        body_      = brie::gravity::Parser::parsedNaifId(b);
        isDefault_ = false;
        return *this;
    }
    Self& body(const std::string& b)
    {
        body_      = brie::gravity::Parser::parsedNaifId(b);
        isDefault_ = false;
        return *this;
    }

    /** @brief Get the body */
    const NaifId& body() const { return body_; }
    Model model() const { return model_; }
    bool isNrlmsise00() const { return model_ == Model::NRLMSISE00; }
    const std::string& spaceWeatherFile() const { return spaceWeatherFile_; }
    double coverageStartMjdUtc() const { return coverageStartMjdUtc_; }
    double coverageEndMjdUtc() const { return coverageEndMjdUtc_; }
    double nrlHCutoff() const { return nrlHCutoff_; }
    double densityScale() const { return densityScale_; }
    /** @brief Get the body name */
    std::string bodyName() const
    {
        return brie::gravity::Parser::parsedName(body_);
    }

    /** @brief Single-block chainable setters — address segment 0 (the
     *  first/lowest band), creating it if the profile is empty.  These are
     *  the single-block ergonomic API; they are ill-defined on a piecewise
     *  profile and **throw** when there is more than one segment (use
     *  ``segment(i)`` for a specific band, or ``segments([...])`` to set the
     *  whole profile). */
    Self& rho0(const double& v)
    {
        assertSingleBlock_();
        ensureSeg0_().rho0 = v;
        isDefault_         = false;
        return *this;
    }
    Self& h0(const double& v)
    {
        assertSingleBlock_();
        ensureSeg0_().h0 = v;
        isDefault_       = false;
        return *this;
    }
    Self& scaleHeight(const double& v)
    {
        assertSingleBlock_();
        ensureSeg0_().scaleHeight = v;
        isDefault_                = false;
        return *this;
    }
    Self& hCutoff(const double& v)
    {
        assertSingleBlock_();
        ensureSeg0_().hCutoff = v;
        isDefault_            = false;
        return *this;
    }

    /** @brief Single-block getters — segment 0's value (a zero-density
     *  default when the profile is empty; the configured baseline is
     *  literally segment 0, with no persistent fallback object).  Return by
     *  value, and **throw** on a piecewise (>1 band) profile — read a
     *  specific band with ``segment(i)``. */
    double rho0() const { assertSingleBlock_(); return seg0Value_().rho0; }
    double h0() const { assertSingleBlock_(); return seg0Value_().h0; }
    double scaleHeight() const
    {
        assertSingleBlock_();
        return seg0Value_().scaleHeight;
    }
    double hCutoff() const { assertSingleBlock_(); return seg0Value_().hCutoff; }

    /** @brief The full ordered segment list (sorted ascending by ``h0``,
     *  read by paraHPOP-side ``extractAtmospheres_``). */
    const std::vector<Segment>& segments() const { return segments_; }

    /** @brief Number of segments (bands). */
    idx_t nSegments() const { return static_cast<idx_t>(segments_.size()); }

    /** @brief Set the segments from a JSON object (one block) or a JSON
     *  list of block objects.  Sorts ascending by ``h0``.  This is the
     *  piecewise programmatic API (used by the Python ``segments([...])``
     *  setter). */
    Self& segments(const json& v)
    {
        parseExponential_(v);
        isDefault_ = false;
        return *this;
    }

    /** @brief Direct access to band ``i`` (bounds-checked).  Backs the
     *  per-segment view (``exponential.segment(i)``); ``i`` indexes the
     *  current (as-built, pre-``finalize``) order. */
    Segment& segmentRef(const idx_t& i)
    {
        assertSegment_(i);
        return segments_[i];
    }
    const Segment& segmentRef(const idx_t& i) const
    {
        assertSegment_(i);
        return segments_[i];
    }

    /** @brief On-the-fly segment mutators (timeline-style; no auto-sort —
     *  the canonical h0-sort + overlap check runs in @ref finalize at
     *  ``Bodies::make()`` time).  ``append`` adds a band at the end,
     *  ``prepend`` at the front, ``insert`` at an explicit position, and
     *  ``remove`` drops one.  Each accepts a typed ``Segment`` or a JSON
     *  block ``{rho0, h0, scaleHeight, hCutoff}``. */
    Self& append(const Segment& s)
    {
        segments_.push_back(s);
        isDefault_ = false;
        return *this;
    }
    Self& append(const json& block)
    {
        Segment s;
        parseSegment_(block, s);
        return append(s);
    }
    Self& prepend(const Segment& s) { return insert(idx_t{ 0 }, s); }
    Self& prepend(const json& block)
    {
        Segment s;
        parseSegment_(block, s);
        return insert(idx_t{ 0 }, s);
    }
    Self& insert(const idx_t& pos, const Segment& s)
    {
        PARAHPOP_ASSERT(pos <= static_cast<idx_t>(segments_.size()),
            "Atmosphere insert position " + std::to_string(pos)
                + " out of range (have " + std::to_string(segments_.size())
                + " segment(s)).");
        segments_.insert(segments_.begin() + pos, s);
        isDefault_ = false;
        return *this;
    }
    Self& insert(const idx_t& pos, const json& block)
    {
        Segment s;
        parseSegment_(block, s);
        return insert(pos, s);
    }
    Self& remove(const idx_t& i)
    {
        assertSegment_(i);
        segments_.erase(segments_.begin() + i);
        return *this;
    }

    /** @brief Canonicalise the profile: sort ascending by ``h0`` and reject
     *  overlapping bands.  Idempotent; run at the end of every JSON parse
     *  and at config-finalize (``Bodies::make()``), so the backend always
     *  sees a sorted, non-overlapping profile regardless of the order in
     *  which segments were added programmatically.
     *
     *  Overlap = the next band's ``h0`` lies *below* the current band's
     *  ``hCutoff`` (the two density laws would claim the same altitudes).
     *  Exact contiguity (``h0_next == hCutoff_prev``) and gaps
     *  (``h0_next > hCutoff_prev``, read as a vacuum shell) are allowed. */
    void finalize()
    {
        std::sort(segments_.begin(), segments_.end(),
            [](const Segment& a, const Segment& b) { return a.h0 < b.h0; });
        for (idx_t i = 1; i < static_cast<idx_t>(segments_.size()); ++i)
            PARAHPOP_ASSERT(segments_[i].h0 >= segments_[i - 1].hCutoff,
                "Overlapping atmosphere segments: a band with h0="
                    + std::to_string(segments_[i].h0)
                    + " km starts below the previous band's hCutoff="
                    + std::to_string(segments_[i - 1].hCutoff)
                    + " km. Bands must be non-overlapping (a gap is allowed "
                      "and reads as vacuum; exact contiguity h0==hCutoff is "
                      "allowed).");
    }

    /** @brief Whether this is a default (no-op) atmosphere */
    bool isDefault() const { return isDefault_; }

    /** @brief Serialise to JSON — symmetric with ``operator=(const json&)``.
     *
     *  Emits ``{"exponential": <object>}`` for a single segment and
     *  ``{"exponential": [<object>, ...]}`` for a piecewise profile, so the
     *  round-trip reproduces the input shape. */
    json to_json() const
    {
        json out;
        if (model_ == Model::NRLMSISE00) {
            json p;
            p["spaceWeatherFile"] = spaceWeatherFile_;
            p["hCutoff"] = nrlHCutoff_;
            p["densityScale"] = densityScale_;
            if (coverageStartMjdUtc_ >= 0.0)
                p["coverageStartMjdUtc"] = coverageStartMjdUtc_;
            if (coverageEndMjdUtc_ >= 0.0)
                p["coverageEndMjdUtc"] = coverageEndMjdUtc_;
            out["nrlmsise00"] = std::move(p);
            return out;
        }
        if (segments_.size() > 1) {
            json arr = json::array();
            for (const auto& s : segments_)
                arr.push_back(segmentJson_(s));
            out["exponential"] = std::move(arr);
        } else {
            out["exponential"] = segmentJson_(seg0Value_());
        }
        return out;
    }

protected:
    /** @brief Parse the NRLMSISE-00 model configuration.
     *
     *  ``coverageStartMjdUtc``/``coverageEndMjdUtc`` are optional absolute
     *  UTC MJD limits.  When supplied, the backend keeps the requested days
     *  plus the four preceding days required by the 3-hour Ap history.
     *  When omitted, the complete space-weather file is loaded. */
    void parseNrlmsise00_(const json& p)
    {
        PARAHPOP_ASSERT(p.is_object(),
            "NRLMSISE-00 atmosphere must be an object.");
        bool foundFile = false;
        for (auto it = p.begin(); it != p.end(); ++it) {
            std::string key = it.key();
            brie::util::Strings::lowerCase(key);
            if (key == "spaceweatherfile") {
                spaceWeatherFile_ = it.value().get<std::string>();
                foundFile = !spaceWeatherFile_.empty();
            } else if (key == "coveragestartmjdutc") {
                coverageStartMjdUtc_ = it.value().get<double>();
            } else if (key == "coverageendmjdutc") {
                coverageEndMjdUtc_ = it.value().get<double>();
            } else if (key == "hcutoff") {
                nrlHCutoff_ = it.value().get<double>();
            } else if (key == "densityscale") {
                densityScale_ = it.value().get<double>();
            } else {
                PARAHPOP_THROW(std::runtime_error,
                    "Invalid NRLMSISE-00 atmosphere field: '" + it.key()
                        + "'. Valid fields are: [spaceWeatherFile, "
                          "coverageStartMjdUtc, coverageEndMjdUtc, hCutoff, "
                          "densityScale].");
            }
        }
        PARAHPOP_ASSERT(foundFile,
            "NRLMSISE-00 spaceWeatherFile must be specified.");
        PARAHPOP_ASSERT(nrlHCutoff_ > 0.0,
            "NRLMSISE-00 hCutoff must be positive.");
        PARAHPOP_ASSERT(densityScale_ > 0.0,
            "NRLMSISE-00 densityScale must be positive.");
        PARAHPOP_ASSERT(coverageStartMjdUtc_ < 0.0
                || coverageEndMjdUtc_ < 0.0
                || coverageEndMjdUtc_ >= coverageStartMjdUtc_,
            "NRLMSISE-00 coverageEndMjdUtc must not precede the start.");
        segments_.clear();
    }

    /** @brief Parse the ``exponential`` value: a single object (one
     *  segment) or a list of objects (piecewise).  Sorts ascending by
     *  ``h0`` and replaces the stored segments. */
    void parseExponential_(const json& v)
    {
        model_ = Model::EXPONENTIAL;
        std::vector<Segment> segs;
        if (v.is_array()) {
            PARAHPOP_ASSERT(!v.empty(),
                "Piecewise atmosphere list must contain at least one block.");
            for (const auto& el : v) {
                Segment s;
                parseSegment_(el, s);
                segs.push_back(s);
            }
        } else if (v.is_object()) {
            Segment s;
            parseSegment_(v, s);
            segs.push_back(s);
        } else {
            PARAHPOP_THROW(std::runtime_error,
                "Exponential atmosphere must be an object (single block) or "
                "a list of objects (piecewise).  Got: " + v.dump());
        }

        /* Replace the stored profile, then canonicalise (sort ascending by
         * h0 + reject overlapping bands) — the backend and the device band
         * selector require sorted, non-overlapping segments. */
        segments_ = std::move(segs);
        finalize();
    }

    /** @brief Parse one ``{rho0, h0, scaleHeight, hCutoff}`` block.  Strict:
     *  rejects unknown keys; all four parameters are required. */
    void parseSegment_(const json& p, Segment& out)
    {
        using namespace atmosphere;
        using JIT = json::const_iterator;

        PARAHPOP_ASSERT(p.is_object(),
            "Each exponential atmosphere block must be an object.");

        bool foundRho0 = false, foundH0 = false, foundScale = false,
             foundCutoff = false;

        for (JIT jit = p.begin(); jit != p.end(); jit++) {
            std::string field        = jit.key();
            atmosphere::Fields FIELD = atmosphere::parse(field);
            PARAHPOP_ASSERT(FIELD != atmosphere::INVALID,
                Self::ErrorMessage_(field));

            if (FIELD == RHO0) {
                foundRho0 = true;
                out.rho0  = jit.value().get<double>();
            } else if (FIELD == H0) {
                foundH0 = true;
                out.h0  = jit.value().get<double>();
            } else if (FIELD == SCALEHEIGHT) {
                foundScale      = true;
                out.scaleHeight = jit.value().get<double>();
            } else if (FIELD == HCUTOFF) {
                foundCutoff = true;
                out.hCutoff = jit.value().get<double>();
            }
        }

        PARAHPOP_ASSERT(foundRho0,  "Atmosphere rho0 not specified.");
        PARAHPOP_ASSERT(foundH0,    "Atmosphere h0 not specified.");
        PARAHPOP_ASSERT(foundScale, "Atmosphere scaleHeight not specified.");
        PARAHPOP_ASSERT(foundCutoff,"Atmosphere hCutoff not specified.");

        PARAHPOP_ASSERT(out.rho0 >= 0.0,
            "Atmosphere rho0 must be non-negative.");
        PARAHPOP_ASSERT(out.scaleHeight > 0.0,
            "Atmosphere scaleHeight must be positive.");
    }

    /** @brief Serialise one segment block. */
    static json segmentJson_(const Segment& s)
    {
        json o;
        o["rho0"]        = s.rho0;
        o["h0"]          = s.h0;
        o["scaleHeight"] = s.scaleHeight;
        o["hCutoff"]     = s.hCutoff;
        return o;
    }

    /** @brief Create a string with the error message */
    static std::string ErrorMessage_(std::string field)
    {
        std::stringstream ss;
        ss << "Invalid atmosphere field: '" << field;
        ss << "'. Valid fields are: [";
        idx_t it      = 0;
        auto nameiter = atmosphere::NameMap.begin();
        while (it < static_cast<idx_t>(atmosphere::INVALID)) {
            ss << nameiter->first;
            if (it < static_cast<idx_t>(atmosphere::INVALID) - 1)
                ss << ", ";
            it++;
            nameiter++;
        }
        ss << "] (case insensitive)";
        return ss.str();
    }

    /** @brief Segment 0, creating it if the profile is empty (for the
     *  single-block setters). */
    Segment& ensureSeg0_()
    {
        model_ = Model::EXPONENTIAL;
        if (segments_.empty())
            segments_.emplace_back();
        return segments_[0];
    }

    /** @brief Segment 0 by value; a zero-density default block when the
     *  profile is empty.  Used by the single-block getters + ``to_json`` so
     *  there is no persistent fallback object — the configured baseline is
     *  literally segment 0. */
    Segment seg0Value_() const
    {
        return segments_.empty() ? Segment{} : segments_.front();
    }

    /** @brief Guard for the single-block ergonomic accessors: they address
     *  segment 0 and are ill-defined on a piecewise (>1 band) profile. */
    void assertSingleBlock_() const
    {
        PARAHPOP_ASSERT(segments_.size() <= 1,
            "Single-block exponential accessor used on a piecewise "
            "atmosphere with " + std::to_string(segments_.size())
                + " segments. Address a specific band with .segment(i), or "
                  "set the whole profile with .segments([...]).");
    }

    /** @brief Bounds check for the per-segment accessors / ``remove``. */
    void assertSegment_(const idx_t& i) const
    {
        PARAHPOP_ASSERT(i < static_cast<idx_t>(segments_.size()),
            "Atmosphere segment index " + std::to_string(i)
                + " out of range (have " + std::to_string(segments_.size())
                + " segment(s)).");
    }

    NaifId body_ = 0;
    Model model_ = Model::EXPONENTIAL;
    std::vector<Segment> segments_;
    std::string spaceWeatherFile_;
    double coverageStartMjdUtc_ = -1.0;
    double coverageEndMjdUtc_ = -1.0;
    double nrlHCutoff_ = 1000.0;
    double densityScale_ = 1.0;
    bool isDefault_ = true;
};

} // namespace environment
} // namespace model
} // namespace config
} // namespace interface
