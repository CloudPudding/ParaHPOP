#pragma once

/* Body feature vocabulary and leaf toggle/view widgets.
 * Split out of BodySource.h for readability: the Features enum + name
 * map + parse/helpers, and the per-feature Toggle / FieldToggle /
 * atmosphere view classes.  The owned Config lives in BodyConfig.h; the
 * non-owning array views (Gravity/Body/ConstBody) live in BodyView.h. */

#include "interface/config/model/environment/Atmosphere.h"
#include "interface/config/model/environment/SphericalHarmonics.h"
#include "interface/typedefs.h"
#include "interface/util.h"

namespace interface {
namespace config {
namespace model {
namespace environment {
namespace body {

/** @brief Available body modelling features */
enum Features {

    POINTGRAVITY, /* Gravitational effects of the point mass */
    SHAPE,        /* gravity shape model (full spherical harmonics) */
    OBLATENESS,   /* Equatorial bulge perturbation (J2-only) */
    RADIATION,    /* radiation pressure */
    ATMOSPHERE,   /* body has an atmosphere (drag force on spacecraft) */
    OCCULTING,      /* body casts a shadow: attenuates SRP and is an eligible
                     occulter for eclipse events */

    /* INCLUDE OTHER RELEVANT MODELLING FEATURES HERE */

    /* Leave the following item as last -- it only acts as enum size */
    FEATSIZE,

    /* Invalid body modelling feature */
    INVALID

};

using Map = std::map<std::string, Features>;

/** @brief `std::map` to map the body features to the corresponding enum
 */
const Map NameMap = {
    /* All lower case */
    { "oblateness", OBLATENESS }, { "shape", SHAPE },
    { "radiation", RADIATION }, { "atmosphere", ATMOSPHERE },
    { "occulting", OCCULTING }
};

/** @brief Mutable toggle for activation/deactivation of a
 * single model feature */
class Toggle {
public:
    Toggle(bool& flag)
        : flag_{ flag }
    {
    }

    /** @brief Activate */
    void activate() { flag_ = true; }
    /** @brief Deactivate */
    void deactivate() { flag_ = false; }

    /** @brief Get active state */
    bool active() const { return flag_; }
    /** @brief Set active state */
    Toggle& active(bool v)
    {
        flag_ = v;
        return *this;
    }

private:
    bool& flag_;
};

/** @brief Read-only toggle to inspect the activation status only */
class ConstToggle {
public:
    ConstToggle(const bool& flag)
        : flag_{ flag }
    {
    }

    /** @brief Get active state */
    bool active() const { return flag_; }

private:
    const bool& flag_;
};

namespace single {
using Feature      = Toggle;
using ConstFeature = ConstToggle;
} // namespace single

/* forward declare SphericalHarmonics for field feature */
using SHT = interface::config::model::environment::SphericalHarmonics;

/* forward declare Atmosphere for atmosphere feature */
using AtmT = interface::config::model::environment::Atmosphere;

/* Per-body flattening handle: a feta scalar-array GRef indexed by body
 * index (mirrors the flags `arr_` handle).  Flattening is the body's
 * oblate-ellipsoid shape consumed by the geodetic-altitude solver in the
 * drag path; it is stored in a feta handle array — NOT a raw pointer —
 * for safety and consistency with the flags array. */
using FlatArrT = feta::scalar::Array<double>::GRef;

/** @brief Assert that a body flattening value is in the valid range
 *  ``[0, 1)`` (0 = sphere). */
static inline void assertFlatteningRange_(const double& v)
{
    PARAHPOP_ASSERT(v >= 0.0 && v < 1.0,
        ("Body flattening must be in [0, 1). Got: " + std::to_string(v))
            .c_str());
}

/** @brief Field toggle — wraps both the bool activation flag and the
 *  per-body SphericalHarmonics configuration.  Returned by
 *  Body::gravity().shape() / Config::shape() when SH data is
 *  available. */
class FieldToggle {
public:
    FieldToggle(bool& flag, SHT* sh)
        : flag_{ flag }
        , sh_{ sh }
    {
    }

    /** @brief Activate */
    void activate() { flag_ = true; }
    /** @brief Deactivate */
    void deactivate() { flag_ = false; }
    /** @brief Get active state */
    bool active() const { return flag_; }
    /** @brief Set active state */
    FieldToggle& active(bool v)
    {
        flag_ = v;
        return *this;
    }

    /** @brief Spherical harmonics config chainable setters */
    FieldToggle& degree(const int& d)
    {
        assertSH_();
        sh_->degree(d);
        return *this;
    }
    FieldToggle& order(const int& o)
    {
        assertSH_();
        sh_->order(o);
        return *this;
    }
    FieldToggle& body(const std::string& b)
    {
        assertSH_();
        sh_->body(b);
        return *this;
    }
    FieldToggle& body(const NaifId& b)
    {
        assertSH_();
        sh_->body(b);
        return *this;
    }
    FieldToggle& addFile(const std::string& f)
    {
        assertSH_();
        sh_->files().push_back(f);
        return *this;
    }
    FieldToggle& delimiter(const char& d)
    {
        assertSH_();
        sh_->delimiter(d);
        return *this;
    }
    FieldToggle& headerLines(const idx_t& h)
    {
        assertSH_();
        sh_->headerLines(h);
        return *this;
    }

    /** @brief Spherical harmonics config getters */
    const int& degree() const
    {
        assertSH_();
        return sh_->degree();
    }
    const int& order() const
    {
        assertSH_();
        return sh_->order();
    }
    std::string bodyName() const
    {
        assertSH_();
        return sh_->bodyName();
    }
    const char& delimiter() const
    {
        assertSH_();
        return sh_->delimiter();
    }
    const idx_t& headerLines() const
    {
        assertSH_();
        return sh_->headerLines();
    }

    /** @brief Direct access to the SH config object */
    SHT& config()
    {
        assertSH_();
        return *sh_;
    }
    const SHT& config() const
    {
        assertSH_();
        return *sh_;
    }

private:
    void assertSH_() const
    {
        PARAHPOP_ASSERT(sh_ != nullptr,
            "Spherical harmonics config not available in this context. "
            "Access field config through environment().bodies()[\"...\"].");
    }
    bool& flag_;
    SHT* sh_;
};

namespace single {
using FieldFeature = FieldToggle;
} // namespace single

/** @brief Read-only field toggle inspector */
class ConstFieldToggle {
public:
    ConstFieldToggle(const bool& flag, const SHT* sh)
        : flag_{ flag }
        , sh_{ sh }
    {
    }

    /** @brief Get active state */
    bool active() const { return flag_; }

    /** @brief Spherical harmonics config getters */
    const int& degree() const
    {
        assertSH_();
        return sh_->degree();
    }
    const int& order() const
    {
        assertSH_();
        return sh_->order();
    }
    std::string bodyName() const
    {
        assertSH_();
        return sh_->bodyName();
    }
    const char& delimiter() const
    {
        assertSH_();
        return sh_->delimiter();
    }
    const idx_t& headerLines() const
    {
        assertSH_();
        return sh_->headerLines();
    }

    const SHT& config() const
    {
        assertSH_();
        return *sh_;
    }

private:
    void assertSH_() const
    {
        PARAHPOP_ASSERT(sh_ != nullptr,
            "Spherical harmonics config not available in this context.");
    }
    const bool& flag_;
    const SHT* sh_;
};

namespace single {
using ConstFieldFeature = ConstFieldToggle;
} // namespace single

/** @brief Per-band view — read/write the four parameters of one specific
 *  atmosphere segment.  Reached via ``atmosphere.exponential.segment(i)``;
 *  this is the per-segment path the single-block accessors point at once a
 *  profile is piecewise.  Holds the parent ``AtmT*`` + the band index;
 *  setters write that band directly (no single-block gate — addressing a
 *  band is always explicit here).  Bounds are checked by ``AtmT`` on each
 *  access. */
class SegmentView {
public:
    SegmentView(AtmT* atm, idx_t seg)
        : atm_{ atm }
        , seg_{ seg }
    {
    }

    SegmentView& rho0(const double& v)
    {
        assertAtm_();
        atm_->segmentRef(seg_).rho0 = v;
        return *this;
    }
    SegmentView& h0(const double& v)
    {
        assertAtm_();
        atm_->segmentRef(seg_).h0 = v;
        return *this;
    }
    SegmentView& scaleHeight(const double& v)
    {
        assertAtm_();
        atm_->segmentRef(seg_).scaleHeight = v;
        return *this;
    }
    SegmentView& hCutoff(const double& v)
    {
        assertAtm_();
        atm_->segmentRef(seg_).hCutoff = v;
        return *this;
    }

    double rho0() const { assertAtm_(); return atm_->segmentRef(seg_).rho0; }
    double h0() const { assertAtm_(); return atm_->segmentRef(seg_).h0; }
    double scaleHeight() const
    {
        assertAtm_();
        return atm_->segmentRef(seg_).scaleHeight;
    }
    double hCutoff() const
    {
        assertAtm_();
        return atm_->segmentRef(seg_).hCutoff;
    }

private:
    void assertAtm_() const
    {
        PARAHPOP_ASSERT(atm_ != nullptr,
            "Atmosphere segment view not available in this context.");
    }
    AtmT* atm_;
    idx_t seg_;
};

/** @brief Read-only per-band view. */
class ConstSegmentView {
public:
    ConstSegmentView(const AtmT* atm, idx_t seg)
        : atm_{ atm }
        , seg_{ seg }
    {
    }

    double rho0() const { assertAtm_(); return atm_->segmentRef(seg_).rho0; }
    double h0() const { assertAtm_(); return atm_->segmentRef(seg_).h0; }
    double scaleHeight() const
    {
        assertAtm_();
        return atm_->segmentRef(seg_).scaleHeight;
    }
    double hCutoff() const
    {
        assertAtm_();
        return atm_->segmentRef(seg_).hCutoff;
    }

private:
    void assertAtm_() const
    {
        PARAHPOP_ASSERT(atm_ != nullptr,
            "Atmosphere segment view not available in this context.");
    }
    const AtmT* atm_;
    idx_t seg_;
};

/** @brief Exponential-atmosphere model view — chainable setters /
 *  getters for the four exponential parameters (``rho0``, ``h0``,
 *  ``scaleHeight``, ``hCutoff``).
 *
 *  The oblate-ellipsoid ``flattening`` used for geodetic altitude is a
 *  *body* property (``Body::flattening()``), not an atmosphere parameter.
 *
 *  Reached via ``AtmosphereToggle::exponential()``.  Holds a non-owning
 *  pointer to the per-body ``AtmT`` (= ``Atmosphere``) config; the
 *  caller must keep the parent ``Bodies`` alive for the duration of any
 *  view use. */
class ExponentialAtmosphereView {
public:
    ExponentialAtmosphereView(AtmT* atm)
        : atm_{ atm }
    {
    }

    /** @brief Exponential model chainable setters */
    ExponentialAtmosphereView& rho0(const double& v)
    {
        assertAtm_();
        atm_->rho0(v);
        return *this;
    }
    ExponentialAtmosphereView& h0(const double& v)
    {
        assertAtm_();
        atm_->h0(v);
        return *this;
    }
    ExponentialAtmosphereView& scaleHeight(const double& v)
    {
        assertAtm_();
        atm_->scaleHeight(v);
        return *this;
    }
    ExponentialAtmosphereView& hCutoff(const double& v)
    {
        assertAtm_();
        atm_->hCutoff(v);
        return *this;
    }

    /** @brief Set the full piecewise profile from a JSON object (one block)
     *  or a JSON list of block objects — the backend sorts ascending by
     *  ``h0``.  The single-block setters above address segment 0; use this
     *  for a layered (piecewise) atmosphere. */
    ExponentialAtmosphereView& segments(const json& v)
    {
        assertAtm_();
        atm_->segments(v);
        return *this;
    }

    /** @brief Per-band sub-view for band ``i`` (bounds-checked by ``AtmT``).
     *  This is the explicit per-segment path; the single-block setters above
     *  throw on a piecewise profile and point here. */
    SegmentView segment(idx_t i)
    {
        assertAtm_();
        return SegmentView{ atm_, i };
    }

    /** @brief On-the-fly segment mutators (timeline-style; the backend sorts
     *  by ``h0`` + rejects overlaps at config-finalize).  Each takes a JSON
     *  block ``{rho0, h0, scale_height, h_cutoff}``. */
    ExponentialAtmosphereView& append(const json& block)
    {
        assertAtm_();
        atm_->append(block);
        return *this;
    }
    ExponentialAtmosphereView& prepend(const json& block)
    {
        assertAtm_();
        atm_->prepend(block);
        return *this;
    }
    ExponentialAtmosphereView& insert(idx_t pos, const json& block)
    {
        assertAtm_();
        atm_->insert(pos, block);
        return *this;
    }
    ExponentialAtmosphereView& remove(idx_t i)
    {
        assertAtm_();
        atm_->remove(i);
        return *this;
    }

    /** @brief Exponential model getters (segment 0; throw on a piecewise
     *  profile — read a band with ``segment(i)``). */
    double rho0() const
    {
        assertAtm_();
        return atm_->rho0();
    }
    double h0() const
    {
        assertAtm_();
        return atm_->h0();
    }
    double scaleHeight() const
    {
        assertAtm_();
        return atm_->scaleHeight();
    }
    double hCutoff() const
    {
        assertAtm_();
        return atm_->hCutoff();
    }

    /** @brief Number of segments (bands) in the profile (1 for a single
     *  block). */
    idx_t nSegments() const
    {
        assertAtm_();
        return atm_->nSegments();
    }

private:
    void assertAtm_() const
    {
        PARAHPOP_ASSERT(atm_ != nullptr,
            "Exponential atmosphere config not available in this context. "
            "Access atmosphere config through environment().bodies()[\"...\"].");
    }
    AtmT* atm_;
};

/** @brief Read-only exponential-atmosphere view */
class ConstExponentialAtmosphereView {
public:
    ConstExponentialAtmosphereView(const AtmT* atm)
        : atm_{ atm }
    {
    }

    double rho0() const
    {
        assertAtm_();
        return atm_->rho0();
    }
    double h0() const
    {
        assertAtm_();
        return atm_->h0();
    }
    double scaleHeight() const
    {
        assertAtm_();
        return atm_->scaleHeight();
    }
    double hCutoff() const
    {
        assertAtm_();
        return atm_->hCutoff();
    }

    /** @brief Per-band read-only sub-view for band ``i``. */
    ConstSegmentView segment(idx_t i) const
    {
        assertAtm_();
        return ConstSegmentView{ atm_, i };
    }

    /** @brief Number of segments (bands) in the profile. */
    idx_t nSegments() const
    {
        assertAtm_();
        return atm_->nSegments();
    }

private:
    void assertAtm_() const
    {
        PARAHPOP_ASSERT(atm_ != nullptr,
            "Exponential atmosphere config not available in this context.");
    }
    const AtmT* atm_;
};

/** @brief Atmosphere toggle — activation flag + per-body atmosphere
 *  configuration dispatch.  Returned by Body::atmosphere() /
 *  Config::atmosphere() when atmosphere data is available.
 *
 *  Nested-model API: the toggle itself owns the activation flag and the
 *  body identity; per-model parameter access goes through a model-named
 *  sub-view (currently only ``exponential()``).  When more atmosphere
 *  models land, additional sub-views will be added here (e.g. ``msis()``,
 *  ``marsgram()``). */
class AtmosphereToggle {
public:
    AtmosphereToggle(bool& flag, AtmT* atm)
        : flag_{ flag }
        , atm_{ atm }
    {
    }

    /** @brief Activate */
    void activate() { flag_ = true; }
    /** @brief Deactivate */
    void deactivate() { flag_ = false; }
    /** @brief Get active state */
    bool active() const { return flag_; }
    /** @brief Set active state */
    AtmosphereToggle& active(bool v)
    {
        flag_ = v;
        return *this;
    }

    /** @brief Body-identity setters (programmatic; the JSON path uses
     *  the surrounding ``bodies.<name>`` scope to set this). */
    AtmosphereToggle& body(const std::string& b)
    {
        assertAtm_();
        atm_->body(b);
        return *this;
    }
    AtmosphereToggle& body(const NaifId& b)
    {
        assertAtm_();
        atm_->body(b);
        return *this;
    }
    std::string bodyName() const
    {
        assertAtm_();
        return atm_->bodyName();
    }

    /** @brief Exponential-atmosphere sub-view (mutable). */
    ExponentialAtmosphereView exponential()
    {
        assertAtm_();
        return ExponentialAtmosphereView{ atm_ };
    }
    /** @brief Exponential-atmosphere sub-view (read-only). */
    ConstExponentialAtmosphereView exponential() const
    {
        assertAtm_();
        return ConstExponentialAtmosphereView{ atm_ };
    }

    /** @brief Direct access to the atmosphere config object */
    AtmT& config()
    {
        assertAtm_();
        return *atm_;
    }
    const AtmT& config() const
    {
        assertAtm_();
        return *atm_;
    }

private:
    void assertAtm_() const
    {
        PARAHPOP_ASSERT(atm_ != nullptr,
            "Atmosphere config not available in this context. "
            "Access atmosphere config through environment().bodies()[\"...\"].");
    }
    bool& flag_;
    AtmT* atm_;
};

/** @brief Read-only atmosphere toggle inspector */
class ConstAtmosphereToggle {
public:
    ConstAtmosphereToggle(const bool& flag, const AtmT* atm)
        : flag_{ flag }
        , atm_{ atm }
    {
    }

    /** @brief Get active state */
    bool active() const { return flag_; }

    /** @brief Body-identity getter */
    std::string bodyName() const
    {
        assertAtm_();
        return atm_->bodyName();
    }

    /** @brief Exponential-atmosphere sub-view (read-only). */
    ConstExponentialAtmosphereView exponential() const
    {
        assertAtm_();
        return ConstExponentialAtmosphereView{ atm_ };
    }

    const AtmT& config() const
    {
        assertAtm_();
        return *atm_;
    }

private:
    void assertAtm_() const
    {
        PARAHPOP_ASSERT(atm_ != nullptr,
            "Atmosphere config not available in this context.");
    }
    const bool& flag_;
    const AtmT* atm_;
};

/* forward declarations */
class Config;
class Body;
class ConstBody;
class Gravity;
class ConstGravity;

/* backward-compatible aliases */
using Feature          = Config;
using RefFeature       = Body;
using ConstRefFeature  = ConstBody;
using BodyGravity      = Gravity;
using ConstBodyGravity = ConstGravity;

/** @brief Body feature parser */
static Features parse(std::string name)
{
    /* make lower case */
    brie::util::Strings::lowerCase(name);

    /* find the corresponding iterator */
    auto iterator = NameMap.find(name);
    if (iterator != NameMap.end())
        return iterator->second;

    /* no valid source was parsed */
    return INVALID;
}

/** @brief Whether the feature is a gravity sub-feature */
static bool isGravityFeature_(Features f)
{
    return f == SHAPE || f == OBLATENESS;
}

/** @brief Error message for invalid gravity feature key */
static std::string GravityErrorMessage_(std::string name)
{
    return "Invalid gravity feature: '" + name
        + "'. Valid gravity features are: [oblateness, shape] "
          "(case insensitive)";
}

/** @brief Error message for invalid top-level body feature key */
static std::string TopLevelErrorMessage_(std::string name)
{
    return "Invalid top-level body feature key: '" + name
        + "'. Valid keys are: [gravity, radiation, atmosphere, occulting, "
          "flattening, orientation] "
          "(case insensitive). "
          "Gravity features (oblateness, shape) must be nested "
          "under the 'gravity' key.";
}

/** @brief Determine whether a JSON feature-config value means 'activate'.
 *  Valid values: true (activate), false (deactivate), null (deactivate),
 *  "default" case-insensitive (activate), {} or {config} (activate). */
static bool shouldActivateFromValue_(const json& val)
{
    if (val.is_boolean())
        return val.get<bool>();
    if (val.is_null())
        return false;
    if (val.is_object())
        return true;
    if (val.is_string()) {
        std::string s = val.get<std::string>();
        brie::util::Strings::lowerCase(s);
        PARAHPOP_ASSERT(s == "default",
            ("Invalid feature value string: '" + val.get<std::string>()
                + "'. Only \"default\" is supported.")
                .c_str());
        return true;
    }
    PARAHPOP_THROW(
        std::runtime_error, "Unsupported feature value type in body features.");
    return false;
}

} // namespace body
} // namespace environment
} // namespace model
} // namespace config
} // namespace interface
