#pragma once

/* Standalone owned body feature configuration (Config).
 * Split out of BodySource.h; depends on the feature vocabulary and leaf
 * widgets in FeatureToggles.h.  Non-owning array views are in BodyView.h. */

#include "interface/config/model/environment/bodies/FeatureToggles.h"

namespace interface {
namespace config {
namespace model {
namespace environment {
namespace body {

/** @brief Individual body configuration — standalone owned flags */
class Config {

    using FlagsT = feta::vector::Item<bool, FEATSIZE>;

    friend class Body;
    friend class ConstBody;

public:
    /* Default flags - only Point Gravity is active */
    static FlagsT DefaultFlags()
    {
        FlagsT flags;
        flags.get<POINTGRAVITY>() = true;
        return flags;
    }

    /** @brief Default constructor sets them all to false */
    Config()
        : flags_{}
        , fieldConfig_{}
        , atmosphere_{}
    {
    }

    /** @brief Copy constructor from the given flags */
    Config(const FlagsT& flags)
        : flags_{ flags }
        , fieldConfig_{}
        , atmosphere_{}
    {
    }

    /** @brief Copy constructor */
    Config(const Config& other)
        : flags_{ other.flags_ }
        , fieldConfig_{ other.fieldConfig_ }
        , atmosphere_{ other.atmosphere_ }
        , flattening_{ other.flattening_ }
        , orientation_{ other.orientation_ }
    {
    }

    /** @brief Copy construct from the given ref body source */
    inline Config(const RefFeature& ref);

    /** @brief Copy construct from the given const ref body source */
    inline Config(const ConstRefFeature& ref);

    /** @brief Move constructor from the given flags */
    Config(FlagsT&& flags)
        : flags_{ std::move(flags) }
        , fieldConfig_{}
        , atmosphere_{}
    {
    }

    /** @brief Move constructor */
    Config(Config&& other)
        : flags_{ std::move(other.flags_) }
        , fieldConfig_{ std::move(other.fieldConfig_) }
        , atmosphere_{ std::move(other.atmosphere_) }
        , flattening_{ other.flattening_ }
        , orientation_{ std::move(other.orientation_) }
    {
    }

    /** @brief Construct from json containing the given source names.
     *  Supports: array ["gravity", "radiation"], object {"gravity":{...}, ...},
     *  boolean (true=defaults, false=all off), null (all off),
     *  or "default" string (defaults).
     *  Gravity sub-features (oblateness, shape) must be nested under the
     *  "gravity" key.  The gravity key itself controls point-mass gravity.
     *  Radiation stays at top level. */
    Config(const json& j)
    {
        if (j.is_array()) {
            for (const auto& item : j) {
                std::string name = item.get<std::string>();
                brie::util::Strings::lowerCase(name);
                if (name == "gravity")
                    flags_.data()[POINTGRAVITY] = true;
                else if (name == "radiation")
                    flags_.data()[RADIATION] = true;
                else if (name == "atmosphere")
                    flags_.data()[ATMOSPHERE] = true;
                else if (name == "occulting")
                    flags_.data()[OCCULTING] = true;
                else
                    PARAHPOP_ASSERT(false, TopLevelErrorMessage_(name).c_str());
            }
        } else if (j.is_object()) {
            for (auto jit = j.begin(); jit != j.end(); ++jit) {
                std::string key = jit.key();
                brie::util::Strings::lowerCase(key);
                if (key == "gravity")
                    parseGravityFlags_(jit.value());
                else if (key == "radiation")
                    flags_.data()[RADIATION]
                        = shouldActivateFromValue_(jit.value());
                else if (key == "atmosphere") {
                    flags_.data()[ATMOSPHERE]
                        = shouldActivateFromValue_(jit.value());
                    if (jit.value().is_object())
                        atmosphere_ = jit.value();
                } else if (key == "occulting")
                    flags_.data()[OCCULTING]
                        = shouldActivateFromValue_(jit.value());
                else if (key == "flattening")
                    this->flattening(jit.value().get<double>());
                else if (key == "orientation")
                    this->orientation(jit.value().get<std::string>());
                else
                    PARAHPOP_ASSERT(false, TopLevelErrorMessage_(key).c_str());
            }
        } else if (j.is_boolean()) {
            if (!j.get<bool>()) {
                for (idx_t i = 0; i < FEATSIZE; ++i)
                    flags_.data()[i] = false;
            }
        } else if (j.is_null()) {
            /* null → all off */
            for (idx_t i = 0; i < FEATSIZE; ++i)
                flags_.data()[i] = false;
        } else if (j.is_string()) {
            std::string s = j.get<std::string>();
            brie::util::Strings::lowerCase(s);
            PARAHPOP_ASSERT(s == "default",
                ("Invalid body feature string: '" + j.get<std::string>()
                    + "'. Only \"default\" is supported.")
                    .c_str());
        } else {
            PARAHPOP_THROW(
                std::runtime_error, "Unsupported JSON type for body features.");
        }
    }

    /** @brief Copy assignment operator */
    Config& operator=(const Config& other)
    {
        if (this != &other) {
            flags_       = other.flags_;
            fieldConfig_ = other.fieldConfig_;
            atmosphere_  = other.atmosphere_;
            flattening_  = other.flattening_;
            orientation_ = other.orientation_;
        }
        return *this;
    }

    /** @brief Copy assignment operator from the given ref body source */
    inline Config& operator=(const RefFeature& ref);

    /** @brief Copy assignment operator from the given const ref body source */
    inline Config& operator=(const ConstRefFeature& ref);

    /** @brief Move assignment operator */
    Config& operator=(Config&& other) noexcept
    {
        if (this != &other) {
            flags_       = std::move(other.flags_);
            fieldConfig_ = std::move(other.fieldConfig_);
            atmosphere_  = std::move(other.atmosphere_);
            flattening_  = other.flattening_;
            orientation_ = std::move(other.orientation_);
        }
        return *this;
    }

    /** @brief Assignment operator with item-based flags */
    Config& operator=(const FlagsT& flags)
    {
        this->flags_ = flags;
        return *this;
    }

    /** @brief Assignment operator with json-based values.
     *  Supports: array, object, boolean, null, "default" string.
     *  Gravity sub-features must be nested under the "gravity" key. */
    Config& operator=(const json& j)
    {
        if (j.is_array()) {
            for (const auto& item : j) {
                std::string name = item.get<std::string>();
                brie::util::Strings::lowerCase(name);
                if (name == "gravity")
                    flags_.get<POINTGRAVITY>() = true;
                else if (name == "radiation")
                    this->radiation().activate();
                else if (name == "atmosphere")
                    this->atmosphere().activate();
                else if (name == "occulting")
                    this->occulting().activate();
                else
                    PARAHPOP_ASSERT(false, TopLevelErrorMessage_(name).c_str());
            }
        } else if (j.is_object()) {
            for (auto jit = j.begin(); jit != j.end(); ++jit) {
                std::string key = jit.key();
                brie::util::Strings::lowerCase(key);
                if (key == "gravity") {
                    assignGravityFromJson_(jit.value());
                } else if (key == "radiation") {
                    bool active = shouldActivateFromValue_(jit.value());
                    if (active)
                        this->radiation().activate();
                    else
                        this->radiation().deactivate();
                } else if (key == "atmosphere") {
                    bool active = shouldActivateFromValue_(jit.value());
                    if (active)
                        this->atmosphere().activate();
                    else
                        this->atmosphere().deactivate();
                    if (jit.value().is_object())
                        atmosphere_ = jit.value();
                } else if (key == "occulting") {
                    if (shouldActivateFromValue_(jit.value()))
                        this->occulting().activate();
                    else
                        this->occulting().deactivate();
                } else if (key == "flattening") {
                    this->flattening(jit.value().get<double>());
                } else if (key == "orientation") {
                    this->orientation(jit.value().get<std::string>());
                } else {
                    PARAHPOP_ASSERT(false, TopLevelErrorMessage_(key).c_str());
                }
            }
        } else if (j.is_boolean()) {
            if (!j.get<bool>()) {
                for (idx_t i = 0; i < FEATSIZE; ++i)
                    flags_.data()[i] = false;
            }
        } else if (j.is_null()) {
            /* null → all off */
            for (idx_t i = 0; i < FEATSIZE; ++i)
                flags_.data()[i] = false;
        } else if (j.is_string()) {
            std::string s = j.get<std::string>();
            brie::util::Strings::lowerCase(s);
            PARAHPOP_ASSERT(s == "default",
                ("Invalid body feature string: '" + j.get<std::string>()
                    + "'. Only \"default\" is supported.")
                    .c_str());
        } else {
            PARAHPOP_THROW(
                std::runtime_error, "Unsupported JSON type for body features.");
        }
        return *this;
    }

    /** @brief Expose whether the point gravity is active */
    Toggle pointgravity() { return Toggle(flags_.get<POINTGRAVITY>()); }
    ConstToggle pointgravity() const
    {
        return ConstToggle(flags_.get<POINTGRAVITY>());
    }

    /** @brief Expose the gravity shape model feature with SH config */
    FieldToggle shape()
    {
        return FieldToggle(flags_.get<SHAPE>(), &fieldConfig_);
    }
    ConstFieldToggle shape() const
    {
        return ConstFieldToggle(flags_.get<SHAPE>(), &fieldConfig_);
    }

    /** @brief Expose the oblateness (J2) perturbation feature */
    Toggle oblateness() { return Toggle(flags_.get<OBLATENESS>()); }
    ConstToggle oblateness() const
    {
        return ConstToggle(flags_.get<OBLATENESS>());
    }

    /** @brief Expose whether the radiation pressure is active */
    Toggle radiation() { return Toggle(flags_.get<RADIATION>()); }
    ConstToggle radiation() const
    {
        return ConstToggle(flags_.get<RADIATION>());
    }

    /** @brief Expose whether the body casts a shadow — when active the body
     *  attenuates solar radiation pressure on the spacecraft and is an
     *  eligible occulter for eclipse events. */
    Toggle occulting() { return Toggle(flags_.get<OCCULTING>()); }
    ConstToggle occulting() const { return ConstToggle(flags_.get<OCCULTING>()); }

    /** @brief Expose the atmosphere feature (body property; the kernel
     *  layer turns this into a drag acceleration on the spacecraft). */
    AtmosphereToggle atmosphere()
    {
        return AtmosphereToggle(flags_.get<ATMOSPHERE>(), &atmosphere_);
    }
    ConstAtmosphereToggle atmosphere() const
    {
        return ConstAtmosphereToggle(flags_.get<ATMOSPHERE>(), &atmosphere_);
    }

    /** @brief Direct access to field config */
    SHT& fieldConfig() { return fieldConfig_; }
    const SHT& fieldConfig() const { return fieldConfig_; }

    /** @brief Direct access to atmosphere config */
    AtmT& atmosphereConfig() { return atmosphere_; }
    const AtmT& atmosphereConfig() const { return atmosphere_; }

    /** @brief Body oblate-ellipsoid flattening (dimensionless, in
     *  ``[0, 1)``; ``0`` = sphere) used for geodetic altitude in the drag
     *  path.  A body shape property, not an atmosphere parameter.
     *  Typical values: WGS84 Earth ``0.00335281066474748``, IAU 2009 Mars
     *  ``0.005888934691714269``. */
    Config& flattening(const double& v)
    {
        assertFlatteningRange_(v);
        flattening_ = v;
        return *this;
    }
    const double& flattening() const { return flattening_; }

    /** @brief Per-body orientation frame selection: the name (or alias) of
     *  the frame whose rotation the force models (SH, J2, drag) use for this
     *  body.  Empty (the default) means auto-best — the highest-accuracy
     *  loaded frame for the body, falling back to its IAU polynomial.  The
     *  name is validated (and the resolution performed) at environment
     *  construction, never here, since custom frames register only once the
     *  environment's orientations are made. */
    Config& orientation(const std::string& v)
    {
        orientation_ = v;
        return *this;
    }
    const std::string& orientation() const { return orientation_; }

    /** @brief Serialise to JSON — symmetric with ``operator=(const json&)``.
     *
     *  Emits an object form whose keys are the activated features.  The
     *  output is canonical: each active feature's value is either ``true``
     *  (no extra config) or the feature's config object (SH coefficients,
     *  atmosphere parameters).  Inactive features are omitted, NOT emitted
     *  as ``false`` — this matches the parser default-state semantics
     *  (default flags = ``POINTGRAVITY`` only), so an all-default body
     *  dumps to ``{}`` which the parser correctly re-reads as default.
     *  The dump is therefore lossy for the "config-present-but-flag-off"
     *  programmatic edge case, which is unreachable from a parser input.
     *
     *  Gravity sub-features (shape, oblateness) are nested under the
     *  ``"gravity"`` key to mirror the parser's ``assignGravityFromJson_``
     *  contract.  Point-mass gravity is left implicit (default-on) — when
     *  the user explicitly disables it programmatically, the dump emits
     *  ``"gravity": false``. */
    json to_json() const
    {
        const bool pg = flags_.template get<POINTGRAVITY>();
        const bool sh = flags_.template get<SHAPE>();
        const bool ob = flags_.template get<OBLATENESS>();
        const bool rd = flags_.template get<RADIATION>();
        const bool at = flags_.template get<ATMOSPHERE>();
        const bool ec = flags_.template get<OCCULTING>();

        json out = json::object();

        if (!pg) {
            /* Explicitly disable the full gravity stack. */
            out["gravity"] = false;
        } else if (sh || ob) {
            json grav = json::object();
            if (sh) {
                if (!fieldConfig_.isDefault())
                    grav["shape"] = fieldConfig_.to_json();
                else
                    grav["shape"] = true;
            }
            if (ob)
                grav["oblateness"] = true;
            out["gravity"] = grav;
        }

        if (rd)
            out["radiation"] = true;

        if (at) {
            if (!atmosphere_.isDefault())
                out["atmosphere"] = atmosphere_.to_json();
            else
                out["atmosphere"] = true;
        }

        if (ec)
            out["occulting"] = true;

        /* Body shape: emit only when non-spherical (matches the
         * omit-defaults convention; flattening == 0 re-reads as default). */
        if (flattening_ != 0.0)
            out["flattening"] = flattening_;

        /* Per-body orientation: emit only when explicitly set (empty =
         * auto-best, re-reads as default). */
        if (!orientation_.empty())
            out["orientation"] = orientation_;

        return out;
    }

private:
    FlagsT flags_ = DefaultFlags();
    SHT fieldConfig_;
    AtmT atmosphere_;
    double flattening_ = 0.0;
    std::string orientation_;

    /** @brief Parse gravity sub-object and set flags directly (for ctor) */
    void parseGravityFlags_(const json& gj)
    {
        if (gj.is_object()) {
            for (auto git = gj.begin(); git != gj.end(); ++git) {
                Features SRC = parse(git.key());
                PARAHPOP_ASSERT(SRC != INVALID && isGravityFeature_(SRC),
                    GravityErrorMessage_(git.key()).c_str());
                flags_.data()[SRC] = shouldActivateFromValue_(git.value());
            }
        } else if (gj.is_array()) {
            for (const auto& name : gj) {
                Features SRC = parse(name);
                PARAHPOP_ASSERT(SRC != INVALID && isGravityFeature_(SRC),
                    GravityErrorMessage_(name).c_str());
                flags_.data()[SRC] = true;
            }
        } else if (gj.is_boolean()) {
            if (gj.get<bool>()) {
                flags_.data()[POINTGRAVITY] = true;
            } else {
                flags_.data()[POINTGRAVITY] = false;
                flags_.data()[SHAPE]        = false;
                flags_.data()[OBLATENESS]   = false;
            }
        } else if (gj.is_null()) {
            flags_.data()[POINTGRAVITY] = false;
            flags_.data()[SHAPE]        = false;
            flags_.data()[OBLATENESS]   = false;
        } else if (gj.is_string()) {
            std::string s = gj.get<std::string>();
            brie::util::Strings::lowerCase(s);
            PARAHPOP_ASSERT(s == "default",
                ("Invalid gravity value string: '" + gj.get<std::string>()
                    + "'. Only \"default\" is supported.")
                    .c_str());
            flags_.data()[POINTGRAVITY] = true;
        } else {
            PARAHPOP_THROW(std::runtime_error,
                "Unsupported JSON type for gravity features.");
        }
    }

    /** @brief Parse gravity sub-object via proxy accessors (for operator=) */
    void assignGravityFromJson_(const json& gj)
    {
        if (gj.is_object()) {
            for (auto git = gj.begin(); git != gj.end(); ++git) {
                Features SRC = parse(git.key());
                PARAHPOP_ASSERT(SRC != INVALID && isGravityFeature_(SRC),
                    GravityErrorMessage_(git.key()).c_str());
                bool active = shouldActivateFromValue_(git.value());
                if (SRC == SHAPE) {
                    if (active)
                        this->shape().activate();
                    else
                        this->shape().deactivate();
                } else if (SRC == OBLATENESS) {
                    if (active)
                        this->oblateness().activate();
                    else
                        this->oblateness().deactivate();
                }
            }
        } else if (gj.is_array()) {
            for (const auto& name : gj) {
                Features SRC = parse(name);
                PARAHPOP_ASSERT(SRC != INVALID && isGravityFeature_(SRC),
                    GravityErrorMessage_(name).c_str());
                if (SRC == SHAPE)
                    this->shape().activate();
                else if (SRC == OBLATENESS)
                    this->oblateness().activate();
            }
        } else if (gj.is_boolean()) {
            if (gj.get<bool>()) {
                flags_.data()[POINTGRAVITY] = true;
            } else {
                flags_.data()[POINTGRAVITY] = false;
                flags_.data()[SHAPE]        = false;
                flags_.data()[OBLATENESS]   = false;
            }
        } else if (gj.is_null()) {
            flags_.data()[POINTGRAVITY] = false;
            flags_.data()[SHAPE]        = false;
            flags_.data()[OBLATENESS]   = false;
        } else if (gj.is_string()) {
            std::string s = gj.get<std::string>();
            brie::util::Strings::lowerCase(s);
            PARAHPOP_ASSERT(s == "default",
                ("Invalid gravity value string: '" + gj.get<std::string>()
                    + "'. Only \"default\" is supported.")
                    .c_str());
            flags_.data()[POINTGRAVITY] = true;
        } else {
            PARAHPOP_THROW(std::runtime_error,
                "Unsupported JSON type for gravity features.");
        }
    }
};

} // namespace body
} // namespace environment
} // namespace model
} // namespace config
} // namespace interface
