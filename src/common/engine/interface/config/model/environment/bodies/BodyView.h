#pragma once

/* Non-owning views of a body within the bodies array (Gravity/ConstGravity,
 * Body/ConstBody) plus the cross-type Config<->Body conversions.  Split out
 * of BodySource.h; depends on the owned Config in BodyConfig.h. */

#include "interface/config/model/environment/bodies/BodyConfig.h"

namespace interface {
namespace config {
namespace model {
namespace environment {
namespace body {

/** @brief Non-owning gravity manager for a single body.
 *  Provides activate/deactivate/active for point gravity (implicit),
 *  plus .oblateness() and .shape() sub-accessors.  Constructed
 *  transiently by Body::gravity(). */
class Gravity {
    using FlagArrT = feta::vector::Array<bool, FEATSIZE>::GRef;

    friend class Config;
    friend class Body;
    friend class ConstBody;

public:
    Gravity(const FlagArrT& arr, const idx_t& idx, SHT* shConfig)
        : arr_{ arr }
        , idx_{ idx }
        , fieldConfig_{ shConfig }
    {
    }

    /** @brief Activate point gravity for this body */
    void activate() { arr_.template get<POINTGRAVITY>(idx_) = true; }

    /** @brief Deactivate point gravity for this body */
    void deactivate() { arr_.template get<POINTGRAVITY>(idx_) = false; }

    /** @brief Get active state */
    bool active() const { return arr_.template get<POINTGRAVITY>(idx_); }
    /** @brief Set active state */
    Gravity& active(bool v)
    {
        arr_.template get<POINTGRAVITY>(idx_) = v;
        return *this;
    }

    /** @brief Spherical harmonics shape model with SH config */
    FieldToggle shape()
    {
        return FieldToggle(arr_.template get<SHAPE>(idx_), fieldConfig_);
    }
    ConstFieldToggle shape() const
    {
        return ConstFieldToggle(arr_.template get<SHAPE>(idx_), fieldConfig_);
    }

    /** @brief Oblateness (J2) perturbation feature */
    Toggle oblateness() { return Toggle(arr_.template get<OBLATENESS>(idx_)); }
    ConstToggle oblateness() const
    {
        return ConstToggle(arr_.template get<OBLATENESS>(idx_));
    }

private:
    FlagArrT arr_;
    idx_t idx_;
    SHT* fieldConfig_;
};

/** @brief Read-only gravity manager for a single body. */
class ConstGravity {
    using FlagArrT = feta::vector::Array<bool, FEATSIZE>::GRef;

    friend class Config;
    friend class Body;
    friend class ConstBody;

public:
    ConstGravity(const FlagArrT& arr, const idx_t& idx, const SHT* shConfig)
        : arr_{ arr }
        , idx_{ idx }
        , fieldConfig_{ shConfig }
    {
    }

    /** @brief Get active state */
    bool active() const { return arr_.template get<POINTGRAVITY>(idx_); }

    /** @brief Spherical harmonics shape model */
    ConstFieldToggle shape() const
    {
        return ConstFieldToggle(arr_.template get<SHAPE>(idx_), fieldConfig_);
    }

    /** @brief Oblateness (J2) perturbation feature */
    ConstToggle oblateness() const
    {
        return ConstToggle(arr_.template get<OBLATENESS>(idx_));
    }

private:
    FlagArrT arr_;
    idx_t idx_;
    const SHT* fieldConfig_;
};

/** @brief Non-owning view of a body's physical features within the
 *  broad features array.
 *  Stores a GRef (non-owning pointer into host memory) by value together
 *  with the body index, so it is safe to return from functions without
 *  dangling-reference concerns.
 *
 *  Gravity-related features (pointgravity, shape, oblateness) are accessed
 *  through the grouped .gravity() accessor.  Radiation stays flat. */
class Body {
    using FlagArrT   = feta::vector::Array<bool, FEATSIZE>::GRef;
    using FlagsItemT = feta::vector::Item<bool, FEATSIZE>;

    friend class Config;
    friend class ConstBody;

public:
    /** @brief Default construct is forbidden */
    Body() = delete;

    /** @brief Construct from a GRef and body index (no SH / atmosphere config) */
    Body(const FlagArrT& arr, const idx_t& idx)
        : arr_{ arr }
        , idx_{ idx }
        , fieldConfig_{ nullptr }
        , atmosphere_{ nullptr }
        , flatArr_{}
    {
    }

    /** @brief Construct from a GRef, body index, and per-body SH config
     *  (atmosphere unset).  Kept for back-compat with sites that don't
     *  carry atmosphere config yet. */
    Body(const FlagArrT& arr, const idx_t& idx, SHT* shConfig)
        : arr_{ arr }
        , idx_{ idx }
        , fieldConfig_{ shConfig }
        , atmosphere_{ nullptr }
        , flatArr_{}
    {
    }

    /** @brief Construct from a GRef, body index, per-body SH config,
     *  per-body atmosphere config, and the per-body flattening handle.
     *  Production constructor used by Bodies::operator[].  The flattening
     *  handle is a feta scalar-array GRef indexed by ``idx`` (default empty
     *  for the back-compat sites above). */
    Body(const FlagArrT& arr, const idx_t& idx, SHT* shConfig, AtmT* atmosphere,
        const FlatArrT& flatArr = FlatArrT{}, std::string* orientation = nullptr)
        : arr_{ arr }
        , idx_{ idx }
        , fieldConfig_{ shConfig }
        , atmosphere_{ atmosphere }
        , flatArr_{ flatArr }
        , orientation_{ orientation }
    {
    }

    /** @brief Copy constructor */
    Body(const Body& other) = default;

    /** @brief Move constructor */
    Body(Body&& other) = default;

    /** @brief Copy assignment operator — writes through to the referenced
     *  array element-by-element */
    Body& operator=(const Body& other)
    {
        if (this != &other) {
            arr_.template get<POINTGRAVITY>(idx_)
                = other.arr_.template get<POINTGRAVITY>(other.idx_);
            arr_.template get<SHAPE>(idx_)
                = other.arr_.template get<SHAPE>(other.idx_);
            arr_.template get<OBLATENESS>(idx_)
                = other.arr_.template get<OBLATENESS>(other.idx_);
            arr_.template get<RADIATION>(idx_)
                = other.arr_.template get<RADIATION>(other.idx_);
            arr_.template get<ATMOSPHERE>(idx_)
                = other.arr_.template get<ATMOSPHERE>(other.idx_);
            arr_.template get<OCCULTING>(idx_)
                = other.arr_.template get<OCCULTING>(other.idx_);
            if (fieldConfig_ && other.fieldConfig_)
                *fieldConfig_ = *other.fieldConfig_;
            if (atmosphere_ && other.atmosphere_)
                *atmosphere_ = *other.atmosphere_;
            if (idx_ < flatArr_.size()
                && other.idx_ < other.flatArr_.size())
                flatArr_[idx_] = other.flatArr_[other.idx_];
            if (orientation_ && other.orientation_)
                *orientation_ = *other.orientation_;
        }
        return *this;
    }

    /** @brief Copy assignment operator from the given body source */
    Body& operator=(const Config& body)
    {
        arr_.template get<POINTGRAVITY>(idx_) = body.flags_.get<POINTGRAVITY>();
        arr_.template get<SHAPE>(idx_)        = body.flags_.get<SHAPE>();
        arr_.template get<OBLATENESS>(idx_)   = body.flags_.get<OBLATENESS>();
        arr_.template get<RADIATION>(idx_)    = body.flags_.get<RADIATION>();
        arr_.template get<ATMOSPHERE>(idx_)   = body.flags_.get<ATMOSPHERE>();
        arr_.template get<OCCULTING>(idx_)      = body.flags_.get<OCCULTING>();
        if (fieldConfig_)
            *fieldConfig_ = body.fieldConfig_;
        if (atmosphere_)
            *atmosphere_ = body.atmosphere_;
        if (idx_ < flatArr_.size())
            flatArr_[idx_] = body.flattening_;
        if (orientation_)
            *orientation_ = body.orientation_;
        return *this;
    }

    /** @brief Copy assignment operator from the given const ref body source */
    inline Body& operator=(const ConstBody& body);

    /** @brief Move assignment operator */
    Body& operator=(Body&& other) noexcept
    {
        if (this != &other) {
            arr_.template get<POINTGRAVITY>(idx_)
                = other.arr_.template get<POINTGRAVITY>(other.idx_);
            arr_.template get<SHAPE>(idx_)
                = other.arr_.template get<SHAPE>(other.idx_);
            arr_.template get<OBLATENESS>(idx_)
                = other.arr_.template get<OBLATENESS>(other.idx_);
            arr_.template get<RADIATION>(idx_)
                = other.arr_.template get<RADIATION>(other.idx_);
            arr_.template get<ATMOSPHERE>(idx_)
                = other.arr_.template get<ATMOSPHERE>(other.idx_);
            arr_.template get<OCCULTING>(idx_)
                = other.arr_.template get<OCCULTING>(other.idx_);
            if (fieldConfig_ && other.fieldConfig_)
                *fieldConfig_ = std::move(*other.fieldConfig_);
            if (atmosphere_ && other.atmosphere_)
                *atmosphere_ = std::move(*other.atmosphere_);
            if (idx_ < flatArr_.size()
                && other.idx_ < other.flatArr_.size())
                flatArr_[idx_] = other.flatArr_[other.idx_];
            if (orientation_ && other.orientation_)
                *orientation_ = std::move(*other.orientation_);
        }
        return *this;
    }

    /** @brief Assignment operator with item-based flags — write the flags
     * values in-place */
    Body& operator=(const FlagsItemT& flags)
    {
        arr_.template get<POINTGRAVITY>(idx_) = flags.get<POINTGRAVITY>();
        arr_.template get<SHAPE>(idx_)        = flags.get<SHAPE>();
        arr_.template get<OBLATENESS>(idx_)   = flags.get<OBLATENESS>();
        arr_.template get<RADIATION>(idx_)    = flags.get<RADIATION>();
        arr_.template get<ATMOSPHERE>(idx_)   = flags.get<ATMOSPHERE>();
        arr_.template get<OCCULTING>(idx_)      = flags.get<OCCULTING>();
        return *this;
    }

    /** @brief Update the flags by parsing the given json value.
     *  Supports: array, object, boolean, null, "default" string.
     *  Gravity features must be nested under the "gravity" key. */
    Body& operator=(const json& j)
    {
        if (j.is_array()) {
            for (const auto& item : j) {
                std::string name = item.get<std::string>();
                brie::util::Strings::lowerCase(name);
                if (name == "gravity")
                    arr_.template get<POINTGRAVITY>(idx_) = true;
                else if (name == "radiation")
                    arr_.template get<RADIATION>(idx_) = true;
                else if (name == "atmosphere")
                    arr_.template get<ATMOSPHERE>(idx_) = true;
                else if (name == "occulting")
                    arr_.template get<OCCULTING>(idx_) = true;
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
                    arr_.template get<RADIATION>(idx_)
                        = shouldActivateFromValue_(jit.value());
                } else if (key == "atmosphere") {
                    arr_.template get<ATMOSPHERE>(idx_)
                        = shouldActivateFromValue_(jit.value());
                    if (jit.value().is_object() && atmosphere_)
                        *atmosphere_ = jit.value();
                } else if (key == "occulting") {
                    arr_.template get<OCCULTING>(idx_)
                        = shouldActivateFromValue_(jit.value());
                } else if (key == "flattening") {
                    /* Write only when a valid flattening handle is bound
                     * (full Body from Bodies::operator[]); the flags-only
                     * view used by the Bodies(json) ctor skips here and the
                     * write is done by Bodies::parseFlattening_. */
                    if (idx_ < flatArr_.size()) {
                        const double f = jit.value().get<double>();
                        assertFlatteningRange_(f);
                        flatArr_[idx_] = f;
                    }
                } else if (key == "orientation") {
                    /* Write only when an orientation handle is bound (full
                     * Body from Bodies::operator[]); the flags-only view used
                     * by the Bodies(json) ctor skips here and the write is
                     * done by Bodies::parseOrientation_. */
                    if (orientation_)
                        *orientation_ = jit.value().get<std::string>();
                } else {
                    PARAHPOP_ASSERT(false, TopLevelErrorMessage_(key).c_str());
                }
            }
        } else if (j.is_boolean()) {
            if (!j.get<bool>()) {
                arr_.template get<POINTGRAVITY>(idx_) = false;
                arr_.template get<SHAPE>(idx_)        = false;
                arr_.template get<OBLATENESS>(idx_)   = false;
                arr_.template get<RADIATION>(idx_)    = false;
                arr_.template get<ATMOSPHERE>(idx_)   = false;
                arr_.template get<OCCULTING>(idx_)      = false;
            }
        } else if (j.is_null()) {
            arr_.template get<POINTGRAVITY>(idx_) = false;
            arr_.template get<SHAPE>(idx_)        = false;
            arr_.template get<OBLATENESS>(idx_)   = false;
            arr_.template get<RADIATION>(idx_)    = false;
            arr_.template get<ATMOSPHERE>(idx_)   = false;
            arr_.template get<OCCULTING>(idx_)      = false;
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

    /** @brief Grouped gravity accessor (pointgravity, shape, oblateness) */
    Gravity gravity() { return Gravity(arr_, idx_, fieldConfig_); }
    ConstGravity gravity() const
    {
        return ConstGravity(arr_, idx_, fieldConfig_);
    }

    /** @brief Radiation pressure feature (flat) */
    Toggle radiation() { return Toggle(arr_.template get<RADIATION>(idx_)); }
    ConstToggle radiation() const
    {
        return ConstToggle(arr_.template get<RADIATION>(idx_));
    }

    /** @brief Shadow-casting feature (flat): SRP attenuation + occulter
     *  eligibility for eclipse events. */
    Toggle occulting() { return Toggle(arr_.template get<OCCULTING>(idx_)); }
    ConstToggle occulting() const
    {
        return ConstToggle(arr_.template get<OCCULTING>(idx_));
    }

    /** @brief Atmosphere body property (kernel layer applies drag) */
    AtmosphereToggle atmosphere()
    {
        return AtmosphereToggle(
            arr_.template get<ATMOSPHERE>(idx_), atmosphere_);
    }
    ConstAtmosphereToggle atmosphere() const
    {
        return ConstAtmosphereToggle(
            arr_.template get<ATMOSPHERE>(idx_), atmosphere_);
    }

    /** @brief Body oblate-ellipsoid flattening (dimensionless, in
     *  ``[0, 1)``; ``0`` = sphere) used for geodetic altitude in the drag
     *  path.  A body shape property, not an atmosphere parameter.  Written
     *  through the per-body flattening handle. */
    Body& flattening(const double& v)
    {
        assertFlat_();
        assertFlatteningRange_(v);
        flatArr_[idx_] = v;
        return *this;
    }
    double flattening() const
    {
        assertFlat_();
        return flatArr_[idx_];
    }

    /** @brief Per-body orientation frame selection (name/alias; empty =
     *  auto-best).  Written through the per-body orientation handle.  See
     *  ``Config::orientation`` for the resolution semantics. */
    Body& orientation(const std::string& v)
    {
        assertOrient_();
        *orientation_ = v;
        return *this;
    }
    std::string orientation() const
    {
        assertOrient_();
        return *orientation_;
    }

private:
    FlagArrT arr_;
    idx_t idx_;
    SHT* fieldConfig_;
    AtmT* atmosphere_;
    FlatArrT flatArr_;
    /* Per-body orientation name handle (non-owning, into Bodies::orientations_);
     * null for the flags-only views from Accelerations::operator[]. */
    std::string* orientation_ = nullptr;

    /** @brief Whether a valid per-body flattening handle is bound (true for
     *  Body views from Bodies::operator[]; false for the flags-only views
     *  produced by Accelerations::operator[]). */
    void assertFlat_() const
    {
        PARAHPOP_ASSERT(idx_ < flatArr_.size(),
            "Body flattening not available in this context. "
            "Access flattening through environment().bodies()[\"...\"].");
    }

    /** @brief Whether a valid per-body orientation handle is bound (true for
     *  Body views from Bodies::operator[]; false for the flags-only views). */
    void assertOrient_() const
    {
        PARAHPOP_ASSERT(orientation_ != nullptr,
            "Body orientation not available in this context. "
            "Access orientation through environment().bodies()[\"...\"].");
    }

    /** @brief Parse gravity sub-object and write flags through the array ref */
    void assignGravityFromJson_(const json& gj)
    {
        if (gj.is_object()) {
            for (auto git = gj.begin(); git != gj.end(); ++git) {
                Features SRC = parse(git.key());
                PARAHPOP_ASSERT(SRC != INVALID && isGravityFeature_(SRC),
                    GravityErrorMessage_(git.key()).c_str());
                bool active = shouldActivateFromValue_(git.value());
                if (SRC == SHAPE)
                    arr_.template get<SHAPE>(idx_) = active;
                else if (SRC == OBLATENESS)
                    arr_.template get<OBLATENESS>(idx_) = active;
            }
        } else if (gj.is_array()) {
            for (const auto& name : gj) {
                Features SRC = parse(name);
                PARAHPOP_ASSERT(SRC != INVALID && isGravityFeature_(SRC),
                    GravityErrorMessage_(name).c_str());
                if (SRC == SHAPE)
                    arr_.template get<SHAPE>(idx_) = true;
                else if (SRC == OBLATENESS)
                    arr_.template get<OBLATENESS>(idx_) = true;
            }
        } else if (gj.is_boolean()) {
            if (gj.get<bool>()) {
                arr_.template get<POINTGRAVITY>(idx_) = true;
            } else {
                arr_.template get<POINTGRAVITY>(idx_) = false;
                arr_.template get<SHAPE>(idx_)        = false;
                arr_.template get<OBLATENESS>(idx_)   = false;
            }
        } else if (gj.is_null()) {
            arr_.template get<POINTGRAVITY>(idx_) = false;
            arr_.template get<SHAPE>(idx_)        = false;
            arr_.template get<OBLATENESS>(idx_)   = false;
        } else if (gj.is_string()) {
            std::string s = gj.get<std::string>();
            brie::util::Strings::lowerCase(s);
            PARAHPOP_ASSERT(s == "default",
                ("Invalid gravity value string: '" + gj.get<std::string>()
                    + "'. Only \"default\" is supported.")
                    .c_str());
            arr_.template get<POINTGRAVITY>(idx_) = true;
        } else {
            PARAHPOP_THROW(std::runtime_error,
                "Unsupported JSON type for gravity features.");
        }
    }
};

/** @brief Const view of a body's physical features.
 *  Same value-based storage as Body but with read-only access.
 *
 *  Gravity-related features accessed through .gravity() accessor.
 *  Radiation stays flat. */
class ConstBody {
    using FlagArrT   = feta::vector::Array<bool, FEATSIZE>::GRef;
    using FlagsItemT = feta::vector::Item<bool, FEATSIZE>;

    friend class Config;
    friend class Body;

public:
    /** @brief Default construct is forbidden */
    ConstBody() = delete;

    /** @brief Construct from a GRef and body index (no SH / atmosphere config) */
    ConstBody(const FlagArrT& arr, const idx_t& idx)
        : arr_{ arr }
        , idx_{ idx }
        , fieldConfig_{ nullptr }
        , atmosphere_{ nullptr }
        , flatArr_{}
    {
    }

    /** @brief Construct from a GRef, body index, and per-body SH config
     *  (atmosphere unset).  Kept for back-compat. */
    ConstBody(const FlagArrT& arr, const idx_t& idx, const SHT* shConfig)
        : arr_{ arr }
        , idx_{ idx }
        , fieldConfig_{ shConfig }
        , atmosphere_{ nullptr }
        , flatArr_{}
    {
    }

    /** @brief Construct from a GRef, body index, per-body SH config,
     *  per-body atmosphere config, and the per-body flattening handle.
     *  Production constructor. */
    ConstBody(const FlagArrT& arr, const idx_t& idx, const SHT* shConfig,
        const AtmT* atmosphere, const FlatArrT& flatArr = FlatArrT{},
        const std::string* orientation = nullptr)
        : arr_{ arr }
        , idx_{ idx }
        , fieldConfig_{ shConfig }
        , atmosphere_{ atmosphere }
        , flatArr_{ flatArr }
        , orientation_{ orientation }
    {
    }

    /** @brief Copy constructor */
    ConstBody(const ConstBody& other) = default;

    /** @brief Move constructor */
    ConstBody(ConstBody&& other) = default;

    /** @brief Assignment operators are forbidden (const view) */
    ConstBody& operator=(const ConstBody& other)  = delete;
    ConstBody& operator=(const Config& body)      = delete;
    ConstBody& operator=(const Body& body)        = delete;
    ConstBody& operator=(ConstBody&& other)       = delete;
    ConstBody& operator=(const FlagsItemT& flags) = delete;

    /** @brief Grouped gravity accessor (pointgravity, shape, oblateness) */
    ConstGravity gravity() const
    {
        return ConstGravity(arr_, idx_, fieldConfig_);
    }

    /** @brief Radiation pressure feature (flat) */
    ConstToggle radiation() const
    {
        return ConstToggle(arr_.template get<RADIATION>(idx_));
    }

    /** @brief Shadow-casting feature (flat, read-only). */
    ConstToggle occulting() const
    {
        return ConstToggle(arr_.template get<OCCULTING>(idx_));
    }

    /** @brief Atmosphere body property (read-only) */
    ConstAtmosphereToggle atmosphere() const
    {
        return ConstAtmosphereToggle(
            arr_.template get<ATMOSPHERE>(idx_), atmosphere_);
    }

    /** @brief Body oblate-ellipsoid flattening (read-only).  Dimensionless,
     *  in ``[0, 1)``; ``0`` = sphere.  A body shape property used for
     *  geodetic altitude in the drag path. */
    double flattening() const
    {
        assertFlat_();
        return flatArr_[idx_];
    }

    /** @brief Per-body orientation frame selection (read-only; name/alias,
     *  empty = auto-best). */
    std::string orientation() const
    {
        assertOrient_();
        return *orientation_;
    }

private:
    FlagArrT arr_;
    idx_t idx_;
    const SHT* fieldConfig_;
    const AtmT* atmosphere_;
    FlatArrT flatArr_;
    const std::string* orientation_ = nullptr;

    /** @brief Assert a valid per-body flattening handle is bound. */
    void assertFlat_() const
    {
        PARAHPOP_ASSERT(idx_ < flatArr_.size(),
            "Body flattening not available in this context. "
            "Access flattening through environment().bodies()[\"...\"].");
    }

    /** @brief Assert a valid per-body orientation handle is bound. */
    void assertOrient_() const
    {
        PARAHPOP_ASSERT(orientation_ != nullptr,
            "Body orientation not available in this context. "
            "Access orientation through environment().bodies()[\"...\"].");
    }
};

/* finalize the implementations of copy constructors and assignment operators */
/* Config from Body: copy element-by-element from the array ref */
inline Config::Config(const Body& ref)
{
    flags_.get<POINTGRAVITY>() = ref.arr_.template get<POINTGRAVITY>(ref.idx_);
    flags_.get<SHAPE>()        = ref.arr_.template get<SHAPE>(ref.idx_);
    flags_.get<OBLATENESS>()   = ref.arr_.template get<OBLATENESS>(ref.idx_);
    flags_.get<RADIATION>()    = ref.arr_.template get<RADIATION>(ref.idx_);
    flags_.get<ATMOSPHERE>()   = ref.arr_.template get<ATMOSPHERE>(ref.idx_);
    flags_.get<OCCULTING>()      = ref.arr_.template get<OCCULTING>(ref.idx_);
    if (ref.fieldConfig_)
        fieldConfig_ = *ref.fieldConfig_;
    if (ref.atmosphere_)
        atmosphere_ = *ref.atmosphere_;
    if (ref.idx_ < ref.flatArr_.size())
        flattening_ = ref.flatArr_[ref.idx_];
    if (ref.orientation_)
        orientation_ = *ref.orientation_;
}
inline Config::Config(const ConstBody& ref)
{
    flags_.get<POINTGRAVITY>() = ref.arr_.template get<POINTGRAVITY>(ref.idx_);
    flags_.get<SHAPE>()        = ref.arr_.template get<SHAPE>(ref.idx_);
    flags_.get<OBLATENESS>()   = ref.arr_.template get<OBLATENESS>(ref.idx_);
    flags_.get<RADIATION>()    = ref.arr_.template get<RADIATION>(ref.idx_);
    flags_.get<ATMOSPHERE>()   = ref.arr_.template get<ATMOSPHERE>(ref.idx_);
    flags_.get<OCCULTING>()      = ref.arr_.template get<OCCULTING>(ref.idx_);
    if (ref.fieldConfig_)
        fieldConfig_ = *ref.fieldConfig_;
    if (ref.atmosphere_)
        atmosphere_ = *ref.atmosphere_;
    if (ref.idx_ < ref.flatArr_.size())
        flattening_ = ref.flatArr_[ref.idx_];
    if (ref.orientation_)
        orientation_ = *ref.orientation_;
}
inline Config& Config::operator=(const Body& ref)
{
    flags_.get<POINTGRAVITY>() = ref.arr_.template get<POINTGRAVITY>(ref.idx_);
    flags_.get<SHAPE>()        = ref.arr_.template get<SHAPE>(ref.idx_);
    flags_.get<OBLATENESS>()   = ref.arr_.template get<OBLATENESS>(ref.idx_);
    flags_.get<RADIATION>()    = ref.arr_.template get<RADIATION>(ref.idx_);
    flags_.get<ATMOSPHERE>()   = ref.arr_.template get<ATMOSPHERE>(ref.idx_);
    flags_.get<OCCULTING>()      = ref.arr_.template get<OCCULTING>(ref.idx_);
    if (ref.fieldConfig_)
        fieldConfig_ = *ref.fieldConfig_;
    if (ref.atmosphere_)
        atmosphere_ = *ref.atmosphere_;
    if (ref.idx_ < ref.flatArr_.size())
        flattening_ = ref.flatArr_[ref.idx_];
    if (ref.orientation_)
        orientation_ = *ref.orientation_;
    return *this;
}
inline Config& Config::operator=(const ConstBody& ref)
{
    flags_.get<POINTGRAVITY>() = ref.arr_.template get<POINTGRAVITY>(ref.idx_);
    flags_.get<SHAPE>()        = ref.arr_.template get<SHAPE>(ref.idx_);
    flags_.get<OBLATENESS>()   = ref.arr_.template get<OBLATENESS>(ref.idx_);
    flags_.get<RADIATION>()    = ref.arr_.template get<RADIATION>(ref.idx_);
    flags_.get<ATMOSPHERE>()   = ref.arr_.template get<ATMOSPHERE>(ref.idx_);
    flags_.get<OCCULTING>()      = ref.arr_.template get<OCCULTING>(ref.idx_);
    if (ref.fieldConfig_)
        fieldConfig_ = *ref.fieldConfig_;
    if (ref.atmosphere_)
        atmosphere_ = *ref.atmosphere_;
    if (ref.idx_ < ref.flatArr_.size())
        flattening_ = ref.flatArr_[ref.idx_];
    if (ref.orientation_)
        orientation_ = *ref.orientation_;
    return *this;
}

/* Body from ConstBody */
inline Body& Body::operator=(const ConstBody& ref)
{
    arr_.template get<POINTGRAVITY>(idx_)
        = ref.arr_.template get<POINTGRAVITY>(ref.idx_);
    arr_.template get<SHAPE>(idx_) = ref.arr_.template get<SHAPE>(ref.idx_);
    arr_.template get<OBLATENESS>(idx_)
        = ref.arr_.template get<OBLATENESS>(ref.idx_);
    arr_.template get<RADIATION>(idx_)
        = ref.arr_.template get<RADIATION>(ref.idx_);
    arr_.template get<ATMOSPHERE>(idx_)
        = ref.arr_.template get<ATMOSPHERE>(ref.idx_);
    arr_.template get<OCCULTING>(idx_)
        = ref.arr_.template get<OCCULTING>(ref.idx_);
    if (fieldConfig_ && ref.fieldConfig_)
        *fieldConfig_ = *ref.fieldConfig_;
    if (atmosphere_ && ref.atmosphere_)
        *atmosphere_ = *ref.atmosphere_;
    if (idx_ < flatArr_.size() && ref.idx_ < ref.flatArr_.size())
        flatArr_[idx_] = ref.flatArr_[ref.idx_];
    if (orientation_ && ref.orientation_)
        *orientation_ = *ref.orientation_;
    return *this;
}

} // namespace body

/* backward-compatible namespace alias */
namespace bodies = body;

} // namespace environment
} // namespace model
} // namespace config
} // namespace interface
