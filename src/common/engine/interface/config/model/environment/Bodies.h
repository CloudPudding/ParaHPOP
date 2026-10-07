#pragma once

#include "brie/constants/ConstantsJson.h"
#include "interface/config/model/Accelerations.h"
#include "interface/config/model/environment/Atmosphere.h"
#include "interface/config/model/environment/SphericalHarmonics.h"

namespace interface {
namespace config {
namespace model {
namespace environment {

/** @brief Active bodies Manager */
class Bodies {
    using Self      = Bodies;
    using Parser    = brie::gravity::Parser;
    using FeaturesT = model::Accelerations;
    using SHT       = environment::SphericalHarmonics;
    using AtmT      = environment::Atmosphere;
    /* Per-body flattening store: a feta scalar handle array (one double per
     * body), mirroring the flags array in `features_`.  Handed to the body
     * views as a GRef handle — not a raw pointer. */
    using FlatT     = feta::scalar::Array<double>;

public:
    using VecT   = std::vector<NaifId>;
    using NamesT = std::vector<std::string>;

    /** @brief Inline-declared physical constants for a body (``gm`` / ``radius``
     *  / ``soi`` / ``aMean`` / ``j2``), an alternative to a constants file for declaring a (typically
     *  massless) body's constants.  Only the explicitly-set fields are recorded,
     *  round-tripped, and merged; unset fields default to ``0`` and merge as
     *  such — so a body declared inline cannot partially override one already
     *  present in a constants file (that is a conflict, see
     *  ``brie::Constants::merged``).  The merge keeps the universal constants
     *  (AU, CLIGHT) from the loaded constants files, so a base constants file is
     *  still required for those. */
    struct InlineConstants {
        bool hasGm     = false;
        bool hasRadius = false;
        bool hasSoi    = false;
        bool hasAMean  = false;
        bool hasJ2     = false;
        double gm      = 0.0;
        double radius  = 0.0;
        double soi     = 0.0;
        double aMean   = 0.0;
        double j2      = 0.0;
        bool any() const
        {
            return hasGm || hasRadius || hasSoi || hasAMean || hasJ2;
        }
    };

    /** @brief Default constructor */
    Bodies() = default;

    /** @brief Copy constructors */
    Bodies(Bodies& other) { *this = other; }
    Bodies(const Bodies& other) { *this = other; }

    /** @brief Move constructor */
    Bodies(Bodies&& other) { *this = std::move(other); }

    /** @brief Construct from the given size */
    Bodies(const idx_t& size)
        : features_{ size }
        , fieldConfigs_(size)
        , atmospheres_(size)
        , orientations_(size)
        , inlineConstants_(size)
        , flattenings_{ size }
    {
        bodies_.reserve(size);
    }

    /** @brief Construct from vector of Naif ID*/
    Bodies(const VecT& bodies)
        : Bodies{ static_cast<idx_t>(bodies.size()) }
    {
        for (idx_t i = 0; i < bodies.size(); i++)
            add(bodies[i]);
    }

    /** @brief Construct from vector of strings  */
    Bodies(const NamesT& bodies)
        : Bodies{ static_cast<idx_t>(bodies.size()) }
    {
        for (idx_t i = 0; i < bodies.size(); i++)
            add(bodies[i]);
    }

    /** @brief Construct from json - potentially heterogeneous */
    Bodies(const json& j)
        : Bodies{ static_cast<idx_t>(j.size()) }
    {
        if (j.is_array()) {
            /* array of mixed strings / numbers — add each element */
            for (const auto& item : j)
                add(item);
        } else if (j.is_object()) {
            /* object with body names/ids as keys and (optionally) body
             * features as values */

            /* collect entries and sort by NAIF ID for deterministic body
             * order (nlohmann::json alphabetises object keys, which would
             * otherwise affect floating-point accumulation order) */
            struct Entry {
                std::string key;
                NaifId id;
                json features;
            };
            std::vector<Entry> entries;
            entries.reserve(j.size());
            for (auto jit = j.begin(); jit != j.end(); ++jit)
                entries.push_back({ jit.key(), Parser::parsedNaifId(jit.key()),
                    jit.value() });
            std::sort(entries.begin(), entries.end(),
                [](const Entry& a, const Entry& b) { return a.id < b.id; });

            for (idx_t i = 0; i < static_cast<idx_t>(entries.size()); i++) {
                add(entries[i].key);
                /* Inline physical constants (gm/radius/soi) are a Bodies-level
                 * concern parsed by parseConstants_; strip them from the value
                 * before the strict body-feature key validation in the
                 * features_ assignment, which would otherwise reject them. */
                features_[i] = withoutInlineConstants_(entries[i].features);
                parseFieldConfig_(i, entries[i].key, entries[i].features);
                parseAtmosphereConfig_(i, entries[i].key, entries[i].features);
                parseFlattening_(i, entries[i].features);
                parseOrientation_(i, entries[i].features);
                parseConstants_(i, entries[i].features);
            }
        } else {
            PARAHPOP_THROW(std::runtime_error,
                "Bodies must be specified as a json object (with body "
                "names/ids as keys) or a json array of names/ids.");
        }
    }

    /** @brief Copy assignment operators */
    Self& operator=(const Bodies& other)
    {
        bodies_       = other.bodies_;
        features_     = other.features_;
        fieldConfigs_ = other.fieldConfigs_;
        atmospheres_  = other.atmospheres_;
        orientations_ = other.orientations_;
        inlineConstants_ = other.inlineConstants_;
        flattenings_  = std::move(other.flattenings_.clone());
        showInfo_     = other.showInfo_;
        dirty_        = other.dirty_;
        return *this;
    }

    /** @brief Move assignment operator */
    Self& operator=(Bodies&& other)
    {
        bodies_       = std::move(other.bodies_);
        features_     = std::move(other.features_);
        fieldConfigs_ = std::move(other.fieldConfigs_);
        atmospheres_  = std::move(other.atmospheres_);
        orientations_ = std::move(other.orientations_);
        inlineConstants_ = std::move(other.inlineConstants_);
        flattenings_  = std::move(other.flattenings_);
        showInfo_     = other.showInfo_;
        dirty_        = other.dirty_;

        /* Reset other flags */
        other.showInfo_ = false;

        return *this;
    }

    /** @brief Add a body */
    Self& add(const NaifId& body)
    {
        if (!this->contains(body)) {
            bodies_.push_back(Parser::parsedNaifId(body));
            if (fieldConfigs_.size() < bodies_.size())
                fieldConfigs_.resize(bodies_.size());
            if (atmospheres_.size() < bodies_.size())
                atmospheres_.resize(bodies_.size());
            if (orientations_.size() < bodies_.size())
                orientations_.resize(bodies_.size());
            if (inlineConstants_.size() < bodies_.size())
                inlineConstants_.resize(bodies_.size());
            dirty_ = true;
        } else {
            PARAHPOP_WARN("Duplicate body: %s", Parser::parsedName(body).c_str());
        }
        return *this;
    }
    Self& add(const std::string& body)
    {
        if (!this->contains(body)) {
            bodies_.push_back(Parser::parsedNaifId(body));
            if (fieldConfigs_.size() < bodies_.size())
                fieldConfigs_.resize(bodies_.size());
            if (atmospheres_.size() < bodies_.size())
                atmospheres_.resize(bodies_.size());
            if (orientations_.size() < bodies_.size())
                orientations_.resize(bodies_.size());
            if (inlineConstants_.size() < bodies_.size())
                inlineConstants_.resize(bodies_.size());
            dirty_ = true;
        } else {
            PARAHPOP_WARN("Duplicate body: %s", Parser::parsedName(body).c_str());
        }
        return *this;
    }
    Self& add(const json& body)
    {
        if (!this->contains(body)) {
            /* input json may be either a plain string/number (if no force
             * sources are attached to it) or an object. In this last case we
             * need to read the key only */
            if (body.is_object()) {
                if (body.size() != 1) {
                    PARAHPOP_THROW(std::runtime_error,
                        "Body objects must contain exactly one key (the body "
                        "name/id)");
                }
                auto item = body.begin();
                bodies_.push_back(Self::parse(item.key()));
            } else
                bodies_.push_back(Self::parse(body));
            if (fieldConfigs_.size() < bodies_.size())
                fieldConfigs_.resize(bodies_.size());
            if (atmospheres_.size() < bodies_.size())
                atmospheres_.resize(bodies_.size());
            if (orientations_.size() < bodies_.size())
                orientations_.resize(bodies_.size());
            if (inlineConstants_.size() < bodies_.size())
                inlineConstants_.resize(bodies_.size());
            dirty_ = true;
        } else {
            PARAHPOP_WARN("Duplicate body: %s",
                Parser::parsedName(Self::parse(body)).c_str());
        }
        return *this;
    }

    /** @brief Expose the managed bodies */
    VecT& bodies() { return bodies_; }
    const VecT& bodies() const { return bodies_; }

    /** @brief Clear all the items */
    Self& clear()
    {
        bodies_.clear();
        dirty_ = true;
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

    /** @brief Process the bodies and create a NaifID array */
    Self& make()
    {
        if (!brieBodiesLoaded_) {
            load(std::move(brie::core::NaifIdArray(bodies_)));
        }
        /* Canonicalise every body's atmosphere profile (sort ascending by
         * h0 + reject overlapping bands).  Runs here, after any programmatic
         * insert/prepend/append, so the backend (extractAtmospheres_) always
         * reads sorted, non-overlapping segments.  Idempotent — JSON-parsed
         * profiles are already finalized at parse. */
        for (auto& atm : atmospheres_)
            atm.finalize();
        return *this;
    }

    /** @brief Parse the given body */
    static inline NaifId parse(const json& item)
    {
        NaifId out = 0;
        if (item.is_string()) {
            std::string i(item);
            out = Parser::parsedNaifId(i);
        } else if (item.is_number()) {
            NaifId i(item);
            out = Parser::parsedNaifId(i);
        } else {
            PARAHPOP_THROW(std::runtime_error,
                "Cannot parse items that are not numbers or strings.");
        }
        return out;
    }

    /** @brief Expose the brie bodies */
    const brie::core::NaifIdArray& brieBodies() const { return brieBodies_; }

    /** @brief Return the number of active bodies */
    idx_t size() const { return bodies_.size(); }

    /** @brief Check if these active bodies contain the given body */
    bool contains(const NaifId& id) const
    {
        return std::find(
                   bodies_.begin(), bodies_.end(), Parser::parsedNaifId(id))
            != bodies_.end();
    }
    bool contains(const std::string& id) const
    {
        return std::find(
                   bodies_.begin(), bodies_.end(), Parser::parsedNaifId(id))
            != bodies_.end();
    }
    bool contains(const json& jid) const { return contains(Self::parse(jid)); }

    /** @brief Expose the bodies - i.e. the naif IDs */
    const VecT& ids() const { return bodies_; }

    /** @brief Return the body names */
    NamesT names()
    {
        NamesT n;
        n.reserve(bodies_.size());
        for (const auto& id : bodies_)
            n.push_back(Parser::parsedName(id));
        return n;
    }

    /** @brief Expose the features (read-only \u2014 mutate via operator[]) */
    const FeaturesT& features() const { return features_; }

    /** @brief Access body features by name or string NAIF ID (e.g. "Earth",
     * "399"). Case-insensitive. Non-const access marks bodies as dirty. */
    body::Body operator[](const std::string& key)
    {
        dirty_    = true;
        idx_t idx = indexOf_(Parser::parsedNaifId(key));
        return body::Body(features_.flags().ref(), idx, &fieldConfigs_[idx],
            &atmospheres_[idx], flattenings_.ref(), &orientations_[idx]);
    }
    body::ConstBody operator[](const std::string& key) const
    {
        idx_t idx = indexOf_(Parser::parsedNaifId(key));
        return body::ConstBody(features_.flags().ref(), idx,
            &fieldConfigs_[idx], &atmospheres_[idx], flattenings_.ref(),
            &orientations_[idx]);
    }

    /** @brief Access body features by integer NAIF ID (e.g. 399) */
    body::Body operator[](const NaifId& key)
    {
        dirty_    = true;
        idx_t idx = indexOf_(Parser::parsedNaifId(key));
        return body::Body(features_.flags().ref(), idx, &fieldConfigs_[idx],
            &atmospheres_[idx], flattenings_.ref(), &orientations_[idx]);
    }
    body::ConstBody operator[](const NaifId& key) const
    {
        idx_t idx = indexOf_(Parser::parsedNaifId(key));
        return body::ConstBody(features_.flags().ref(), idx,
            &fieldConfigs_[idx], &atmospheres_[idx], flattenings_.ref(),
            &orientations_[idx]);
    }

    /** @brief Whether body features have been modified since last sync */
    bool dirty() const { return dirty_; }

    /** @brief Clear the dirty flag (called by Model after refresh) */
    void clearDirty() { dirty_ = false; }

    /** @brief Move-load the given brie::core::NaifIdArray */
    void load(brie::core::NaifIdArray&& bodies)
    {
        brieBodies_       = std::move(bodies);
        brieBodiesLoaded_ = true;
    }

    /** @brief Move-dump the given brie::core::NaifIdArray */
    brie::core::NaifIdArray dump()
    {
        brie::core::NaifIdArray out = std::move(brieBodies_);
        brieBodiesLoaded_           = false;
        return out;
    }

    /** @brief Load all per-body spherical harmonics coefficients */
    void loadFieldConfigs()
    {
        for (auto& sh : fieldConfigs_)
            sh.make();
    }

    /** @brief Load dumped per-body SH coefficients back into the matching
     *  body's field config (identified by the coefficients' body NAIF ID). */
    void loadFieldCoefficients(std::vector<
        interface::config::model::sphericalharmonics::Coefficients>&& allCoeffs)
    {
        for (auto& coeffs : allCoeffs) {
            NaifId shBody = coeffs.hostRef(true).body();
            if (shBody == 0)
                continue;
            for (idx_t i = 0; i < bodies_.size(); i++) {
                if (bodies_[i] == shBody) {
                    fieldConfigs_[i].load(std::move(coeffs));
                    break;
                }
            }
        }
    }

    /** @brief Expose per-body field configs (for backend consumption) */
    std::vector<SHT>& fieldConfigs() { return fieldConfigs_; }
    const std::vector<SHT>& fieldConfigs() const { return fieldConfigs_; }

    /** @brief Expose per-body atmosphere configs (for backend consumption) */
    std::vector<AtmT>& atmospheres() { return atmospheres_; }
    const std::vector<AtmT>& atmospheres() const { return atmospheres_; }

    /** @brief Expose the per-body flattening handle array (for backend
     *  consumption — read by paraHPOP's ``Environment::extractFlattenings_``).
     *  Index by body index: ``flattenings().ref()[i]``. */
    FlatT& flattenings() { return flattenings_; }
    const FlatT& flattenings() const { return flattenings_; }

    /** @brief Expose the per-body orientation frame names (for backend
     *  resolution at paraHPOP Environment construction).  Index by body index;
     *  empty string at ``i`` means auto-best for that body. */
    std::vector<std::string>& orientationNames() { return orientations_; }
    const std::vector<std::string>& orientationNames() const
    {
        return orientations_;
    }

    /** @brief Per-body physical constants for every body that declares inline
     *  ``gm`` / ``radius`` / ``soi``: a ``BodyConstants`` entry merged into the
     *  loaded constants set at Environment load, so a (typically massless) body
     *  can be declared without a constants file.  Bodies with no inline constant
     *  are omitted.  Unset fields default to ``0`` (a body declared inline
     *  cannot partially override one already in a constants file —
     *  ``brie::Constants::merged`` throws on a conflicting redefinition). */
    std::vector<brie::detail::BodyConstants> physicalBodyConstants() const
    {
        std::vector<brie::detail::BodyConstants> out;
        for (idx_t i = 0; i < static_cast<idx_t>(bodies_.size()); ++i) {
            if (i >= static_cast<idx_t>(inlineConstants_.size())
                || !inlineConstants_[i].any())
                continue;
            const InlineConstants& c = inlineConstants_[i];
            brie::detail::BodyConstants b;
            b.naifId = bodies_[i];
            b.gm     = c.gm;
            b.r      = c.radius;
            b.aMean  = c.aMean;
            b.j2     = c.j2;
            b.soi    = c.soi;
            out.push_back(b);
        }
        return out;
    }

    /** @brief Whether any body declares inline ``gm`` / ``radius`` / ``soi``
     *  constants (drives the constants-merge step on the propagation path). */
    bool hasInlineConstants() const
    {
        for (const auto& c : inlineConstants_)
            if (c.any())
                return true;
        return false;
    }

    /** @brief Serialise to JSON (inverse of ``Bodies(const json&)``).
     *
     *  Schema: when no body has features configured beyond the default
     *  state (POINTGRAVITY-only, no SH or atmosphere config), returns the
     *  compact array form (``["EARTH", "MOON", "SUN"]``).  When any body
     *  has features configured, returns the dict form
     *  (``{"EARTH": {<features>}, ...}``) symmetric with the parser's
     *  object-form input.  Each per-body value is the JSON produced by
     *  ``body::Config::to_json()``.
     *
     *  This is reconstructed from the live state of ``features_``,
     *  ``fieldConfigs_``, and ``atmospheres_`` — NOT from the raw JSON
     *  cached at parse time, which can become stale after programmatic
     *  setters fire. */
    json to_json() const
    {
        if (!anyFeaturesConfigured_())
            return json(bodies_);

        json out = json::object();
        for (idx_t i = 0; i < static_cast<idx_t>(bodies_.size()); ++i) {
            const std::string name = Parser::parsedName(bodies_[i]);
            body::ConstBody bview  = (*this)[bodies_[i]];
            body::Config cfg(bview);
            out[name] = cfg.to_json();
            /* Inline-declared physical constants round-trip the set fields
             * only (omit-defaults convention, symmetric with the parser). */
            if (i < static_cast<idx_t>(inlineConstants_.size())
                && inlineConstants_[i].any()) {
                const InlineConstants& c = inlineConstants_[i];
                if (c.hasGm)
                    out[name]["gm"] = c.gm;
                if (c.hasRadius)
                    out[name]["radius"] = c.radius;
                if (c.hasSoi)
                    out[name]["soi"] = c.soi;
                if (c.hasAMean)
                    out[name]["aMean"] = c.aMean;
                if (c.hasJ2)
                    out[name]["j2"] = c.j2;
            }
        }
        return out;
    }

protected:
    /** @brief Predicate for to_json(): is any body in a non-default state?
     *  Default means POINTGRAVITY-only + isDefault() SH config + isDefault()
     *  atmosphere. */
    bool anyFeaturesConfigured_() const
    {
        const auto flagsRef = features_.flags().ref();
        const auto flatRef  = flattenings_.ref();
        for (idx_t i = 0; i < static_cast<idx_t>(bodies_.size()); ++i) {
            if (!flagsRef.template get<body::POINTGRAVITY>(i))
                return true;
            if (flagsRef.template get<body::SHAPE>(i))
                return true;
            if (flagsRef.template get<body::OBLATENESS>(i))
                return true;
            if (flagsRef.template get<body::RADIATION>(i))
                return true;
            if (flagsRef.template get<body::ATMOSPHERE>(i))
                return true;
            if (flagsRef.template get<body::OCCULTING>(i))
                return true;
            if (!fieldConfigs_[i].isDefault())
                return true;
            if (!atmospheres_[i].isDefault())
                return true;
            if (flatRef[i] != 0.0)
                return true;
            if (!orientations_[i].empty())
                return true;
            if (i < static_cast<idx_t>(inlineConstants_.size())
                && inlineConstants_[i].any())
                return true;
        }
        return false;
    }

    /** @brief Find the index of the given canonical NAIF ID in bodies_ */
    idx_t indexOf_(const NaifId& canonicalId) const
    {
        auto it = std::find(bodies_.begin(), bodies_.end(), canonicalId);
        PARAHPOP_ASSERT(it != bodies_.end(),
            ("Body not found in active bodies: "
                + Parser::parsedName(canonicalId))
                .c_str());
        return static_cast<idx_t>(std::distance(bodies_.begin(), it));
    }

    /** @brief std::vector of bodies */
    brie::core::NaifIdArray brieBodies_ = brie::core::NaifIdArray(0);
    bool brieBodiesLoaded_              = false;
    VecT bodies_;
    FeaturesT features_;
    std::vector<SHT> fieldConfigs_;
    std::vector<AtmT> atmospheres_;
    /* Per-body orientation frame selection (name/alias; empty = auto-best).
     * Consumed once at paraHPOP Environment construction to resolve the per-body
     * rotation target; mirrors atmospheres_ (owned here, viewed by Body). */
    std::vector<std::string> orientations_;
    /* Per-body inline physical constants (gm/radius/soi), parsed from the
     * body payload; merged into the loaded constants at Environment load via
     * physicalBodyConstants().  Indexed by body index, like orientations_. */
    std::vector<InlineConstants> inlineConstants_;
    FlatT flattenings_ = FlatT::flexible();
    bool showInfo_ = false;
    bool dirty_    = true;

    /** @brief Parse per-body shape (SH) config from JSON.
     *  If features is an object with a "gravity" key whose value is an object
     *  containing a "shape" key with a non-empty object value, parse it as SH
     *  config and store in fieldConfigs_[i]. */
    void parseFieldConfig_(
        idx_t i, const std::string& bodyKey, const json& features)
    {
        if (!features.is_object())
            return;
        for (auto fit = features.begin(); fit != features.end(); ++fit) {
            std::string key = fit.key();
            brie::util::Strings::lowerCase(key);
            if (key == "gravity" && fit.value().is_object()) {
                for (auto git = fit.value().begin(); git != fit.value().end();
                    ++git) {
                    std::string gkey = git.key();
                    brie::util::Strings::lowerCase(gkey);
                    if (gkey == "shape" && git.value().is_object()
                        && !git.value().empty()) {
                        json shJson      = git.value();
                        shJson["body"]   = bodyKey;
                        fieldConfigs_[i] = SHT(shJson);
                        return;
                    }
                }
            }
        }
    }

    /** @brief Parse per-body atmosphere config from JSON.
     *  If features is an object with an "atmosphere" key whose value is a
     *  non-empty object, parse it as an Atmosphere config and store in
     *  atmospheres_[i]. */
    void parseAtmosphereConfig_(
        idx_t i, const std::string& bodyKey, const json& features)
    {
        if (!features.is_object())
            return;
        for (auto fit = features.begin(); fit != features.end(); ++fit) {
            std::string key = fit.key();
            brie::util::Strings::lowerCase(key);
            if (key == "atmosphere" && fit.value().is_object()
                && !fit.value().empty()) {
                /* Parse the model-dispatch dict (e.g. {exponential: {...}})
                 * via Atmosphere::operator=(json&).  The atmosphere block
                 * does NOT carry a `body` field; body identity comes from
                 * the surrounding bodies.<name> scope and is stamped onto
                 * the AtmT POD after parsing. */
                atmospheres_[i] = AtmT(fit.value());
                atmospheres_[i].body(bodyKey);
                return;
            }
        }
    }

    /** @brief Parse per-body flattening from JSON.  If ``features`` is an
     *  object with a numeric ``flattening`` key, validate it lies in
     *  ``[0, 1)`` and write the per-body flattening handle at index ``i``.
     *  Mirrors ``parseAtmosphereConfig_``: writes the container store
     *  directly, independent of the flags-only view the ``features_[i] =``
     *  path uses. */
    void parseFlattening_(idx_t i, const json& features)
    {
        if (!features.is_object())
            return;
        for (auto fit = features.begin(); fit != features.end(); ++fit) {
            std::string key = fit.key();
            brie::util::Strings::lowerCase(key);
            if (key == "flattening") {
                const double f = fit.value().get<double>();
                body::assertFlatteningRange_(f);
                flattenings_.ref()[i] = f;
                return;
            }
        }
    }

    /** @brief Parse the per-body orientation frame name from JSON.  If
     *  ``features`` is an object with a string ``orientation`` key, store the
     *  name in ``orientations_[i]``.  Mirrors ``parseAtmosphereConfig_``:
     *  writes the container store directly, independent of the flags-only
     *  view the ``features_[i] =`` path uses.  The name is validated (and the
     *  rotation target resolved) only at paraHPOP Environment construction. */
    void parseOrientation_(idx_t i, const json& features)
    {
        if (!features.is_object())
            return;
        for (auto fit = features.begin(); fit != features.end(); ++fit) {
            std::string key = fit.key();
            brie::util::Strings::lowerCase(key);
            if (key == "orientation") {
                orientations_[i] = fit.value().get<std::string>();
                return;
            }
        }
    }

    /** @brief Parse inline body physical constants (``gm`` / ``radius`` /
     *  ``soi``) from JSON.  Lets a (typically massless) body declare its
     *  constants without a constants file; the present keys are recorded and
     *  merged into the loaded constants set at Environment load (via
     *  ``physicalBodyConstants`` → ``brie::Constants::merged``).  Only keys that
     *  appear are recorded; unset fields stay default (``0``).  Mirrors
     *  ``parseFlattening_``: writes the container store directly at index
     *  ``i``. */
    /** @brief Return a copy of the body-features value with the inline
     *  physical-constant keys (gm/radius/soi, case-insensitive) removed, so the
     *  strict body-feature key validation in the features assignment does not
     *  reject them — they are parsed separately by parseConstants_. Non-object
     *  values (e.g. an empty feature array) pass through unchanged. */
    static json withoutInlineConstants_(const json& features)
    {
        if (!features.is_object())
            return features;
        json out = json::object();
        for (auto fit = features.begin(); fit != features.end(); ++fit) {
            std::string k = fit.key();
            brie::util::Strings::lowerCase(k);
            if (k == "gm" || k == "radius" || k == "soi" || k == "amean"
                || k == "j2")
                continue;
            out[fit.key()] = fit.value();
        }
        return out;
    }

    void parseConstants_(idx_t i, const json& features)
    {
        if (!features.is_object())
            return;
        InlineConstants& c = inlineConstants_[i];
        for (auto fit = features.begin(); fit != features.end(); ++fit) {
            std::string key = fit.key();
            brie::util::Strings::lowerCase(key);
            if (key == "gm") {
                c.gm    = fit.value().get<double>();
                c.hasGm = true;
            } else if (key == "radius") {
                c.radius    = fit.value().get<double>();
                c.hasRadius = true;
            } else if (key == "soi") {
                c.soi    = fit.value().get<double>();
                c.hasSoi = true;
            } else if (key == "amean") {
                c.aMean    = fit.value().get<double>();
                c.hasAMean = true;
            } else if (key == "j2") {
                c.j2    = fit.value().get<double>();
                c.hasJ2 = true;
            }
        }
    }
};

} // namespace environment

/* TEMPORARY FOR TEST AND COMPATIBILITY */
/* finish the definition of accelerations object core */
inline Accelerations::Accelerations(const environment::Bodies& abodies)
    : Accelerations{ abodies.features() }
{
}

} // namespace model
} // namespace config
} // namespace interface