#pragma once

#include <algorithm>
#include <optional>
#include <set>
#include <string>
#include <vector>
#include "interface/config/detail/FieldCodec.h"
#include "interface/config/model/environment/Bodies.h"
#include "interface/config/model/environment/Constants.h"
#include "interface/config/model/environment/EarthCorrections.h"
#include "interface/config/model/environment/Ephemeris.h"
#include "interface/config/model/environment/Orientations.h"
#include "interface/naming/orientations/Resolver.h"
#include "interface/config/model/environment/detail/keys.h"

namespace interface {
namespace config {
namespace model {

namespace environment {

enum Fields {
    EPHEMERIS,
    CONSTANTS,
    ORIENTATIONS,
    BODIES,
    PREFERSSB,
    EARTHCORRECTIONS,
    
    INVALID
};

using Map = std::map<std::string, Fields>;

const Map NameMap = {
    
    { "ephemeris", EPHEMERIS }, { "constants", CONSTANTS },
    { keys::orientations, ORIENTATIONS }, { "bodies", BODIES },
    { "preferssb", PREFERSSB },
    { "earthcorrections", EARTHCORRECTIONS }
};

inline Fields parse(std::string name)
{
    return interface::config::detail::parseField(
        std::move(name), NameMap, INVALID);
}

} // namespace environment

class Environment {
    using Self = Environment;

public:
    using BodiesT = environment::Bodies;
    using EphT    = environment::Ephemeris;
    using CT      = environment::Constants;
    using OrientationsT = environment::Orientations;
    using EarthCorrectionsT = environment::EarthCorrections;
    using SHT     = environment::SphericalHarmonics;

    Environment() = default;

    Environment(const BodiesT& bodies, const EphT& ephemeris,
        const CT& constants, const OrientationsT& orientations = OrientationsT(),
        const bool& SunToSSB = true)
        : bodies_{ bodies }
        , ephemeris_{ ephemeris }
        , constants_{ constants }
        , orientations_{ orientations }
        , preferSSB_{ SunToSSB }
    {
    }
    Environment(const Environment& other) { *this = other; }

    Environment(BodiesT&& bodies, EphT&& ephemeris, CT&& constants,
        OrientationsT&& orientations = OrientationsT(), const bool& SunToSSB = true)
        : bodies_{ std::move(bodies) }
        , ephemeris_{ std::move(ephemeris) }
        , constants_{ std::move(constants) }
        , orientations_{ std::move(orientations) }
        , preferSSB_{ SunToSSB }
    {
    }
    Environment(Environment&& other) { *this = std::move(other); }

    Self& operator=(const Environment& other)
    {
        bodies_    = other.bodies_;
        ephemeris_ = other.ephemeris_;
        constants_ = other.constants_;
        orientations_ = other.orientations_;
        earthCorrections_ = other.earthCorrections_;
        preferSSB_  = other.preferSSB_;

        return *this;
    }

    Self& operator=(Environment&& other)
    {
        bodies_    = std::move(other.bodies_);
        ephemeris_ = std::move(other.ephemeris_);
        constants_ = std::move(other.constants_);
        orientations_ = std::move(other.orientations_);
        earthCorrections_ = std::move(other.earthCorrections_);
        preferSSB_  = std::exchange(other.preferSSB_, true);

        return *this;
    }

    Environment(const json& j)
    {
        using namespace environment;
        using JIT = json::const_iterator;

        for (JIT jit = j.begin(); jit != j.end(); jit++) {
            std::string field = jit.key();
            Fields FIELD      = parse(field);
            PARAHPOP_ASSERT(FIELD != INVALID, std::string("Unsupported environment field: ")+field);
            if (FIELD == EPHEMERIS)
                this->ephemeris_ = std::move(EphT(jit.value()));
            else if (FIELD == CONSTANTS)
                this->constants_ = std::move(CT(jit.value()));
            else if (FIELD == ORIENTATIONS)
                this->orientations_
                    = std::move(OrientationsT(jit.value()));
            else if (FIELD == PREFERSSB)
                this->preferSSB_ = jit.value();
            else if (FIELD == BODIES)
                this->bodies_ = std::move(BodiesT(jit.value()));
            else if (FIELD == EARTHCORRECTIONS)
                this->earthCorrections_ = EarthCorrectionsT(jit.value());
        }
        // Solid tides need prepared Sun/Moon positions, even when their
        // third-body gravity is disabled. Do not implicitly enable forces.
        if (earthCorrections_.solidEarthTides()) {
            const auto& ids = bodies_.ids();
            auto missing = [&](NaifId id) {
                return std::find(ids.begin(), ids.end(), id) == ids.end();
            };
            if (missing(10) || missing(301)) {
                json entries = bodies_.to_json();
                if (entries.is_array()) {
                    json object = json::object();
                    for (const auto& id : entries)
                        object[std::to_string(id.get<NaifId>())] = true;
                    entries = std::move(object);
                }
                for (NaifId id : { NaifId{10}, NaifId{301} })
                    if (missing(id)) entries[std::to_string(id)] = {{"gravity", false}};
                bodies_ = BodiesT(entries);
            }
        }
    }

    inline Self& showInfo()
    {
        bodies_.showInfo();
        ephemeris_.showInfo();
        constants_.showInfo();
        orientations_.showInfo();
        return *this;
    }

    inline Self& noInfo()
    {
        bodies_.noInfo();
        ephemeris_.noInfo();
        constants_.noInfo();
        orientations_.noInfo();
        return *this;
    }

    const bool& isSSBPreferred() const { return preferSSB_; }

    Self& preferSSB()
    {
        preferSSB_ = true;
        return *this;
    }
    Self& preferSun()
    {
        preferSSB_ = false;
        return *this;
    }

    BodiesT& bodies() { return bodies_; }
    const BodiesT& bodies() const { return bodies_; }

    EphT& ephemeris() { return ephemeris_; }
    const EphT& ephemeris() const { return ephemeris_; }

    CT& constants() { return constants_; }
    const CT& constants() const { return constants_; }

    OrientationsT& orientations() { return orientations_; }
    const OrientationsT& orientations() const { return orientations_; }

    std::string resolvedOrientationName(const NaifId& body) const
    {
        namespace orient = naming::orientations;
        using OId         = orient::OrientationId;
        PARAHPOP_ASSERT(orientations_.loaded(),
            "resolvedOrientationName requires the environment's orientations "
            "to be loaded first (call load()).");

        const auto& ids   = bodies_.ids();
        const auto& names = bodies_.orientationNames();
        std::string explicitName;
        bool found = false;
        for (idx_t i = 0; i < static_cast<idx_t>(ids.size()); i++) {
            if (ids[i] == body) {
                if (i < static_cast<idx_t>(names.size()))
                    explicitName = names[i];
                found = true;
                break;
            }
        }
        PARAHPOP_ASSERT(found,
            "Body " + std::to_string(body)
                + " is not among the active bodies.");

        const OId target = orient::resolveBodyOrientationTarget(body,
            explicitName,
            [this](const OId& t) { return orientations_.hasTarget(t); });
        const auto entry = orient::Registry::instance().findByTarget(target);
        return entry ? entry->name : ("FRAME_" + std::to_string(target));
    }

    EarthCorrectionsT& earthCorrections() { return earthCorrections_; }
    const EarthCorrectionsT& earthCorrections() const { return earthCorrections_; }

    brie::gravity::Tree makeGravityTree() const
    {
        BodiesT& b = const_cast<BodiesT&>(bodies_);
        return brie::gravity::Tree::fromActiveBodies(
            b.make().brieBodies(), preferSSB_);
    }

    json to_json() const
    {
        json j;
        j["Bodies"]    = bodies_.to_json();
        j["Ephemeris"] = ephemeris_.to_json();
        j["Constants"] = constants_.to_json();
        j["Orientations"] = orientations_.to_json();
        
        j["PreferSSB"] = preferSSB_;
        if (earthCorrections_.any())
            j["EarthCorrections"] = earthCorrections_.to_json();
        return j;
    }

    Self& load()
    {
        validateRadiationOnlyOnSun();
        bodies_.make();
        bodies_.loadFieldConfigs();
        this->makeEphemeris_();
        constants_.make();
        this->injectInlineConstants_();
        orientations_.make();
        return *this;
    }

    Self& preparePropagationData()
    {
        bodies_.make();
        this->makeEphemeris_();
        if (bodies_.hasInlineConstants()
            || constants_.hasInlineUniversals()) {
            constants_.make();
            this->injectInlineConstants_();
        }
        return *this;
    }

    void validateRadiationOnlyOnSun() const
    {
        constexpr NaifId SUN_NAIF_ID = 10;
        const auto flags = bodies_.features().flags().ref();
        for (idx_t i = 0; i < bodies_.size(); ++i) {
            const NaifId id = bodies_.ids()[i];
            if (id == SUN_NAIF_ID)
                continue;
            if (flags.template get<environment::body::RADIATION>(i)) {
                PARAHPOP_THROW(std::runtime_error,
                    std::string(
                        "Radiation is currently enabled as Solar Radiation "
                        "Pressure and only for the Sun. Albedo and/or "
                        "ionizing radiation for generic bodies may be added "
                        "in future releases.")
                        + " Offending body: '"
                        + brie::gravity::Parser::parsedName(id) + "'.");
            }
        }
    }

    void assertEpochCoverage(const Real& loMjd, const Real& hiMjd,
        const std::string& context) const
    {
        if (ephemeris_.loaded())
            ephemeris_.assertEpochCoverage(loMjd, hiMjd, context);
        if (orientations_.loaded())
            orientations_.assertEpochCoverage(loMjd, hiMjd, context);
    }

private:
    void makeEphemeris_() {
        const auto& active=bodies_.brieBodies();
        if (!earthCorrections_.any()) { ephemeris_.make(active); return; }
        auto ref=active.hostRef();
        PARAHPOP_ASSERT(ref.contains(NaifId{399}),"Earth corrections require Earth.");
        std::vector<NaifId> ids;
        for(idx_t i=0;i<active.size();++i) ids.push_back(ref[i]);
        auto add=[&](NaifId id) {
            if(std::find(ids.begin(),ids.end(),id)==ids.end()) ids.push_back(id);
        };
        if(earthCorrections_.tidesActive()) {add(10);add(301);}
        ephemeris_.make(brie::core::NaifIdArray(ids));
    }
    void injectInlineConstants_() {
        auto extra=bodies_.physicalBodyConstants();
        if(!extra.empty()) constants_.mergeBodies(extra);
    }
    BodiesT bodies_;
    EphT ephemeris_;
    CT constants_;
    OrientationsT orientations_;
    EarthCorrectionsT earthCorrections_;
    bool preferSSB_=true;
};
}}}
