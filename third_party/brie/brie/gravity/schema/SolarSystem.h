#pragma once

#include "brie/gravity/schema/detail/Info.h"

namespace brie {
namespace gravity {
namespace schema {
namespace sun {

/** @brief Counters for the bodies in the System */
enum class Bodies {
    MERCURY,
    VENUS,
    EARTH,
    MARS,
    JUPITER,
    SATURN,
    URANUS,
    NEPTUNE,
    PLUTO,
    /* size */
    SIZE,
    /* Sun */
    SUN,
    /* invalid item */
    INVALID
};

/** @brief Counters for the barycentres in the System */
enum class Barys {
    MERCURYB,
    VENUSB,
    EARTHB,
    MARSB,
    JUPITERB,
    SATURNB,
    URANUSB,
    NEPTUNEB,
    PLUTOB,
    /* size */
    SIZE,
    /* SSB */
    SSB,
    /* invalid item */
    INVALID
};

/** @brief Body Maps */
struct MapsT {

    /** @brief Enum class for Bodies */
    using BodiesT = Bodies;
    /** @brief Enum class for Barys */
    using BarysT = Barys;

    /** @brief Body Naif ID to enum map */
    using BodyIdToEnumT = std::map<NaifId, BodiesT>;
    /** @brief Barycentre Naif ID to enum map */
    using BaryIdToEnumT = std::map<NaifId, BarysT>;
    /** @brief Body enum to Naif ID map */
    using EnumToBodyIdT = std::map<BodiesT, NaifId>;
    /** @brief Barycentre enum to Naif ID map */
    using EnumToBaryIdT = std::map<BarysT, NaifId>;
    /** @brief Body name string to enum map */
    using BodyStringToEnumT = std::map<std::string, BodiesT>;
    /** @brief Barycentre name string to enum map */
    using BaryStringToEnumT = std::map<std::string, BarysT>;
    /** @brief Body enum to name string map */
    using EnumToBodyStringT = std::map<BodiesT, std::string>;
    /** @brief Barycentre enum to name string map */
    using EnumToBaryStringT = std::map<BarysT, std::string>;

    /** @brief Maps */
    static BodyIdToEnumT ibo2ebo()
    {
        return { { 199, Bodies::MERCURY }, { 299, Bodies::VENUS },
            { 399, Bodies::EARTH }, { 499, Bodies::MARS },
            { 599, Bodies::JUPITER }, { 699, Bodies::SATURN },
            { 799, Bodies::URANUS }, { 899, Bodies::NEPTUNE },
            { 999, Bodies::PLUTO }, { 10, Bodies::SUN } };
    }
    static BaryIdToEnumT iba2eba()
    {
        return { { 1, Barys::MERCURYB }, { 2, Barys::VENUSB },
            { 3, Barys::EARTHB }, { 4, Barys::MARSB }, { 5, Barys::JUPITERB },
            { 6, Barys::SATURNB }, { 7, Barys::URANUSB },
            { 8, Barys::NEPTUNEB }, { 9, Barys::PLUTOB }, { 0, Barys::SSB } };
    }
    static EnumToBodyIdT ebo2ibo()
    {
        return { { Bodies::MERCURY, 199 }, { Bodies::VENUS, 299 },
            { Bodies::EARTH, 399 }, { Bodies::MARS, 499 },
            { Bodies::JUPITER, 599 }, { Bodies::SATURN, 699 },
            { Bodies::URANUS, 799 }, { Bodies::NEPTUNE, 899 },
            { Bodies::PLUTO, 999 }, { Bodies::SUN, 10 } };
    }
    static EnumToBaryIdT eba2iba()
    {
        return { { Barys::MERCURYB, 1 }, { Barys::VENUSB, 2 },
            { Barys::EARTHB, 3 }, { Barys::MARSB, 4 }, { Barys::JUPITERB, 5 },
            { Barys::SATURNB, 6 }, { Barys::URANUSB, 7 },
            { Barys::NEPTUNEB, 8 }, { Barys::PLUTOB, 9 }, { Barys::SSB, 0 } };
    }
    static BodyStringToEnumT sbo2ebo()
    {
        return { { "mercury", Bodies::MERCURY }, { "venus", Bodies::VENUS },
            { "earth", Bodies::EARTH }, { "mars", Bodies::MARS },
            { "jupiter", Bodies::JUPITER }, { "saturn", Bodies::SATURN },
            { "uranus", Bodies::URANUS }, { "neptune", Bodies::NEPTUNE },
            { "pluto", Bodies::PLUTO }, { "sun", Bodies::SUN },
            { "199", Bodies::MERCURY }, { "299", Bodies::VENUS },
            { "399", Bodies::EARTH }, { "499", Bodies::MARS },
            { "599", Bodies::JUPITER }, { "699", Bodies::SATURN },
            { "799", Bodies::URANUS }, { "899", Bodies::NEPTUNE },
            { "999", Bodies::PLUTO }, { "10", Bodies::SUN } };
    }
    static BaryStringToEnumT sba2eba()
    {
        return { { "mercuryb", Barys::MERCURYB }, { "venusb", Barys::VENUSB },
            { "earthb", Barys::EARTHB }, { "marsb", Barys::MARSB },
            { "jupiterb", Barys::JUPITERB }, { "saturnb", Barys::SATURNB },
            { "uranusb", Barys::URANUSB }, { "neptuneb", Barys::NEPTUNEB },
            { "plutob", Barys::PLUTOB }, { "ssb", Barys::SSB },
            { "solarsystemb", Barys::SSB },
            { "mercurybarycentre", Barys::MERCURYB },
            { "venusbarycentre", Barys::VENUSB },
            { "earthbarycentre", Barys::EARTHB },
            { "marsbarycentre", Barys::MARSB },
            { "jupiterbarycentre", Barys::JUPITERB },
            { "saturnbarycentre", Barys::SATURNB },
            { "uranusbarycentre", Barys::URANUSB },
            { "neptunebarycentre", Barys::NEPTUNEB },
            { "plutobarycentre", Barys::PLUTOB },
            { "solarsystembarycentre", Barys::SSB },
            { "mercurybarycenter", Barys::MERCURYB },
            { "venusbarycenter", Barys::VENUSB },
            { "earthbarycenter", Barys::EARTHB },
            { "marsbarycenter", Barys::MARSB },
            { "jupiterbarycenter", Barys::JUPITERB },
            { "saturnbarycenter", Barys::SATURNB },
            { "uranusbarycenter", Barys::URANUSB },
            { "neptunebarycenter", Barys::NEPTUNEB },
            { "plutobarycenter", Barys::PLUTOB },
            { "solarsystembarycenter", Barys::SSB },
            { "mercurybary", Barys::MERCURYB }, { "venusbary", Barys::VENUSB },
            { "earthbary", Barys::EARTHB }, { "marsbary", Barys::MARSB },
            { "jupiterbary", Barys::JUPITERB },
            { "saturnbary", Barys::SATURNB }, { "uranusbary", Barys::URANUSB },
            { "neptunebary", Barys::NEPTUNEB }, { "plutobary", Barys::PLUTOB },
            { "solarsystembary", Barys::PLUTOB }, { "1", Barys::MERCURYB },
            { "2", Barys::VENUSB }, { "3", Barys::EARTHB },
            { "4", Barys::MARSB }, { "5", Barys::JUPITERB },
            { "6", Barys::SATURNB }, { "7", Barys::URANUSB },
            { "8", Barys::NEPTUNEB }, { "9", Barys::PLUTOB },
            { "0", Barys::SSB } };
    }
    static EnumToBodyStringT ebo2sbo()
    {
        return { { Bodies::MERCURY, "Mercury" }, { Bodies::VENUS, "Venus" },
            { Bodies::EARTH, "Earth" }, { Bodies::MARS, "Mars" },
            { Bodies::JUPITER, "Jupiter" }, { Bodies::SATURN, "Saturn" },
            { Bodies::URANUS, "Uranus" }, { Bodies::NEPTUNE, "Neptune" },
            { Bodies::PLUTO, "Pluto" }, { Bodies::SUN, "Sun" } };
    }
    static EnumToBaryStringT eba2sba()
    {
        return { { Barys::MERCURYB, "MercuryBarycenter" },
            { Barys::VENUSB, "VenusBarycenter" },
            { Barys::EARTHB, "EarthBarycenter" },
            { Barys::MARSB, "MarsBarycenter" },
            { Barys::JUPITERB, "JupiterBarycenter" },
            { Barys::SATURNB, "SaturnBarycenter" },
            { Barys::URANUSB, "UranusBarycenter" },
            { Barys::NEPTUNEB, "NeptuneBarycenter" },
            { Barys::PLUTOB, "PlutoBarycenter" },
            { Barys::SSB, "SolarSystemBarycenter" } };
    }
};

/** @brief Information structure for this system */
struct Info : public detail::Info<MapsT, Info> {
    friend struct detail::Info<MapsT, Info>;

protected:
    /** @brief The main body of this system */
    static constexpr NaifId main_ = 10;
    /** @brief The bary of this system */
    static constexpr NaifId bary_ = 0;
};

} // namespace sun
} // namespace schema
} // namespace gravity
} // namespace brie