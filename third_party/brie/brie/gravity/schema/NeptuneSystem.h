#pragma once

#include "brie/gravity/schema/detail/Info.h"

namespace brie {
namespace gravity {
namespace schema {
namespace neptune {

/** @brief Counters for the bodies in the system */
enum class Bodies {
    TRITON,
    NEREID,
    NAIAD,
    THALASSA,
    DESPINA,
    GALATEA,
    LARISSA,
    PROTEUS,
    HALIMEDE,
    PSAMATHE,
    SAO,
    LAOMEDEIA,
    NESO,
    /* SIZE */
    SIZE,
    /* invalid item */
    INVALID
};

/** @brief Counters for the barycentres in the System */
enum class Barys {
    /* SIZE */
    SIZE,
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
        return { { 801, Bodies::TRITON }, { 802, Bodies::NEREID },
            { 803, Bodies::NAIAD }, { 804, Bodies::THALASSA },
            { 805, Bodies::DESPINA }, { 806, Bodies::GALATEA },
            { 807, Bodies::LARISSA }, { 808, Bodies::PROTEUS },
            { 809, Bodies::HALIMEDE }, { 810, Bodies::PSAMATHE },
            { 811, Bodies::SAO }, { 812, Bodies::LAOMEDEIA },
            { 813, Bodies::NESO } };
    }
    static BaryIdToEnumT iba2eba() { return {}; }
    static EnumToBodyIdT ebo2ibo()
    {
        return { { Bodies::TRITON, 801 }, { Bodies::NEREID, 802 },
            { Bodies::NAIAD, 803 }, { Bodies::THALASSA, 804 },
            { Bodies::DESPINA, 805 }, { Bodies::GALATEA, 806 },
            { Bodies::LARISSA, 807 }, { Bodies::PROTEUS, 808 },
            { Bodies::HALIMEDE, 809 }, { Bodies::PSAMATHE, 810 },
            { Bodies::SAO, 811 }, { Bodies::LAOMEDEIA, 812 },
            { Bodies::NESO, 813 } };
    }
    static EnumToBaryIdT eba2iba() { return {}; }
    static BodyStringToEnumT sbo2ebo()
    {
        return { { "triton", Bodies::TRITON }, { "nereid", Bodies::NEREID },
            { "naiad", Bodies::NAIAD }, { "thalassa", Bodies::THALASSA },
            { "despina", Bodies::DESPINA }, { "galatea", Bodies::GALATEA },
            { "larissa", Bodies::LARISSA }, { "proteus", Bodies::PROTEUS },
            { "halimede", Bodies::HALIMEDE }, { "psamathe", Bodies::PSAMATHE },
            { "sao", Bodies::SAO }, { "laomedeia", Bodies::LAOMEDEIA },
            { "neso", Bodies::NESO }, { "801", Bodies::TRITON },
            { "802", Bodies::NEREID }, { "803", Bodies::NAIAD },
            { "804", Bodies::THALASSA }, { "805", Bodies::DESPINA },
            { "806", Bodies::GALATEA }, { "807", Bodies::LARISSA },
            { "808", Bodies::PROTEUS }, { "809", Bodies::HALIMEDE },
            { "810", Bodies::PSAMATHE }, { "811", Bodies::SAO },
            { "812", Bodies::LAOMEDEIA }, { "813", Bodies::NESO } };
    }
    static BaryStringToEnumT sba2eba() { return {}; }
    static EnumToBodyStringT ebo2sbo()
    {
        return { { Bodies::TRITON, "Triton" }, { Bodies::NEREID, "Nereid" },
            { Bodies::NAIAD, "Naiad" }, { Bodies::THALASSA, "Thalassa" },
            { Bodies::DESPINA, "Despina" }, { Bodies::GALATEA, "Galatea" },
            { Bodies::LARISSA, "Larissa" }, { Bodies::PROTEUS, "Proteus" },
            { Bodies::HALIMEDE, "Halimede" }, { Bodies::PSAMATHE, "Psamathe" },
            { Bodies::SAO, "Sao" }, { Bodies::LAOMEDEIA, "Laomedeia" },
            { Bodies::NESO, "Neso" } };
    }
    static EnumToBaryStringT eba2sba() { return {}; }
};

/** @brief Information structure for this system */
struct Info : public detail::Info<MapsT, Info> {
    friend struct detail::Info<MapsT, Info>;

protected:
    /** @brief The main body of this system */
    static constexpr NaifId main_ = 899;
    /** @brief The bary of this system */
    static constexpr NaifId bary_ = 8;
};

} // namespace neptune
} // namespace schema
} // namespace gravity
} // namespace brie