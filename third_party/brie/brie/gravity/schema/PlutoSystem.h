#pragma once

#include "brie/gravity/schema/detail/Info.h"

namespace brie {
namespace gravity {
namespace schema {
namespace pluto {

/** @brief Counters for the bodies in the system */
enum class Bodies {
    CHARON,
    NIX,
    HYDRA,
    KERBEROS,
    STYX,
    /* SIZE */
    SIZE,
    /* invalid item */
    INVALID
};

/** @brief Counters for the barycentres in the system */
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
        return { { 901, Bodies::CHARON }, { 902, Bodies::NIX },
            { 903, Bodies::HYDRA }, { 904, Bodies::KERBEROS },
            { 905, Bodies::STYX } };
    }
    static BaryIdToEnumT iba2eba() { return {}; }
    static EnumToBodyIdT ebo2ibo()
    {
        return { { Bodies::CHARON, 901 }, { Bodies::NIX, 902 },
            { Bodies::HYDRA, 903 }, { Bodies::KERBEROS, 904 },
            { Bodies::STYX, 905 } };
    }
    static EnumToBaryIdT eba2iba() { return {}; }
    static BodyStringToEnumT sbo2ebo()
    {
        return { { "charon", Bodies::CHARON }, { "nix", Bodies::NIX },
            { "hydra", Bodies::HYDRA }, { "kerberos", Bodies::KERBEROS },
            { "styx", Bodies::STYX }, { "901", Bodies::CHARON },
            { "902", Bodies::NIX }, { "903", Bodies::HYDRA },
            { "904", Bodies::KERBEROS }, { "905", Bodies::STYX } };
    }
    static BaryStringToEnumT sba2eba() { return {}; }
    static EnumToBodyStringT ebo2sbo()
    {
        return { { Bodies::CHARON, "Charon" }, { Bodies::NIX, "Nix" },
            { Bodies::HYDRA, "Hydra" }, { Bodies::KERBEROS, "Kerberos" },
            { Bodies::STYX, "Styx" } };
    }
    static EnumToBaryStringT eba2sba() { return {}; }
};

/** @brief Information structure for this system */
struct Info : public detail::Info<MapsT, Info> {
    friend struct detail::Info<MapsT, Info>;

protected:
    /** @brief The main body of this system */
    static constexpr NaifId main_ = 999;
    /** @brief The bary of this system */
    static constexpr NaifId bary_ = 9;
};

} // namespace pluto
} // namespace schema
} // namespace gravity
} // namespace brie