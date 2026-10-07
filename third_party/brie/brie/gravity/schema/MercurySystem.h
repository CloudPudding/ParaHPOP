#pragma once

#include "brie/typedefs.h"
#include "brie/util.h"

namespace brie {
namespace gravity {
namespace schema {
namespace mercury {

/** @brief Counters for the bodies in the system */
enum class Bodies {
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
    static BodyIdToEnumT ibo2ebo() { return {}; }
    static BaryIdToEnumT iba2eba() { return {}; }
    static EnumToBodyIdT ebo2ibo() { return {}; }
    static EnumToBaryIdT eba2iba() { return {}; }
    static BodyStringToEnumT sbo2ebo() { return {}; }
    static BaryStringToEnumT sba2eba() { return {}; }
    static EnumToBodyStringT ebo2sbo() { return {}; }
    static EnumToBaryStringT eba2sba() { return {}; }
};

/** @brief Information structure for this system */
struct Info : public detail::Info<MapsT, Info> {
    friend struct detail::Info<MapsT, Info>;

protected:
    /** @brief The main body of this system */
    static constexpr NaifId main_ = 199;
    /** @brief The bary of this system */
    static constexpr NaifId bary_ = 1;
};

} // namespace mercury
} // namespace schema
} // namespace gravity
} // namespace brie