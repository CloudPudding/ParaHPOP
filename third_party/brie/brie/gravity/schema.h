#pragma once

#include "brie/gravity/schema/EarthSystem.h"
#include "brie/gravity/schema/JupiterSystem.h"
#include "brie/gravity/schema/MarsSystem.h"
#include "brie/gravity/schema/MercurySystem.h"
#include "brie/gravity/schema/NeptuneSystem.h"
#include "brie/gravity/schema/PlutoSystem.h"
#include "brie/gravity/schema/SaturnSystem.h"
#include "brie/gravity/schema/SolarSystem.h"
#include "brie/gravity/schema/UranusSystem.h"
#include "brie/gravity/schema/VenusSystem.h"

namespace brie {
namespace gravity {
namespace schema {

/** @brief The available systems */
enum Systems {
    SUN,
    MERCURY,
    VENUS,
    EARTH,
    MARS,
    JUPITER,
    SATURN,
    URANUS,
    NEPTUNE,
    PLUTO,
    /* Size */
    SYSTEMSSIZE
};

/** @brief System info switcher */
template<Systems SYS>
struct SystemSwitcher {
    using Info = sun::Info;
};

/** @brief Template specializations */
template<>
struct SystemSwitcher<MERCURY> {
    using Info = mercury::Info;
};
template<>
struct SystemSwitcher<VENUS> {
    using Info = venus::Info;
};
template<>
struct SystemSwitcher<EARTH> {
    using Info = earth::Info;
};
template<>
struct SystemSwitcher<MARS> {
    using Info = mars::Info;
};
template<>
struct SystemSwitcher<JUPITER> {
    using Info = jupiter::Info;
};
template<>
struct SystemSwitcher<SATURN> {
    using Info = saturn::Info;
};
template<>
struct SystemSwitcher<URANUS> {
    using Info = uranus::Info;
};
template<>
struct SystemSwitcher<NEPTUNE> {
    using Info = neptune::Info;
};
template<>
struct SystemSwitcher<PLUTO> {
    using Info = pluto::Info;
};


/** @brief Body parser */
struct Parser {

    /** @brief Search for the given gravity body identifier (string) scanning
     * all the available systems, and possibly return its Naif ID */
    static NaifId parsedNaifId(std::string item)
    {
        /* lower case string */
        util::Strings::lowerCase(item);

        /* search the item in the available bodies */
        bool found = false;
        NaifId id  = 0;

        /* run the search */
        constexpr Systems S = static_cast<Systems>(SYSTEMSSIZE - 1);
        stringParser_<S>(found, id, item);

        /* Raise error if body is not found */
        if (!found) {
            std::stringstream ss;
            ss << "Could not parse: " << item;
            std::string s = ss.str();
            BRIE_THROW(std::runtime_error, s.c_str());
        }

        return id;
    }
    /** @brief Search for the given gravity body identifier (Naif ID) scanning
     * all the available systems, and possibly return its Naif ID */
    static NaifId parsedNaifId(NaifId item)
    {
        /* search the item in the available bodies */
        bool found = false;
        NaifId id  = 0;

        /* run the search */
        constexpr Systems S = static_cast<Systems>(SYSTEMSSIZE - 1);
        naifParser_<S>(found, id, item);

        /* Raise error if body is not found */
        if (!found) {
            std::stringstream ss;
            ss << "Could not parse: " << item;
            std::string s = ss.str();
            BRIE_THROW(std::runtime_error, s.c_str());
        }

        return id;
    }

    /** @brief Search for the given gravity body identifier (string) scanning
     * all the available systems, and possibly return its Naif ID */
    static std::string parsedName(std::string item)
    {
        /* lower case string */
        util::Strings::lowerCase(item);

        /* search the item in the available bodies */
        bool found       = false;
        std::string name = "";

        /* run the search */
        constexpr Systems S = static_cast<Systems>(SYSTEMSSIZE - 1);
        stringParser_<S>(found, name, item);

        /* Raise error if body is not found */
        if (!found) {
            std::stringstream ss;
            ss << "Could not parse: " << item;
            std::string s = ss.str();
            BRIE_THROW(std::runtime_error, s.c_str());
        }

        return name;
    }
    /** @brief Search for the given gravity body identifier (Naif ID) scanning
     * all the available systems, and possibly return its Naif ID */
    static std::string parsedName(NaifId item)
    {
        /* search the item in the available bodies */
        bool found       = false;
        std::string name = "";

        /* run the search */
        constexpr Systems S = static_cast<Systems>(SYSTEMSSIZE - 1);
        naifParser_<S>(found, name, item);

        /* Raise error if body is not found */
        if (!found) {
            std::stringstream ss;
            ss << "Could not parse: " << item;
            std::string s = ss.str();
            BRIE_THROW(std::runtime_error, s.c_str());
        }

        return name;
    }

private:
    /** @brief Parse the given item string to get the valid Naif ID */
    template<Systems SYS>
    static void stringParser_(bool& found, NaifId& id, const std::string& item)
    {
        /* previous recursion step */
        if constexpr (SYS >= 1) {
            stringParser_<static_cast<Systems>(SYS - 1)>(found, id, item);
        }
        /* this recursion step */
        using SInfo = typename SystemSwitcher<SYS>::Info;
        if (!found) {
            if (SInfo::bodiesContain(item)) {
                found = true;
                id    = SInfo::getNaifId(item);
            }
        }
        if (!found) {
            if (SInfo::barycentresContain(item)) {
                found = true;
                id    = SInfo::getNaifId(item);
            }
        }
    }
    /** @brief Parse the given item Naif ID to get the valid Naif ID */
    template<Systems SYS>
    static void naifParser_(bool& found, NaifId& id, const NaifId& item)
    {
        /* previous recursion step */
        if constexpr (SYS >= 1) {
            naifParser_<static_cast<Systems>(SYS - 1)>(found, id, item);
        }

        /* this recursion step */
        using SInfo = typename SystemSwitcher<SYS>::Info;
        if (!found) {
            if (SInfo::bodiesContain(item)) {
                found = true;
                id    = SInfo::getNaifId(item);
            }
        }
        if (!found) {
            if (SInfo::barycentresContain(item)) {
                found = true;
                id    = SInfo::getNaifId(item);
            }
        }
    }
    /** @brief Parse the given item string to get the valid string*/
    template<Systems SYS>
    static void stringParser_(
        bool& found, std::string& name, const std::string& item)
    {
        /* previous recursion step */
        if constexpr (SYS >= 1) {
            stringParser_<static_cast<Systems>(SYS - 1)>(found, name, item);
        }
        /* this recursion step */
        using SInfo = typename SystemSwitcher<SYS>::Info;
        if (!found) {
            if (SInfo::bodiesContain(item)) {
                found = true;
                name  = SInfo::getName(item);
            }
        }
        if (!found) {
            if (SInfo::barycentresContain(item)) {
                found = true;
                name  = SInfo::getName(item);
            }
        }
    }
    /** @brief Parse the given item Naif ID to get the valid string */
    template<Systems SYS>
    static void naifParser_(bool& found, std::string& name, const NaifId& item)
    {
        /* previous recursion step */
        if constexpr (SYS >= 1) {
            naifParser_<static_cast<Systems>(SYS - 1)>(found, name, item);
        }

        /* this recursion step */
        using SInfo = typename SystemSwitcher<SYS>::Info;
        if (!found) {
            if (SInfo::bodiesContain(item)) {
                found = true;
                name  = SInfo::getName(item);
            }
        }
        if (!found) {
            if (SInfo::barycentresContain(item)) {
                found = true;
                name  = SInfo::getName(item);
            }
        }
    }
};

} // namespace schema
} // namespace gravity
} // namespace brie