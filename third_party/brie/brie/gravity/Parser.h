#pragma once

#include "brie/gravity/schema.h"

namespace brie {
namespace gravity {

using namespace schema;

/** @brief Body parser */
struct Parser {

    /** @brief Non-throwing probe: search for the given gravity body identifier
     * (string) scanning all the available systems, and possibly return its
     * Naif ID.
     *
     * @param item  Body name to look up (will be lower-cased internally).
     * @param id    Output Naif ID, written only on success.
     * @return      true if the body was found, false otherwise (no exception). */
    static bool tryParsedNaifId(std::string item, NaifId& id)
    {
        /* lower case string */
        util::Strings::lowerCase(item);

        /* search the item in the available bodies */
        bool found     = false;
        NaifId result  = 0;

        /* run the search */
        constexpr Systems S = static_cast<Systems>(SYSTEMSSIZE - 1);
        stringParser_<S>(found, result, item);

        if (found) { id = result; }
        return found;
    }

    /** @brief Non-throwing probe: search for the given gravity body identifier
     * (Naif ID) scanning all the available systems, and possibly return its
     * Naif ID.
     *
     * @param item  Naif ID to look up.
     * @param id    Output Naif ID, written only on success.
     * @return      true if the body was found, false otherwise (no exception). */
    static bool tryParsedNaifId(const NaifId& item, NaifId& id)
    {
        /* search the item in the available bodies */
        bool found    = false;
        NaifId result = 0;

        /* run the search */
        constexpr Systems S = static_cast<Systems>(SYSTEMSSIZE - 1);
        naifParser_<S>(found, result, item);

        if (found) { id = result; }
        return found;
    }

    /** @brief Non-throwing probe: search for the given gravity body identifier
     * (string) scanning all the available systems, and possibly return its
     * canonical name.
     *
     * @param item  Body name to look up (will be lower-cased internally).
     * @param name  Output canonical name, written only on success.
     * @return      true if the body was found, false otherwise (no exception). */
    static bool tryParsedName(std::string item, std::string& name)
    {
        /* lower case string */
        util::Strings::lowerCase(item);

        /* search the item in the available bodies */
        bool found        = false;
        std::string result = "";

        /* run the search */
        constexpr Systems S = static_cast<Systems>(SYSTEMSSIZE - 1);
        stringParser_<S>(found, result, item);

        if (found) { name = result; }
        return found;
    }

    /** @brief Non-throwing probe: search for the given gravity body identifier
     * (Naif ID) scanning all the available systems, and possibly return its
     * canonical name.
     *
     * @param item  Naif ID to look up.
     * @param name  Output canonical name, written only on success.
     * @return      true if the body was found, false otherwise (no exception). */
    static bool tryParsedName(const NaifId& item, std::string& name)
    {
        /* search the item in the available bodies */
        bool found        = false;
        std::string result = "";

        /* run the search */
        constexpr Systems S = static_cast<Systems>(SYSTEMSSIZE - 1);
        naifParser_<S>(found, result, item);

        if (found) { name = result; }
        return found;
    }

    /** @brief Search for the given gravity body identifier (string) scanning
     * all the available systems, and possibly return its Naif ID */
    static NaifId parsedNaifId(std::string item)
    {
        NaifId id = 0;
        if (!tryParsedNaifId(item, id)) {
            /* lower case for the error message (matches historical behaviour) */
            util::Strings::lowerCase(item);
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
        NaifId id = 0;
        if (!tryParsedNaifId(item, id)) {
            std::stringstream ss;
            ss << "Could not parse: " << item;
            std::string s = ss.str();
            BRIE_THROW(std::runtime_error, s.c_str());
        }
        return id;
    }

    /** @brief Search for the given gravity body identifier (string) scanning
     * all the available systems, and possibly return its canonical name */
    static std::string parsedName(std::string item)
    {
        std::string name;
        if (!tryParsedName(item, name)) {
            /* lower case for the error message (matches historical behaviour) */
            util::Strings::lowerCase(item);
            std::stringstream ss;
            ss << "Could not parse: " << item;
            std::string s = ss.str();
            BRIE_THROW(std::runtime_error, s.c_str());
        }
        return name;
    }

    /** @brief Search for the given gravity body identifier (Naif ID) scanning
     * all the available systems, and possibly return its canonical name */
    static std::string parsedName(NaifId item)
    {
        std::string name;
        if (!tryParsedName(item, name)) {
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

} // namespace gravity
} // namespace brie