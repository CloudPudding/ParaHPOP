/* String manipulation utilities */
#pragma once

#include <string>

#include "parm/typedefs.h"

namespace parm {
namespace util {

using nlohmann::json;

struct Strings {

    /** @brief Convert the given string to lower case */
    static void lowerCase(std::string& str)
    {
        for (idx_t i = 0; i < str.size(); i++)
            str[i] = std::tolower(str[i]);
    }

    /** @brief Capitalize the fisrt leter of the given string */
    static void capitalize(std::string& str) { str[0] = std::toupper(str[0]); }

    /** @brief Create a new `nlohmann::json` object with all keys converted to
    lower
     * case. This is a recursive function that can span different level of a
     given
     * Json.
     *
     */
    static json lowerCaseJson(json& jup)
    {

        json jlow;

        /* If jup is an object, modify all the keys */
        if (jup.is_object()) {
            for (auto& el : jup.items()) {
                /* extract key and value */
                std::string key = el.key();
                auto val        = el.value();
                /* make key to lower case */
                lowerCase(key);

                /* if `val` is a nested struct or an array, recursively call
                 * this function */
                if (val.is_object() || val.is_array()) {
                    jlow[key] = Strings::lowerCaseJson(val);
                }
                /* just add the value to the lower-case json otherwise */
                else {
                    jlow[key] = val;
                }
            }
        }
        /* if jup is an array, then just recursively call for all the array
         * elements */
        else if (jup.is_array()) {
            std::vector<json> vec;
            for (auto item : jup) {
                vec.push_back(Strings::lowerCaseJson(item));
            }
            jlow = vec;
        }
        /* Just return the element otherwise */
        else {
            jlow = jup;
        }


        return jlow;
    }

    /** @brief Parse the given string to extrapolate the given number */
    template<typename DataT = Real>
    static DataT parse(const char* str, char** endPtr = nullptr)
    {
        if constexpr (std::is_same<DataT, int>::value) {
            return static_cast<DataT>(std::strtol(str, endPtr, 10));
        } else if constexpr (std::is_same<DataT, long>::value) {
            return static_cast<DataT>(std::strtol(str, endPtr, 10));
        } else if constexpr (std::is_same<DataT, long long>::value) {
            return static_cast<DataT>(std::strtoll(str, endPtr, 10));
        } else if constexpr (std::is_same<DataT, float>::value) {
            return static_cast<DataT>(std::strtof(str, endPtr));
        } else if constexpr (std::is_same<DataT, double>::value) {
            return static_cast<DataT>(std::strtod(str, endPtr));
        } else if constexpr (std::is_same<DataT, long double>::value) {
            return static_cast<DataT>(std::strtold(str, endPtr));
        } else {
            static_assert(feta::core::expr::alwaysFalse<DataT>::value,
                "Unsupported data type for parsing from string");
        }
    }

    /** @brief Split the given string using the given delimiter */
    static std::vector<std::string> split(
        const std::string& str, char delimiter)
    {
        std::vector<std::string> tokens;
        std::string token;
        std::stringstream strStream(str);

        while (std::getline(strStream, token, delimiter)) {
            token = trim(token);
            if (!token.empty()) {
                tokens.push_back(token);
            }
        }
        return tokens;
    }

    /** @brief Remove the leading whitespace from the given string */
    static std::string ltrim(const std::string& s)
    {
        static const std::string whitespace = " \n\r\t\f\v";
        size_t start                        = s.find_first_not_of(whitespace);
        return (start == std::string::npos) ? "" : s.substr(start);
    }

    /** @brief Remove the trailing whitespace from the given string */
    static std::string rtrim(const std::string& s)
    {
        static const std::string whitespace = " \n\r\t\f\v";
        size_t end                          = s.find_last_not_of(whitespace);
        return (end == std::string::npos) ? "" : s.substr(0, end + 1);
    }

    /** @brief Remove leading and trailing whitespace from the given string */
    static std::string trim(const std::string& s) { return rtrim(ltrim(s)); }
};

} // namespace util
} // namespace parm