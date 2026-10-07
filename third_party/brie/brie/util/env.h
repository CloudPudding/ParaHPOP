#pragma once

#include <string>
#include <vector>

namespace brie {
namespace util {
/**
 * @brief Retrieves an environment variable. Returns empty string if variable
 * is not defined.
 *
 * @param name Name of the environment variable
 * @return std::string Value of the environment variable
 */
inline std::string getEnvVar(const std::string& name)
{
    const char* var = std::getenv(name.c_str());
    return var ? std::string(var) : "";
}


/**
 * @brief Sets an environment variable to the desired value. If the value
 * exists, it overwrites it.
 *
 * @param name Name of the environment variable
 * @param value Value of the environment variable
 */
inline void setEnvVar(const std::string& name, const std::string& value)
{
    setenv(name.c_str(), value.c_str(), true);
}

/**
 * @brief Split `subject` on every occurrence of `delim` and return the
 * resulting segments.
 */
inline std::vector<std::string> split(
    const std::string& subject, const char delim)
{
    std::vector<std::string> result;
    std::stringstream ss(subject);
    std::string item;

    while (getline(ss, item, delim)) {
        result.push_back(item);
    }

    return result;
}

} // namespace util
} // namespace brie
