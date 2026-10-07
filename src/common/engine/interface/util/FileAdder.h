#pragma once

#include "interface/typedefs.h"
#include "interface/util/err.h"

namespace interface {
namespace util {

struct FileAdder {

    /** @brief Add files to the given vector of files */
    static void append(
        std::vector<std::string>& to, const std::vector<std::string>& from)
    {
        to.insert(to.end(), from.begin(), from.end());
    }

    /** @brief Add the given files from json input to the given vector of files
     */
    static void add(std::vector<std::string>& to, const json& j)
    {
        std::vector<std::string> from;
        recurseFlatten_(from, j);
        append(to, from);
    }

private:
    /** @brief Flatten the given json, validating its items type */
    static void recurseFlatten_(std::vector<std::string>& files, const json& j)
    {
        if (j.is_string()) {
            files.push_back((std::string)j);
        } else if (j.is_array()) {
            for (auto jit : j) {
                recurseFlatten_(files, jit);
            }
        } else {
            PARAHPOP_THROW(std::runtime_error,
                "Only strings or arrays/list of strings are valid json "
                "arguments.");
        }
    }
};

} // namespace util
} // namespace interface