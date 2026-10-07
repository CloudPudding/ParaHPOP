#pragma once

#include <filesystem>
#include <vector>

#include "brie/util/env.h"
#include "brie/util/throw.h"

namespace brie {
namespace util {

using path = std::filesystem::path;

/**
 * The `Paths` class is analogous in function to the $PATH environment variable
 * in Linux systems.
 * It represents a collection of filesystem paths where files are searched for.
 */
class Paths {
public:
    static const char delimiter = ':';

    /** @brief Return a string representation of the paths, separated by `:` */
    std::string toString() const
    {
        std::string out = "";
        for (const auto& p : paths_) {
            if (!out.empty())
                out += delimiter;
            out += p.string();
        }
        return out;
    }

    /**
     * @brief Search the paths for the given file (non-recursively).
     *
     * This checks whether a regular file exists by concatenating the given path
     * `file` to each stored path in turn.
     *
     * If `file` is an absolute path, only that path is checked for existence.
     *
     * @return If a regular file is found, return the path to the first such
     * occurrence. Otherwise, return an empty path.
     */
    path findFile(const path& file) const
    {
        auto isFile = [](const path& p) -> bool {
            return std::filesystem::is_regular_file(p);
        };

        if (file.is_absolute()) {
            return isFile(file) ? file : "";
        }

        for (const path& p : paths_) {
            const path combined = p / file;
            if (isFile(combined)) {
                return combined;
            }
        }
        return "";
    }

    /**
     * @brief Analogous to `findFile`, but throws an exception if not found.
     *
     * @throws `std::runtime_error` if the file could not be found.
     * @return Path to the located file.
     */
    path requireFile(const path& file) const
    {
        auto lazyErrMsg = [&file](const Paths& self) {
            std::stringstream ss;
            ss << "File not found: '" << file << "'. Searched PATH: '"
               << self.toString() << "'";
            return ss.str();
        };
        const util::path path = this->findFile(file);
        BRIE_ASSERT(!path.empty(), lazyErrMsg(*this));
        return path;
    }


    /** @brief Remove all paths. */
    void clear() { paths_.clear(); }

    /** @brief Add a path. Does nothing if the path is empty.
     * Note: does not parse the path for multiple `:`-delimited paths. See
     * `addPaths`.
     */
    void addPath(const path& path)
    {
        if (!path.empty())
            paths_.push_back(path);
    }

    /**
     * @brief Read an environment variable and add paths from its value.
     *
     * If the environment variable is empty or not set, nothing happens.
     * The environment variable may contain multiple paths separated by `:`.
     *
     * @param varName Name of the environment variable to read.
     */
    void addEnv(const std::string& varName)
    {
        const std::string value = getEnvVar(varName);
        if (value.empty())
            return;
        addString(value);
    }

    /**
     * @brief Add one or more paths from a string.
     *
     * If the string is empty, nothing happens.
     * The string may contain multiple paths separated by `:`.
     *
     * @param paths String to parse for paths.
     */
    void addString(const std::string& paths)
    {
        for (const std::string& s : split(paths, delimiter)) {
            if (!s.empty())
                paths_.push_back(s);
        }
    }

private:
    std::vector<path> paths_;
};

struct paths {

    /** @brief Returns the default BRIE paths. Includes env{BRIE_PATH} */
    static Paths brieDefaultPath()
    {
        Paths path;
        path.addString(".");
        path.addEnv("BRIE_PATH");
        return path;
    }
};

} // namespace util
} // namespace brie
