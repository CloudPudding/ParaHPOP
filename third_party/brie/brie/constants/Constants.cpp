#include "brie/constants/Constants.h"

namespace brie {

/** @brief Factory method. Read constants from a JSON file. */
Constants Constants::fromFile(const std::string& file, const util::Paths& paths)
{
    const util::path path     = paths.requireFile(file);
    const nlohmann::json data = Load::json(path);

    detail::UniversalConstants univ = detail::Find::universalConstants(data);
    std::vector<detail::BodyConstants> bodies
        = detail::Find::bodyConstants(data);

    return Constants(univ, bodies);
}

/** @brief Factory method. Read constants from multiple JSON files. */
Constants Constants::fromFiles(
    const std::vector<std::string>& files, const util::Paths& paths)
{
    BRIE_ASSERT(!files.empty(), "Files list may not be empty");

    if (files.size() == 1)
        return fromFile(files[0], paths);

    std::vector<util::path> filePaths;
    for (auto f : files) {
        filePaths.push_back(paths.requireFile(f));
    }

    // TODO: merge multiple files
    BRIE_THROW(std::runtime_error, "Not yet implemented");
}

} // namespace brie