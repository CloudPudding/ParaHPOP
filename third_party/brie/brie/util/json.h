#pragma once
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <string>

#include "brie/util.h"
#include "brie/util/throw.h"

namespace brie {

/** @brief File loading functions */
struct Load {

    /**
     * @brief Read binary CBOR-formatted file and return `nlohmann::json`
     * structure
     *
     * @param fileName
     * @param paths (Optional) Path manager
     * @return jsonStructure
     *
     */
    static nlohmann::json cbor(const std::string fileName,
        const util::Paths& paths = util::paths::brieDefaultPath())
    {
        util::path filePath = paths.requireFile(fileName);
        /* Open file  */
        std::ifstream ifs(filePath, std::ios::binary);
        BRIE_ASSERT(ifs.is_open(), "Failed to open file: " + filePath.string());
        BRIE_ASSERT(ifs.good(), "Failed to read file: " + filePath.string());
        /* load data into buffer */
        std::vector<unsigned char> buffer(
            std::istreambuf_iterator<char>(ifs), {});
        /* read json and return */
        try {
            nlohmann::json out = nlohmann::json::from_cbor(buffer);
            while (out.is_array() && out.size() == 1)
                out = out[0];
            return out;
        } catch (const nlohmann::json::exception& e) {
            std::cout << "Failed CBOR parsing of " << filePath.string() << "!"
                      << std::endl;
            BRIE_THROW(std::runtime_error, e.what());
        }
    }

    /**
     * @brief Read JSON-formatted file and return `nlohmann::json` structure
     *
     * @param fileName
     * @param paths (Optional) Path manager
     * @return jsonStructure
     *
     */
    static nlohmann::json json(const std::string fileName,
        const util::Paths& paths = util::paths::brieDefaultPath())
    {
        util::path filePath = paths.requireFile(fileName);
        /* open file */
        std::ifstream ifs(filePath);
        BRIE_ASSERT(ifs.is_open(), "Failed to open file: " + filePath.string());
        BRIE_ASSERT(ifs.good(), "Failed to read file: " + filePath.string());
        try {
            nlohmann::json out = nlohmann::json::parse(ifs);
            while (out.is_array() && out.size() == 1) {
                out = out[0];
            }
            return out;
        } catch (const nlohmann::json::parse_error& e) {
            std::cout << "Failed parsing of " << filePath.string() << "!"
                      << std::endl;
            BRIE_THROW(std::runtime_error, e.what());
        }
    }
};

} // namespace brie