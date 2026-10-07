#include "brie/constants/ConstantsJson.h"

namespace brie {
namespace detail {

/**
 * @brief Scan a JSON constants file for the first universal constants block
 *
 * @throws `std::runtime_error` if no universal constants block is found or
 * if it is missing required fields.
 */
UniversalConstants Find::universalConstants(const nlohmann::json& j)
{
    for (const auto& el : j.items()) {
        if (el.value().contains("CONSTANTS")) {
            try {
                return el.value().template get<UniversalConstants>();
            } catch (const nlohmann::json::out_of_range& e) {
                BRIE_THROW(std::runtime_error,
                    "Invalid universal CONSTANTS block found in json! "
                        + std::string(e.what()));
            }
        }
    }
    BRIE_THROW(std::runtime_error,
        "Unable to find universal constants block in json!");
}

/**
 * @brief Scan a JSON constants file for every per-body constants block.
 *
 * @throws `std::runtime_error` if no body constants block is found or if
 * one is found that is missing required fields.
 */
std::vector<BodyConstants> Find::bodyConstants(const nlohmann::json& j)
{
    std::vector<BodyConstants> constants;
    for (const auto& el : j.items()) {
        if (el.value().contains("NAIFID")) {
            try {
                constants.push_back(el.value().template get<BodyConstants>());
            } catch (const nlohmann::json::out_of_range& e) {
                BRIE_THROW(std::runtime_error,
                    "Invalid body constants block found in json! "
                        + std::string(e.what()));
            }
        }
    }
    BRIE_ASSERT(
        !constants.empty(), "Unable to find any body constants in json!");
    return constants;
}

} // namespace detail
} // namespace brie
