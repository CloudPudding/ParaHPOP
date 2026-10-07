/**
 * Some helpers for parsing the JSON constants files. See
 * `brie::Constants::fromFile`.
 */
#pragma once

#include "brie/util//json.h"

namespace brie {
namespace detail {

/** @brief Universal constants intermediate structure */
struct UniversalConstants {
    double au;
    double clight;
};

/** @brief Body constants intermediate structure */
struct BodyConstants {
    idx_t naifId;
    double gm;
    double r;
    double aMean;
    double j2;
    double soi;
};

/** @brief Universal constants to json */
inline void to_json(nlohmann::json& j, const UniversalConstants& u)
{
    j = nlohmann::json{ { "CONSTANTS", "" }, { "AU", u.au },
        { "CLIGHT", u.clight } };
}

/** @brief Universal constants from json */
inline void from_json(const nlohmann::json& j, UniversalConstants& u)
{
    j.at("AU").get_to(u.au);
    j.at("CLIGHT").get_to(u.clight);
}

/** @brief Body constants to json */
inline void to_json(nlohmann::json& j, const BodyConstants& b)
{
    j = nlohmann::json{
        { "NAIFID", b.naifId }, //
        { "GM", b.gm },         //
        { "R", b.r },           //
        { "A_MEAN", b.aMean },  //
        { "J2", b.j2 },         //
        { "SOI", b.soi }        //
    };
}

/** @brief Body constants from json */
inline void from_json(const nlohmann::json& j, BodyConstants& b)
{
    j.at("NAIFID").get_to(b.naifId);
    j.at("GM").get_to(b.gm);
    j.at("R").get_to(b.r);
    j.at("A_MEAN").get_to(b.aMean);

    if (j.contains("J2")) {
        j.at("J2").get_to(b.j2);
    } else {
        b.j2 = 0;
    }

    if (j.contains("SOIOld")) {
        j.at("SOIOld").get_to(b.soi);
    } else if (j.contains("SOI")) {
        j.at("SOI").get_to(b.soi);
    } else {
        b.soi = 1e20;
    }
}

/** @brief Constants find routine wrapper structure */
struct Find {

    /**
     * @brief Scan a JSON constants file for the first universal constants block
     *
     * @throws `std::runtime_error` if no universal constants block is found or
     * if it is missing required fields.
     */
    static UniversalConstants universalConstants(const nlohmann::json& j);

    /**
     * @brief Scan a JSON constants file for every per-body constants block.
     *
     * @throws `std::runtime_error` if no body constants block is found or if
     * one is found that is missing required fields.
     */
    static std::vector<BodyConstants> bodyConstants(const nlohmann::json& j);
};

} // namespace detail
} // namespace brie
