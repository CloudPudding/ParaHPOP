#pragma once

#include <algorithm>
#include <cctype>
#include <string>
#include "interface/typedefs.h"
#include "interface/util.h"

namespace interface { namespace config { namespace model { namespace environment {

/** Optional native implementation of the three MATLAB HPOP 4.2.2 Earth
 * corrections. EOP-All uses the same thirteen-column format as MATLAB IERS.m.
 * IERS 2010/FES2004 corrected equations; legacy model string is an alias. */
class EarthCorrections {
public:
    EarthCorrections() = default;
    explicit EarthCorrections(const json& j)
    {
        PARAHPOP_ASSERT(j.is_object(), "environment.earthCorrections must be an object.");
        for (auto it = j.begin(); it != j.end(); ++it) {
            std::string key = it.key();
            std::transform(key.begin(), key.end(), key.begin(),
                [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            key.erase(std::remove(key.begin(), key.end(), '_'), key.end());
            if (key == "model") model_ = it.value().get<std::string>();
            else if (key == "solidearthtides") solidEarthTides_ = it.value().get<bool>();
            else if (key == "oceantides") oceanTides_ = it.value().get<bool>();
            else if (key == "relativity") relativity_ = it.value().get<bool>();
            else if (key == "tidesystem") {
                const std::string value=it.value().get<std::string>();
                PARAHPOP_ASSERT(value=="zero_tide" || value=="tide_free",
                    "earthCorrections.tideSystem must be zero_tide or tide_free.");
                zeroTide_=value=="zero_tide";
            }
            else if (key == "eopfile") eopFile_ = it.value().get<std::string>();
            else PARAHPOP_THROW(std::runtime_error,
                "Unknown environment.earthCorrections field: " + it.key());
        }
        PARAHPOP_ASSERT(model_ == "iers2010_fes2004" || model_ == "matlab_hpop_4_2_2",
            "earthCorrections.model must be iers2010_fes2004 (legacy alias supported).");
        model_ = "iers2010_fes2004";
        PARAHPOP_ASSERT(!tidesActive() || !eopFile_.empty(),
            "Earth tides require earthCorrections.eopFile (HPOP EOP-All format).");
    }

    bool any() const { return solidEarthTides_ || oceanTides_ || relativity_; }
    bool tidesActive() const { return solidEarthTides_ || oceanTides_; }
    bool solidEarthTides() const { return solidEarthTides_; }
    bool oceanTides() const { return oceanTides_; }
    bool relativity() const { return relativity_; }
    bool zeroTide() const { return zeroTide_; }
    const std::string& model() const { return model_; }
    const std::string& eopFile() const { return eopFile_; }
    json to_json() const
    {
        return {{"model", model_}, {"solidEarthTides", solidEarthTides_},
            {"oceanTides", oceanTides_}, {"relativity", relativity_},
            {"eopFile", eopFile_}, {"tideSystem", zeroTide_ ? "zero_tide" : "tide_free"}};
    }

private:
    std::string model_ = "iers2010_fes2004";
    bool solidEarthTides_ = false;
    bool oceanTides_ = false;
    bool relativity_ = false;
    bool zeroTide_ = true;
    std::string eopFile_;
};

}}}} // namespace interface::config::model::environment
