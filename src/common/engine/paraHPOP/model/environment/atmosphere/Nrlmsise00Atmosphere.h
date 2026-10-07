#pragma once

#include "paraHPOP/typedefs.h"
#include "interface/config/model/environment/Atmosphere.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace paraHPOP {
namespace model {
namespace environment {
namespace atmosphere {

/** One daily CelesTrak SW-All record, stored as an SoA vector on the GPU. */
enum NrlWeatherComponent : idx_t {
    OBS_F107 = 0,
    OBS_F107A = 1,
    ADJ_F107 = 2,
    ADJ_F107A = 3,
    AVG_AP = 4,
    AP0 = 5,
    AP1 = 6,
    AP2 = 7,
    AP3 = 8,
    AP4 = 9,
    AP5 = 10,
    AP6 = 11,
    AP7 = 12,
    NRL_WEATHER_COMPONENTS = 13
};

/** Per-sample NRLMSISE input cache written by the preprocessor kernel. */
enum NrlInputComponent : idx_t {
    NRL_DOY = 0,
    NRL_SEC = 1,
    NRL_ALT = 2,
    NRL_LAT = 3,
    NRL_LON = 4,
    NRL_LST = 5,
    NRL_F107A = 6,
    NRL_F107 = 7,
    NRL_AVG_AP = 8,
    NRL_AP0 = 9,
    NRL_AP1 = 10,
    NRL_AP2 = 11,
    NRL_AP3 = 12,
    NRL_AP4 = 13,
    NRL_AP5 = 14,
    NRL_AP6 = 15,
    NRL_VALID = 16,
    NRL_INPUT_COMPONENTS = 17
};

using NrlWeatherArrayT = feta::vector::Array<Real, NRL_WEATHER_COMPONENTS>;
using NrlInputCacheStoreT = feta::core::memory::Container<Real,
    feta::core::memory::Device::CUDA_DEVICE, NRL_INPUT_COMPONENTS>;
using NrlInputArrayT = feta::vector::Array<Real, NRL_INPUT_COMPONENTS>;

struct NrlSettings {
    Real weatherStartMjdUtc = Real{ 0 };
    Real hCutoff = Real{ 1000 };
    Real densityScale = Real{ 1 };
    bool useAdjustedFlux = true;
    bool useHpopLocalTime = true;
};

/** Host owner for one body's cropped space-weather table. */
class Nrlmsise00Atmosphere {
public:
    using InterfaceAtmosphere =
        interface::config::model::environment::Atmosphere;

    /** Lightweight backend-specific view used by RefEnvironment.  The
     *  weather handle points at host memory in the CPU propagator and at
     *  device memory in the CUDA propagator; settings and the active flag
     *  are copied by value. */
    struct GRef {
        NrlWeatherArrayT::GRef weather{};
        NrlSettings settings{};
        bool active = false;

        DEVICEHOST()
        bool operator==(const GRef& other) const
        {
            return weather.data() == other.weather.data()
                && weather.size() == other.weather.size()
                && settings.weatherStartMjdUtc
                    == other.settings.weatherStartMjdUtc
                && settings.hCutoff == other.settings.hCutoff
                && settings.densityScale == other.settings.densityScale
                && settings.useAdjustedFlux == other.settings.useAdjustedFlux
                && settings.useHpopLocalTime == other.settings.useHpopLocalTime
                && active == other.active;
        }

        DEVICEHOST()
        bool operator!=(const GRef& other) const { return !(*this == other); }
    };

    Nrlmsise00Atmosphere() = default;

    explicit Nrlmsise00Atmosphere(const InterfaceAtmosphere& cfg)
        : active_{ cfg.isNrlmsise00() }
    {
        if (!active_)
            return;
        settings_.hCutoff = static_cast<Real>(cfg.nrlHCutoff());
        settings_.densityScale = static_cast<Real>(cfg.densityScale());
        load_(cfg.spaceWeatherFile(), cfg.coverageStartMjdUtc(),
            cfg.coverageEndMjdUtc());
    }

    Nrlmsise00Atmosphere(const Nrlmsise00Atmosphere&) = delete;
    Nrlmsise00Atmosphere& operator=(const Nrlmsise00Atmosphere&) = delete;
    Nrlmsise00Atmosphere(Nrlmsise00Atmosphere&&) noexcept = default;
    Nrlmsise00Atmosphere& operator=(Nrlmsise00Atmosphere&&) noexcept = default;

    bool active() const { return active_; }
    idx_t size() const { return weather_.size(); }
    const NrlSettings& settings() const { return settings_; }
    NrlWeatherArrayT::GRef hostWeather() const { return weather_.hostRef(); }
    NrlWeatherArrayT::GRef deviceWeather() const { return weather_.deviceRef(); }
    GRef hostRef() const
    {
        return { hostWeather(), settings_, active_ };
    }
    GRef deviceRef() const
    {
        return { deviceWeather(), settings_, active_ };
    }
    void upload(const cudaStream_t& stream = 0)
    {
        if (active_)
            weather_.upload(stream);
    }
    void clearDevice()
    {
        if (active_)
            weather_.clearDevice();
    }

private:
    struct HostRecord {
        int year = 0, month = 0, day = 0;
        int averageAp = 0;
        int ap[8]{};
        double observedF107 = 0.0;
        double observedCenteredF81 = 0.0;
        double adjustedF107 = 0.0;
        double adjustedCenteredF81 = 0.0;
        double mjdUtc = 0.0;
    };

    static double calendarMjd_(int year, int month, int day)
    {
        const int a = (14 - month) / 12;
        const int y = year + 4800 - a;
        const int m = month + 12 * a - 3;
        const long jdn = day + (153 * m + 2) / 5 + 365L * y + y / 4
            - y / 100 + y / 400 - 32045;
        return static_cast<double>(jdn) - 2400001.0;
    }

    static bool parseRecord_(const std::string& line, HostRecord& r)
    {
        std::istringstream in(line);
        int bsrn = 0, nd = 0, kp[8]{}, sumKp = 0;
        double cp = 0.0;
        int c9 = 0, isn = 0, q = 0;
        double adjustedLastF81 = 0.0, observedLastF81 = 0.0;
        if (!(in >> r.year >> r.month >> r.day >> bsrn >> nd))
            return false;
        for (int& value : kp)
            if (!(in >> value)) return false;
        if (!(in >> sumKp)) return false;
        for (int& value : r.ap)
            if (!(in >> value)) return false;
        if (!(in >> r.averageAp >> cp >> c9 >> isn >> r.adjustedF107 >> q
                >> r.adjustedCenteredF81 >> adjustedLastF81 >> r.observedF107
                >> r.observedCenteredF81 >> observedLastF81))
            return false;
        r.mjdUtc = calendarMjd_(r.year, r.month, r.day);
        return true;
    }

    void load_(const std::string& path, double requestedStart,
        double requestedEnd)
    {
        std::ifstream file(path);
        PARAHPOP_ASSERT(file.good(),
            "Unable to open NRLMSISE-00 space-weather file: " + path);

        std::vector<HostRecord> all;
        std::string line;
        while (std::getline(file, line)) {
            HostRecord r;
            if (parseRecord_(line, r))
                all.push_back(r);
        }
        PARAHPOP_ASSERT(!all.empty(),
            "No CelesTrak SW-All records found in: " + path);

        const double keepStart = requestedStart >= 0.0
            ? std::floor(requestedStart) - 4.0 : all.front().mjdUtc;
        const double keepEnd = requestedEnd >= 0.0
            ? std::floor(requestedEnd) + 1.0 : all.back().mjdUtc;
        std::vector<HostRecord> selected;
        for (const auto& r : all)
            if (r.mjdUtc >= keepStart && r.mjdUtc <= keepEnd)
                selected.push_back(r);
        PARAHPOP_ASSERT(selected.size() >= 5,
            "NRLMSISE-00 weather coverage is missing or too short; at least "
            "four look-back days plus the propagation day are required.");
        for (idx_t i = 1; i < static_cast<idx_t>(selected.size()); ++i)
            PARAHPOP_ASSERT(selected[i].mjdUtc == selected[i - 1].mjdUtc + 1.0,
                "NRLMSISE-00 space-weather records must be consecutive days.");

        settings_.weatherStartMjdUtc = static_cast<Real>(selected.front().mjdUtc);
        weather_ = NrlWeatherArrayT(
            static_cast<idx_t>(selected.size()), Real{ 0 });
        auto dst = weather_.hostRef();
        for (idx_t i = 0; i < static_cast<idx_t>(selected.size()); ++i) {
            dst.template get<OBS_F107>(i) = selected[i].observedF107;
            dst.template get<OBS_F107A>(i) = selected[i].observedCenteredF81;
            dst.template get<ADJ_F107>(i) = selected[i].adjustedF107;
            dst.template get<ADJ_F107A>(i) = selected[i].adjustedCenteredF81;
            dst.template get<AVG_AP>(i) = static_cast<Real>(selected[i].averageAp);
            for (idx_t k = 0; k < 8; ++k)
                dst.data()[(AP0 + k) * dst.size() + i]
                    = static_cast<Real>(selected[i].ap[k]);
        }
    }

    bool active_ = false;
    NrlSettings settings_{};
    NrlWeatherArrayT weather_ = NrlWeatherArrayT::flexible();
};

} // namespace atmosphere
} // namespace environment
} // namespace model
} // namespace paraHPOP
