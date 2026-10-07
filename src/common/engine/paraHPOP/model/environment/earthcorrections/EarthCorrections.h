#pragma once

#include "paraHPOP/typedefs.h"
#include "interface/config/model/environment/EarthCorrections.h"
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace paraHPOP { namespace model { namespace environment { namespace earthcorrections {

struct Flags {
    bool solidEarthTides = false;
    bool oceanTides = false;
    bool relativity = false;
    bool zeroTide = true;
    DEVICEHOST() bool any() const
    { return solidEarthTides || oceanTides || relativity; }
    DEVICEHOST() bool tidesActive() const { return solidEarthTides || oceanTides; }
};

struct TimeData {
    Real mjdUtc;
    Real ut1Utc;       // seconds
    Real ttUtc;        // seconds
    Real xpArcsec;     // arcseconds, not radians
    Real ypArcsec;     // arcseconds, not radians
    bool valid;
};

enum EopComponent : idx_t { XP = 0, YP = 1, UT1_UTC = 2, TAI_UTC = 3 };
using EopArray = feta::vector::Array<Real, 4>;

/** Read-only SoA view: identical O(1) daily lookup on CPU and GPU. No per-target
 * allocations, host callbacks, runtime MATLAB calls, or interpolated DAT. */
struct View {
    Flags flags{};
    EopArray::GRef eop{};
    Real firstMjdUtc = Real{0};

    DEVICEHOST() bool any() const { return flags.any(); }
    DEVICEHOST() bool tidesActive() const { return flags.tidesActive(); }

    /** Exact seven-term Mjday_TDB.m convention used by the local MATLAB HPOP. */
    DEVICEHOST() static Real tdbMinusTt(Real mjdTt)
    {
        const Real t = (mjdTt - Real{51544.5}) / Real{36525};
        return Real{0.001657} * sin(Real{628.3076} * t + Real{6.2401})
            + Real{0.000022} * sin(Real{575.3385} * t + Real{4.2970})
            + Real{0.000014} * sin(Real{1256.6152} * t + Real{6.1969})
            + Real{0.000005} * sin(Real{606.9777} * t + Real{4.0212})
            + Real{0.000005} * sin(Real{52.9691} * t + Real{0.4444})
            + Real{0.000002} * sin(Real{21.3299} * t + Real{5.5431})
            + Real{0.000010} * sin(Real{628.3076} * t + Real{4.2490});
    }

    DEVICEHOST() static TimeData invalidTime()
    {
        const Real q = static_cast<Real>(nan(""));
        return {q, q, q, q, q, false};
    }

    /** Linear interpolation matches IERS(eop,mjd,'l'), including its direct
     * UT1-UTC interpolation and piecewise-constant daily TAI-UTC. The final
     * EOP row is a right interpolation guard, not an extrapolation day. */
    DEVICEHOST() TimeData timeAtMjdUtc(Real mjdUtc) const
    {
        if (eop.size() < 2 || !(mjdUtc >= firstMjdUtc)
            || !(mjdUtc < firstMjdUtc + static_cast<Real>(eop.size() - 1))
            ) return invalidTime();
        const idx_t i = static_cast<idx_t>(floor(mjdUtc - firstMjdUtc));
        const Real f = mjdUtc - floor(mjdUtc);
        const Real xp = eop.template get<XP>(i);
        const Real yp = eop.template get<YP>(i);
        const Real dut = eop.template get<UT1_UTC>(i);
        return {mjdUtc,
            dut + f * (eop.template get<UT1_UTC>(i + 1) - dut),
            Real{32.184} + eop.template get<TAI_UTC>(i),
            xp + f * (eop.template get<XP>(i + 1) - xp),
            yp + f * (eop.template get<YP>(i + 1) - yp), true};
    }

    /** TDB seconds since J2000 noon -> UTC, using the loaded EOP DAT column.
     * Two fixed-point iterations invert Mjday_TDB. DAT day selection is then
     * corrected on UTC midnight boundaries before interpolation. */
    DEVICEHOST() TimeData timeAt(Real epochEt) const
    {
        if (eop.size() < 2 || !isfinite(epochEt)) return invalidTime();
        Real ttEt = epochEt;
        for (int j = 0; j < 2; ++j)
            ttEt = epochEt - tdbMinusTt(Real{51544.5} + ttEt / Real{86400});
        Real utc = Real{51544.5} + (ttEt - Real{69.184}) / Real{86400};
        // Clamp only the provisional DAT lookup. Final UTC coverage is strict.
        for (int j = 0; j < 3; ++j) {
            Real day = floor(utc - firstMjdUtc);
            if (day < Real{0}) day = Real{0};
            if (day > static_cast<Real>(eop.size() - 1))
                day = static_cast<Real>(eop.size() - 1);
            const idx_t i = static_cast<idx_t>(day);
            utc = Real{51544.5} + (ttEt - Real{32.184}
                - eop.template get<TAI_UTC>(i)) / Real{86400};
        }
        return timeAtMjdUtc(utc);
    }
};

/** Own once per environment; only the EOP table crosses to the GPU. */
class EarthCorrections {
public:
    using Config = interface::config::model::environment::EarthCorrections;
    using GRef = View;
    EarthCorrections() = default;
    explicit EarthCorrections(const Config& cfg)
        : flags_{cfg.solidEarthTides(), cfg.oceanTides(), cfg.relativity(), cfg.zeroTide()}
    {
        if (flags_.tidesActive()) load_(cfg.eopFile());
    }
    EarthCorrections(const EarthCorrections&) = delete;
    EarthCorrections& operator=(const EarthCorrections&) = delete;
    EarthCorrections(EarthCorrections&&) noexcept = default;
    EarthCorrections& operator=(EarthCorrections&&) noexcept = default;
    bool any() const { return flags_.any(); }
    bool tidesActive() const { return flags_.tidesActive(); }
    View hostRef() const { return {flags_, eop_.hostRef(), firstMjdUtc_}; }
    View deviceRef() const { return {flags_, eop_.deviceRef(), firstMjdUtc_}; }
    void upload(const cudaStream_t& stream = 0)
    { if (tidesActive()) eop_.upload(stream); }
    void clearDevice() { if (tidesActive()) eop_.clearDevice(); }

    void assertEpochCoverageEt(Real lo, Real hi) const
    {
        if (!tidesActive()) return;
        PARAHPOP_ASSERT(hi >= lo && hostRef().timeAt(lo).valid
                && hostRef().timeAt(hi).valid,
            "Earth-tide EOP coverage is insufficient for the requested epochs; "
            "include a complete next UTC day for interpolation.");
    }

private:
    struct Record { double mjd, xp, yp, dut, dat; };
    void load_(const std::string& path)
    {
        std::ifstream in(path);
        PARAHPOP_ASSERT(in.good(), "Cannot open Earth-tide EOP file: " + path);
        std::vector<Record> records;
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream row(line);
            int year, month, day;
            double lod, dpsi, deps, dx, dy;
            Record r{};
            if (!(row >> year >> month >> day >> r.mjd >> r.xp >> r.yp
                    >> r.dut >> lod >> dpsi >> deps >> dx >> dy >> r.dat))
                continue;
            PARAHPOP_ASSERT(year >= 1960 && month >= 1 && month <= 12
                    && day >= 1 && day <= 31 && std::isfinite(r.mjd)
                    && std::isfinite(r.xp) && std::isfinite(r.yp)
                    && std::isfinite(r.dut) && std::isfinite(r.dat)
                    && r.mjd == std::floor(r.mjd),
                "Invalid Earth-tide EOP record in: " + path);
            PARAHPOP_ASSERT(records.empty() || r.mjd == records.back().mjd + 1.0,
                "Earth-tide EOP records must be sorted, unique, consecutive days: " + path);
            records.push_back(r);
        }
        PARAHPOP_ASSERT(records.size() >= 2,
            "Earth tides require at least two HPOP EOP-All records in: " + path);
        firstMjdUtc_ = static_cast<Real>(records.front().mjd);
        eop_ = EopArray(static_cast<idx_t>(records.size()), Real{0});
        auto out = eop_.hostRef();
        for (idx_t i = 0; i < static_cast<idx_t>(records.size()); ++i) {
            out.template get<XP>(i) = records[i].xp;
            out.template get<YP>(i) = records[i].yp;
            out.template get<UT1_UTC>(i) = records[i].dut;
            out.template get<TAI_UTC>(i) = records[i].dat;
        }
    }
    Flags flags_{};
    Real firstMjdUtc_ = Real{0};
    EopArray eop_ = EopArray::flexible();
};

}}}} // namespace paraHPOP::model::environment::earthcorrections
