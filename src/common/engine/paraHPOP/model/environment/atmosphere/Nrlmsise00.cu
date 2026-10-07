#include "paraHPOP/model/environment/atmosphere/Nrlmsise00.h"
#include "paraHPOP/model/environment/Environment.h"

#include <math.h>
#include <stdio.h>

namespace paraHPOP {
namespace model {
namespace environment {
namespace atmosphere {
namespace {

struct nrlmsise_flags { int switches[24]; double sw[24]; double swc[24]; };
struct ap_array { double a[7]; };
struct nrlmsise_input {
    int year, doy;
    double sec, alt, g_lat, g_long, lst, f107A, f107, ap;
    ap_array* ap_a;
};
struct nrlmsise_output { double d[9]; double t[2]; };

#define pt CudajNrlPt
#define pd CudajNrlPd
#define ps CudajNrlPs
#define pdl CudajNrlPdl
#define ptl CudajNrlPtl
#define pma CudajNrlPma
#define sam CudajNrlSam
#define ptm CudajNrlPtm
#define pdm CudajNrlPdm
#define pavgm CudajNrlPavgm
#define NRL_COEFFICIENT __device__ __constant__
#define NRL_FUNCTION __device__
#include "paraHPOP/model/environment/atmosphere/detail/Nrlmsise00Data.inl"
struct NrlModel {
#include "paraHPOP/model/environment/atmosphere/detail/Nrlmsise00Model.inl"
};
#undef NRL_COEFFICIENT
#undef NRL_FUNCTION
#undef pt
#undef pd
#undef ps
#undef pdl
#undef ptl
#undef pma
#undef sam
#undef ptm
#undef pdm
#undef pavgm

/* Compile the same NRLMSISE-00 equations for the CPU.  The coefficient
 * storage differs by backend, but the algorithm source is shared so host and
 * device cannot silently drift to different physical models. */
namespace cpu_nrl {
#define pt CudajNrlHostPt
#define pd CudajNrlHostPd
#define ps CudajNrlHostPs
#define pdl CudajNrlHostPdl
#define ptl CudajNrlHostPtl
#define pma CudajNrlHostPma
#define sam CudajNrlHostSam
#define ptm CudajNrlHostPtm
#define pdm CudajNrlHostPdm
#define pavgm CudajNrlHostPavgm
#define NRL_COEFFICIENT static
#define NRL_FUNCTION
#include "paraHPOP/model/environment/atmosphere/detail/Nrlmsise00Data.inl"
struct NrlModel {
#include "paraHPOP/model/environment/atmosphere/detail/Nrlmsise00Model.inl"
};
#undef NRL_COEFFICIENT
#undef NRL_FUNCTION
#undef pt
#undef pd
#undef ps
#undef pdl
#undef ptl
#undef pma
#undef sam
#undef ptm
#undef pdm
#undef pavgm
} // namespace cpu_nrl

constexpr double kSecondsPerDay = 86400.0;
constexpr double kMjdJ2000 = 51544.5;
constexpr double kRadToDeg = 57.2957795130823208768;
constexpr double kDegToRad = 0.0174532925199432957692;
constexpr double kTwoPi = 6.28318530717958647693;

DEVICEHOST() double taiMinusUtc(double mjd)
{
    double dat = 10.0;
    const double dates[] = { 41499,41683,42048,42413,42778,43144,43509,
        43874,44239,44786,45151,45516,46247,47161,47892,48257,48804,
        49169,49534,50083,50630,51179,53736,54832,56109,57204,57754 };
    for (int i = 0; i < 27; ++i)
        if (mjd >= dates[i]) dat = 11.0 + i;
    return dat;
}

/** SPICE ET (TDB seconds past J2000 noon) to UTC MJD. */
DEVICEHOST() double etToMjdUtc(double et)
{
    const double days = et / kSecondsPerDay;
    const double g = 6.240075674 + 0.01720197034 * days;
    const double tdbMinusTt = 0.001657 * sin(g) + 0.000022 * sin(2.0 * g);
    const double mjdTt = kMjdJ2000 + (et - tdbMinusTt) / kSecondsPerDay;
    const double first = mjdTt - (32.184 + taiMinusUtc(mjdTt)) / kSecondsPerDay;
    return mjdTt - (32.184 + taiMinusUtc(first)) / kSecondsPerDay;
}

DEVICEHOST() void calendarDate(double mjd, int& year, int& month, int& day,
    int& hour, int& minute, double& second)
{
    const long a = static_cast<long>(floor(mjd + 2400001.0));
    long b = 0, c = 0;
    if (a < 2299161) c = a + 1524;
    else { b = static_cast<long>((a - 1867216.25) / 36524.25); c = a + b - b / 4 + 1525; }
    const long d = static_cast<long>((c - 122.1) / 365.25);
    const long e = 365 * d + d / 4;
    const long f = static_cast<long>((c - e) / 30.6001);
    day = static_cast<int>(c - e - static_cast<long>(30.6001 * f));
    month = static_cast<int>(f - 1 - 12 * (f / 14));
    year = static_cast<int>(d - 4715 - (7 + month) / 10);
    double sod = (mjd - floor(mjd)) * kSecondsPerDay;
    if (sod < 0.0) sod += kSecondsPerDay;
    hour = static_cast<int>(sod / 3600.0);
    minute = static_cast<int>((sod - hour * 3600.0) / 60.0);
    second = sod - hour * 3600.0 - minute * 60.0;
}

DEVICEHOST() int dayOfYear(int year, int month, int day)
{
    int days[12] = {31,28,31,30,31,30,31,31,30,31,30,31};
    if ((year % 4 == 0 && year % 100 != 0) || year % 400 == 0) days[1] = 29;
    int out = day;
    for (int m = 1; m < month; ++m) out += days[m - 1];
    return out;
}

DEVICEHOST() double wrapHours(double hours)
{
    hours = fmod(hours, 24.0);
    return hours < 0.0 ? hours + 24.0 : hours;
}

/** Greenwich apparent sidereal time used by HPOP's Density_NRL path.
 *
 * HPOP evaluates GMST with UT1 and adds the IAU-1980 equation of the
 * equinoxes.  The atmosphere object does not own the EOP table, so UT1 is
 * approximated by UTC here (the resulting error is below one sidereal
 * second).  The four dominant IAU-1980 nutation terms retain the part that
 * matters to NRLMSISE's hour-scale local-time response without copying a
 * 106-term Earth-orientation series into every GPU thread. */
DEVICEHOST() double hpopGastHours(double mjdUtc)
{
    const double mjdUt1 = mjdUtc;
    const double mjd0 = floor(mjdUt1);
    const double ut1 = kSecondsPerDay * (mjdUt1 - mjd0);
    const double t0 = (mjd0 - kMjdJ2000) / 36525.0;
    const double t = (mjdUt1 - kMjdJ2000) / 36525.0;
    const double gmstSeconds = 24110.54841 + 8640184.812866 * t0
        + 1.002737909350795 * ut1
        + (0.093104 - 6.2e-6 * t) * t * t;
    const double gmstRadians
        = kTwoPi * (gmstSeconds / kSecondsPerDay
            - floor(gmstSeconds / kSecondsPerDay));

    const double mjdTt = mjdUtc
        + (32.184 + taiMinusUtc(mjdUtc)) / kSecondsPerDay;
    const double tt = (mjdTt - kMjdJ2000) / 36525.0;
    const double omega = (125.04452 - 1934.136261 * tt) * kDegToRad;
    const double sunLongitude = (280.4665 + 36000.7698 * tt) * kDegToRad;
    const double moonLongitude = (218.3165 + 481267.8813 * tt) * kDegToRad;
    const double dpsiArcsec = -17.20 * sin(omega)
        - 1.32 * sin(2.0 * sunLongitude)
        - 0.23 * sin(2.0 * moonLongitude)
        + 0.21 * sin(2.0 * omega);
    const double meanObliquity = (23.43929111
        - (46.8150 + (0.00059 - 0.001813 * tt) * tt) * tt / 3600.0)
        * kDegToRad;
    const double equationOfEquinoxes
        = dpsiArcsec / 3600.0 * kDegToRad * cos(meanObliquity);
    return wrapHours((gmstRadians + equationOfEquinoxes)
        * 24.0 / kTwoPi);
}

DEVICEHOST() double nrlLocalTimeHours(double mjdUtc, double secondsOfDay,
    double longitudeRadians, bool useHpopLocalTime)
{
    const double longitudeHours = longitudeRadians * kRadToDeg / 15.0;
    if (useHpopLocalTime)
        return wrapHours(hpopGastHours(mjdUtc) + longitudeHours);
    return wrapHours(secondsOfDay / 3600.0 + longitudeHours);
}

DEVICEHOST() double weatherAt(NrlWeatherArrayT::GRef weather, int component,
    int record)
{
    return weather.data()[component * weather.size() + record];
}

DEVICEHOST() double weatherApAtLag(NrlWeatherArrayT::GRef weather, int record,
    int slot, int lag)
{
    /* The old four-day history array stored the current day at indices
     * 24..31.  For slot 0..7 and lag 1..19 the index is always 5..30:
     * address the identical daily record/bin without per-thread scratch. */
    const int historyIndex = 24 + slot - lag;
    return weatherAt(weather, AP0 + historyIndex % 8,
        record - 3 + historyIndex / 8);
}

DEVICEHOST() bool fillWeather(NrlWeatherArrayT::GRef weather,
    const NrlSettings& settings, double mjdUtc, double& f107,
    double& f107a, double& averageAp, double ap[7])
{
    const int record = static_cast<int>(floor(mjdUtc - settings.weatherStartMjdUtc));
    if (record < 3 || record >= static_cast<int>(weather.size())) return false;
    int slot = static_cast<int>(floor((mjdUtc - floor(mjdUtc)) * 8.0));
    if (slot < 0) slot = 0; if (slot > 7) slot = 7;
    /* HPOP uses adjusted flux by default; observed flux remains selectable
     * for standards-oriented NRLMSISE-00 studies. */
    const int dailyFlux = settings.useAdjustedFlux ? ADJ_F107 : OBS_F107;
    const int averageFlux = settings.useAdjustedFlux ? ADJ_F107A : OBS_F107A;
    f107 = weatherAt(weather, dailyFlux, record - 1);
    f107a = weatherAt(weather, averageFlux, record);
    averageAp = weatherAt(weather, AVG_AP, record);
    ap[0] = averageAp;
    ap[1] = weatherAt(weather, AP0 + slot, record);
    ap[2] = weatherApAtLag(weather, record, slot, 1);
    ap[3] = weatherApAtLag(weather, record, slot, 2);
    ap[4] = weatherApAtLag(weather, record, slot, 3);
    ap[5] = ap[6] = 0.0;
    /* Preserve the original FP64 addition order in both eight-bin means. */
    for (int d = 4; d <= 11; ++d) ap[5] += weatherApAtLag(weather, record, slot, d);
    for (int d = 12; d <= 19; ++d) ap[6] += weatherApAtLag(weather, record, slot, d);
    ap[5] /= 8.0; ap[6] /= 8.0;
    return true;
}

DEVICEHOST() bool geodeticFixed(const Vec3R& fixed, double radius,
    double flattening, double& lat, double& lon, double& alt)
{
    const double x = fixed.template get<0>();
    const double y = fixed.template get<1>();
    const double z = fixed.template get<2>();
    const double rho2 = x * x + y * y;
    if (!(rho2 + z * z > 0.0)) return false;
    const double e2 = flattening * (2.0 - flattening);
    double dz = e2 * z, zdz = z + dz, n = radius, nh = 0.0;
    for (int iteration = 0; iteration < 20; ++iteration) {
        zdz = z + dz;
        nh = sqrt(rho2 + zdz * zdz);
        const double sinPhi = zdz / nh;
        n = radius / sqrt(1.0 - e2 * sinPhi * sinPhi);
        const double next = n * e2 * sinPhi;
        if (fabs(dz - next) < 1.0e-12 * radius) { dz = next; break; }
        dz = next;
    }
    zdz = z + dz;
    nh = sqrt(rho2 + zdz * zdz);
    lon = atan2(y, x); lat = atan2(zdz, sqrt(rho2)); alt = nh - n;
    return isfinite(lat) && isfinite(lon) && isfinite(alt);
}

} // namespace

Real evaluateNrlmsise00HostDensity(const Vec3R& fixedPosition,
    Real epochEt, const Nrlmsise00Atmosphere::GRef& atmosphere,
    Real equatorialRadius, Real flattening)
{
    if (!atmosphere.active)
        return Real{ 0 };

    const double mjdUtc = etToMjdUtc(static_cast<double>(epochEt));
    double lat = 0.0, lon = 0.0, alt = 0.0;
    if (!geodeticFixed(fixedPosition,
            static_cast<double>(equatorialRadius),
            static_cast<double>(flattening), lat, lon, alt))
        return static_cast<Real>(NAN);
    if (alt > static_cast<double>(atmosphere.settings.hCutoff))
        return Real{ 0 };

    double f107 = 0.0, f107a = 0.0, averageAp = 0.0, ap[7]{};
    if (!fillWeather(atmosphere.weather, atmosphere.settings, mjdUtc,
            f107, f107a, averageAp, ap))
        return static_cast<Real>(NAN);

    int year = 0, month = 0, day = 0, hour = 0, minute = 0;
    double second = 0.0;
    calendarDate(mjdUtc, year, month, day, hour, minute, second);
    const double sec = hour * 3600.0 + minute * 60.0 + second;
    const double lst = nrlLocalTimeHours(
        mjdUtc, sec, lon, atmosphere.settings.useHpopLocalTime);

    ap_array apHistory{};
    for (int k = 0; k < 7; ++k)
        apHistory.a[k] = ap[k];
    nrlmsise_input input{};
    input.year = 0;
    input.doy = dayOfYear(year, month, day);
    input.sec = sec;
    input.alt = alt;
    input.g_lat = lat * kRadToDeg;
    input.g_long = lon * kRadToDeg;
    input.lst = lst;
    input.f107A = f107a;
    input.f107 = f107;
    input.ap = averageAp;
    input.ap_a = &apHistory;

    nrlmsise_flags flags{};
    for (int k = 0; k < 24; ++k)
        flags.switches[k] = 1;
    flags.switches[9] = -1;
    nrlmsise_output output{};
    cpu_nrl::NrlModel model{};
    model.gtd7d(&input, &flags, &output);
    if (!isfinite(output.d[5]) || output.d[5] < 0.0)
        return static_cast<Real>(NAN);

    /* switch 0 == 1 gives kg/m^3; paraHPOP uses kg/km^3. */
    return static_cast<Real>(output.d[5] * 1.0e9)
        * atmosphere.settings.densityScale;
}

namespace kernel {

__global__ __launch_bounds__(NrlPreprocessLaunch::maxBlockSize,
    NrlPreprocessLaunch::minBlocksPerSM) void prepareNrlmsise00(
    NrlInputArrayT::GRef inputs,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef pos,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef ephcache,
    GRID_CONSTANT() feta::vector::Array<Real, 4>::GRef quatcache,
    GRID_CONSTANT() feta::scalar::Array<Real>::GRef::HandleT epochs,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated,
    GRID_CONSTANT() NrlWeatherArrayT::GRef weather,
    GRID_CONSTANT() NrlSettings settings,
    GRID_CONSTANT() Real equatorialRadius,
    GRID_CONSTANT() Real flattening)
{
    const SampleIndex i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (i.global() >= inputs.size()) return;
    if (terminated[i]) { inputs.template get<NRL_VALID>(i) = Real{ 0 }; return; }

    const double mjdUtc = etToMjdUtc(epochs[i]);
    const Vec3R relative = (pos - ephcache)[i];
    OrientationsT::RotationT rotation = { quatcache[i] };
    const Vec3R fixed = rotation.rotate(relative);
    double lat = 0.0, lon = 0.0, alt = 0.0;
    if (!geodeticFixed(fixed, equatorialRadius, flattening, lat, lon, alt)) {
        inputs.template get<NRL_VALID>(i) = Real{ -1 }; return;
    }
    if (alt > settings.hCutoff) {
        inputs.template get<NRL_VALID>(i) = Real{ 2 }; return;
    }
    double f107 = 0.0, f107a = 0.0, averageAp = 0.0, ap[7]{};
    if (!fillWeather(weather, settings, mjdUtc, f107, f107a, averageAp, ap)) {
        inputs.template get<NRL_VALID>(i) = Real{ -1 }; return;
    }
    int year, month, day, hour, minute; double second;
    calendarDate(mjdUtc, year, month, day, hour, minute, second);
    const double sec = hour * 3600.0 + minute * 60.0 + second;
    const double lst
        = nrlLocalTimeHours(mjdUtc, sec, lon, settings.useHpopLocalTime);

    inputs.template get<NRL_DOY>(i) = static_cast<Real>(dayOfYear(year, month, day));
    inputs.template get<NRL_SEC>(i) = sec;
    inputs.template get<NRL_ALT>(i) = alt;
    inputs.template get<NRL_LAT>(i) = lat * kRadToDeg;
    inputs.template get<NRL_LON>(i) = lon * kRadToDeg;
    inputs.template get<NRL_LST>(i) = lst;
    inputs.template get<NRL_F107A>(i) = f107a;
    inputs.template get<NRL_F107>(i) = f107;
    inputs.template get<NRL_AVG_AP>(i) = averageAp;
    for (int k = 0; k < 7; ++k)
        inputs.data()[(NRL_AP0 + k) * inputs.size() + i.global()] = ap[k];
    inputs.template get<NRL_VALID>(i) = Real{ 1 };
}

__global__ __launch_bounds__(NrlDensityLaunch::maxBlockSize,
    NrlDensityLaunch::minBlocksPerSM) void evaluateNrlmsise00(
    GRID_CONSTANT() NrlInputArrayT::GRef inputs,
    feta::scalar::Array<Real>::GRef::HandleT density,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated,
    GRID_CONSTANT() Real densityScale)
{
    const SampleIndex i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (i.global() >= inputs.size()) return;
    const Real valid = inputs.template get<NRL_VALID>(i);
    if (terminated[i] || valid == Real{ 0 } || valid == Real{ 2 }) {
        density[i] = Real{ 0 }; return;
    }
    if (valid < Real{ 0 }) { density[i] = static_cast<Real>(NAN); return; }

    ap_array apHistory{};
    for (int k = 0; k < 7; ++k)
        apHistory.a[k] = inputs.data()[(NRL_AP0 + k) * inputs.size() + i.global()];
    nrlmsise_input input{};
    input.year = 0;
    input.doy = static_cast<int>(inputs.template get<NRL_DOY>(i));
    input.sec = inputs.template get<NRL_SEC>(i);
    input.alt = inputs.template get<NRL_ALT>(i);
    input.g_lat = inputs.template get<NRL_LAT>(i);
    input.g_long = inputs.template get<NRL_LON>(i);
    input.lst = inputs.template get<NRL_LST>(i);
    input.f107A = inputs.template get<NRL_F107A>(i);
    input.f107 = inputs.template get<NRL_F107>(i);
    input.ap = inputs.template get<NRL_AVG_AP>(i);
    input.ap_a = &apHistory;
    nrlmsise_flags flags{};
    for (int k = 0; k < 24; ++k) flags.switches[k] = 1;
    flags.switches[9] = -1;
    nrlmsise_output output{};
    NrlModel model{};
    model.gtd7d(&input, &flags, &output);
    /* switch 0 == 1 gives kg/m^3; paraHPOP uses kg/km^3. */
    density[i] = (!isfinite(output.d[5]) || output.d[5] < 0.0)
        ? static_cast<Real>(NAN)
        : static_cast<Real>(output.d[5] * 1.0e9) * densityScale;
}

} // namespace kernel
} // namespace atmosphere
} // namespace environment
} // namespace model
} // namespace paraHPOP
