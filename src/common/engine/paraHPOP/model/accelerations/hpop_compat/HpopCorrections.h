#pragma once

// Corrected native Earth corrections (2026-09-24).
// IERS 2010: solid degree 2/3 + induced degree 4, frequency corrections,
// mean-pole solid pole tide; FES2004 ocean waves truncated at degree/order 6;
// Earth Schwarzschild relativity. Units: SI; E: inertial -> Earth-fixed.
// Historical hpop namespace is retained for source compatibility, NOT parity
// with the erroneous formulas in the former local MATLAB reference.
#include <cmath>

#if defined(__CUDACC__)
#define HPOP_HD __host__ __device__
#else
#define HPOP_HD
#endif

namespace paraHPOP { namespace model { namespace accelerations { namespace hpop {

struct Input {
    double r[3], v[3], sun[3], moon[3];
    double E[9];
    double mjdUtc, ut1Utc, ttUtc, xpArcsec, ypArcsec;
};

struct Constants {
    double muEarth = 398600.4415e9;
    double radius = 6378.1363e3; // GGM03C reference, different from WGS84 below.
    double muSun = 132712440041.279419e9;
    double muMoon = 398600.4415e9 / 81.3005682214972154;
    double c = 299792457.999999984;
    double geodeticRadius = 6378.137e3;
    double flattening = 1.0 / 298.257223563;
    bool zeroTide = true; // static GGM03C; false for a conventional tide-free field
};

struct Coefficients {
    double C[7][7] = {};
    double S[7][7] = {};
};

struct Outputs {
    double solid[3] = {};
    double ocean[3] = {};
    double relativity[3] = {};
};

namespace detail {
constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double twoPi = 2.0 * pi;
constexpr double arcsecToRad = pi / (180.0 * 3600.0);

struct FrequencyTerm { signed char multiplier[5]; double inPhase, outPhase; };
namespace hostTables {
#define HPOP_FREQUENCY_TABLE(name, count) static const FrequencyTerm name[count]
#include "HpopFrequencyTerms.inc"
#undef HPOP_FREQUENCY_TABLE
}
#if defined(__CUDACC__)
namespace deviceTables {
// Exactly one definition, owned by EarthCorrections.cu. Avoid per-TU copies.
extern __device__ __constant__ const FrequencyTerm terms20[21];
extern __device__ __constant__ const FrequencyTerm terms21[48];
extern __device__ __constant__ const FrequencyTerm terms22[2];
}
#endif

HPOP_HD inline FrequencyTerm frequencyTerm(int order, int index) {
#if defined(__CUDA_ARCH__)
    if (order == 0) return deviceTables::terms20[index];
    if (order == 1) return deviceTables::terms21[index];
    return deviceTables::terms22[index];
#else
    if (order == 0) return hostTables::terms20[index];
    if (order == 1) return hostTables::terms21[index];
    return hostTables::terms22[index];
#endif
}

HPOP_HD inline double dot(const double a[3], const double b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

HPOP_HD inline void rotate(const double E[9], const double x[3], double out[3]) {
    for (int i=0; i<3; ++i)
        out[i] = E[3*i]*x[0] + E[3*i+1]*x[1] + E[3*i+2]*x[2];
}

HPOP_HD inline double positiveMod(double x, double period) {
    double a = ::fmod(x, period);
    return a < 0.0 ? a + period : a;
}

// Preserve the MJD split and operation ordering of local iauEra00/iauGmst06.
HPOP_HD inline double gmst06(double mjdUt1, double mjdTt) {
    const double eraT = mjdUt1 + (2400000.5 - 2451545.0);
    const double f = ::fmod(mjdUt1, 1.0) + ::fmod(2400000.5, 1.0);
    const double era = positiveMod(twoPi*(f + 0.7790572732640
                                    + 0.00273781191135448*eraT), twoPi);
    const double t = ((2400000.5 - 2451545.0) + mjdTt)/36525.0;
    return positiveMod(era + (0.014506 + (4612.156534 + (1.3915817
                    + (-0.00000044 + (-0.000029956 - 0.0000000368*t)*t)*t)*t)*t)
                    *arcsecToRad, twoPi);
}

HPOP_HD inline void legendreValues(int nmax, double phi, double p[7][7]) {
    for (int n=0;n<7;++n) for (int m=0;m<7;++m) p[n][m]=0.0;
    const double s=::sin(phi), c=::cos(phi);
    p[0][0]=1.0;
    if (nmax == 0) return;
    p[1][1]=::sqrt(3.0)*c;
    for (int n=2;n<=nmax;++n)
        p[n][n]=::sqrt((2.0*n+1)/(2.0*n))*c*p[n-1][n-1];
    for (int n=1;n<=nmax;++n)
        p[n][n-1]=::sqrt(2.0*n+1)*s*p[n-1][n-1];
    for (int m=0;m<=nmax;++m) for (int n=m+2;n<=nmax;++n)
        p[n][m]=::sqrt((2.0*n+1)/((n-m)*double(n+m)))
            *(::sqrt(2.0*n-1)*s*p[n-1][m]
            -::sqrt(((n+m-1)*double(n-m-1))/(2.0*n-3))*p[n-2][m]);
}

HPOP_HD inline void legendre(int nmax, double phi, double p[7][7], double dp[7][7]) {
    for (int n=0;n<7;++n) for (int m=0;m<7;++m) {p[n][m]=0.0;dp[n][m]=0.0;}
    const double s=::sin(phi), c=::cos(phi);
    p[0][0]=1.0;
    if (nmax == 0) return;
    p[1][1]=::sqrt(3.0)*c;
    dp[1][1]=-::sqrt(3.0)*s;
    for (int n=2;n<=nmax;++n) {
        const double k=::sqrt((2.0*n+1)/(2.0*n));
        p[n][n]=k*c*p[n-1][n-1];
        dp[n][n]=k*(c*dp[n-1][n-1]-s*p[n-1][n-1]);
    }
    for (int n=1;n<=nmax;++n) {
        const double k=::sqrt(2.0*n+1);
        p[n][n-1]=k*s*p[n-1][n-1];
        dp[n][n-1]=k*(c*p[n-1][n-1]+s*dp[n-1][n-1]);
    }
    for (int m=0;m<=nmax;++m) for (int n=m+2;n<=nmax;++n) {
        const double k=::sqrt((2.0*n+1)/((n-m)*double(n+m)));
        const double a=::sqrt(2.0*n-1);
        const double b=::sqrt(((n+m-1)*double(n-m-1))/(2.0*n-3));
        p[n][m]=k*(a*s*p[n-1][m]-b*p[n-2][m]);
        dp[n][m]=k*(a*s*dp[n-1][m]+a*c*p[n-1][m]-b*dp[n-2][m]);
    }
}

// Tide-generating bodies require geocentric latitude, at ANY altitude.
HPOP_HD inline double bodyLatitude(const double r[3], const Constants&) {
    return ::atan2(r[2],::hypot(r[0],r[1]));
}

struct BodyGeometry {
    double moonLongitude, sunLongitude;
    double moonRatio, sunRatio;
    double moonP[7][7], sunP[7][7];
};

HPOP_HD inline BodyGeometry bodyGeometry(const Input& in, const Constants& k, int nmax) {
    BodyGeometry b;
    double moon[3],sun[3];
    rotate(in.E,in.moon,moon);rotate(in.E,in.sun,sun);
    b.moonLongitude=::atan2(moon[1],moon[0]);
    b.sunLongitude=::atan2(sun[1],sun[0]);
    b.moonRatio=k.radius/::sqrt(dot(moon,moon));
    b.sunRatio=k.radius/::sqrt(dot(sun,sun));
    legendreValues(nmax,bodyLatitude(moon,k),b.moonP);
    legendreValues(nmax,bodyLatitude(sun,k),b.sunP);
    return b;
}

HPOP_HD inline void clear(Coefficients& out) {
    for(int n=0;n<7;++n) for(int m=0;m<7;++m) out.C[n][m]=out.S[n][m]=0.0;
}

HPOP_HD inline void tideArguments(const Input& in,double args[5],double& thetaG) {
    const double mjdTt=in.mjdUtc+in.ttUtc/86400.0;
    const double t=(mjdTt-51544.5)/36525.0,t2=t*t,t3=t2*t,t4=t3*t;
    args[0]=positiveMod(485868.249036+1717915923.2178*t+31.8792*t2+0.051635*t3-0.00024470*t4,1296000.0)*arcsecToRad;
    args[1]=positiveMod(1287104.79305+129596581.0481*t-0.5532*t2+0.000136*t3-0.00001149*t4,1296000.0)*arcsecToRad;
    args[2]=positiveMod(335779.526232+1739527262.8478*t-12.7512*t2-0.001037*t3+0.00000417*t4,1296000.0)*arcsecToRad;
    args[3]=positiveMod(1072260.70369+1602961601.2090*t-6.3706*t2+0.006593*t3-0.00003169*t4,1296000.0)*arcsecToRad;
    args[4]=positiveMod(450160.398036-6962890.5431*t+7.4722*t2+0.007702*t3-0.00005939*t4,1296000.0)*arcsecToRad;
    thetaG=gmst06(in.mjdUtc+in.ut1Utc/86400.0,mjdTt);
}
 
// IERS 2010 registered mean pole, Table 7.7; output in arcseconds.
HPOP_HD inline void poleWobble(const Input& in,double& m1,double& m2) {
    const double t=(in.mjdUtc+in.ttUtc/86400.0-51544.5)/365.25;
    double x,y;
    if (t<10.0) {
        x=55.974+t*(1.8243+t*(0.18413+t*0.007024));
        y=346.346+t*(1.7896+t*(-0.10729-t*0.000908));
    } else {
        x=23.513+7.6141*t;
        y=358.891-0.6287*t;
    }
    m1=in.xpArcsec-1e-3*x;
    m2=-(in.ypArcsec-1e-3*y);
}

HPOP_HD inline void solidFromGeometry(const Input& in, const Constants& k,
                                     const BodyGeometry& b, Coefficients& out) {
    clear(out);
    const double moonFactor=k.muMoon/k.muEarth*::pow(b.moonRatio,3.0);
    const double sunFactor=k.muSun/k.muEarth*::pow(b.sunRatio,3.0);
    double qC[3],qS[3];
    for (int m=0;m<=2;++m) {
        qC[m]=moonFactor*b.moonP[2][m]*::cos(m*b.moonLongitude)
             +sunFactor*b.sunP[2][m]*::cos(m*b.sunLongitude);
        qS[m]=moonFactor*b.moonP[2][m]*::sin(m*b.moonLongitude)
             +sunFactor*b.sunP[2][m]*::sin(m*b.sunLongitude);
    }
    out.C[2][0]=0.30190/5.0*qC[0];
    out.C[2][1]=0.29830/5.0*qC[1]-0.00144/5.0*qS[1];
    out.S[2][1]=0.00144/5.0*qC[1]+0.29830/5.0*qS[1];
    out.C[2][2]=0.30102/5.0*qC[2]-0.00130/5.0*qS[2];
    out.S[2][2]=0.00130/5.0*qC[2]+0.30102/5.0*qS[2];
    out.C[4][0]=-0.00089/5.0*qC[0];
    out.C[4][1]=-0.00080/5.0*qC[1];out.S[4][1]=-0.00080/5.0*qS[1];
    out.C[4][2]=-0.00057/5.0*qC[2];out.S[4][2]=-0.00057/5.0*qS[2];

    // Direct degree-3 solid tide, IERS (6.6), Table 6.3.
    const double moon3=k.muMoon/k.muEarth*::pow(b.moonRatio,4.0);
    const double sun3=k.muSun/k.muEarth*::pow(b.sunRatio,4.0);
    for (int m=0;m<=3;++m) {
        const double factor=(m==3 ? 0.094 : 0.093)/7.0;
        out.C[3][m]=factor*(moon3*b.moonP[3][m]*::cos(m*b.moonLongitude)
                           +sun3*b.sunP[3][m]*::cos(m*b.sunLongitude));
        if (m) out.S[3][m]=factor*(moon3*b.moonP[3][m]*::sin(m*b.moonLongitude)
                                  +sun3*b.sunP[3][m]*::sin(m*b.sunLongitude));
    }

    double args[5],thetaG;
    tideArguments(in,args,thetaG);
    for (int order=0;order<=2;++order) {
        const int count=order==0?21:(order==1?48:2);
        double dC=0.0,dS=0.0;
        for (int i=0;i<count;++i) {
            const FrequencyTerm term=frequencyTerm(order,i);
            double arg=0.0;
            for (int j=0;j<5;++j) arg+=term.multiplier[j]*args[j];
            const double phase=order*(thetaG+pi)-arg;
            const double c=::cos(phase),s=::sin(phase);
            if (order==0) dC+=1e-12*(term.inPhase*c-term.outPhase*s);
            else if (order==1) {
                dC+=1e-12*(term.inPhase*s+term.outPhase*c);
                dS+=1e-12*(term.inPhase*c-term.outPhase*s);
            } else {
                dC+=1e-12*term.inPhase*c;
                dS-=1e-12*term.inPhase*s;
            }
        }
        out.C[2][order]+=dC;
        out.S[2][order]+=dS;
    }
    // IERS (6.13): remove permanent deformation ONCE for a zero-tide
    // static field. A tide-free static field receives the full tide instead.
    const double permanent=4.4228e-8*(-0.31460)*0.30190;
    if (k.zeroTide) out.C[2][0]-=permanent;
    double m1,m2;
    poleWobble(in,m1,m2);
    out.C[2][1]+=-1.333e-9*(m1+0.0115*m2);
    out.S[2][1]+=-1.333e-9*(m2-0.0115*m1);
}

// Embedded official FES2004 Stokes amplitudes, dimensionless (file * 1e-11).
// Each constituent is evaluated once; coefficients only are summed per n,m.
struct OceanWave { signed char doodson[6]; unsigned short begin,end; };
struct OceanTerm { unsigned char n,m; double cp,sp,cm,sm; };
namespace oceanHost {
#define FES_WAVES(name,count) static const OceanWave name[count]
#define FES_TERMS(name,count) static const OceanTerm name[count]
#include "Fes2004Terms.inc"
#undef FES_WAVES
#undef FES_TERMS
}
#if defined(__CUDACC__)
namespace oceanDevice {
extern __device__ __constant__ const OceanWave waves[FES2004_WAVE_COUNT];
extern __device__ __constant__ const OceanTerm terms[FES2004_TERM_COUNT];
}
#endif
HPOP_HD inline OceanWave oceanWave(int i) {
#if defined(__CUDA_ARCH__)
    return oceanDevice::waves[i];
#else
    return oceanHost::waves[i];
#endif
}
HPOP_HD inline OceanTerm oceanTerm(int i) {
#if defined(__CUDA_ARCH__)
    return oceanDevice::terms[i];
#else
    return oceanHost::terms[i];
#endif
}

HPOP_HD inline void oceanFromEpoch(const Input& in,Coefficients& out) {
    clear(out);
    double f[5],gmst;
    tideArguments(in,f,gmst);
    const double s=f[2]+f[4], h=s-f[3];
    const double beta[6]={gmst+pi-s,s,h,s-f[0],-f[4],h-f[1]};
    for (int w=0;w<FES2004_WAVE_COUNT;++w) {
        const OceanWave wave=oceanWave(w);
        double theta=0.0;
        for (int j=0;j<6;++j) theta+=wave.doodson[j]*beta[j];
        const double co=::cos(theta),si=::sin(theta);
        for (int i=wave.begin;i<wave.end;++i) {
            const OceanTerm t=oceanTerm(i);
            // IERS (6.15): (Cp-iSp)e^(i theta)+(Cm+iSm)e^(-i theta).
            out.C[t.n][t.m]+=(t.cp+t.cm)*co+(t.sp+t.sm)*si;
            // FES zonal convention: retrograde zero, prograde doubled;
            // never double again, and S[n][0] must remain exactly zero.
            if (t.m) out.S[t.n][t.m]+=(t.sp-t.sm)*co-(t.cp-t.cm)*si;
        }
    }
    // Degree-2 ocean pole tide, IERS (6.24); explicitly truncated.
    double m1,m2;poleWobble(in,m1,m2);
    out.C[2][1]+=-2.1778e-10*(m1-0.01724*m2);
    out.S[2][1]+=-1.7232e-10*(m2-0.03365*m1);
}

} // namespace detail

HPOP_HD inline void solidCoefficients(const Input& in,const Constants& k,Coefficients& out) {
    const detail::BodyGeometry b=detail::bodyGeometry(in,k,3);
    detail::solidFromGeometry(in,k,b,out);
}

HPOP_HD inline void oceanCoefficients(const Input& in,const Constants& k,Coefficients& out) {
    (void)k;
    detail::oceanFromEpoch(in,out);
}

// Same fully-normalized Legendre gradient as MATLAB, but only the low-degree
// corrections, never the full static gravity field. As in MATLAB, the
// longitude form requires a position not exactly on the terrestrial z-axis.
HPOP_HD inline void acceleration(const Input& in,const Constants& k,
                                const Coefficients& coeff,double out[3],int maxDegree=6) {
    double r[3];detail::rotate(in.E,in.r,r);
    const double d=::sqrt(detail::dot(r,r));
    const double phi=::asin(r[2]/d),lon=::atan2(r[1],r[0]);
    double p[7][7],dp[7][7];detail::legendre(maxDegree,phi,p,dp);
    double dUdr=0.0,dUdlat=0.0,dUdlon=0.0;
    double cosM[7],sinM[7];
    for(int m=0;m<=maxDegree;++m){cosM[m]=::cos(m*lon);sinM[m]=::sin(m*lon);}
    for(int n=0;n<=maxDegree;++n) {
        const double power=::pow(k.radius/d,double(n));
        const double b1=(-k.muEarth/(d*d))*power*(n+1);
        const double b2=(k.muEarth/d)*power;
        double q1=0.0,q2=0.0,q3=0.0;
        for(int m=0;m<=n;++m) {
            const double value=coeff.C[n][m]*cosM[m]+coeff.S[n][m]*sinM[m];
            q1+=p[n][m]*value;
            q2+=dp[n][m]*value;
            q3+=m*p[n][m]*(coeff.S[n][m]*cosM[m]-coeff.C[n][m]*sinM[m]);
        }
        dUdr+=q1*b1;dUdlat+=q2*b2;dUdlon+=q3*b2;
    }
    const double xy2=r[0]*r[0]+r[1]*r[1],xy=::sqrt(xy2);
    const double q=(1.0/d*dUdr-r[2]/(d*d*xy)*dUdlat);
    const double bf[3]={q*r[0]-(1.0/xy2*dUdlon)*r[1],
                        q*r[1]+(1.0/xy2*dUdlon)*r[0],
                        1.0/d*dUdr*r[2]+xy/(d*d)*dUdlat};
    for(int i=0;i<3;++i) out[i]=in.E[i]*bf[0]+in.E[3+i]*bf[1]+in.E[6+i]*bf[2];
}

HPOP_HD inline void relativity(const Input& in,const Constants& k,double out[3]) {
    const double r=::sqrt(detail::dot(in.r,in.r));
    const double v=::sqrt(detail::dot(in.v,in.v));
    const double rv=detail::dot(in.r,in.v);
    // IERS (10.12), Schwarzschild term; correction ADDED to Newtonian RHS.
    const double f=k.muEarth/(k.c*k.c*::pow(r,3.0));
    for(int i=0;i<3;++i) out[i]=f*((4.0*k.muEarth/r-v*v)*in.r[i]+4.0*rv*in.v[i]);
}

HPOP_HD inline Outputs evaluate(const Input& in,const Constants& k,
                               bool solid,bool ocean,bool relativistic) {
    Outputs out;
    Coefficients coeff;
    if(solid) {
        solidCoefficients(in,k,coeff);
        acceleration(in,k,coeff,out.solid,4);
    }
    if(ocean) {
        oceanCoefficients(in,k,coeff);
        acceleration(in,k,coeff,out.ocean,6);
    }
    if(relativistic) relativity(in,k,out.relativity);
    return out;
}

HPOP_HD inline Outputs evaluate(const Input& in,const Constants& k) {
    return evaluate(in,k,true,true,true);
}

}}}} // namespace paraHPOP::model::accelerations::hpop

#undef HPOP_HD
