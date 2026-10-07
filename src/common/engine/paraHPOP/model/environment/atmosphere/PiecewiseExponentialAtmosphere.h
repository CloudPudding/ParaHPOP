#pragma once

#include "paraHPOP/model/environment/atmosphere/Atmosphere.h"
#include "paraHPOP/typedefs.h"

#include <utility> // std::exchange

namespace paraHPOP {
namespace model {
namespace environment {
namespace atmosphere {

/**
 * @brief Piecewise exponential atmosphere density model.
 *
 * Generalises @ref ExponentialAtmosphere to an altitude-layered profile:
 * the atmosphere is a stack of self-contained exponential *segments*, each
 * a ``{rho0, h0, scaleHeight, hCutoff}`` block.  Segments are sorted
 * ascending by base altitude ``h0`` at configuration time; at runtime the
 * band for a given altitude ``h`` is the segment with the largest
 * ``h0 <= h`` (band 0 when ``h`` is below the lowest ``h0`` — downward
 * extrapolation, matching the single-block behaviour).  Within the selected
 * band the density is the ordinary exponential
 * ``rho0 * exp(-(h - h0) / scaleHeight)``, suppressed to 0 above that band's
 * own ``hCutoff``.
 *
 * A 1-segment array reproduces @ref ExponentialAtmosphere bit-for-bit:
 * band selection collapses to band 0 (no ``h``-branch) and the eval is the
 * identical expression tree.
 *
 * Storage mirrors the per-body spherical-harmonics ``Coefficients`` type:
 * the owning object holds one feta vector array of segments (one Dim=NCOMP
 * vector per segment, SoA), is move-only, and hands the drag kernel a small
 * trivially-copyable @ref GRef (a feta ``RefArray`` handle — no raw
 * pointers).  The kernel never sees the owning object.
 *
 * Body shape (``flattening``, used by the geodetic-altitude solver) is a
 * *body* property delivered separately (see ``RefEnvironment::flattening``),
 * NOT a per-segment field — which is exactly what keeps the segment array
 * to four components.
 */

/** @brief Per-segment component layout (the vector dimension of the
 *  segment array).  Ordered to match the interface ``atmosphere::Fields``
 *  enum (RHO0, H0, SCALEHEIGHT, HCUTOFF) so the paraHPOP-side extract loop is
 *  a straight copy. */
enum SegmentComponent : idx_t {
    RHO0        = 0, /* reference density at h0 [kg/km^3] */
    H0          = 1, /* base / reference altitude [km] (also the band edge) */
    SCALEHEIGHT = 2, /* scale height [km] */
    HCUTOFF     = 3, /* per-band cutoff altitude [km] */
    /* size marker — the feta vector dimension */
    NCOMP       = 4
};

/**
 * @brief Single-block exponential density policy.
 *
 * The degenerate 1-band case carried as four scalars (no segment array, no
 * band scan).  This is the policy the graph pulls for a body whose
 * atmosphere has exactly one segment: the ``dragExp`` kernel takes it by
 * value (constant-bank params), so the common single-block path pays
 * nothing for band machinery and reads no per-body segment table.
 *
 * It satisfies the same band contract (``selectBand`` / ``cutoffOf`` /
 * ``densityInBand``) as @ref PiecewiseExponentialAtmosphere::GRef, so
 * ``Drag::eval`` and the templated drag kernel body are shared verbatim
 * between the single-block and piecewise kernels — ``selectBand`` here
 * folds to the constant 0, recovering today's straight single exponential
 * bit-for-bit.
 */
struct ExpBlock {
    Real rho0        = Real{ 0 };
    Real h0          = Real{ 0 };
    Real scaleHeight = Real{ 1 };
    Real hCutoff     = Real{ 0 };

    /** @brief Always band 0 (single block); folds to a constant. */
    DEVICEHOST() idx_t selectBand(const Real&) const { return idx_t{ 0 }; }

    /** @brief The single block's cutoff altitude [km]. */
    DEVICEHOST() Real cutoffOf(const idx_t&) const { return hCutoff; }

    /** @brief ``rho0 * exp(-(h-h0)/H)``, 0 above the cutoff — identical
     *  expression tree to the piecewise band-0 eval. */
    DEVICEHOST()
    Real densityInBand(const Real& h, const idx_t&) const
    {
        if (h > hCutoff)
            return Real{ 0 };
        return rho0 * feta::math::exp(-(h - h0) / scaleHeight);
    }

    /** @brief Convenience single-call density (host RHS / unit tests). */
    DEVICEHOST()
    Real density(const Real& h) const { return densityInBand(h, idx_t{ 0 }); }
};

struct PiecewiseExponentialAtmosphere {
    /** @brief One Dim=NCOMP vector per segment; UseTexture=false (read via
     *  ``__ldg`` — a tiny per-body table). */
    using SegArrayT = feta::vector::Array<Real, NCOMP>;

    /** @brief Kernel-held handle.  Holds the feta segment-array ref (a
     *  POD: pointer + sizes) and exposes the band-selection + density eval.
     *  Trivially copyable + default-constructible so it can live in a
     *  ``feta::scalar::Array<GRef>`` (the per-body refs array) and pass by
     *  value as a kernel argument. */
    struct GRef {
        typename SegArrayT::GRef segs;

        /** @brief Number of segments (bands). */
        DEVICEHOST() idx_t nSegments() const { return segs.size(); }

        /** @brief Band index for altitude ``h``: the largest ``h0 <= h``;
         *  band 0 when ``h`` is below the lowest ``h0`` (downward
         *  extrapolation).
         *
         *  Branchless forward count: ``band = #{ s in [1,n) : h0[s] <= h }``.
         *  Segments are sorted ascending by ``h0`` at config time, so the
         *  count equals the largest index with ``h0 <= h`` — bit-identical
         *  to an early-break forward scan, but with **uniform control flow
         *  across a warp** (no data-dependent ``break`` → no divergence;
         *  the predicate lowers to a predicated integer add).  Reuses the
         *  single ``band`` accumulator, so STACK stays 0.  For ``n == 1``
         *  the loop is zero-trip and ``band`` is the constant 0 — the
         *  single-block fast path (also served by the dedicated ``dragExp``
         *  kernel + @ref ExpBlock). */
        DEVICEHOST()
        idx_t selectBand(const Real& h) const
        {
            const idx_t n = segs.size();
            idx_t band    = 0;
            for (idx_t s = 1; s < n; ++s)
                band += static_cast<idx_t>(segs.template get<H0>(s) <= h);
            return band;
        }

        /** @brief This band's cutoff altitude [km] (the drag kernel's
         *  per-sample early-exit reads it after ``selectBand``). */
        DEVICEHOST()
        Real cutoffOf(const idx_t& band) const
        {
            return segs.template get<HCUTOFF>(band);
        }

        /** @brief Density in a known band: ``rho0 * exp(-(h-h0)/H)``, or 0
         *  above this band's cutoff.  Self-safe (re-checks the cutoff) so
         *  callers that skip the early-exit still get the zero. */
        DEVICEHOST()
        Real densityInBand(const Real& h, const idx_t& band) const
        {
            const Real hCut = segs.template get<HCUTOFF>(band);
            if (h > hCut)
                return Real{ 0 };
            const Real rho0 = segs.template get<RHO0>(band);
            const Real h0   = segs.template get<H0>(band);
            const Real H    = segs.template get<SCALEHEIGHT>(band);
            return rho0 * feta::math::exp(-(h - h0) / H);
        }

        /** @brief Convenience single-call density (band scan + eval).  Used
         *  by unit tests and the host RHS; the device kernel uses the
         *  split form to keep its high-altitude early-exit. */
        DEVICEHOST()
        Real density(const Real& h) const
        {
            return densityInBand(h, selectBand(h));
        }

        /** @brief Identity equality (for ``feta::scalar::Array<GRef>``
         *  storage — mirrors ``RefCoefficients``): two handles are the same
         *  iff they point at the same segment data. */
        DEVICEHOST()
        bool operator==(const GRef& o) const
        {
            return segs.data() == o.segs.data();
        }
        DEVICEHOST()
        bool operator!=(const GRef& o) const { return !(*this == o); }
    };

    /** @brief Default: empty (no segments) — a body without an atmosphere. */
    PiecewiseExponentialAtmosphere() = default;

    /** @brief Allocate ``nSegments`` zero-initialised segments; fill via
     *  @ref setSegment. */
    explicit PiecewiseExponentialAtmosphere(const idx_t& nSegments)
        : segments_{ nSegments, Real{ 0 } }
    {
    }

    /** @brief Copy is forbidden (feta arrays are move-only). */
    PiecewiseExponentialAtmosphere(const PiecewiseExponentialAtmosphere&)
        = delete;
    PiecewiseExponentialAtmosphere& operator=(
        const PiecewiseExponentialAtmosphere&)
        = delete;

    /** @brief Move ctor / assignment (noexcept so ``std::vector`` of these
     *  reallocates by move — never by the deleted copy). */
    PiecewiseExponentialAtmosphere(
        PiecewiseExponentialAtmosphere&& other) noexcept
        : segments_{ std::exchange(other.segments_, SegArrayT::flexible()) }
    {
    }
    PiecewiseExponentialAtmosphere& operator=(
        PiecewiseExponentialAtmosphere&& other) noexcept
    {
        if (this != &other)
            segments_ = std::exchange(other.segments_, SegArrayT::flexible());
        return *this;
    }

    /** @brief Write one segment's four parameters (host side, pre-upload). */
    void setSegment(const idx_t& band, const Real& rho0, const Real& h0,
        const Real& scaleHeight, const Real& hCutoff)
    {
        auto ref                            = segments_.hostRef();
        ref.template get<RHO0>(band)        = rho0;
        ref.template get<H0>(band)          = h0;
        ref.template get<SCALEHEIGHT>(band) = scaleHeight;
        ref.template get<HCUTOFF>(band)     = hCutoff;
    }

    /** @brief Number of segments. */
    idx_t nSegments() const { return segments_.size(); }

    /** @brief Whether this is a non-trivial atmosphere (any band with a
     *  positive cutoff).  A default-constructed or hCutoff==0 body reads as
     *  inactive — the same sentinel the single-block model used, so
     *  ``Environment::anyAtmosphere`` keeps its meaning. */
    bool active() const
    {
        const idx_t n = segments_.size();
        if (n == 0)
            return false;
        auto ref = segments_.hostRef();
        for (idx_t s = 0; s < n; ++s)
            if (ref.template get<HCUTOFF>(s) > Real{ 0 })
                return true;
        return false;
    }

    /** @brief Kernel-facing handles (mirror ``Coefficients``). */
    GRef hostRef() const { return GRef{ segments_.hostRef() }; }
    GRef deviceRef() const { return GRef{ segments_.deviceRef() }; }

    /** @brief Band-0 parameters as a single-block @ref ExpBlock (host side).
     *  Feeds the ``dragExp`` kernel for single-segment bodies — the four
     *  scalars are config constants, read here from the host-resident
     *  segment table.  Caller guarantees ``nSegments() >= 1`` (the
     *  Environment always builds at least one band per body). */
    ExpBlock block0() const
    {
        const GRef g = hostRef();
        return ExpBlock{ g.segs.template get<RHO0>(idx_t{ 0 }),
            g.segs.template get<H0>(idx_t{ 0 }),
            g.segs.template get<SCALEHEIGHT>(idx_t{ 0 }),
            g.segs.template get<HCUTOFF>(idx_t{ 0 }) };
    }

    /** @brief Device residency (segment table is immutable post-build, so no
     *  download is needed). */
    void upload(const cudaStream_t& stream = 0) { segments_.upload(stream); }
    void clearDevice() { segments_.clearDevice(); }

    /** @brief Deep copy (feta arrays are move-only; clone to duplicate). */
    PiecewiseExponentialAtmosphere clone() const
    {
        PiecewiseExponentialAtmosphere out;
        out.segments_ = std::move(segments_.clone());
        return out;
    }

    /** @brief Segments array (one Dim=NCOMP vector per band). */
    SegArrayT segments_ = SegArrayT::flexible();
};

} // namespace atmosphere
} // namespace environment
} // namespace model
} // namespace paraHPOP
