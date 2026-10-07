#pragma once

#include "brie/core/Polynomial.h"
#include "brie/core/Type2BodyUnit.h"
#include "brie/orientations/RotationModelUnit.h"
#include "brie/orientations/metadata/EphUnit.h"

namespace brie {
namespace orientations {

using core::Type2BodyUnit;

static constexpr Real secToDays      = 86400.0;
static constexpr Real secToCenturies = 3155760000.0;


/**
 * @brief Non-owning reference class for a `brie::EphUnit`. Enables several
 * operations on data and metadata, including the actual core computation of
 * Position and Velocities.
 *
 */
template<bool work, bool UseTexture, bool MaybeVolatile = false>
class RefEphUnit {
    /* The evaluator */
    /** TODO: Update for types != chebyshev. May involve the re-definition of
     * ephunit and the introduction of an upper level container type that manges
     * ephemeris data of different interpolation types */
    using PolynomialT = core::Polynomial<UseTexture>;

    /** @brief reference short hand */
    template<bool iwork>
    using VecRef =
        typename feta::vector::Array<Real, 3>::template Ref<iwork>::HandleT;

    /** @brief Dim=4 (quaternion) shmem-scratch handle for the direct
     *  ``unit_type == 4`` quaternion output. */
    template<bool iwork>
    using QuatVecRef =
        typename feta::vector::Array<Real, 4>::template Ref<iwork>::HandleT;

    /** @brief Scalar (Dim=1) shmem-scratch handle used by the ``<iwork>``
     *  leaf accessor overloads. Mirrors ``Polynomial::ScalarRef``. */
    template<bool iwork>
    using ScalarRef =
        typename feta::scalar::Array<Real>::template Ref<iwork>::HandleT;

public:
    /** @brief Volatile is allowed only with work */
    static constexpr bool Volatile = MaybeVolatile && work;
    using MetadataT
        = typename metadata::EphUnit::template Ref<work, MaybeVolatile>;
    using DataT = typename feta::scalar::texture::Array<Real,
        UseTexture>::template Ref<work, MaybeVolatile>;

    /**
     * @brief Factory method to construct from data members
     *
     */
    DEVICEHOST()
    static RefEphUnit make(const MetadataT& metadata, const DataT& data)
    {
        return { metadata, data };
    }

    /**
     * @brief Return the number of bodies in this `brie::EphUnit`
     *
     * @return Nbodies
     *
     */
    DEVICEHOST() vecdim_t nBodyUnits() const { return metadata_.size(); }

    /**
     * @brief Return the total data size
     *
     * @return totDataSize
     *
     */
    DEVICEHOST() idx_t size() const { return data_.size(); }

    /** @brief Prepare the polynomial */
    DEVICEHOST()
    inline void preparePolynomial(PolynomialT& poly, const idx_t& offset) const
    {
#ifdef __CUDA_ARCH__
        if constexpr (UseTexture) {
            poly.ptr       = this->data_.tex();
            poly.texOffset = this->data_.texOffset() + offset;
        } else {
            poly.ptr = this->data_.data() + offset;
        }
#else
        poly.ptr = this->data_.data() + offset;
#endif
    }

    /** @brief Fill the given polynomial for the right ascension terms, for the
     * given body count */
    DEVICEHOST()
    void fillPolynomialRA(PolynomialT& poly, const idx_t& bodyCount) const
    {
        fillPolynomial_<AngleComp::RA>(poly, bodyCount);
    }

    /** @brief build a polynomial and evaluate */
    DEVICEHOST()
    Real RA(const idx_t& bodyCount, const Real& t) const
    {
        PolynomialT poly;
        this->fillPolynomialRA(poly, bodyCount);
        return poly.eval(t / secToCenturies); // requires time in centuries
    }

    /** @brief Fill the given polynomial for the declination terms, for the
     * given body count */
    DEVICEHOST()
    void fillPolynomialDEC(PolynomialT& poly, const idx_t& bodyCount) const
    {
        fillPolynomial_<AngleComp::DEC>(poly, bodyCount);
    }

    /** @brief build a polynomial and evaluate */
    DEVICEHOST()
    Real DEC(const idx_t& bodyCount, const Real& t) const
    {
        PolynomialT poly;
        this->fillPolynomialDEC(poly, bodyCount);
        return poly.eval(t / secToCenturies); // requires time in centuries
    }


    /** @brief Fill the given polynomial for the prime meridian terms, for the
     * given body count */
    DEVICEHOST()
    void fillPolynomialPM(PolynomialT& poly, const idx_t& bodyCount) const
    {
        fillPolynomial_<AngleComp::PM>(poly, bodyCount);
    }

    /** @brief build a polynomial and evaluate */
    DEVICEHOST()
    Real PM(const idx_t& bodyCount, const Real& t) const
    {
        PolynomialT poly;
        this->fillPolynomialPM(poly, bodyCount);
        return poly.eval(t / secToDays); // requires time in days
    }

    /** @brief dRA/dt at the given t, in deg/s.
     * Chain rule on the secToCenturies time scaling: the polynomial is fit
     * for time in centuries, the API takes seconds, so the derivative gets
     * one more 1/secToCenturies factor than the value. */
    DEVICEHOST()
    Real RA_rate(const idx_t& bodyCount, const Real& t) const
    {
        PolynomialT poly;
        this->fillPolynomialRA(poly, bodyCount);
        return poly.evalDeriv(t / secToCenturies) / secToCenturies;
    }

    /** @brief dDEC/dt at the given t, in deg/s. */
    DEVICEHOST()
    Real DEC_rate(const idx_t& bodyCount, const Real& t) const
    {
        PolynomialT poly;
        this->fillPolynomialDEC(poly, bodyCount);
        return poly.evalDeriv(t / secToCenturies) / secToCenturies;
    }

    /** @brief dPM/dt at the given t, in deg/s.
     * PM is fit per-day so the chain factor is 1/secToDays. */
    DEVICEHOST()
    Real PM_rate(const idx_t& bodyCount, const Real& t) const
    {
        PolynomialT poly;
        this->fillPolynomialPM(poly, bodyCount);
        return poly.evalDeriv(t / secToDays) / secToDays;
    }

    /** @brief Fill the given polynomial for the given NP right ascension terms,
     * for the given body count */
    DEVICEHOST()
    void fillNPPolynomialRA(PolynomialT& poly, const idx_t& bodyCount) const
    {
        fillNPPolynomial_<AngleComp::RA>(poly, bodyCount);
    }

    /** @brief Fill the given polynomial for the given NP declination terms, for
     * the given body count */
    DEVICEHOST()
    void fillNPPolynomialDEC(PolynomialT& poly, const idx_t& bodyCount) const
    {
        fillNPPolynomial_<AngleComp::DEC>(poly, bodyCount);
    }

    /** @brief Fill the given polynomial for the given NP prime meridian terms,
     * for the given body count */
    DEVICEHOST()
    void fillNPPolynomialPM(PolynomialT& poly, const idx_t& bodyCount) const
    {
        fillNPPolynomial_<AngleComp::PM>(poly, bodyCount);
    }

    /** @brief Fill the given NP polynomial (barycentre) */
    DEVICEHOST()
    void fillNPPolynomial(PolynomialT& poly, const idx_t& baryCount) const
    {
        preparePolynomial(poly, metadata_.getDataOffset(baryCount));
        poly.degPlusOne_
            = metadata_.template getInt<metadata::bary::PDEG_NP>(baryCount) + 1;
    }

    /** @brief NP Right Ascension evaluation */
    DEVICEHOST()
    Real NP_RA(
        const idx_t& bodyCount, const idx_t& baryCount, const Real& t) const
    {
        PolynomialT poly, np_poly;
        this->fillNPPolynomial(poly, baryCount);
        this->fillNPPolynomialRA(np_poly, bodyCount);
        return np_poly.sinEval(t / secToCenturies, poly);
    }

    /** @brief NP Declination evaluation */
    DEVICEHOST()
    Real NP_DEC(
        const idx_t& bodyCount, const idx_t& baryCount, const Real& t) const
    {
        PolynomialT poly, np_poly;
        this->fillNPPolynomial(poly, baryCount);
        this->fillNPPolynomialDEC(np_poly, bodyCount);
        return np_poly.cosEval(t / secToCenturies, poly);
    }

    /** @brief NP Prime Meridian evaluation */
    DEVICEHOST()
    Real NP_PM(
        const idx_t& bodyCount, const idx_t& baryCount, const Real& t) const
    {
        PolynomialT poly, np_poly;
        this->fillNPPolynomial(poly, baryCount);
        this->fillNPPolynomialPM(np_poly, bodyCount);
        return np_poly.sinEval(t / secToCenturies, poly);
    }

    /** @brief Time derivative of NP_RA, in deg/s.
     * Chain rule: d/dt[Σ a_i sin(arg_i(t/τ)·DEG2RAD)] picks up an extra
     * 1/τ from the inner sub-argument, on top of the chain factors that
     * sinEvalDeriv already folds in. τ = secToCenturies for NP terms. */
    DEVICEHOST()
    Real NP_RA_rate(
        const idx_t& bodyCount, const idx_t& baryCount, const Real& t) const
    {
        PolynomialT poly, np_poly;
        this->fillNPPolynomial(poly, baryCount);
        this->fillNPPolynomialRA(np_poly, bodyCount);
        return np_poly.sinEvalDeriv(t / secToCenturies, poly) / secToCenturies;
    }

    /** @brief Time derivative of NP_DEC, in deg/s. */
    DEVICEHOST()
    Real NP_DEC_rate(
        const idx_t& bodyCount, const idx_t& baryCount, const Real& t) const
    {
        PolynomialT poly, np_poly;
        this->fillNPPolynomial(poly, baryCount);
        this->fillNPPolynomialDEC(np_poly, bodyCount);
        return np_poly.cosEvalDeriv(t / secToCenturies, poly) / secToCenturies;
    }

    /** @brief Time derivative of NP_PM, in deg/s. */
    DEVICEHOST()
    Real NP_PM_rate(
        const idx_t& bodyCount, const idx_t& baryCount, const Real& t) const
    {
        PolynomialT poly, np_poly;
        this->fillNPPolynomial(poly, baryCount);
        this->fillNPPolynomialPM(np_poly, bodyCount);
        return np_poly.sinEvalDeriv(t / secToCenturies, poly) / secToCenturies;
    }

    /* ================================================================
     * ``<iwork>`` leaf-accessor overloads — write the scalar result to
     * a shmem-resident slot at ``out[idx]`` instead of returning a
     * register-resident ``Real``. The body delegates to the scalar
     * accessor + stores the result through ``out[idx]``, so bit-identity
     * vs the scalar API is by construction.
     *
     * These provide the API surface used by the Frames-level cascade
     * (``getRADecPMRateInRadPerSec_<iwork>``, ``bodyAngularVelocity<iwork>``)
     * and any future kernel that wants to consume a single rotation
     * component (RA / DEC / PM or their rates) directly through the
     * shmem-scratch pipeline.
     * ================================================================ */

/* Stamp a single-component ``<iwork>`` leaf overload from its scalar twin.
 * Token-identical to the hand-written ``out[idx] = this->NAME(...)`` body, so
 * the value/iwork pair can no longer drift. ``_T`` = the (bodyCount, t) angle
 * accessors; ``_NP`` = the (bodyCount, baryCount, t) nutation/precession ones. */
#define BRIE_ROT_IWORK_LEAF_T(NAME)                                            \
    template<bool iwork>                                                       \
    DEVICEHOST() void NAME(const SampleIndex& idx, ScalarRef<iwork>& out,      \
        const idx_t& bodyCount, const Real& t) const                          \
    {                                                                          \
        out[idx] = this->NAME(bodyCount, t);                                  \
    }

#define BRIE_ROT_IWORK_LEAF_NP(NAME)                                          \
    template<bool iwork>                                                       \
    DEVICEHOST() void NAME(const SampleIndex& idx, ScalarRef<iwork>& out,      \
        const idx_t& bodyCount, const idx_t& baryCount, const Real& t) const  \
    {                                                                          \
        out[idx] = this->NAME(bodyCount, baryCount, t);                       \
    }

    BRIE_ROT_IWORK_LEAF_T(RA)
    BRIE_ROT_IWORK_LEAF_T(DEC)
    BRIE_ROT_IWORK_LEAF_T(PM)
    BRIE_ROT_IWORK_LEAF_T(RA_rate)
    BRIE_ROT_IWORK_LEAF_T(DEC_rate)
    BRIE_ROT_IWORK_LEAF_T(PM_rate)
    BRIE_ROT_IWORK_LEAF_NP(NP_RA)
    BRIE_ROT_IWORK_LEAF_NP(NP_DEC)
    BRIE_ROT_IWORK_LEAF_NP(NP_PM)
    BRIE_ROT_IWORK_LEAF_NP(NP_RA_rate)
    BRIE_ROT_IWORK_LEAF_NP(NP_DEC_rate)
    BRIE_ROT_IWORK_LEAF_NP(NP_PM_rate)

#undef BRIE_ROT_IWORK_LEAF_T
#undef BRIE_ROT_IWORK_LEAF_NP

    DEVICEHOST()
    Type2BodyUnit<UseTexture, work> getRotBinFrameUnit(
        const idx_t& bodyCount) const
    {
        using namespace metadata::binary;
        return Type2BodyUnit<UseTexture, work>::make(data_.data(),
            metadata_.getDataSize(bodyCount),
            metadata_.getDataOffset(bodyCount),
            metadata_.template getInt<NINTERVALS>(bodyCount),
            metadata_.template getInt<PDEG>(bodyCount), data_.tex(),
            data_.texOffset());
    }

    /** @brief Build the model-based orientation unit (``unit_type == 4``,
     *  the native IPF → IERS2000 family) for the given body count. Mirrors
     *  @ref getRotBinFrameUnit: the whole-array data reference plus this
     *  unit's data offset and the two Lagrange table node counts. */
    DEVICEHOST()
    IERS2000Unit<UseTexture, work> getRotModelUnit(
        const idx_t& bodyCount) const
    {
        using namespace metadata::iers;
        return IERS2000Unit<UseTexture, work>::make(data_,
            metadata_.getDataOffset(bodyCount),
            metadata_.template getInt<N_NODES_NUT>(bodyCount),
            metadata_.template getInt<N_NODES_ERP>(bodyCount));
    }

    /** @brief Direct quaternion (ICRF→target) for a ``unit_type == 4``
     *  model-based unit, bypassing the RA/Dec/PM round-trip (pole-singular
     *  for Earth) — the composed matrix yields the quaternion via Shepperd. */
    DEVICEHOST()
    Vec4R getQuaternionDirect(const Real& epoch, const NaifId& target) const
    {
        idx_t tc = metadata_.getTargetBodyCount(target);
        Vec4R q;
        Vec3R w;
        getRotModelUnit(tc).quatAndOmega(q, w, epoch);
        return q;
    }
    template<bool iwork>
    DEVICEHOST()
    void getQuaternionDirect(const SampleIndex& idx, QuatVecRef<iwork>& out,
        const Real& epoch, const NaifId& target) const
    {
        out[idx] = getQuaternionDirect(epoch, target);
    }

    /** @brief Direct angular velocity (of target w.r.t. ICRF, inertial
     *  components, rad/s) for a ``unit_type == 4`` unit — the exact ω from
     *  the analytic matrix derivative, no Euler-rate kinematics. */
    DEVICEHOST()
    Vec3R getAngularVelocityDirect(
        const Real& epoch, const NaifId& target) const
    {
        idx_t tc = metadata_.getTargetBodyCount(target);
        /* lean ω-only composition path (no Ṙ materialized) — the cudaj
         * omegaResolveKernel spill fix (task #28). */
        return getRotModelUnit(tc).getAngularVelocity(epoch);
    }
    template<bool iwork>
    DEVICEHOST()
    void getAngularVelocityDirect(const SampleIndex& idx, VecRef<iwork>& out,
        const Real& epoch, const NaifId& target) const
    {
        out[idx] = getAngularVelocityDirect(epoch, target);
    }

    /** @brief run all checks */
    DEVICEHOST()
    void allChecks(const Real& epoch, const NaifId& target) const
    {
        checkTarget(target);
        checkEpoch(epoch, target);
    }

    /**
     * @brief Check if the given target is contained in this
     * `brie::RotEphUnit`
     *
     * @param target
     * @return bool True if check is passed
     *
     */
    DEVICEHOST()
    void checkTarget([[maybe_unused]] const NaifId& target) const
    {
#ifdef BRIE_DEBUG_MODE
#ifdef __CUDA_ARCH__
        if (!metadata_.hasBody(target)) {
            BRIE_GPU_THROW(err::TARGET_BODY_NOT_IN_UNIT);
        }
#else
        if (!metadata_.hasBody(target)) {
            BRIE_THROW(
                std::runtime_error, "TARGET body not in the current unit");
        }
#endif
#endif
    }

    /**
     * @brief Check if the requested epoch is available for the target
     *
     * @param epoch
     * @param target
     * @return bool True if all the checks are passed
     *
     */
    DEVICEHOST()
    void checkEpoch([[maybe_unused]] const Real& epoch,
        [[maybe_unused]] const NaifId& target) const
    {
#ifdef BRIE_DEBUG_MODE
        const auto utForCheck
            = metadata_.getUnitType(metadata_.getTargetBodyCount(target));
        if (utForCheck == 3 || utForCheck == 4) {
            // For binary / model-based units, we only check that the epoch is
            // non-negative, as the data is not organized in fixed time
            // intervals and there are no barycentric corrections, so there
            // is no risk of interpolation outside of available data
#ifdef __CUDA_ARCH__
            if (!metadata_.isEpochInRange(epoch, target)) {
                BRIE_GPU_THROW(err::OUT_OF_RANGE_TARGET_EPOCH);
            }
#else
            if (!metadata_.isEpochInRange(epoch, target)) {
                BRIE_THROW(std::runtime_error,
                    "Epoch for TARGET not in the available range");
            }
#endif
        }
#endif
    }

    /**
     * @brief calculate all rotation angles for target body at time epoch
     *
     * @param epoch
     * @param target
     * @return rotationAngles
     *
     */
    DEVICEHOST()
    Vec3R getRADecPM(const Real& epoch, const NaifId& target) const
    {
        allChecks(epoch, target);
        Vec3R out;
        getRADecPM_(out, epoch, target);
        return out;
    }
    template<bool iwork>
    DEVICEHOST()
    void getRADecPM(const SampleIndex& idx, VecRef<iwork>& out,
        const Real& epoch, const NaifId& target) const
    {
        allChecks(epoch, target);
        getRADecPM_<iwork>(idx, out, epoch, target);
    }

    /**
     * @brief Calculate dRA/dt, dDEC/dt, dPM/dt for target body at epoch.
     *
     * Units mirror getRADecPM: deg/s for Type 1 (IAU) bodies, rad/s for
     * Type 3 (binary frame). Frames::getRADecPMRateInRadPerSec_ unifies
     * the output to rad/s.
     */
    DEVICEHOST()
    Vec3R getRADecPMRate(const Real& epoch, const NaifId& target) const
    {
        allChecks(epoch, target);
        Vec3R out;
        getRADecPMRate_(out, epoch, target);
        return out;
    }
    template<bool iwork>
    DEVICEHOST()
    void getRADecPMRate(const SampleIndex& idx, VecRef<iwork>& out,
        const Real& epoch, const NaifId& target) const
    {
        allChecks(epoch, target);
        getRADecPMRate_<iwork>(idx, out, epoch, target);
    }

    /** @brief NAIF id of the base (inertial) frame the unit's angles are
     * expressed against: Type-3 binary-frame units carry it in their
     * INERT_FRAME metadata (1 = J2000/ICRF, 17 = ECLIPJ2000), Type-1 IAU
     * units are J2000-based by construction. `frames::Frames` composes
     * the constant ecliptic bias on top of ECLIPJ2000-based units so its
     * public API always answers ICRF-relative. */
    DEVICEHOST()
    NaifId baseInertFrame(const idx_t& target_count) const
    {
        if (metadata_.getUnitType(target_count) == 3) {
            return metadata_.template getInt<metadata::binary::INERT_FRAME>(
                target_count);
        }
        return 1;
    }

    /** @brief Expose the metadata contained in this
     * `brie::orientations::RefEphUnit`
     */
    DEVICEHOST() const MetadataT& metadata() const { return metadata_; }

    /** @brief Expose the raw data contained in this
     * `brie::orientations::RefEphUnit`
     */
    DEVICEHOST() const DataT& data() const { return data_; }

    /** @brief Clone this reference rotation unit */
    DEVICEHOST() RefEphUnit clone() const { return *this; }

    /* Data members made public for PODification */

    /** @brief Non-owning reference to `brie::EphUnit.metadata_` **/
    MetadataT metadata_;
    /** @brief Non-owning reference to `brie::EphUnit.data_` **/
    DataT data_;

private:
    /** @brief Rotation-angle component selector for the Type-1 polynomial
     *  offset-walkers (right ascension / declination / prime meridian). The
     *  underlying value indexes how many leading PDEG/NP terms precede this
     *  component's coefficients in the packed body record. */
    enum class AngleComp : idx_t { RA = 0, DEC = 1, PM = 2 };

    /** @brief Fill `poly` with the Type-1 IAU coefficients for component `C`.
     *  Single-sources `fillPolynomial{RA,DEC,PM}`: the data offset walks past
     *  the PDEG blocks of the earlier components; the degree is this
     *  component's PDEG + 1. `if constexpr` over the compile-time `C` makes the
     *  generated body token-identical to the former hand-written triple. */
    template<AngleComp C>
    DEVICEHOST()
    void fillPolynomial_(PolynomialT& poly, const idx_t& bodyCount) const
    {
        constexpr idx_t c = static_cast<idx_t>(C);
        idx_t offset      = metadata_.getDataOffset(bodyCount);
        if constexpr (c >= static_cast<idx_t>(AngleComp::DEC))
            offset += metadata_.template getInt<metadata::body::PDEG_RA>(
                          bodyCount)
                + 1;
        if constexpr (c >= static_cast<idx_t>(AngleComp::PM))
            offset += metadata_.template getInt<metadata::body::PDEG_DEC>(
                          bodyCount)
                + 1;
        preparePolynomial(poly, offset);
        if constexpr (C == AngleComp::RA)
            poly.degPlusOne_
                = metadata_.template getInt<metadata::body::PDEG_RA>(bodyCount)
                + 1;
        else if constexpr (C == AngleComp::DEC)
            poly.degPlusOne_
                = metadata_.template getInt<metadata::body::PDEG_DEC>(bodyCount)
                + 1;
        else
            poly.degPlusOne_
                = metadata_.template getInt<metadata::body::PDEG_PM>(bodyCount)
                + 1;
    }

    /** @brief Fill `poly` with the nutation/precession coefficients for
     *  component `C`. Single-sources `fillNPPolynomial{RA,DEC,PM}`: the offset
     *  walks past all three PDEG blocks, then past the NP-term blocks of the
     *  earlier components; the degree is this component's NP-term count (no
     *  `+1`, unlike the polynomial degree above). */
    template<AngleComp C>
    DEVICEHOST()
    void fillNPPolynomial_(PolynomialT& poly, const idx_t& bodyCount) const
    {
        constexpr idx_t c = static_cast<idx_t>(C);
        idx_t offset      = metadata_.getDataOffset(bodyCount);
        offset += metadata_.template getInt<metadata::body::PDEG_RA>(bodyCount)
            + 1;
        offset += metadata_.template getInt<metadata::body::PDEG_DEC>(bodyCount)
            + 1;
        offset += metadata_.template getInt<metadata::body::PDEG_PM>(bodyCount)
            + 1;
        if constexpr (c >= static_cast<idx_t>(AngleComp::DEC))
            offset += metadata_.template getInt<metadata::body::NUM_NP_TERMS_RA>(
                bodyCount);
        if constexpr (c >= static_cast<idx_t>(AngleComp::PM))
            offset
                += metadata_.template getInt<metadata::body::NUM_NP_TERMS_DEC>(
                    bodyCount);
        preparePolynomial(poly, offset);
        if constexpr (C == AngleComp::RA)
            poly.degPlusOne_
                = metadata_.template getInt<metadata::body::NUM_NP_TERMS_RA>(
                    bodyCount);
        else if constexpr (C == AngleComp::DEC)
            poly.degPlusOne_
                = metadata_.template getInt<metadata::body::NUM_NP_TERMS_DEC>(
                    bodyCount);
        else
            poly.degPlusOne_
                = metadata_.template getInt<metadata::body::NUM_NP_TERMS_PM>(
                    bodyCount);
    }

    /**
     * @brief calculate all rotation angles for target body at time epoch
     *
     * @param epoch
     * @param target
     * @return rotationAngles
     *
     */
    DEVICEHOST()
    void getRADecPM_(Vec3R& out, const Real& epoch, const NaifId& target) const
    {
        // get the body unit for the target body
        idx_t target_count = metadata_.getTargetBodyCount(target);

        // Dispatch based on unit type
        if (metadata_.getUnitType(target_count) == 1) {
            getRADecPM_Type1_(out, epoch, target_count);
        } else if (metadata_.getUnitType(target_count) == 3) {
            getRADecPM_Type3_(out, epoch, target_count);
        }
    }
    template<bool iwork>
    DEVICEHOST()
    void getRADecPM_(const SampleIndex& idx, VecRef<iwork>& out,
        const Real& epoch, const NaifId& target) const
    {
        // get the body unit for the target body
        idx_t target_count = metadata_.getTargetBodyCount(target);

        // Dispatch based on unit type
        if (metadata_.getUnitType(target_count) == 1) {
            getRADecPM_Type1_<iwork>(idx, out, epoch, target_count);
        } else if (metadata_.getUnitType(target_count) == 3) {
            getRADecPM_Type3_<iwork>(idx, out, epoch, target_count);
        }
    }

    /** @brief Internal implementation for Type 1 (IAU Models) */
    DEVICEHOST()
    void baryCorrect_(Vec3R& out, Real epoch, const idx_t& target_count) const
    {
        // get the barycenter unit for the target body's barycenter
        idx_t bary_count = metadata_.getTargetBodyCount(
            metadata_.template getInt<metadata::body::BARYCENTER>(
                target_count));
        if (bary_count < metadata_.size()) {
            out.get<0>() += NP_RA(target_count, bary_count, epoch);
            out.get<1>() += NP_DEC(target_count, bary_count, epoch);
            out.get<2>() += NP_PM(target_count, bary_count, epoch);
        }
    }
    template<bool iwork>
    DEVICEHOST()
    void baryCorrect_(const SampleIndex& idx, VecRef<iwork>& out, Real epoch,
        const idx_t& target_count) const
    {
        // get the barycenter unit for the target body's barycenter
        idx_t bary_count = metadata_.getTargetBodyCount(
            metadata_.template getInt<metadata::body::BARYCENTER>(
                target_count));
        if (bary_count < metadata_.size()) {
            out.template get<0>(idx) += NP_RA(target_count, bary_count, epoch);
            out.template get<1>(idx) += NP_DEC(target_count, bary_count, epoch);
            out.template get<2>(idx) += NP_PM(target_count, bary_count, epoch);
        }
    }

    /** @brief First part of type 1 evaluation - the normal body part */
    DEVICEHOST()
    void bodyPart_(
        Vec3R& out, const Real& epoch, const idx_t& target_count) const
    {
        out.get<2>() = PM(target_count, epoch);
        out.get<0>() = RA(target_count, epoch);
        out.get<1>() = DEC(target_count, epoch);
    }
    template<bool iwork>
    DEVICEHOST()
    void bodyPart_(const SampleIndex& idx, VecRef<iwork>& out,
        const Real& epoch, const idx_t& target_count) const
    {
        out.template get<2>(idx) = PM(target_count, epoch);
        out.template get<0>(idx) = RA(target_count, epoch);
        out.template get<1>(idx) = DEC(target_count, epoch);
    }

    /** @brief Do a barycentric correction for Type 1 (IAU) Models */
    DEVICEHOST()
    void getRADecPM_Type1_(
        Vec3R& out, const Real& epoch, const idx_t& target_count) const
    {
        /* ---------------- Body part ------------------------------*/
        bodyPart_(out, epoch, target_count);

        /* ---------------- Barycentric Corrections ---------------- */
        baryCorrect_(out, epoch, target_count);

        /* ---------------- Final Scaling ---------------- */
        normalizeAngles(out);
    }
    template<bool iwork>
    DEVICEHOST()
    void getRADecPM_Type1_(const SampleIndex& idx, VecRef<iwork>& out,
        const Real& epoch, const idx_t& target_count) const
    {
        /* ---------------- Body part ------------------------------*/
        bodyPart_<iwork>(idx, out, epoch, target_count);

        /* ---------------- Barycentric Corrections ---------------- */
        baryCorrect_<iwork>(idx, out, epoch, target_count);

        /* ---------------- Final Scaling ---------------- */
        normalizeAngles<iwork>(idx, out);
    }


    /** @brief Normalize the rotation angles to [0, 360) degrees */
    DEVICEHOST()
    static void normalizeAngles(Vec3R& angles)
    {
        angles.get<0>() = fmod(angles.get<0>(), Real(360.0));
        angles.get<1>() = fmod(angles.get<1>(), Real(360.0));
        angles.get<2>() = fmod(angles.get<2>(), Real(360.0));
    }
    template<bool iwork>
    DEVICEHOST()
    static void normalizeAngles(const SampleIndex& idx, VecRef<iwork>& angles)
    {
        angles.template get<0>(idx)
            = fmod(angles.template get<0>(idx), Real(360.0));
        angles.template get<1>(idx)
            = fmod(angles.template get<1>(idx), Real(360.0));
        angles.template get<2>(idx)
            = fmod(angles.template get<2>(idx), Real(360.0));
    }

    /** @brief Internal implementation for Type 3 (Binary Models) */
    DEVICEHOST()
    void getRADecPM_Type3_(
        Vec3R& out, const Real& epoch, const idx_t& target_count) const
    {
        out = getRotBinFrameUnit(target_count).getValues(epoch);

        out.get<0>() -= M_PI / 2;
        out.get<1>() = M_PI / 2 - out.get<1>();
        out.get<2>() = fmod(out.get<2>(), 2 * M_PI);
    }
    template<bool iwork>
    DEVICEHOST()
    void getRADecPM_Type3_(const SampleIndex& idx, VecRef<iwork>& out,
        const Real& epoch, const idx_t& target_count) const
    {
        out[idx] = getRotBinFrameUnit(target_count).getValues(epoch);

        out.template get<0>(idx) -= M_PI / 2;
        out.template get<1>(idx) = M_PI / 2 - out.template get<1>(idx);
        out.template get<2>(idx)
            = fmod(out.template get<2>(idx), 2 * M_PI);
    }

    /** @brief Internal getRADecPMRate dispatcher by unit type. */
    DEVICEHOST()
    void getRADecPMRate_(
        Vec3R& out, const Real& epoch, const NaifId& target) const
    {
        idx_t target_count = metadata_.getTargetBodyCount(target);

        if (metadata_.getUnitType(target_count) == 1) {
            getRADecPMRate_Type1_(out, epoch, target_count);
        } else if (metadata_.getUnitType(target_count) == 3) {
            getRADecPMRate_Type3_(out, epoch, target_count);
        }
    }
    template<bool iwork>
    DEVICEHOST()
    void getRADecPMRate_(const SampleIndex& idx, VecRef<iwork>& out,
        const Real& epoch, const NaifId& target) const
    {
        idx_t target_count = metadata_.getTargetBodyCount(target);

        if (metadata_.getUnitType(target_count) == 1) {
            getRADecPMRate_Type1_<iwork>(idx, out, epoch, target_count);
        } else if (metadata_.getUnitType(target_count) == 3) {
            getRADecPMRate_Type3_<iwork>(idx, out, epoch, target_count);
        }
    }

    /** @brief Type 1 body rate part — RA_rate / DEC_rate / PM_rate. */
    DEVICEHOST()
    void bodyPart_Rate_(
        Vec3R& out, const Real& epoch, const idx_t& target_count) const
    {
        out.get<2>() = PM_rate(target_count, epoch);
        out.get<0>() = RA_rate(target_count, epoch);
        out.get<1>() = DEC_rate(target_count, epoch);
    }
    template<bool iwork>
    DEVICEHOST()
    void bodyPart_Rate_(const SampleIndex& idx, VecRef<iwork>& out,
        const Real& epoch, const idx_t& target_count) const
    {
        out.template get<2>(idx) = PM_rate(target_count, epoch);
        out.template get<0>(idx) = RA_rate(target_count, epoch);
        out.template get<1>(idx) = DEC_rate(target_count, epoch);
    }

    /** @brief Type 1 NP rate corrections, additive into out. */
    DEVICEHOST()
    void baryCorrect_Rate_(
        Vec3R& out, Real epoch, const idx_t& target_count) const
    {
        idx_t bary_count = metadata_.getTargetBodyCount(
            metadata_.template getInt<metadata::body::BARYCENTER>(
                target_count));
        if (bary_count < metadata_.size()) {
            out.get<0>() += NP_RA_rate(target_count, bary_count, epoch);
            out.get<1>() += NP_DEC_rate(target_count, bary_count, epoch);
            out.get<2>() += NP_PM_rate(target_count, bary_count, epoch);
        }
    }
    template<bool iwork>
    DEVICEHOST()
    void baryCorrect_Rate_(const SampleIndex& idx, VecRef<iwork>& out,
        Real epoch, const idx_t& target_count) const
    {
        idx_t bary_count = metadata_.getTargetBodyCount(
            metadata_.template getInt<metadata::body::BARYCENTER>(
                target_count));
        if (bary_count < metadata_.size()) {
            out.template get<0>(idx)
                += NP_RA_rate(target_count, bary_count, epoch);
            out.template get<1>(idx)
                += NP_DEC_rate(target_count, bary_count, epoch);
            out.template get<2>(idx)
                += NP_PM_rate(target_count, bary_count, epoch);
        }
    }

    /** @brief Type 1 (IAU) rate dispatch.
     * No normalize — rates aren't modular even when the underlying
     * angles are. */
    DEVICEHOST()
    void getRADecPMRate_Type1_(
        Vec3R& out, const Real& epoch, const idx_t& target_count) const
    {
        bodyPart_Rate_(out, epoch, target_count);
        baryCorrect_Rate_(out, epoch, target_count);
    }
    template<bool iwork>
    DEVICEHOST()
    void getRADecPMRate_Type1_(const SampleIndex& idx, VecRef<iwork>& out,
        const Real& epoch, const idx_t& target_count) const
    {
        bodyPart_Rate_<iwork>(idx, out, epoch, target_count);
        baryCorrect_Rate_<iwork>(idx, out, epoch, target_count);
    }

    /** @brief Type 3 (binary frame) rate from chain-interpolator
     * derivatives. The DEC sign flip mirrors the (M_PI/2 - DEC)
     * transform applied to the value path. */
    DEVICEHOST()
    void getRADecPMRate_Type3_(
        Vec3R& out, const Real& epoch, const idx_t& target_count) const
    {
        out = getRotBinFrameUnit(target_count).getDerivatives(epoch);
        out.get<1>() = -out.get<1>();
    }
    template<bool iwork>
    DEVICEHOST()
    void getRADecPMRate_Type3_(const SampleIndex& idx, VecRef<iwork>& out,
        const Real& epoch, const idx_t& target_count) const
    {
        out[idx] = getRotBinFrameUnit(target_count).getDerivatives(epoch);
        out.template get<1>(idx) = -out.template get<1>(idx);
    }
};

/* Explicit instantiation */
extern template class RefEphUnit<false, false>;
extern template class RefEphUnit<true, false>;
extern template class RefEphUnit<true, true>;
extern template class RefEphUnit<false, true>;

} // namespace orientations
} // namespace brie