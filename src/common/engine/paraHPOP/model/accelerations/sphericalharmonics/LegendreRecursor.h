#pragma once

#include "paraHPOP/typedefs.h"
#include "paraHPOP/util.h"

namespace paraHPOP {
namespace model {
namespace accelerations {
namespace sphericalharmonics {

/** @brief POD struct for latitude trigonometric values.
 *
 * Stored externally so it can be computed once and reused,
 * reducing register pressure in the inner loops.
 */
struct PhiState {
    Real sinPhi = 0.0; /**< sin(phi) */
    Real cosPhi = 1.0; /**< cos(phi) */
    Real invCosPhi = 0.0; /**< Zero selects the original division. */

    /** @brief Factory method from position and pre-computed norm @p rn.
     *
     * Uses sinφ = z/r and cosφ = ρ/r where ρ = sqrt(x²+y²) is computed
     * directly from x, y.  This avoids the catastrophic cancellation that
     * sqrt(1 − sin²φ) suffers when sinφ ≈ ±1 within machine epsilon —
     * which would silently round cosφ to zero several decades before the
     * true pole and trigger a 0/0 in @ref LegendreRecursor::divCosPnm.
     *
     * Templated on @p VecT so the call site does not need to import the
     * concrete position type into this header.
     */
    template<typename VecT>
    DEVICEHOST()
    static PhiState fromPosition(const VecT& position, const Real& rn,
        bool useCorrectedDivision = false)
    {
        PhiState state;
        state.sinPhi = position.template get<2>() / rn;
        state.cosPhi = position.template head<2>().norm() / rn;
#ifdef __CUDA_ARCH__
        // Keep exact poles and a conservative near-pole region on division.
        if (useCorrectedDivision && state.cosPhi >= 0x1p-20
            && state.cosPhi <= 1.0)
            state.invCosPhi = 1.0 / state.cosPhi;
#endif
        return state;
    }
};

/** @brief POD struct for diagonal Legendre values P_{m,m} and dP_{m,m}/dphi.
 *
 * Maintained in the outer (m) loop and used to initialize the recursor
 * for each degree loop. This allows the compiler to spill this state
 * between m iterations.
 */
struct DiagonalState {
    Real Pmm  = 1.0; /**< P_{m,m} */
    Real dPmm = 0.0; /**< dP_{m,m}/dphi */

    /** @brief Advance to the next diagonal P_{m+1,m+1}
     * @param cm Precomputed factor c_{m+1} for the diagonal recurrence
     * @param phi The latitude state (sinPhi, cosPhi)
     */
    DEVICEHOST() void advance(const Real& cm, const PhiState& phi)
    {
        const Real newPmm  = cm * phi.cosPhi * Pmm;
        const Real newdPmm = cm * (phi.cosPhi * dPmm - phi.sinPhi * Pmm);
        Pmm                = newPmm;
        dPmm               = newdPmm;
    }
};

/** @brief POD struct containing only the Legendre values needed for
 * accumulation.
 *
 * Extracting these values via a noinline function creates a register barrier,
 * allowing the compiler to spill the full recursor state before accumulation.
 */
struct LegendreValues {
    Real Pnm;       /**< Current P_{n,m} */
    Real dPnmDphi;  /**< Current dP_{n,m}/dphi */
    Real divCosPnm; /**< P_{n,m} / cos(phi) for longitudinal term */
};

/** @brief Lightweight Legendre recursor for degree loop computation.
 *
 * This class is designed to be initialized fresh at the start of each
 * degree loop from a DiagonalState, rather than being passed around
 * between iterations of the order (m) loop. This significantly reduces
 * register pressure by keeping the recursor state local to the inner loop.
 *
 * The recursor only stores the minimal state needed for vertical recurrence:
 * - Current and previous P values for the three-term recurrence
 * - Current and previous dP values for derivative recurrence
 * - Reference to phi state (passed to methods, not stored)
 *
 * Usage pattern:
 *   PhiState phi = PhiState::fromPosition(position, rn);
 *   DiagonalState diag;  // Starts at P_{0,0} = 1
 *   for (m = 0; m <= maxOrder; m++) {
 *       LegendreRecursor leg = LegendreRecursor::fromDiagonal(diag);
 *       // ... degree loop using leg.advance() ...
 *       diag.advance(cm, phi);  // Move to next diagonal
 *   }
 */
class LegendreRecursor {
    using Self = LegendreRecursor;

public:
    /** @brief Initialize recursor from diagonal state P_{m,m}
     * @param diag The diagonal state containing P_{m,m} and dP_{m,m}/dphi
     */
    DEVICEHOST()
    static LegendreRecursor fromDiagonal(const DiagonalState& diag)
    {
        LegendreRecursor rec;
        rec.Pnm_  = diag.Pmm;
        rec.dPnm_ = diag.dPmm;
        // History initialized to zero - first advance will work correctly
        // since for n=m+1: P_{m+1,m} = a_{m+1,m} * sinPhi * P_{m,m} - 0
        rec.Pn1m_  = 0.0;
        rec.dPn1m_ = 0.0;
        return rec;
    }

    /** @brief Default constructor - initializes to P_{0,0} = 1 */
    DEVICEHOST() LegendreRecursor() { }

    /** @brief Advance to degree n+1 within fixed order m (vertical step)
     * @param anm Precomputed factor a_{n+1,m} for the vertical recurrence
     * @param bnm Precomputed factor b_{n+1,m} for the vertical recurrence
     * @param phi The latitude state (sinPhi, cosPhi)
     */
    FORCEINLINE()
    DEVICEHOST() void advance(
        const Real& anm, const Real& bnm, const PhiState& phi)
    {
        // Shift history for P values
        {
            const Real Pn2m = Pn1m_;
            Pn1m_           = Pnm_;
            // Vertical step: P_{n+1,m} from P_{n,m} and P_{n-1,m}
            Pnm_ = anm * phi.sinPhi * Pn1m_ - bnm * Pn2m;
        }

        // Shift history for derivative values
        {
            const Real dPn2m = dPn1m_;
            dPn1m_           = dPnm_;
            // Derivative using recurrence
            dPnm_ = anm * (phi.sinPhi * dPn1m_ + phi.cosPhi * Pn1m_)
                - bnm * dPn2m;
        }
    }

    /** @brief Return current P_{n,m} */
    DEVICEHOST() inline Real Pnm() const { return Pnm_; }

    /** @brief Return current dP_{n,m}/dphi */
    DEVICEHOST() inline Real dPnmDphi() const { return dPnm_; }

    /** @brief Compute P_{n,m} / cos(phi).
     *
     * For m = 0 the longitudinal acceleration term carries an explicit m
     * factor downstream, so we return 0 to skip the division.
     *
     * For m ≥ 1 the direct division is numerically well-conditioned at
     * every latitude except cosφ exactly zero, because Pnm_ carries an
     * intrinsic cos^m(φ) factor from the sectorial seed P_{m,m} =
     * (2m−1)!!·cos^m(φ) which the vertical recurrence preserves.
     *
     * The cosφ = 0 case (literal pole — only reachable when the caller's
     * position has x = y = 0 exactly) is guarded here; the m = 1
     * analytical limit substitution lives in @ref DegreeLoop::applyTerms_.
     */
    DEVICEHOST() inline Real divCosPnm(const idx_t& m, const Real& cPhi) const
    {
        if (m == 0 || cPhi == 0.0)
            return 0.0;
        return Pnm_ / cPhi;
    }

    DEVICEHOST() inline Real divCosPnm(const idx_t& m, const PhiState& phi) const
    {
        if (m == 0 || phi.cosPhi == 0.0)
            return 0.0;
#ifdef __CUDA_ARCH__
        // Leave very small/large values, subnormals and nonfinite data on
        // the original path. The exponent margin protects the FMA residual.
        const Real magnitude = fabs(Pnm_);
        if (phi.invCosPhi != 0.0 && magnitude >= 0x1p-900
            && magnitude <= 0x1p900) {
            const Real quotient = __dmul_rn(Pnm_, phi.invCosPhi);
            const Real residual = __fma_rn(-quotient, phi.cosPhi, Pnm_);
            return __fma_rn(residual, phi.invCosPhi, quotient);
        }
#endif
        return Pnm_ / phi.cosPhi;
    }

    /** @brief Extract values needed for accumulation into a POD struct.
     * @param m Current order (needed for divCosPnm computation)
     * @param phi The latitude state
     */
    FORCEINLINE()
    DEVICEHOST() LegendreValues
        extractValues(const idx_t& m, const Real& cPhi) const
    {
        LegendreValues vals;
        vals.Pnm       = Pnm_;
        vals.dPnmDphi  = dPnm_;
        vals.divCosPnm = divCosPnm(m, cPhi);
        return vals;
    }

    FORCEINLINE()
    DEVICEHOST() LegendreValues
        extractValues(const idx_t& m, const PhiState& phi) const
    {
        return { Pnm_, dPnm_, divCosPnm(m, phi) };
    }

private:
    Real Pnm_  = 1.0; /**< Current P_{n,m} */
    Real dPnm_ = 0.0; /**< Current dP_{n,m}/dphi */

    // History for vertical recurrence (local to degree loop)
    Real Pn1m_  = 0.0; /**< P_{n-1,m} */
    Real dPn1m_ = 0.0; /**< dP_{n-1,m}/dphi for derivative recurrence */
};

// Keep DiagonalHistory as alias for backwards compatibility if needed
using DiagonalHistory = DiagonalState;

} // namespace sphericalharmonics
} // namespace accelerations
} // namespace model
} // namespace paraHPOP
