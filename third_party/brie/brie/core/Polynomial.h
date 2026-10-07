#pragma once

#include "brie/typedefs.h"
#include "brie/util.h"

namespace brie {
namespace core {

static constexpr Real DEG2RAD = M_PI / 180.0;

/** @brief feta handle switcher */
template<bool UseTexture>
struct Handle {
    using T = typename feta::scalar::texture::detail::RefArray<Real, false,
        UseTexture>::HandleT;
};

/** @brief Polynomial component */
template<bool UseTexture>
class Polynomial : public Handle<UseTexture>::T {
    using ParentT = Handle<UseTexture>::T;

public:
    /** @brief Factory method to construct from parent and degree */
    DEVICEHOST()
    static Polynomial make(const ParentT& parent, const idx_t& degree)
    {
        return Polynomial{ parent, degree };
    }

    /** @brief Evaluate the polynomial at the given t */
    DEVICEHOST()
    Real eval(const Real& t) const
    {
        Real result = 0;
        for (idx_t i = 0; i < degPlusOne_; i++) {
            result = fma(result, t, revAccess(i));
        }
        return result;
    }

    /** @brief Sine/cosine-modulated Horner:
     *  ``Σ_i (*this)[i] · trig(p_i(t)·DEG2RAD)`` where ``trig`` is ``sin``
     *  (``IsSin``) or ``cos``. Single-sources the former ``sinEval`` /
     *  ``cosEval`` twins (they differed only by the ``sin``↔``cos`` token);
     *  the ``if constexpr (IsSin)`` keeps codegen bit-identical to the two
     *  hand-written originals. ``p_i`` is the ``inner`` polynomial advanced
     *  by ``inner.next()`` each outer step. */
    template<bool IsSin>
    DEVICEHOST()
    Real trigEval(const Real t, Polynomial inner) const
    {
        Real result           = 0;
        const idx_t inner_deg = inner.degPlusOne_;
        for (idx_t i = 0; i < degPlusOne_; i++) {
            /* Manually inline inner.eval(t) to avoid function call overhead */
            Real inner_result = 0;
            Real tpow         = 1;
            for (idx_t j = 0; j < inner_deg; j++) {
                inner_result = fma(inner[j], tpow, inner_result);
                tpow *= t;
            }
            /* Apply sine/cosine and accumulate */
            if constexpr (IsSin) {
                result = fma((*this)[i],
                    feta::math::sin(inner_result * DEG2RAD), result);
            } else {
                result = fma((*this)[i],
                    feta::math::cos(inner_result * DEG2RAD), result);
            }

            /* Shift to next inner polynomial */
            inner.next();
        }
        return result;
    }

    /** @brief Run a sine-based evaluation using the other polynomial as inner
     * item */
    DEVICEHOST()
    Real sinEval(const Real t, Polynomial inner) const
    {
        return trigEval<true>(t, inner);
    }

    /** @brief Run a cosine-based evaluation using the other polynomial as inner
     * item */
    DEVICEHOST()
    Real cosEval(const Real t, Polynomial inner) const
    {
        return trigEval<false>(t, inner);
    }

    /** @brief Evaluate the polynomial time-derivative at the given t.
     *
     * Same reverse-Horner traversal as eval(), but each non-constant
     * coefficient is scaled by its monomial power. The constant term
     * (revAccess(degPlusOne_ - 1)) has zero derivative and is therefore
     * skipped — the loop processes degPlusOne_ - 1 terms instead of
     * degPlusOne_, otherwise the final fma would scale the result by an
     * extra factor of t. */
    DEVICEHOST()
    Real evalDeriv(const Real& t) const
    {
        Real result = 0;
        for (idx_t i = 0; i + 1 < degPlusOne_; i++) {
            const Real coeff = revAccess(i) * Real(degPlusOne_ - 1 - i);
            result = fma(result, t, coeff);
        }
        return result;
    }

    /** @brief Time-derivative of @ref trigEval:
     *  ``d/dt[Σ_i a_i sin(p_i(t)·DEG2RAD)]
     *    = Σ_i a_i cos(p_i(t)·DEG2RAD)·p_i'(t)·DEG2RAD`` (``IsSin``), or the
     *  sign-flipped cos counterpart (``d/du cos(u) = −sin(u)``). Single-sources
     *  the former ``sinEvalDeriv`` / ``cosEvalDeriv`` twins (they differed only
     *  by the ``cos``↔``−sin`` chain factor); ``if constexpr`` keeps codegen
     *  bit-identical. inner.eval and inner.evalDeriv are computed in a single
     *  forward walk to avoid duplicating the monomial loop; the DEG2RAD chain
     *  factor treats inner_eval as degrees. */
    template<bool IsSin>
    DEVICEHOST()
    Real trigEvalDeriv(const Real t, Polynomial inner) const
    {
        Real result           = 0;
        const idx_t inner_deg = inner.degPlusOne_;
        for (idx_t i = 0; i < degPlusOne_; i++) {
            Real inner_eval  = 0;
            Real inner_deriv = 0;
            Real tpow        = 1;
            for (idx_t j = 0; j < inner_deg; j++) {
                inner_eval = fma(inner[j], tpow, inner_eval);
                if (j + 1 < inner_deg) {
                    inner_deriv
                        = fma(inner[j + 1] * Real(j + 1), tpow, inner_deriv);
                }
                tpow *= t;
            }
            Real chain;
            if constexpr (IsSin) {
                chain = feta::math::cos(inner_eval * DEG2RAD) * inner_deriv
                    * DEG2RAD;
            } else {
                chain = -feta::math::sin(inner_eval * DEG2RAD) * inner_deriv
                    * DEG2RAD;
            }
            result = fma((*this)[i], chain, result);

            inner.next();
        }
        return result;
    }

    /** @brief Time-derivative of sinEval. */
    DEVICEHOST()
    Real sinEvalDeriv(const Real t, Polynomial inner) const
    {
        return trigEvalDeriv<true>(t, inner);
    }

    /** @brief Time-derivative of cosEval. */
    DEVICEHOST()
    Real cosEvalDeriv(const Real t, Polynomial inner) const
    {
        return trigEvalDeriv<false>(t, inner);
    }

    /** @brief Per-thread shmem scratch handle, Dim=1, used by the
     *  ``<iwork>`` overloads below to write the polynomial walker's
     *  result into a shmem-resident slot instead of returning a
     *  register-resident ``Real``. Mirrors the
     *  ``feta::vector::Array<Real, 3>::Ref<iwork>::HandleT`` pattern
     *  already used by ``brie::frames::Frames`` rotation entry points. */
    template<bool iwork>
    using ScalarRef =
        typename feta::scalar::Array<Real>::template Ref<iwork>::HandleT;

    /** @brief ``<iwork>`` Horner evaluation — writes ``poly(t)`` to
     *  ``out[idx]`` instead of returning. Same FMA chain as the scalar
     *  ``eval(t)``; only the result transport changes. The cascade
     *  preserves bit-identity (identical operations in identical order)
     *  and lets downstream ``RefEphUnit`` leaves (RA / DEC / PM) hand
     *  the result directly to shmem-resident output slots instead of
     *  routing through a stack-temporary. */
    template<bool iwork>
    DEVICEHOST()
    void eval(
        const SampleIndex& idx, ScalarRef<iwork>& out, const Real& t) const
    {
        out[idx] = this->eval(t);
    }

    /** @brief ``<iwork>`` derivative — writes ``poly'(t)`` to ``out[idx]``.
     *  Used by the rate-accessor cascade (``RA_rate<iwork>``,
     *  ``DEC_rate<iwork>``, ``PM_rate<iwork>``). */
    template<bool iwork>
    DEVICEHOST()
    void evalDeriv(
        const SampleIndex& idx, ScalarRef<iwork>& out, const Real& t) const
    {
        out[idx] = this->evalDeriv(t);
    }

    /** @brief ``<iwork>`` sine-modulated Horner — same Σ ``a_i sin(p_i(t)·DEG2RAD)``
     *  result as the scalar ``sinEval``, written to ``out[idx]``. */
    template<bool iwork>
    DEVICEHOST()
    void sinEval(const SampleIndex& idx, ScalarRef<iwork>& out, const Real t,
        Polynomial inner) const
    {
        out[idx] = this->sinEval(t, inner);
    }

    /** @brief ``<iwork>`` cosine-modulated Horner — counterpart of
     *  ``sinEval<iwork>``. */
    template<bool iwork>
    DEVICEHOST()
    void cosEval(const SampleIndex& idx, ScalarRef<iwork>& out, const Real t,
        Polynomial inner) const
    {
        out[idx] = this->cosEval(t, inner);
    }

    /** @brief ``<iwork>`` time-derivative of ``sinEval``. Used by the NP
     *  rate cascade (``NP_RA_rate<iwork>``, ``NP_PM_rate<iwork>``). */
    template<bool iwork>
    DEVICEHOST()
    void sinEvalDeriv(const SampleIndex& idx, ScalarRef<iwork>& out,
        const Real t, Polynomial inner) const
    {
        out[idx] = this->sinEvalDeriv(t, inner);
    }

    /** @brief ``<iwork>`` time-derivative of ``cosEval``. Used by the NP
     *  rate cascade (``NP_DEC_rate<iwork>``). */
    template<bool iwork>
    DEVICEHOST()
    void cosEvalDeriv(const SampleIndex& idx, ScalarRef<iwork>& out,
        const Real t, Polynomial inner) const
    {
        out[idx] = this->cosEvalDeriv(t, inner);
    }

    /** @brief Return the reverse index */
    DEVICEHOST() idx_t revIndex(const idx_t& i) const
    {
        return degPlusOne_ - i - 1;
    }

    /** @brief Reverse access */
    DEVICEHOST() Real revAccess(const idx_t& i) const
    {
        return (*this)[revIndex(i)];
    }

    /** @brief advance either the pointer or the texture offset */
    DEVICEHOST()
    void next()
    {
        if constexpr (UseTexture) {
#ifdef __CUDA_ARCH__
            this->texOffset += degPlusOne_;
#endif
        } else {
            this->ptr += degPlusOne_;
        }
    }

    /** @brief The polynomial degree */
    idx_t degPlusOne_ = 0;
};

} // namespace core
} // namespace brie