#pragma once
#include "parm/integrate/rk/coefs/tableau.h"
#include <feta/vector/PacketItem.h>
#include <feta/vector/PacketTile.h>
#include <feta/vector/Tile.h>
namespace parm::integrate::rk::host {
namespace detail {

template<typename RealT, typename ScalarT>
inline RealT mulScalar(const RealT& a, const ScalarT b)
{
    if constexpr (std::is_arithmetic_v<RealT>) {
        return a * b;
    } else {
        
        return a * RealT::broadcast(b);
    }
}

template<typename V, idx_t K>
struct KStorageT {
    using type = feta::vector::Tile<typename V::ComponentT, V::VecDims, K>;
};

template<typename T, dims_t D, idx_t W, idx_t K>
struct KStorageT<feta::vector::PacketItem<T, D, W>, K> {
    using type = feta::vector::PacketTile<T, D, K, W>;
};

template<typename V, idx_t K>
using KStorageFor = typename KStorageT<V, K>::type;

template<idx_t row, idx_t col, typename Tableau, typename V, typename K>
struct AccumStackK {
    template<typename RealT>
    static inline void apply(V& acc, const K& k, RealT dt)
    {
        if constexpr (col >= 1) {
            AccumStackK<row, col - 1, Tableau, V, K>::apply(acc, k, dt);
        }
        if constexpr (Tableau::template a<row, col>() != 0) {
            
            const RealT coef  = mulScalar(dt, Tableau::template a<row, col>());
            auto* accD        = acc.data();
            const auto* kD    = k.template slotPtr<col>();
            constexpr idx_t D = V::VecDims;
            for (idx_t d = 0; d < D; ++d)
                accD[d] += coef * kD[d];
        }
    }
};

template<idx_t stage, typename Tableau>
struct RecurseEvalStack {
    
    template<typename V, typename K, typename RealT, typename RHS,
        typename IndexT>
    static inline void eval(RHS&& rhs, const V& state, K& k, RealT dt,
        RealT epoch0, const IndexT& i)
    {
        if constexpr (stage >= 1) {
            RecurseEvalStack<stage - 1, Tableau>::eval(
                rhs, state, k, dt, epoch0, i);
        }

        V stagedState = state;
        if constexpr (stage >= 1) {
            AccumStackK<stage - 1, stage - 1, Tableau, V, K>::apply(
                stagedState, k, dt);
        }

        RealT stagedTime = epoch0;
        if constexpr (stage >= 1) {
            stagedTime += mulScalar(dt, Tableau::template c<stage - 1>());
        }

        V kStage          = rhs(i, stagedState, stagedTime);
        const auto* src   = kStage.data();
        auto* dst         = k.template slotPtr<stage>();
        constexpr idx_t D = V::VecDims;
        for (idx_t d = 0; d < D; ++d)
            dst[d] = src[d];
    }
};

template<idx_t stage, typename Tableau>
struct AccumFinalSum {
    template<typename V, typename K, typename RealT>
    static inline void apply(V& stateNext, const K& k, RealT dt)
    {
        if constexpr (stage >= 1) {
            AccumFinalSum<stage - 1, Tableau>::apply(
                stateNext, k, dt);
        }
        auto* yD          = stateNext.data();
        const auto* kD    = k.template slotPtr<stage>();
        constexpr idx_t D = V::VecDims;
        if constexpr (Tableau::template b<stage>() != 0) {
            const RealT coefB = mulScalar(dt, Tableau::template b<stage>());
            for (idx_t d = 0; d < D; ++d)
                yD[d] += coefB * kD[d];
        }

    }
};

} // namespace detail
template<typename Tableau, typename V, typename RealT, typename RHS>
inline void stepFixed(
    const SampleIndex& i, V& state, RealT dt, RealT epoch0, RHS&& rhs)
{
    constexpr idx_t nStages = Tableau::nStages();

    detail::KStorageFor<V, nStages> k;

    detail::RecurseEvalStack<nStages - 1, Tableau>::eval(
        rhs, state, k, dt, epoch0, i);

    V stateNext = state;
    detail::AccumFinalSum<nStages - 1, Tableau>::apply(
        stateNext, k, dt);

    state = stateNext;
}

}
