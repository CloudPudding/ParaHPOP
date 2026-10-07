#pragma once

/**
 * @file ScalarEval.h
 * @brief Scalar tiled evaluation — `eval`/`evalParallel`, the `tiledFor`
 *        family, scalar `capture`/`assign`, the `TileBuffer` gather/scatter
 *        machinery, and `bufferedTiledFor`/`bufferedTiledForParallel`.
 *
 * Part of the `TiledEval.h` facade (block 1 — no packet dependency).
 */

#include "feta/core/simd/L2Cache.h"
#include "feta/typedefs.h"
#include "feta/vector/Item.h"
#include "feta/vector/expr/OperationsDetail.h"

#include <algorithm>
#include <omp.h>
#include <tuple>

namespace feta {
namespace cpu {

/** @brief Default tile size in samples.
 *
 * For a pipeline touching ~13 doubles per sample (10 read + 3 write),
 * 256 samples × 13 × 8 bytes = 26 KB — fits comfortably in L1 cache.
 */
inline constexpr idx_t DEFAULT_TILE_SIZE = 256;

/**
 * @brief Evaluate a simple expression into a destination array, processing
 * samples in cache-friendly tiles.
 *
 * Best for expressions where no sub-expression appears more than once.
 * For expressions with shared sub-expressions (e.g. `v_world` used in both
 * a sum and a cross product), prefer `tiledFor` with a lambda that
 * materialises intermediates.
 *
 * @tparam ExprL  Destination array/ref type (must be writable).
 * @tparam ExprR  Source expression type.
 * @param dest    Destination (e.g. `outR` from `Array::ref()`).
 * @param expr    Lazy expression to evaluate.
 * @param N       Number of samples to process.
 * @param tileSize Tile size in samples (default: `DEFAULT_TILE_SIZE`).
 */
template<typename ExprL, typename ExprR>
inline void eval(ExprL& dest, const ExprR& expr, idx_t N,
    idx_t tileSize = DEFAULT_TILE_SIZE)
{
    static_assert(ExprL::VecDims == ExprR::VecDims,
        "Destination and expression must have the same number of dimensions!");
    for (idx_t tileStart = 0; tileStart < N; tileStart += tileSize) {
        const idx_t tileEnd = std::min(tileStart + tileSize, N);
        for (idx_t i = tileStart; i < tileEnd; ++i) {
            vector::expr::detail::RecursiveAssign<ExprL::VecDims - 1, ExprL,
                ExprR>::eval(i, dest, expr);
        }
    }
}

/**
 * @brief Serial tiled loop over a sample range.
 *
 * Calls `body(i)` for every sample index in `[begin, end)`, processing
 * samples in tiles of `tileSize` for cache locality. The body lambda
 * can build and evaluate arbitrarily complex expression trees, including
 * materialising intermediates into `Item` registers.
 *
 * @tparam Func   Callable with signature `void(idx_t)`.
 * @param begin   First sample index (inclusive).
 * @param end     Past-the-end sample index.
 * @param body    Lambda to invoke per sample.
 * @param tileSize Tile size in samples (default: `DEFAULT_TILE_SIZE`).
 */
template<typename Func>
inline void tiledFor(
    idx_t begin, idx_t end, Func&& body, idx_t tileSize = DEFAULT_TILE_SIZE)
{
    for (idx_t tileStart = begin; tileStart < end; tileStart += tileSize) {
        const idx_t tileEnd = std::min(tileStart + tileSize, end);
        for (idx_t i = tileStart; i < tileEnd; ++i) {
            body(i);
        }
    }
}

/**
 * @brief OpenMP-parallel tiled loop over a sample range.
 *
 * Tiles are distributed across threads. Each thread processes its tiles
 * sequentially, preserving cache locality within each tile.
 *
 * @tparam Func   Callable with signature `void(idx_t)`.
 * @param begin   First sample index (inclusive).
 * @param end     Past-the-end sample index.
 * @param body    Lambda to invoke per sample.
 * @param tileSize Tile size in samples (default: `DEFAULT_TILE_SIZE`).
 */
template<typename Func>
inline void tiledForParallel(
    idx_t begin, idx_t end, Func&& body, idx_t tileSize = DEFAULT_TILE_SIZE)
{
    const idx_t nTiles = (end - begin + tileSize - 1) / tileSize;
#pragma omp parallel for schedule(static)
    for (idx_t t = 0; t < nTiles; ++t) {
        const idx_t tileStart = begin + t * tileSize;
        const idx_t tileEnd   = std::min(tileStart + tileSize, end);
        for (idx_t i = tileStart; i < tileEnd; ++i) {
            body(i);
        }
    }
}

/**
 * @brief OpenMP-parallel tiled evaluation of a simple expression.
 *
 * @tparam ExprL  Destination array/ref type (must be writable).
 * @tparam ExprR  Source expression type.
 * @param dest    Destination.
 * @param expr    Lazy expression to evaluate.
 * @param N       Number of samples to process.
 * @param tileSize Tile size in samples (default: `DEFAULT_TILE_SIZE`).
 */
template<typename ExprL, typename ExprR>
inline void evalParallel(ExprL& dest, const ExprR& expr, idx_t N,
    idx_t tileSize = DEFAULT_TILE_SIZE)
{
    static_assert(ExprL::VecDims == ExprR::VecDims,
        "Destination and expression must have the same number of dimensions!");
    const idx_t nTiles = (N + tileSize - 1) / tileSize;
#pragma omp parallel for schedule(static)
    for (idx_t t = 0; t < nTiles; ++t) {
        const idx_t tileStart = t * tileSize;
        const idx_t tileEnd   = std::min(tileStart + tileSize, N);
        for (idx_t i = tileStart; i < tileEnd; ++i) {
            vector::expr::detail::RecursiveAssign<ExprL::VecDims - 1, ExprL,
                ExprR>::eval(i, dest, expr);
        }
    }
}

/**
 * @brief Materialise a lazy expression into a register-local Item at index i.
 *
 * Use this inside a `tiledFor` lambda to avoid redundant evaluation of
 * shared sub-expressions. Returns an `Item` with the expression's value
 * at sample `i`.
 *
 * @par Example
 * @code
 *   feta::cpu::tiledFor(0, N, [&](idx_t i) {
 *       auto vw   = feta::cpu::capture<Real>(qR.quatRotate(vR), i);
 *       auto expr = vw + omR.cross(vw);
 *       feta::cpu::assign(outR, expr, i);
 *   });
 * @endcode
 *
 * @tparam DataT  Scalar type (e.g. `double`).
 * @tparam Expr   Expression type (deduced).
 * @param expr    Lazy expression to evaluate at index `i`.
 * @param i       Sample index.
 * @return `Item<DataT, Expr::VecDims>` with the materialised value.
 */
template<typename DataT, typename Expr>
inline vector::Item<DataT, Expr::VecDims> capture(const Expr& expr, idx_t i)
{
    vector::Item<DataT, Expr::VecDims> item;
    vector::expr::detail::RecursiveAssign<Expr::VecDims - 1,
        vector::Item<DataT, Expr::VecDims>, const Expr>::eval(i, item, expr);
    return item;
}

/**
 * @brief Assign an expression to a destination array at index i.
 *
 * Convenience wrapper around `RecursiveAssign` for use in `tiledFor` lambdas.
 *
 * @tparam ExprL  Destination type.
 * @tparam ExprR  Source expression type.
 * @param dest    Destination array/ref.
 * @param expr    Expression to evaluate.
 * @param i       Sample index.
 */
template<typename ExprL, typename ExprR>
inline void assign(ExprL& dest, const ExprR& expr, idx_t i)
{
    static_assert(ExprL::VecDims == ExprR::VecDims,
        "Destination and expression must have the same number of dimensions!");
    vector::expr::detail::RecursiveAssign<ExprL::VecDims - 1, ExprL,
        ExprR>::eval(i, dest, expr);
}

// ─── Buffered tiled evaluation ──────────────────────────────────────────────

/**
 * @brief Stack-allocated tile buffer for cache-friendly SoA access on CPU.
 *
 * Gathers a tile of samples from a SoA `RefArray` into a contiguous buffer
 * where `dimOffset_ = tileSize`, so all components of the tile fit within a
 * few cache lines. The user computes on the tile ref (which is a normal
 * RefArray with small dimOffset), then scatters results back.
 *
 * @tparam DataT     Scalar type (e.g. `double`).
 * @tparam VecDims   Number of vector components.
 * @tparam MaxTile   Maximum tile size (stack allocation size).
 */
template<typename DataT, dims_t VecDims, idx_t MaxTile = 256>
class TileBuffer {
    using RefT = vector::texture::detail::RefArray<DataT, VecDims, false, false>;

public:
    /**
     * @brief Gather a tile from a SoA ref into this buffer.
     *
     * Copies `[tileStart, tileEnd)` from `src` into the contiguous buffer.
     * After this call, `ref()` returns a ref whose index 0 maps to
     * `tileStart` in the source.
     */
    template<typename SrcRef>
    inline void gather(const SrcRef& src, idx_t tileStart, idx_t tileEnd)
    {
        tileLen_ = tileEnd - tileStart;
        for (dims_t d = 0; d < VecDims; ++d) {
            const DataT* srcPtr = src.data() + src.dimOffset() * d + tileStart;
            DataT* dstPtr       = buf_ + MaxTile * d;
            std::copy(srcPtr, srcPtr + tileLen_, dstPtr);
        }
    }

    /**
     * @brief Scatter this buffer back into a SoA ref.
     *
     * Copies the tile contents back to `[tileStart, tileEnd)` in `dst`.
     */
    template<typename DstRef>
    inline void scatter(DstRef& dst, idx_t tileStart) const
    {
        for (dims_t d = 0; d < VecDims; ++d) {
            DataT* dstPtr       = dst.data() + dst.dimOffset() * d + tileStart;
            const DataT* srcPtr = buf_ + MaxTile * d;
            std::copy(srcPtr, srcPtr + tileLen_, dstPtr);
        }
    }

    /** @brief Return a ref to the tile buffer for use in expressions. */
    inline RefT ref()
    {
        RefT r;
        r.data_      = buf_;
        r.nVecs_     = tileLen_;
        r.dimOffset_ = MaxTile;
        r.tex_       = 0;
        r.texOffset_ = 0;
        return r;
    }

    /** @brief Return a const ref to the tile buffer. */
    inline RefT ref() const
    {
        RefT r;
        r.data_      = const_cast<DataT*>(buf_);
        r.nVecs_     = tileLen_;
        r.dimOffset_ = MaxTile;
        r.tex_       = 0;
        r.texOffset_ = 0;
        return r;
    }

    /** @brief Number of samples currently in the tile. */
    inline idx_t size() const { return tileLen_; }

private:
    DataT buf_[VecDims * MaxTile];
    idx_t tileLen_ = 0;
};

/**
 * @brief Buffered tiled loop: gather inputs into contiguous tile buffers,
 * compute, then scatter outputs back to SoA.
 *
 * The body lambda receives tile-local refs (dimOffset = tileSize) and the
 * tile length. All indices inside the body are tile-local (0 to tileLen-1).
 *
 * @par Example
 * @code
 *   feta::cpu::bufferedTiledFor<Real>(0, N, tileSize,
 *       std::tie(qR, vR, omR),    // inputs to gather
 *       std::tie(outR),           // outputs to scatter
 *       [](auto& ins, auto& outs, idx_t tileLen) {
 *           auto& [tqR, tvR, tomR] = ins;
 *           auto& [toutR]          = outs;
 *           for (idx_t i = 0; i < tileLen; ++i) {
 *               auto vw   = feta::cpu::capture<Real>(tqR.quatRotate(tvR), i);
 *               auto expr = vw + tomR.cross(vw);
 *               feta::cpu::assign(toutR, expr, i);
 *           }
 *       });
 * @endcode
 */

namespace detail {

/** @brief Gather one ref into its tile buffer. */
template<typename DataT, dims_t VecDims, idx_t MaxTile, typename SrcRef>
inline void gatherOne(TileBuffer<DataT, VecDims, MaxTile>& buf,
    const SrcRef& src, idx_t tileStart, idx_t tileEnd)
{
    buf.gather(src, tileStart, tileEnd);
}

/** @brief Scatter one tile buffer into its ref. */
template<typename DataT, dims_t VecDims, idx_t MaxTile, typename DstRef>
inline void scatterOne(const TileBuffer<DataT, VecDims, MaxTile>& buf,
    DstRef& dst, idx_t tileStart)
{
    buf.scatter(dst, tileStart);
}

/** @brief Helper to create a TileBuffer matching a given ref. */
template<idx_t MaxTile, typename RefT>
struct TileBufferFor {
    using type = TileBuffer<typename RefT::DataT, RefT::VecDims, MaxTile>;
};

/** @brief Gather a tuple of refs into a tuple of tile buffers. */
template<idx_t MaxTile, typename... Refs, std::size_t... Is>
inline auto gatherAll(const std::tuple<Refs&...>& refs,
    idx_t tileStart, idx_t tileEnd, std::index_sequence<Is...>)
{
    auto bufs = std::make_tuple(
        typename TileBufferFor<MaxTile, std::remove_const_t<Refs>>::type{}...);
    (gatherOne(std::get<Is>(bufs), std::get<Is>(refs), tileStart, tileEnd),
        ...);
    return bufs;
}

/** @brief Get tile refs from a tuple of tile buffers. */
template<typename... Bufs, std::size_t... Is>
inline auto tileRefs(std::tuple<Bufs...>& bufs, std::index_sequence<Is...>)
{
    return std::make_tuple(std::get<Is>(bufs).ref()...);
}

/** @brief Scatter a tuple of tile buffers back to refs. */
template<typename... Bufs, typename... Refs, std::size_t... Is>
inline void scatterAll(const std::tuple<Bufs...>& bufs,
    std::tuple<Refs&...>& refs, idx_t tileStart,
    std::index_sequence<Is...>)
{
    (scatterOne(std::get<Is>(bufs), std::get<Is>(refs), tileStart), ...);
}

} // namespace detail

/**
 * @brief Buffered tiled evaluation with explicit gather/scatter.
 *
 * For each tile, gathers input SoA arrays into contiguous stack buffers,
 * calls the body with tile-local refs, then scatters output buffers back.
 *
 * @tparam MaxTile  Maximum tile size (default 256).
 * @param begin     First sample index.
 * @param end       Past-the-end sample index.
 * @param inputs    Tuple of input refs (gathered, read-only in body).
 * @param outputs   Tuple of output refs (gathered, written in body, scattered).
 * @param body      Lambda: `(auto& inputRefs, auto& outputRefs, idx_t tileLen)`.
 */
template<idx_t MaxTile = 256, typename InputTuple, typename OutputTuple,
    typename Func>
inline void bufferedTiledFor(idx_t begin, idx_t end,
    InputTuple inputs, OutputTuple outputs, Func&& body)
{
    constexpr auto NIn  = std::tuple_size<InputTuple>::value;
    constexpr auto NOut = std::tuple_size<OutputTuple>::value;

    for (idx_t tileStart = begin; tileStart < end; tileStart += MaxTile) {
        const idx_t tileEnd = std::min(tileStart + MaxTile, end);
        const idx_t tileLen = tileEnd - tileStart;

        // Gather inputs into contiguous tile buffers
        auto inBufs = detail::gatherAll<MaxTile>(
            inputs, tileStart, tileEnd, std::make_index_sequence<NIn>{});
        auto inRefs = detail::tileRefs(
            inBufs, std::make_index_sequence<NIn>{});

        // Gather outputs (may contain prior data needed for read-modify-write)
        auto outBufs = detail::gatherAll<MaxTile>(
            outputs, tileStart, tileEnd, std::make_index_sequence<NOut>{});
        auto outRefs = detail::tileRefs(
            outBufs, std::make_index_sequence<NOut>{});

        // Run computation on tile-local refs
        body(inRefs, outRefs, tileLen);

        // Scatter outputs back to SoA
        detail::scatterAll(
            outBufs, outputs, tileStart, std::make_index_sequence<NOut>{});
    }
}

/**
 * @brief OpenMP-parallel buffered tiled evaluation.
 *
 * Same as `bufferedTiledFor` but distributes tiles across threads.
 */
template<idx_t MaxTile = 256, typename InputTuple, typename OutputTuple,
    typename Func>
inline void bufferedTiledForParallel(idx_t begin, idx_t end,
    InputTuple inputs, OutputTuple outputs, Func&& body)
{
    constexpr auto NIn  = std::tuple_size<InputTuple>::value;
    constexpr auto NOut = std::tuple_size<OutputTuple>::value;

    const idx_t nTiles = (end - begin + MaxTile - 1) / MaxTile;
#pragma omp parallel for schedule(static)
    for (idx_t t = 0; t < nTiles; ++t) {
        const idx_t tileStart = begin + t * MaxTile;
        const idx_t tileEnd   = std::min(tileStart + MaxTile, end);
        const idx_t tileLen   = tileEnd - tileStart;

        auto inBufs = detail::gatherAll<MaxTile>(
            inputs, tileStart, tileEnd, std::make_index_sequence<NIn>{});
        auto inRefs = detail::tileRefs(
            inBufs, std::make_index_sequence<NIn>{});

        auto outBufs = detail::gatherAll<MaxTile>(
            outputs, tileStart, tileEnd, std::make_index_sequence<NOut>{});
        auto outRefs = detail::tileRefs(
            outBufs, std::make_index_sequence<NOut>{});

        body(inRefs, outRefs, tileLen);

        detail::scatterAll(
            outBufs, outputs, tileStart, std::make_index_sequence<NOut>{});
    }
}

} // namespace cpu
} // namespace feta
