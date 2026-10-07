#pragma once

/**
 * @file PacketEval.h
 * @brief SIMD-packet tiled evaluation drivers — `packetEval`/`packetEvalParallel`,
 *        the `packetTiledFor`/`packetFlatFor`/`packetFor`/`packetBatchedFor`
 *        family, `processTile`, and the L2-tuned `optimalTileSize` heuristics.
 *
 * Part of the `TiledEval.h` facade (block 2); needs the packet ops + block 1.
 */

#include "feta/core/simd/L2Cache.h"
#include "feta/typedefs.h"
#include "feta/vector/Item.h"
#include "feta/vector/expr/OperationsDetail.h"

#include <algorithm>
#include <omp.h>
#include <tuple>
#include "feta/vector/expr/cpu/PacketOps.h"
#include "feta/vector/expr/cpu/tiled/ScalarEval.h"

namespace feta {
namespace cpu {

/**
 * @brief Evaluate a simple expression into a destination array using SIMD
 * packets, processing samples in cache-friendly tiles.
 *
 * This is the SIMD-accelerated equivalent of `cpu::eval()`. Within each
 * tile, samples are processed in SIMD-width packets. The last packet in
 * each tile uses masked loads/stores for tail elements.
 *
 * Prefetching is interleaved: while the current packet is being computed,
 * the next packet's memory is prefetched into L1 cache.
 *
 * @tparam ExprL  Destination array/ref type (must be writable).
 * @tparam ExprR  Source expression type.
 * @param dest    Destination (e.g. `outR` from `Array::ref()`).
 * @param expr    Lazy expression to evaluate.
 * @param N       Number of samples to process.
 * @param tileSize Tile size in samples (default: `DEFAULT_TILE_SIZE`).
 */
template<typename ExprL, typename ExprR>
inline void packetEval(ExprL& dest, const ExprR& expr, idx_t N,
    idx_t tileSize = DEFAULT_TILE_SIZE)
{
    static_assert(ExprL::VecDims == ExprR::VecDims,
        "Destination and expression must have the same number of dimensions!");

    using DataT = typename ExprL::ComponentT;
    constexpr idx_t W  = simd::PreferredWidth<DataT>;
    constexpr idx_t D1 = simd::DEFAULT_PREFETCH_L1_DISTANCE;
    constexpr idx_t D2 = simd::DEFAULT_PREFETCH_L2_DISTANCE;

    for (idx_t tileStart = 0; tileStart < N; tileStart += tileSize) {
        const idx_t tileEnd = std::min(tileStart + tileSize, N);

        // Full packets
        idx_t i = tileStart;
        for (; i + W <= tileEnd; i += W) {
            auto pi = PacketIndex<W>::make(i);
            // L2 far-ahead prefetch (read inputs + write-intent output)
            if (i + D2 * W <= tileEnd) {
                packet::prefetchExprL2(expr, i + D2 * W);
                packet::prefetchExprW(dest, i + D2 * W);
            }
            // L1 close prefetch
            if (i + D1 * W <= tileEnd) {
                packet::prefetchExpr(expr, i + D1 * W);
                packet::prefetchExprW(dest, i + D1 * W);
            }
            packet::RecursivePacketAssign<ExprL::VecDims - 1,
                ExprL, ExprR, W>::eval(pi, dest, expr);
        }
        // Tail packet (masked)
        if (i < tileEnd) {
            auto pi = PacketIndex<W>::makeTail(i, tileEnd - i);
            packet::RecursivePacketAssign<ExprL::VecDims - 1,
                ExprL, ExprR, W>::eval(pi, dest, expr);
        }
    }
}

/**
 * @brief OpenMP-parallel SIMD packet evaluation.
 *
 * Tiles are distributed across threads via `omp parallel for`.
 * Within each tile, packets are processed sequentially with SIMD.
 * This is the optimal dispatch level: tiles are large enough to
 * amortize OpenMP overhead, while packets exploit data-level
 * parallelism within each thread.
 *
 * @tparam ExprL  Destination array/ref type.
 * @tparam ExprR  Source expression type.
 * @param dest    Destination.
 * @param expr    Lazy expression to evaluate.
 * @param N       Number of samples to process.
 * @param tileSize Tile size in samples (default: `DEFAULT_TILE_SIZE`).
 */
template<typename ExprL, typename ExprR>
inline void packetEvalParallel(ExprL& dest, const ExprR& expr, idx_t N,
    idx_t tileSize = DEFAULT_TILE_SIZE)
{
    static_assert(ExprL::VecDims == ExprR::VecDims,
        "Destination and expression must have the same number of dimensions!");

    using DataT = typename ExprL::ComponentT;
    constexpr idx_t W  = simd::PreferredWidth<DataT>;
    constexpr idx_t D1 = simd::DEFAULT_PREFETCH_L1_DISTANCE;
    constexpr idx_t D2 = simd::DEFAULT_PREFETCH_L2_DISTANCE;
    const idx_t nTiles = (N + tileSize - 1) / tileSize;

#pragma omp parallel for schedule(static)
    for (idx_t t = 0; t < nTiles; ++t) {
        const idx_t tileStart = t * tileSize;
        const idx_t tileEnd   = std::min(tileStart + tileSize, N);

        idx_t i = tileStart;
        for (; i + W <= tileEnd; i += W) {
            auto pi = PacketIndex<W>::make(i);
            if (i + D2 * W <= tileEnd) {
                packet::prefetchExprL2(expr, i + D2 * W);
                packet::prefetchExprW(dest, i + D2 * W);
            }
            if (i + D1 * W <= tileEnd) {
                packet::prefetchExpr(expr, i + D1 * W);
                packet::prefetchExprW(dest, i + D1 * W);
            }
            packet::RecursivePacketAssign<ExprL::VecDims - 1,
                ExprL, ExprR, W>::eval(pi, dest, expr);
        }
        if (i < tileEnd) {
            auto pi = PacketIndex<W>::makeTail(i, tileEnd - i);
            packet::RecursivePacketAssign<ExprL::VecDims - 1,
                ExprL, ExprR, W>::eval(pi, dest, expr);
        }
    }
}

/**
 * @brief SIMD packet tiled loop with user-provided lambda and prefetching.
 *
 * Calls `body(pi)` for every `PacketIndex` in `[begin, end)`,
 * processing samples in tiles for cache locality, then in SIMD
 * packets within each tile.
 *
 * For each packet, the next packet's memory is prefetched into L1
 * for all expressions passed in `prefetchExprs`. This hides memory
 * latency by overlapping SIMD computation with cache-line fills.
 *
 * @tparam DataT  Scalar type (controls SIMD width).
 * @tparam Func   Callable with signature `void(PacketIndex<W>)`.
 * @tparam Exprs  Expression types to prefetch (deduced).
 * @param begin   First sample index (inclusive).
 * @param end     Past-the-end sample index.
 * @param body    Lambda to invoke per packet.
 * @param tileSize Tile size in samples (default: `DEFAULT_TILE_SIZE`).
 * @param prefetchExprs Expressions whose next-packet data should be
 *                      prefetched into L1 while the current packet computes.
 */
template<typename DataT = double, typename Func, typename... Exprs>
inline void packetTiledFor(
    idx_t begin, idx_t end, Func&& body, idx_t tileSize,
    const Exprs&... prefetchExprs)
{
    constexpr idx_t W  = simd::PreferredWidth<DataT>;
    constexpr idx_t D1 = simd::DEFAULT_PREFETCH_L1_DISTANCE;
    constexpr idx_t D2 = simd::DEFAULT_PREFETCH_L2_DISTANCE;

    for (idx_t tileStart = begin; tileStart < end; tileStart += tileSize) {
        const idx_t tileEnd = std::min(tileStart + tileSize, end);

        idx_t i = tileStart;
        for (; i + W <= tileEnd; i += W) {
            // L2 far-ahead prefetch (DRAM → L2)
            if (i + D2 * W <= tileEnd)
                (packet::prefetchExprL2(prefetchExprs, i + D2 * W), ...);
            // L1 close prefetch (L2 → L1)
            if (i + D1 * W <= tileEnd)
                (packet::prefetchExpr(prefetchExprs, i + D1 * W), ...);
            body(PacketIndex<W>::make(i));
        }
        if (i < tileEnd)
            body(PacketIndex<W>::makeTail(i, tileEnd - i));
    }
}

/**
 * @brief SIMD packet tiled loop (no prefetch, default tile size).
 */
template<typename DataT = double, typename Func>
inline void packetTiledFor(idx_t begin, idx_t end, Func&& body)
{
    packetTiledFor<DataT>(begin, end, std::forward<Func>(body),
        DEFAULT_TILE_SIZE);
}

/**
 * @brief OpenMP-parallel SIMD packet tiled loop with prefetching.
 *
 * Tiles are distributed across threads via `omp parallel for`.
 * Within each tile, packets are processed sequentially with SIMD,
 * and the next packet's data is prefetched for all provided expressions.
 *
 * @tparam DataT  Scalar type (controls SIMD width).
 * @tparam Func   Callable with signature `void(PacketIndex<W>)`.
 * @tparam Exprs  Expression types to prefetch (deduced).
 * @param begin   First sample index (inclusive).
 * @param end     Past-the-end sample index.
 * @param body    Lambda to invoke per packet.
 * @param tileSize Tile size in samples (default: `DEFAULT_TILE_SIZE`).
 * @param prefetchExprs Expressions whose next-packet data should be
 *                      prefetched into L1 while the current packet computes.
 */
template<typename DataT = double, typename Func, typename... Exprs>
inline void packetTiledForParallel(
    idx_t begin, idx_t end, Func&& body, idx_t tileSize,
    const Exprs&... prefetchExprs)
{
    constexpr idx_t W  = simd::PreferredWidth<DataT>;
    constexpr idx_t D1 = simd::DEFAULT_PREFETCH_L1_DISTANCE;
    constexpr idx_t D2 = simd::DEFAULT_PREFETCH_L2_DISTANCE;
    const idx_t nTiles = (end - begin + tileSize - 1) / tileSize;

#pragma omp parallel for schedule(static)
    for (idx_t t = 0; t < nTiles; ++t) {
        const idx_t tileStart = begin + t * tileSize;
        const idx_t tileEnd   = std::min(tileStart + tileSize, end);

        idx_t i = tileStart;
        for (; i + W <= tileEnd; i += W) {
            if (i + D2 * W <= tileEnd)
                (packet::prefetchExprL2(prefetchExprs, i + D2 * W), ...);
            if (i + D1 * W <= tileEnd)
                (packet::prefetchExpr(prefetchExprs, i + D1 * W), ...);
            body(PacketIndex<W>::make(i));
        }
        if (i < tileEnd)
            body(PacketIndex<W>::makeTail(i, tileEnd - i));
    }
}

/**
 * @brief OpenMP-parallel SIMD packet tiled loop (no prefetch, default tile).
 */
template<typename DataT = double, typename Func>
inline void packetTiledForParallel(idx_t begin, idx_t end, Func&& body)
{
    packetTiledForParallel<DataT>(begin, end, std::forward<Func>(body),
        DEFAULT_TILE_SIZE);
}

// ─── Flat (no-tile) packet loops ─────────────────────────────────────────────
// For single-pass fused kernels where all intermediates stay in registers,
// tiling adds unnecessary boundary overhead.  These flat variants iterate
// straight through [begin, end) without tile subdivision.

/**
 * @brief Flat SIMD packet loop — no tiling, no prefetch.
 *
 * Use this for single-pass fused kernels where `packetCapture` keeps all
 * intermediates register-resident.  Avoids tile-boundary overhead of
 * `packetTiledFor` when explicit prefetch is not needed.
 */
template<typename DataT = double, typename Func>
inline void packetFlatFor(idx_t begin, idx_t end, Func&& body)
{
    constexpr idx_t W = simd::PreferredWidth<DataT>;

    idx_t i = begin;
    for (; i + W <= end; i += W)
        body(PacketIndex<W>::make(i));
    if (i < end)
        body(PacketIndex<W>::makeTail(i, end - i));
}

/**
 * @brief OpenMP-parallel flat SIMD packet loop — no tiling, no prefetch.
 *
 * Distributes packets across threads via `omp parallel for`.
 * Each thread processes a contiguous range of full packets.
 */
template<typename DataT = double, typename Func>
inline void packetFlatForParallel(idx_t begin, idx_t end, Func&& body)
{
    constexpr idx_t W = simd::PreferredWidth<DataT>;
    const idx_t nFullPackets = (end - begin) / W;
    const idx_t tailStart    = begin + nFullPackets * W;

#pragma omp parallel for schedule(static)
    for (idx_t p = 0; p < nFullPackets; ++p)
        body(PacketIndex<W>::make(begin + p * W));

    // Tail (if any) — single-threaded, at most W-1 elements
    if (tailStart < end)
        body(PacketIndex<W>::makeTail(tailStart, end - tailStart));
}


// ─── Context-aware packet loops ──────────────────────────────────────────────
// Automatically select the right parallelism strategy based on context:
//  - Already inside an OMP parallel region → `#pragma omp for` (work-sharing)
//  - Outside, `parallel=true`  → `#pragma omp parallel for` (new region)
//  - Outside, `parallel=false` → serial (default, least overhead)

namespace detail {

/** @brief Process a single tile [tileStart, tileEnd) with SIMD packets. */
template<idx_t W, typename Func>
inline void processTile(idx_t tileStart, idx_t tileEnd, Func&& body)
{
    idx_t i = tileStart;
    for (; i + W <= tileEnd; i += W)
        body(PacketIndex<W>::make(i));
    if (i < tileEnd)
        body(PacketIndex<W>::makeTail(i, tileEnd - i));
}

} // namespace detail (tile helper)

/**
 * @brief Context-aware SIMD packet tiled loop.
 *
 * Automatically selects the right OpenMP strategy:
 *  - If already inside an `omp parallel` region, distributes tiles across
 *    existing threads via `#pragma omp for` (work-sharing, no new region).
 *  - If outside and `parallel == true`, opens a new parallel region via
 *    `#pragma omp parallel for`.
 *  - If outside and `parallel == false` (default), runs serially with
 *    SIMD vectorisation only — least overhead for small workloads or
 *    when the caller manages parallelism externally.
 *
 * @tparam DataT     Scalar type (controls SIMD width via PreferredWidth).
 * @tparam Func      Callable with signature `void(PacketIndex<W>)`.
 * @param begin      First sample index (inclusive).
 * @param end        Past-the-end sample index.
 * @param body       Lambda to invoke per packet.
 * @param parallel   When not already in a parallel region, open one if true.
 * @param tileSize   Tile size in samples (default: `DEFAULT_TILE_SIZE`).
 */
template<typename DataT = double, typename Func>
inline void packetFor(idx_t begin, idx_t end, Func&& body,
    bool parallel = false, idx_t tileSize = DEFAULT_TILE_SIZE)
{
    constexpr idx_t W = simd::PreferredWidth<DataT>;
    const idx_t nTilesTotal = (end - begin + tileSize - 1) / tileSize;

    if (omp_in_parallel()) {
        // Already inside a parallel region — distribute tiles via omp for
#pragma omp for schedule(static) nowait
        for (idx_t t = 0; t < nTilesTotal; ++t) {
            const idx_t tileStart = begin + t * tileSize;
            const idx_t tileEnd   = std::min(tileStart + tileSize, end);
            detail::processTile<W>(tileStart, tileEnd, body);
        }
    } else if (parallel) {
        // Outside — open a new parallel region
#pragma omp parallel for schedule(static)
        for (idx_t t = 0; t < nTilesTotal; ++t) {
            const idx_t tileStart = begin + t * tileSize;
            const idx_t tileEnd   = std::min(tileStart + tileSize, end);
            detail::processTile<W>(tileStart, tileEnd, body);
        }
    } else {
        // Serial — SIMD only, no threading overhead
        for (idx_t t = 0; t < nTilesTotal; ++t) {
            const idx_t tileStart = begin + t * tileSize;
            const idx_t tileEnd   = std::min(tileStart + tileSize, end);
            detail::processTile<W>(tileStart, tileEnd, body);
        }
    }
}

/**
 * @brief Context-aware flat SIMD packet loop (no tiling).
 *
 * Same context-detection as `packetFor` but without tile subdivision.
 * Preferred for single-pass fused kernels where all intermediates are
 * register-resident via `packetCapture`.
 */
template<typename DataT = double, typename Func>
inline void packetFlatFor(idx_t begin, idx_t end, Func&& body,
    bool parallel = false)
{
    constexpr idx_t W = simd::PreferredWidth<DataT>;
    const idx_t nFullPackets = (end - begin) / W;
    const idx_t tailStart    = begin + nFullPackets * W;

    if (omp_in_parallel()) {
#pragma omp for schedule(static) nowait
        for (idx_t p = 0; p < nFullPackets; ++p)
            body(PacketIndex<W>::make(begin + p * W));
#pragma omp single nowait
        if (tailStart < end)
            body(PacketIndex<W>::makeTail(tailStart, end - tailStart));
    } else if (parallel) {
#pragma omp parallel for schedule(static)
        for (idx_t p = 0; p < nFullPackets; ++p)
            body(PacketIndex<W>::make(begin + p * W));
        if (tailStart < end)
            body(PacketIndex<W>::makeTail(tailStart, end - tailStart));
    } else {
        for (idx_t p = 0; p < nFullPackets; ++p)
            body(PacketIndex<W>::make(begin + p * W));
        if (tailStart < end)
            body(PacketIndex<W>::makeTail(tailStart, end - tailStart));
    }
}


// ─── L2-tiled packet loop ────────────────────────────────────────────────────
// Same shape as ``packetFor`` but the tile size is derived at runtime from
// the platform's L2 cache and the caller's per-sample working-set estimate.
// Used by host integration loops that want their tile to stay L2-resident
// across substages.

/** @brief Cache-line size in bytes (modern x86-64 / Apple Silicon).
 *
 *  Used by ``optimalTileSize`` to round tile boundaries to a multiple
 *  of one cache line × packet width — prevents false sharing on the
 *  edges between adjacent threads' tiles. */
inline constexpr idx_t CACHE_LINE_BYTES = 64;

/** @brief Fraction of L2 reserved for the working set inside one tile.
 *
 *  Leaves ~half of L2 for state + dStates + per-step scratch that the
 *  body lambda accesses besides the per-sample working set declared
 *  via ``bytesPerSample``. */
inline constexpr double L2_WORKING_SET_FRACTION = 0.5;

/** @brief Lower bound on the tile size in samples — keeps OMP overhead
 *  amortised even when working-set estimates are pathological. */
inline constexpr idx_t MIN_TILE_SAMPLES = 64;

/** @brief Upper bound on the tile size in samples — caps the tile when
 *  working-set estimates are very small, preventing one thread from
 *  monopolising work. */
inline constexpr idx_t MAX_TILE_SAMPLES = 4096;

/**
 * @brief Compute an L2-resident tile size for the caller's working set.
 *
 *  Returns a tile size ``K`` (in samples) such that:
 *  - ``K * bytesPerSample <= L2_WORKING_SET_FRACTION * L2_per_core``
 *  - ``K`` is a multiple of ``CACHE_LINE_BYTES / sizeof(DataT)`` rounded
 *    up to a multiple of the SIMD packet width ``W = PreferredWidth<DataT>``,
 *    so adjacent tiles never share a cache line (false-sharing free).
 *  - ``K`` is clamped to ``[MIN_TILE_SAMPLES, MAX_TILE_SAMPLES]``.
 *
 *  When ``bytesPerSample == 0`` returns ``DEFAULT_TILE_SIZE`` (caller
 *  has no estimate; fall back to the legacy default).
 *
 *  @tparam DataT          Scalar type — controls the SIMD packet width.
 *  @param  bytesPerSample Estimated per-sample working set the body
 *                         touches inside the tile.
 *  @param  l2Override     Optional L2 size in bytes (0 = auto-detect).
 *                         Useful for tests and explicit override.
 */
template<typename DataT = double>
inline idx_t optimalTileSize(idx_t bytesPerSample, idx_t l2Override = 0)
{
    if (bytesPerSample == 0)
        return DEFAULT_TILE_SIZE;

    constexpr idx_t W = simd::PreferredWidth<DataT>;
    constexpr idx_t cacheLineSamples
        = (CACHE_LINE_BYTES + sizeof(DataT) - 1) / sizeof(DataT);
    constexpr idx_t alignment
        = (cacheLineSamples > W) ? cacheLineSamples : W;

    const idx_t l2 = (l2Override > 0) ? l2Override : detail::l2CacheSize();
    const idx_t budget
        = static_cast<idx_t>(L2_WORKING_SET_FRACTION * static_cast<double>(l2));
    idx_t K = budget / bytesPerSample;

    /* Round DOWN to a multiple of alignment so the tile end falls on a
     * cache-line boundary; this prevents threads from sharing the last
     * cache line of one tile with the first of the next. */
    K = (K / alignment) * alignment;

    /* Clamp to sane bounds; ensure the alignment is preserved at the
     * lower end (alignment is always <= MIN_TILE_SAMPLES in practice). */
    if (K < MIN_TILE_SAMPLES)
        K = ((MIN_TILE_SAMPLES + alignment - 1) / alignment) * alignment;
    if (K > MAX_TILE_SAMPLES)
        K = (MAX_TILE_SAMPLES / alignment) * alignment;

    return K;
}

/**
 * @brief Context-aware SIMD packet loop with L2-derived tile sizing.
 *
 *  Thin wrapper over ``packetFor`` that picks the tile size from
 *  ``optimalTileSize<DataT>(bytesPerSample)``.  All other semantics
 *  match ``packetFor`` exactly: nested → ``omp for`` work-share with
 *  ``nowait``; outside + ``parallel=true`` → opens a new region;
 *  outside + ``parallel=false`` → serial scalar tile loop.
 *
 *  @tparam DataT          Scalar type — controls the SIMD packet width.
 *  @tparam Func           Callable with signature ``void(PacketIndex<W>)``.
 *  @param  begin          First sample index (inclusive).
 *  @param  end            Past-the-end sample index.
 *  @param  body           Lambda to invoke per packet.
 *  @param  bytesPerSample Estimated per-sample working set inside the
 *                         body — drives the L2-derived tile size.
 *  @param  parallel       When not already in a parallel region, open
 *                         one if true.
 *  @param  tileOverride   Optional explicit tile size in samples
 *                         (0 = use L2-derived).  Useful for tests.
 */
template<typename DataT = double, typename Func>
inline void packetBatchedFor(idx_t begin, idx_t end, Func&& body,
    idx_t bytesPerSample, bool parallel = false, idx_t tileOverride = 0)
{
    idx_t tile = (tileOverride > 0)
        ? tileOverride
        : optimalTileSize<DataT>(bytesPerSample);

    /* Thread-count clamp: when the L2-derived tile is large enough
     * that the workload would produce fewer tiles than ``nthreads ×
     * 2``, cap it so each thread receives at least 2 tiles for
     * static-schedule load balance.  Without this clamp, small
     * workloads (e.g. an acceptance test with a few hundred
     * samples) leave threads idle.  Tile is rounded down to packet
     * width to preserve SIMD alignment within the tile, with a
     * floor of one packet. */
    if (omp_in_parallel() && (end > begin)) {
        const idx_t nthreads
            = static_cast<idx_t>(omp_get_num_threads());
        const idx_t targetTiles = nthreads * idx_t{ 2 };
        const idx_t maxTile
            = (end - begin + targetTiles - 1) / targetTiles;
        if (maxTile > 0 && tile > maxTile) {
            constexpr idx_t W  = simd::PreferredWidth<DataT>;
            const idx_t aligned = (maxTile / W) * W;
            tile = (aligned > 0) ? aligned : W;
        }
    }

    packetFor<DataT>(
        begin, end, std::forward<Func>(body), parallel, tile);
}

} // namespace cpu
} // namespace feta
