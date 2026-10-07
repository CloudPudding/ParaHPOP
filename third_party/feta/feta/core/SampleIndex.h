#pragma once

#include "feta/err/DeviceError.h"
#include "feta/typedefs.h"

#include "feta/core/simd/PacketTraits.h"

namespace feta {
/**
 * @brief Sample index, encompassing both global indexing as well as indexing in
 * work (i.e. shared or a generic subset of global) memory.
 */
struct SampleIndex {
    /**
     * @brief Construct a `SampleIndex` from GPU built-in thread/block indices.
     *
     * On device, asserts that `threadIdxx == threadIdx.x`, `blockIdxx == blockIdx.x`,
     * and `blockDimx == blockDim.x` to catch accidental mis-ordering of arguments.
     *
     * @param threadIdxx  Value of `threadIdx.x`.
     * @param blockIdxx   Value of `blockIdx.x`.
     * @param blockDimx   Value of `blockDim.x`.
     * @return `SampleIndex` with `global = blockIdxx * blockDimx + threadIdxx`
     *         and `work = threadIdxx`.
     */
    DEVICEHOST()
    static SampleIndex make(
        const idx_t threadIdxx, const idx_t blockIdxx, const idx_t blockDimx)
    {
#ifdef __CUDA_ARCH__
        FETA_GPU_ASSERT(
            threadIdxx == threadIdx.x, ::feta::err::SAMPLE_INDEX_INIT);
        FETA_GPU_ASSERT(
            blockIdxx == blockIdx.x, ::feta::err::SAMPLE_INDEX_INIT);
        FETA_GPU_ASSERT(
            blockDimx == blockDim.x, ::feta::err::SAMPLE_INDEX_INIT);
#endif
        return { blockIdxx * blockDimx + threadIdxx, threadIdxx };
    }

    /**
     * @brief Construct a `SampleIndex` from a flat index with identical global and work values.
     *
     * @param index  Index value used for both `globalIdx_` and `workIdx_`.
     * @return `SampleIndex` with `global == work == index`.
     */
    DEVICEHOST()
    static SampleIndex make(const idx_t index) { return { index, index }; }

    /** @brief Return the global (grid-wide) sample index. */
    DEVICEHOST() inline idx_t global() const { return globalIdx_; }

    /** @brief Return the block-local index used for work (shared) memory arrays. */
    DEVICEHOST() inline idx_t work() const { return workIdx_; }

    /** @brief Return a copy of this index (explicit alternative to the copy constructor). */
    DEVICEHOST() SampleIndex clone() const { return *this; }

    /** @brief Public data members (POD layout required for device use). */
    idx_t globalIdx_;
    idx_t workIdx_;
};

/**
 * @brief Lightweight SIMD packet index — represents a contiguous group of
 * `Width` samples starting at `base_`.
 *
 * This is the CPU SIMD analogue of `SampleIndex`: where `SampleIndex`
 * identifies a single sample, `PacketIndex` identifies a SIMD-width group.
 *
 * POD struct with the same footprint as SampleIndex (two `idx_t` members).
 * The SIMD mask is *not* stored here — it is computed once at the
 * load/store boundary inside `packetGet`/`packetStore`, so interior
 * expression nodes never pay mask overhead.
 *
 * For full packets (the hot path), `active_ == Width` and the
 * `full()` check is trivially branch-predicted away. Tail packets
 * carry `active_ < Width` so that leaf nodes can issue masked loads/stores.
 *
 * @tparam Width  Number of SIMD lanes (compile-time constant).
 */
template<idx_t Width>
struct PacketIndex {
    idx_t base_;     ///< First sample index in this packet
    idx_t active_;   ///< Number of active lanes (Width for full packets)

    static constexpr idx_t width = Width;

    /** @brief Create a full packet (all lanes active). */
    static PacketIndex make(idx_t base) { return {base, Width}; }

    /** @brief Create a tail packet with `remaining` active lanes. */
    static PacketIndex makeTail(idx_t base, idx_t remaining)
    {
        return {base, remaining};
    }

    /** @brief True when all lanes are active (hot path). */
    bool full() const { return active_ == Width; }

    /** @brief Extract a scalar SampleIndex for lane `k`. */
    SampleIndex scalar(idx_t k) const { return SampleIndex::make(base_ + k); }
};

/** @brief Alias for the native-width packet index for a given scalar type. */
template<typename DataT>
using NativePacketIndex = PacketIndex<simd::PreferredWidth<DataT>>;

} // namespace feta
