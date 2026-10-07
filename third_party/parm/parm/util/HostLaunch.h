#pragma once

#include "parm/typedefs.h"
#include <type_traits>

namespace parm {
namespace util {

struct Host {

    /** @brief Safe scalar packet load — uses masked load for tail packets. */
    template<typename DataT, feta::idx_t W>
    static inline feta::simd::Packet<DataT, W> packetLoad(
        const DataT* ptr, const feta::PacketIndex<W>& pi)
    {
        using PacketT = feta::simd::Packet<DataT, W>;
        using MaskT   = feta::simd::PacketMask<DataT, W>;
        if (pi.full())
            return PacketT::load(ptr + pi.base_);
        else
            return PacketT::maskLoad(
                ptr + pi.base_, MaskT::firstN(pi.active_));
    }

    /** @brief GRef/Handle overload — forwards to the raw-pointer version.
     *  Accepts any object exposing ``.data()`` (feta scalar::Array GRef
     *  or its HandleT) so callers don't need a ``T* xPtr = arr.data();``
     *  local. */
    template<typename DataT, feta::idx_t W, typename ArrayLikeT>
        requires (!std::is_pointer_v<std::remove_cvref_t<ArrayLikeT>>)
              && requires(const ArrayLikeT& a) { a.data(); }
    static inline feta::simd::Packet<DataT, W> packetLoad(
        const ArrayLikeT& arr, const feta::PacketIndex<W>& pi)
    {
        return packetLoad<DataT, W>(arr.data(), pi);
    }

    /** @brief Safe scalar packet store — uses masked store for tail packets. */
    template<typename DataT, feta::idx_t W>
    static inline void packetStore(
        DataT* ptr, const feta::PacketIndex<W>& pi,
        const feta::simd::Packet<DataT, W>& val)
    {
        using PacketT = feta::simd::Packet<DataT, W>;
        using MaskT   = feta::simd::PacketMask<DataT, W>;
        if (pi.full())
            PacketT::store(ptr + pi.base_, val);
        else
            PacketT::maskStore(
                ptr + pi.base_, MaskT::firstN(pi.active_), val);
    }

    /** @brief GRef/Handle overload — forwards to the raw-pointer version. */
    template<typename DataT, feta::idx_t W, typename ArrayLikeT>
        requires (!std::is_pointer_v<std::remove_cvref_t<ArrayLikeT>>)
              && requires(const ArrayLikeT& a) { a.data(); }
    static inline void packetStore(
        const ArrayLikeT& arr, const feta::PacketIndex<W>& pi,
        const feta::simd::Packet<DataT, W>& val)
    {
        packetStore<DataT, W>(arr.data(), pi, val);
    }

    /** @brief Build a PacketMask from a contiguous bool array at the given
     *  packet index.  Each lane whose bool is `true` becomes active. */
    template<typename DataT, idx_t W>
    static inline feta::simd::PacketMask<DataT, W> loadMask(
        const bool* flags, const feta::PacketIndex<W>& pi)
    {
        using MaskT = feta::simd::PacketMask<DataT, W>;
        unsigned bits = 0;
        const idx_t n = pi.active_;
        for (idx_t k = 0; k < n; ++k)
            bits |= (static_cast<unsigned>(flags[pi.base_ + k]) << k);
        return MaskT{static_cast<typename MaskT::StorageT>(bits)};
    }

    /** @brief GRef/Handle overload — forwards to the raw-pointer version. */
    template<typename DataT, idx_t W, typename ArrayLikeT>
        requires (!std::is_pointer_v<std::remove_cvref_t<ArrayLikeT>>)
              && requires(const ArrayLikeT& a) { a.data(); }
    static inline feta::simd::PacketMask<DataT, W> loadMask(
        const ArrayLikeT& flags, const feta::PacketIndex<W>& pi)
    {
        return loadMask<DataT, W>(flags.data(), pi);
    }

    /** @brief Dispatch active mask lanes to a scalar kernel. */
    template<idx_t W, typename MaskT, typename KernelFunc>
    static inline void dispatchMasked(
        const feta::PacketIndex<W>& pi, const MaskT& active,
        KernelFunc&& kernel)
    {
        if (!active.anyTrue()) return;
        for (idx_t k = 0; k < pi.active_; ++k) {
            if (active.lane(k))
                kernel(pi.scalar(k));
        }
    }

    /** @brief Combine a mask with the tail mask for partial packets. */
    template<idx_t W>
    static inline feta::simd::PacketMask<Real, W> applyTail(
        feta::simd::PacketMask<Real, W> mask,
        const feta::PacketIndex<W>& pi)
    {
        if (!pi.full())
            mask = mask & feta::simd::PacketMask<Real, W>::firstN(pi.active_);
        return mask;
    }

    /** @brief Launcher wrapper — SIMD packet-based, processes W samples per
     *  iteration.  Uses context-aware ``packetBatchedFor``: work-shares
     *  tiles across existing OMP threads if inside a parallel region,
     *  otherwise runs serially.
     *
     *  @param bytesPerSample  Caller's per-sample working-set estimate.
     *                         ``0`` (default) falls back to feta's
     *                         compile-time ``DEFAULT_TILE_SIZE``.  A
     *                         non-zero value drives runtime L2-derived
     *                         tile sizing — pass the same value across
     *                         every host kernel call in a step so each
     *                         thread's tile slice stays L2-resident
     *                         across kernels (PR-5 cross-phase
     *                         residency). */
    template<typename KernelFunc>
    static inline void launch(const idx_t& size, KernelFunc&& kernel,
        idx_t bytesPerSample = 0)
    {
        feta::cpu::packetBatchedFor<Real>(
            idx_t{0}, size, [&](const auto& pi) {
                for (idx_t k = 0; k < pi.active_; ++k)
                    kernel(pi.scalar(k));
            }, bytesPerSample, /*parallel=*/false);
    }

    /** @brief Launcher wrapper, excluding items where the flag is true.
     *
     *  Loads W boolean flags per packet into a PacketMask and skips
     *  the entire packet when no lanes are active, avoiding per-sample
     *  branch mispredictions.  ``bytesPerSample`` follows the same
     *  semantics as ``launch``. */
    template<typename KernelFunc>
    static inline void launchIfNot(const idx_t& size,
        feta::scalar::Array<bool>::GRef::HandleT terminated,
        KernelFunc&& kernel, idx_t bytesPerSample = 0)
    {
        const bool* flags = terminated.data();
        feta::cpu::packetBatchedFor<Real>(
            idx_t{0}, size, [&](const auto& pi) {
                constexpr idx_t W =
                    std::remove_cvref_t<decltype(pi)>::width;
                auto active = applyTail<W>(
                    ~loadMask<Real, W>(flags, pi), pi);
                dispatchMasked(pi, active, kernel);
            }, bytesPerSample, /*parallel=*/false);
    }

    /** @brief Launcher wrapper, including only items where the flag is true.
     *
     *  Loads W boolean flags per packet into a PacketMask and skips
     *  the entire packet when no lanes are active.  ``bytesPerSample``
     *  follows the same semantics as ``launch``. */
    template<typename KernelFunc>
    static inline void launchIf(const idx_t& size,
        const feta::scalar::Array<bool>::GRef::HandleT terminated,
        KernelFunc&& kernel, idx_t bytesPerSample = 0)
    {
        const bool* flags = terminated.data();
        feta::cpu::packetBatchedFor<Real>(
            idx_t{0}, size, [&](const auto& pi) {
                constexpr idx_t W =
                    std::remove_cvref_t<decltype(pi)>::width;
                auto active = applyTail<W>(
                    loadMask<Real, W>(flags, pi), pi);
                dispatchMasked(pi, active, kernel);
            }, bytesPerSample, /*parallel=*/false);
    }
    /** @brief Packet-aware launch — passes PacketIndex directly to kernel.
     *  The kernel receives a PacketIndex<W> and operates on W samples at once.
     *  ``bytesPerSample`` follows the same semantics as ``launch``. */
    template<typename KernelFunc>
    static inline void packetLaunch(const idx_t& size, KernelFunc&& kernel,
        idx_t bytesPerSample = 0)
    {
        feta::cpu::packetBatchedFor<Real>(
            idx_t{0}, size, kernel, bytesPerSample, /*parallel=*/false);
    }

    /** @brief Packet-aware launch excluding terminated samples.
     *  Skips entire packets where all lanes are terminated.
     *  For mixed packets, the kernel receives the full PacketIndex
     *  and must handle masking internally (or use the provided mask).
     *  ``bytesPerSample`` follows the same semantics as ``launch``. */
    template<typename KernelFunc>
    static inline void packetLaunchIfNot(const idx_t& size,
        feta::scalar::Array<bool>::GRef::HandleT terminated,
        KernelFunc&& kernel, idx_t bytesPerSample = 0)
    {
        const bool* flags = terminated.data();
        feta::cpu::packetBatchedFor<Real>(
            idx_t{0}, size, [&](const auto& pi) {
                constexpr idx_t W =
                    std::remove_cvref_t<decltype(pi)>::width;
                auto active = applyTail<W>(
                    ~loadMask<Real, W>(flags, pi), pi);
                if (active.anyTrue())
                    kernel(pi);
            }, bytesPerSample, /*parallel=*/false);
    }
};

} // namespace util
} // namespace parm
