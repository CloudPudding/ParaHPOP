/**
 * @file PacketMask.h
 *
 * SIMD lane masks for conditional (masked) load/store and early-return
 * semantics on CPU.
 *
 * `PacketMask<DataT, Width>` wraps platform-specific mask storage:
 *  - AVX-512: `__mmask8` / `__mmask16`
 *  - AVX2/SSE2: integer bitmask (converted to SIMD at load/store boundary)
 *  - Scalar (Width==1): plain `bool`
 *
 * Masks support boolean algebra (`&`, `|`, `~`), construction from
 * lane counts (`firstN`), and runtime queries (`anyTrue`, `isAllTrue`).
 */
#pragma once

#include "feta/core/simd/PacketTraits.h"

namespace feta {
namespace simd {

/**
 * @brief SIMD lane mask — primary template for Width > 1.
 *
 * @tparam DataT  The scalar type whose packet this mask accompanies.
 * @tparam Width  Number of SIMD lanes.
 */
template<typename DataT, idx_t Width = ::feta::simd::PreferredWidth<DataT>>
struct PacketMask {

#if defined(__AVX512F__)
    using StorageT = std::conditional_t<(Width <= 8), __mmask8, __mmask16>;
    StorageT mask_;

    PacketMask()
        : mask_{}
    {
    }
    explicit PacketMask(StorageT m)
        : mask_(m)
    {
    }

    static PacketMask allTrue()
    {
        return PacketMask{ static_cast<StorageT>((1u << Width) - 1u) };
    }
    static PacketMask allFalse() { return PacketMask{ StorageT{ 0 } }; }
    static PacketMask firstN(idx_t n)
    {
        return PacketMask{ static_cast<StorageT>((1u << n) - 1u) };
    }

    bool anyTrue() const { return mask_ != 0; }
    bool isAllTrue() const
    {
        return mask_ == static_cast<StorageT>((1u << Width) - 1u);
    }

    PacketMask operator&(const PacketMask& o) const
    {
        return PacketMask{ static_cast<StorageT>(mask_ & o.mask_) };
    }
    PacketMask operator|(const PacketMask& o) const
    {
        return PacketMask{ static_cast<StorageT>(mask_ | o.mask_) };
    }
    PacketMask operator~() const
    {
        return PacketMask{ static_cast<StorageT>(
            ~mask_ & static_cast<StorageT>((1u << Width) - 1u)) };
    }

    bool lane(idx_t k) const { return (mask_ >> k) & 1u; }

#else
    // SSE2/AVX2 and fallback: store mask as integer bitmask.
    // Conversion to SIMD comparison vectors happens at the Packet level.
    using StorageT = unsigned;
    StorageT mask_;

    PacketMask()
        : mask_{ 0 }
    {
    }
    explicit PacketMask(unsigned m)
        : mask_(m)
    {
    }

    static PacketMask allTrue() { return PacketMask{ (1u << Width) - 1u }; }
    static PacketMask allFalse() { return PacketMask{ 0u }; }
    static PacketMask firstN(idx_t n) { return PacketMask{ (1u << n) - 1u }; }

    bool anyTrue() const { return mask_ != 0; }
    bool isAllTrue() const { return mask_ == (1u << Width) - 1u; }

    PacketMask operator&(const PacketMask& o) const
    {
        return PacketMask{ mask_ & o.mask_ };
    }
    PacketMask operator|(const PacketMask& o) const
    {
        return PacketMask{ mask_ | o.mask_ };
    }
    PacketMask operator~() const
    {
        return PacketMask{ ~mask_ & ((1u << Width) - 1u) };
    }

    bool lane(idx_t k) const { return (mask_ >> k) & 1u; }
#endif
};

// ─── Scalar specialisation (Width == 1): zero overhead ──────────────────────

template<typename DataT>
struct PacketMask<DataT, 1> {
    bool mask_;

    PacketMask()
        : mask_{ false }
    {
    }
    explicit PacketMask(bool m)
        : mask_(m)
    {
    }

    static PacketMask allTrue() { return PacketMask{ true }; }
    static PacketMask allFalse() { return PacketMask{ false }; }
    static PacketMask firstN(idx_t n) { return PacketMask{ n > 0 }; }

    bool anyTrue() const { return mask_; }
    bool isAllTrue() const { return mask_; }

    PacketMask operator&(const PacketMask& o) const
    {
        return PacketMask{ mask_ && o.mask_ };
    }
    PacketMask operator|(const PacketMask& o) const
    {
        return PacketMask{ mask_ || o.mask_ };
    }
    PacketMask operator~() const { return PacketMask{ !mask_ }; }

    bool lane(idx_t) const { return mask_; }
};

} // namespace simd
} // namespace feta
