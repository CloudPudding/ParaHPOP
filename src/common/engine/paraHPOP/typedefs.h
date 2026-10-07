#pragma once

#include <brie/brie.h>
#include <feta/feta.h>
#include <interface/typedefs.h>


namespace paraHPOP {
using Real = brie::Real;

using brie::Vec3RArray;

using brie::Vec3R;
using brie::Vec4R;
using brie::Vec6R;
using brie::Vec9R;

using feta::SampleIndex;

/* Naif ID type */
using brie::NaifId;
using brie::core::NaifIdArray;

/* JSON */
using nlohmann::json;

/* Index type */
using idx_t = interface::idx_t;

/* Specific type definition */
using mInt_t  = NaifId; // Metadata long integers
using mReal_t = double; // Metadata double
using mSize_t = idx_t;  // Metadata size

/** @brief Compile-time CUDA launch-bounds traits shared by every device
 *  kernel: the ``__launch_bounds__`` block-size cap and the per-SM block
 *  floor, plus the matching ``__shared__`` stride.  Each kernel family
 *  aliases this with its own tuned ``(maxBlockSize, minBlocksPerSM)`` so the
 *  trait *shape* is single-sourced while the values stay per-kernel. */
template <idx_t MaxBlockSize, idx_t MinBlocksPerSM>
struct KernelLaunchTraits {
    static constexpr idx_t maxBlockSize   = MaxBlockSize;
    static constexpr idx_t minBlocksPerSM = MinBlocksPerSM;
};

} // namespace paraHPOP
