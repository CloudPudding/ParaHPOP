#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "brie/util/Paths.h"

namespace brie {
/* Duplicated from `brie/typedefs.h` to keep this header out of the
 * `feta/feta.h` include chain — cbor.cpp is built as LANGUAGE CXX (glaze
 * needs C++23) and feta's bare __host__/__device__ won't parse under g++. */
using Real = double;

namespace cbor {

/**
 * @brief Layout-2 sizing descriptor recovered by the cheap pass-1 walk over
 * a `.brie`/`.brot` file.
 */
struct EphFileSizes {
    int    layout            = 0; ///< 1 or 2
    int    nBodyUnits        = 0; ///< Layout-2 only (0 for Layout-1)
    size_t intMetadataLen    = 0; ///< Layout-2 only
    size_t doubleMetadataLen = 0; ///< Layout-2 only (0 if absent)
    size_t dataLen           = 0; ///< Layout-2 only
};

/** @brief File contents + the result of pass-1 peek (no payload parsing). */
struct EphFileBuffer {
    std::vector<unsigned char> bytes;
    EphFileSizes               sizes;
};

/**
 * @brief Pass 1 — load the file into memory and recover its layout + array sizes.
 *
 * For Layout-2 files the returned `sizes` is fully populated and the caller can
 * size the EphUnit exactly before calling `loadEphLayout2()`.
 *
 * For Layout-1 files only `layout==1` is set; the caller is expected to fall
 * back to the legacy `nlohmann::json`-based path (`Load::cbor`).
 */
EphFileBuffer peekEphFile(const std::string& fileName,
    const util::Paths& paths = util::paths::brieDefaultPath());

/**
 * @brief Pass 2 — Layout-2 fast path. Decodes CBOR straight into pre-allocated
 * FETA host buffers (no `nlohmann::json` DOM, no `std::vector<double>`
 * intermediate).
 *
 * Pre-condition: capacities EXACTLY match the corresponding lengths from
 * `peekEphFile().sizes`. `doubleMetaDst` may be `nullptr` iff
 * `sizes.doubleMetadataLen == 0`.
 *
 * @throws std::runtime_error on parse failure or capacity mismatch.
 */
void loadEphLayout2(const EphFileBuffer& fb,
    int*  intMetaDst,    std::size_t intMetaCap,
    Real* doubleMetaDst, std::size_t doubleMetaCap,
    Real* dataDst,       std::size_t dataCap);

} // namespace cbor

} // namespace brie
