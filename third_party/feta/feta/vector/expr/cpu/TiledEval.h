/**
 * @file TiledEval.h
 * @brief Facade for CPU tiled (scalar + SIMD-packet) expression evaluation.
 *
 * The definitions are organised into per-concern sub-headers under `tiled/`;
 * this file simply pulls them in (include path unchanged for all consumers).
 * The include order preserves the original layout: scalar tiled eval, then
 * the SIMD packet ops, then the packet eval / capture / reduction machinery.
 */
#pragma once

#include "feta/vector/expr/cpu/tiled/ScalarEval.h"

// ═════════════════════════════════════════════════════════════════════════════
//  SIMD Packet-Based Evaluation (CPU only)
// ═════════════════════════════════════════════════════════════════════════════
#include "feta/vector/expr/cpu/tiled/PacketEval.h"
#include "feta/vector/expr/cpu/tiled/Capture.h"
#include "feta/vector/expr/cpu/tiled/Reductions.h"
