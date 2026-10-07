/**
 * @file PacketOps.h
 * @brief Facade for the CPU SIMD-packet expression operations.
 *
 *  - `RecursivePacketAssign` — compile-time unrolled packet assignment.
 *
 * The definitions are organised into per-concern sub-headers under
 * `packet/`; this file simply pulls them in (include path unchanged for
 * all consumers).
 */
#pragma once

#include "feta/vector/expr/cpu/packet/LoadStore.h"
#include "feta/vector/expr/cpu/packet/Assign.h"
#include "feta/vector/expr/cpu/packet/Prefetch.h"
