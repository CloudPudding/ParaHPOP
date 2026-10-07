#pragma once

#include "brie/typedefs.h"

#ifndef BRIE_CPU_ONLY
#include <cuda.h>
#endif

namespace brie {
namespace orientations {
namespace metadata {
namespace common {

struct Sizes {
    static constexpr vecdim_t Int  = 12;
    static constexpr vecdim_t Real = 2;
};

enum IntMembers {
    /* common metadata */
    UNITTYPE,
    TARGET,
    /* information on data*/
    DATAOFFSET = common::Sizes::Int - 2,
    DATASIZE
};

} // namespace common

namespace body {
/**
 * @brief Enumeration for unit specific rot body unit integer metadata members
 *
 */
enum IntMembers {
    PDEG_RA = 2,
    PDEG_DEC,
    PDEG_PM,
    NUM_DIMS_RADII,
    BARYCENTER,
    NUM_NP_TERMS_RA,
    NUM_NP_TERMS_DEC,
    NUM_NP_TERMS_PM
};
} // namespace body

namespace bary {
/**
 * @brief Enumeration for rot barycenter unit integer metadata members
 *
 */
enum IntMembers { PDEG_NP = 2, NUM_NP_ANGLES };
} // namespace bary


namespace binary {
/**
 * @brief Enumeration for rot bin frame unit integer metadata members
 *
 */
enum IntMembers {
    /* Integer members composing the Ephemeris metadata */
    INERT_FRAME = 2,
    INTERPOL_TYPE,
    NINTERVALS,
    PDEG
};
} // namespace binary


namespace iers {
/**
 * @brief Integer metadata members for the model-based orientation family
 * (``unit_type == 4``).
 *
 * This is a NEW @c unit_type (a model-based orientation family), *not* a new
 * @c dtype: the binary-frame ``INTERPOL_TYPE`` carries the literal NAIF
 * segment data type, which stays free to carry genuine NAIF numbers for
 * SPICE-derived specializations. ``MODEL_ID`` selects the ModelPolicy
 * (0 = IERS2000) and ``INTERP_ID`` the InterpPolicy (0 = uniform Lagrange);
 * the rest describe the dual-table Lagrange payload. The Real envelope is
 * carried in the shared ``RealMembers`` ``INITIALEPOCH`` / ``FINALEPOCH``.
 */
enum IntMembers {
    MODEL_ID = 2,  /**< 0 = IERS2000 (CIO-based ICRF→ITRF) */
    INTERP_ID,     /**< 0 = uniform Lagrange; 1 = non-uniform ERP (the ERP
                    *   node keys are appended to the payload so the device
                    *   searches the actual leap-second grid). Documentation
                    *   only — the InterpPolicy is a compile-time type. */
    N_NODES_NUT,   /**< nutation (X,Y,s) node count */
    N_NODES_ERP,   /**< ERP (xp,yp,ΔUT1,δX,δY) node count */
    LAG_ORDER_NUT, /**< nutation Lagrange degree (5) */
    LAG_ORDER_ERP  /**< ERP Lagrange degree (3) */
};
} // namespace iers


/**
 * @brief Enumeration for Real metadata members
 *
 */
enum RealMembers {
    /* Real members composing the Ephemeris metadata */
    INITIALEPOCH,
    FINALEPOCH,
    /* Leave the following item as last -- it only act as enum size*/
    REALSIZE
};

} // namespace metadata
} // namespace orientations
} // namespace brie