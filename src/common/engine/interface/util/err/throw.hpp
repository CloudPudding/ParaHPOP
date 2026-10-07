#pragma once

#include "interface/typedefs.h"

/**
 * @brief Throw exception from host code.
 *
 * For device code, use PARAHPOP_GPU_THROW.
 */
#define PARAHPOP_THROW(EXCEPTION_TYPE, MESSAGE) FETA_THROW(EXCEPTION_TYPE, MESSAGE)

/**
 * @brief Throw exception from host code if CONDITION is false.
 *
 * For device code, use PARAHPOP_GPU_ASSERT.
 */
#define PARAHPOP_ASSERT(CONDITION, MESSAGE) FETA_ASSERT(CONDITION, MESSAGE)
