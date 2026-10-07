#pragma once

#include <feta/err/throw.h>

/**
 * @brief Throw exception from host code.
 *
 * For device code, use BRIE_GPU_THROW.
 */
#define BRIE_THROW(EXCEPTION_TYPE, MESSAGE) FETA_THROW(EXCEPTION_TYPE, MESSAGE)

/**
 * @brief Throw exception from host code if CONDITION is false.
 *
 * For device code, use BRIE_GPU_ASSERT.
 */
#define BRIE_ASSERT(CONDITION, MESSAGE) FETA_ASSERT(CONDITION, MESSAGE)
