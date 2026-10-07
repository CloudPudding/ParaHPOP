/**
 * @file OperationsDetail.h
 * @brief Facade for the expression-templated vector operation nodes.
 *
 * This header contains implementation details for the expression-templated
 * vector operations exposed in `VectorOperations.cuh`. The definitions are
 * organised into per-concern sub-headers under `nodes/`; this file simply
 * pulls them in (include path unchanged for all consumers).
 */
#pragma once

#include "feta/vector/expr/nodes/Recursion.h"
#include "feta/vector/expr/nodes/Arithmetic.h"
#include "feta/vector/expr/nodes/Geometric.h"
#include "feta/vector/expr/nodes/Quaternion.h"
#include "feta/vector/expr/nodes/AssignMultiLoad.h"
