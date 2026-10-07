#pragma once
// Include in exactly ONE CUDA translation unit per linked program.
// The main library owner is EarthCorrections.cu; standalone tests may own it.
#include "HpopCorrections.h"
#if defined(__CUDACC__)
namespace paraHPOP::model::accelerations::hpop::detail {
namespace deviceTables {
#define HPOP_FREQUENCY_TABLE(name,count) __device__ __constant__ const FrequencyTerm name[count]
#include "HpopFrequencyTerms.inc"
#undef HPOP_FREQUENCY_TABLE
}
namespace oceanDevice {
#define FES_WAVES(name,count) __device__ __constant__ const OceanWave name[count]
#define FES_TERMS(name,count) __device__ __constant__ const OceanTerm name[count]
#include "Fes2004Terms.inc"
#undef FES_WAVES
#undef FES_TERMS
}
}
#endif
