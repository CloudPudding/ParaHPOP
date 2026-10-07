#include "paraHPOP/model/accelerations/AccumulateBodies.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <cuda_runtime_api.h>

namespace paraHPOP {
namespace model {
namespace accelerations {
namespace kernel {

ForceKernelOptions forceKernelOptions(idx_t sampleCount)
{
    struct Overrides {
        ForceKernelOptions value;
        bool shThreads = false, nrlThreads = false, radial = false, division = false;
    };
    static const Overrides overrides = [] {
        Overrides o;
        auto& p = o.value;
        // Former force-schedule/reduction overrides no longer change topology.
        const auto readThreads = [](const char* key, unsigned maximum,
                                    bool& present) -> idx_t {
            const char* raw = std::getenv(key);
            if (!raw || !*raw) return 0;
            present = true;
            if (std::string(raw) == "auto") return 0;
            const std::string value(raw);
            for (unsigned n : {32u, 64u, 128u, 256u})
                if (n <= maximum && value == std::to_string(n)) return n;
            throw std::invalid_argument(std::string("Invalid ") + key);
        };
        p.shThreads = readThreads("PARAHPOP_PERF_SH_THREADS", 256, o.shThreads);
        p.nrlThreads = readThreads("PARAHPOP_PERF_NRL_THREADS", 128, o.nrlThreads);
        const auto readFlag = [](const char* key, bool& present) {
            const char* raw = std::getenv(key);
            if (!raw || !*raw) return false;
            present = true;
            const std::string value(raw);
            if (value == "1") return true;
            if (value == "0") return false;
            throw std::invalid_argument(std::string("Invalid ") + key);
        };
        p.shRadialCache = readFlag("PARAHPOP_PERF_SH_CACHE", o.radial);
        p.shCorrectedDivision = readFlag("PARAHPOP_PERF_SH_DIVISION", o.division);
        return o;
    }();

    ForceKernelOptions p;
    int device = 0;
    cudaDeviceProp props{};
    auto status = cudaGetDevice(&device);
    if (status == cudaSuccess) status = cudaGetDeviceProperties(&props, device);
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("Cannot resolve GPU force policy: ")
            + cudaGetErrorString(status));

    // Empirical profile for the tested 14-SM GTX 1650. Other devices keep
    // the original automatic layout unless an explicit override is given.
    // Thresholds stay in this TU so retuning does not rebuild physics headers.
    if (props.major == 7 && props.minor == 5 && props.multiProcessorCount == 14
        && std::strstr(props.name, "GTX 1650")) {
        p.shThreads = sampleCount <= 4096 ? 32 : 128;
        p.nrlThreads = 32;
        p.shRadialCache = sampleCount > 256;
        p.shCorrectedDivision = true;
    }
    const auto& o = overrides;
    if (o.shThreads) p.shThreads = o.value.shThreads;
    if (o.nrlThreads) p.nrlThreads = o.value.nrlThreads;
    if (o.radial) p.shRadialCache = o.value.shRadialCache;
    if (o.division) p.shCorrectedDivision = o.value.shCorrectedDivision;
    return p;
}

__global__ __launch_bounds__(AccumulateBodiesLaunch::maxBlockSize,
    AccumulateBodiesLaunch::minBlocksPerSM) void zeroBodyAccScratch(
    feta::vector::Array<Real, 3>::GRef bodyAcc,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated)
{
    SampleIndex i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);

    if (i.global() >= bodyAcc.size())
        return;

    if (terminated[i])
        return;

    /* Fully feta — assigns a 3-vector zero through the expression
     * template machinery; no raw pointer arithmetic, no manual loop. */
    bodyAcc[i] = feta::vector::Item<Real, 3>::Zeros();
}

__global__ __launch_bounds__(AccumulateBodiesLaunch::maxBlockSize,
    AccumulateBodiesLaunch::minBlocksPerSM) void reduceBodyAccScratch(
    feta::vector::Array<Real, 3>::GRef acc,
    GRID_CONSTANT() feta::vector::Array<Real, 3>::GRef bodyAcc,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated)
{
    SampleIndex i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);

    if (i.global() >= acc.size())
        return;

    if (terminated[i])
        return;

    /* Feta expression-template `+=` — non-atomic, single writer per
     * address per launch.  Per-body kernels are chained serially in
     * fixed body order in the per-step graph, so the cumulative
     * order of `acc[i] += bodyAcc_b[i]` over b is the body-loop
     * order — same on every replay, matches host. */
    acc[i] += bodyAcc[i];
}

__global__ __launch_bounds__(AccumulateBodiesLaunch::maxBlockSize,
    AccumulateBodiesLaunch::minBlocksPerSM) void reduceForceAccScratch(
    VecGRef acc, GRID_CONSTANT() BodyAccTermsArrayT::GRef::HandleT terms,
    GRID_CONSTANT() idx_t nTerms, GRID_CONSTANT() VecGRef corrections,
    GRID_CONSTANT() bool hasCorrections,
    GRID_CONSTANT() feta::scalar::Array<bool>::GRef::HandleT terminated)
{
    SampleIndex i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);

    if (i.global() >= acc.size())
        return;

    if (terminated[i])
        return;

    Vec3R total = acc[i];
    for (idx_t t = 0; t < nTerms; ++t) {
        const BodyAccTerm term = terms[t];
        Vec3R body = term.base[i];
        if (term.hasShape)
            body += term.shape[i];
        total += body;
    }
    if (hasCorrections)
        total += corrections[i];
    acc[i] = total;
}

} // namespace kernel
} // namespace accelerations
} // namespace model
} // namespace paraHPOP
