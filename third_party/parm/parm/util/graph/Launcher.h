#pragma once

#include "parm/util/graph/Graph.h"

#ifndef PARM_CPU_ONLY

namespace parm {
namespace util {
namespace graph {

/**
 * @brief Instantiated CUDA graph executor.
 *
 * Holds a ``cudaGraphExec_t`` (instantiated from a ``cudaGraph_t``)
 * and provides ``launch()`` / ``synchronize()`` / ``setLogicalSize()``
 * for repeated execution on the associated stream. Constructed by
 * ``Graph::launcher()`` only.
 *
 * Lifetime: shares ownership of the producing ``Graph::Storage`` via
 * ``std::shared_ptr<Graph::Storage> storage_``. The source graph and
 * every captured-child original it owns stay alive for as long as
 * any ``Launcher`` (or the producing ``Graph``) holds a reference.
 * That guarantee keeps the kernel-node handles and ``kernelParams``
 * pointers in ``kernelNodes_`` valid for any subsequent
 * ``setLogicalSize`` call.
 *
 * Multi-launcher fan-out: any number of Launchers can share the same
 * ``Storage``. Each instantiates its own independent
 * ``cudaGraphExec_t``.
 *
 * Copy construction and copy assignment are disabled; use move semantics.
 *
 * @note Available only when ``PARM_CPU_ONLY`` is not defined.
 * @see Graph — builds the graph and produces a Launcher.
 */
class Launcher {
    using Self = Launcher;
    friend class Graph;

public:
    /** @brief Default constructor: empty Launcher (no source storage,
     *  no exec instance). Useful as a member that gets move-assigned
     *  later. */
    Launcher() = default;

    /** @brief Copy constructor is forbidden */
    Launcher(Launcher& other)       = delete;
    Launcher(const Launcher& other) = delete;

    /** @brief Move constructor */
    Launcher(Launcher&& other) noexcept
        : storage_{ std::move(other.storage_) }
        , instance_{ other.instance_ }
        , instantiated_{ other.instantiated_ }
        , stream_{ other.stream_ }
        , kernelNodes_{ std::move(other.kernelNodes_) }
    {
        other.instance_     = nullptr;
        other.instantiated_ = false;
        other.stream_       = 0;
    }

    /** @brief Copy assignment is forbidden */
    Self& operator=(Launcher& other)       = delete;
    Self& operator=(const Launcher& other) = delete;

    /** @brief Move assignment */
    Self& operator=(Launcher&& other) noexcept
    {
        /* Free our exec instance before taking over. The shared_ptr
         * release below decrements our previous Storage's ref-count;
         * if we held the last reference the Storage destructor runs
         * here and tears down the source CUDA graph. */
        destroyInstance_();

        storage_      = std::move(other.storage_);
        instance_     = other.instance_;
        instantiated_ = other.instantiated_;
        stream_       = other.stream_;
        kernelNodes_  = std::move(other.kernelNodes_);

        other.instance_     = nullptr;
        other.instantiated_ = false;
        other.stream_       = 0;
        return *this;
    }

    /** @brief Update the stream */
    Self& stream(const cudaStream_t& stream)
    {
        stream_ = stream;
        return *this;
    }

    /** @brief Expose the stream */
    const cudaStream_t& stream() const { return stream_; }

    /** @brief Launch the graph */
    void launch()
    {
        PARM_KERNEL_PRE();
        PARM_CHECK(cudaGraphLaunch(instance_, stream_));
        PARM_KERNEL_POST();
    }

    /** @brief Synchronize */
    inline void synchronize() { PARM_CHECK(cudaStreamSynchronize(stream_)); }

    /** @brief Patch each registered kernel node's launch params for
     *  the new ``logicalSize``.
     *
     *  Per-record: ``idealBlockSize == 0`` keeps capture-time blockDim
     *  and shrinks only ``gridDim`` (required for fixed-layout
     *  reduction-tree kernels); ``> 0`` runs ``computeBlocks`` to
     *  re-tune both. Microsecond-class — no re-capture.
     *
     *  Never calls ``cudaGraphKernelNodeGetParams`` at runtime: the
     *  exec graph round-trip rejects templated kernels with
     *  ``InvalidDeviceFunction``. Snapshotted ``capturedParams`` is
     *  copied and mutated instead.
     *
     *  Lifetime: walks our own ``kernelNodes_`` snapshot. Both the
     *  node handles and the ``kernelParams`` pointers reference
     *  internals of ``storage_->graph``; safe because ``storage_``
     *  keeps Storage alive for our lifetime.
     */
    void setLogicalSize(idx_t logicalSize)
    {
        PARM_ASSERT(instantiated_,
            "Launcher::setLogicalSize requires an instantiated graph");

        /* Skip altogether if logicalSize is 0 */
        if (logicalSize == 0)
            return;

        for (const KernelNodeRecord& rec : kernelNodes_) {
            /* Layout-locked kernels (e.g. scan-tree recursion levels)
             * advertise ``kFixedSize`` to opt out of patching entirely:
             * their grid/block were sized at capture for a buffer
             * smaller than the launcher's logicalSize, and re-deriving
             * gridDim from logicalSize would over-write internal
             * fixed-size scratch (verified to corrupt
             * ``Slice::scanGraph`` recursion's per-block sums). */
            if (rec.idealBlockSize == kFixedSize)
                continue;

            cudaKernelNodeParams params = rec.capturedParams;
            idx_t newBlock;
            idx_t newGrid;
            if (rec.idealBlockSize == 0) {
                newBlock = rec.capturedParams.blockDim.x;
                newGrid  = (logicalSize + newBlock - 1) / newBlock;
            } else {
                computeBlocks(
                    logicalSize, newGrid, newBlock, rec.idealBlockSize);
            }
            params.blockDim = dim3(newBlock, 1, 1);
            params.gridDim  = dim3(newGrid, 1, 1);
            PARM_CHECK(
                cudaGraphExecKernelNodeSetParams(instance_, rec.node, &params));
        }
    }


    /** @brief Number of registered kernel records (for tests / debug). */
    idx_t kernelNodeCount() const
    {
        return static_cast<idx_t>(kernelNodes_.size());
    }

    /** @brief Read-only accessor to the registered kernel records
     *  (tests / diagnostics). */
    const std::vector<KernelNodeRecord>& kernelNodes() const
    {
        return kernelNodes_;
    }

    /** @brief Destructor — destroys the exec instance.
     *
     *  ``storage_`` (the shared ``Graph::Storage`` reference) is
     *  released after; if we hold the last reference the Storage
     *  destructor runs here and tears down the source CUDA graph. */
    ~Launcher() { destroyInstance_(); }

    /** @brief Expose the source graph (lifetime tied to ``storage_``). */
    cudaGraph_t graph() const
    {
        return storage_ ? storage_->graph : nullptr;
    }

protected:
    /** @brief Construct from shared Storage and a kernelNodes_
     *  snapshot. Called only by ``Graph::launcher()``. */
    explicit Launcher(const cudaStream_t& stream,
        std::shared_ptr<Graph::Storage> storage,
        std::vector<KernelNodeRecord> kernelNodes)
        : storage_{ std::move(storage) }
        , stream_{ stream }
        , kernelNodes_{ std::move(kernelNodes) }
    {
        instantiate_();
    }

    /** @brief Create the instance */
    void instantiate_()
    {
        if (!instantiated_) {
            PARM_ASSERT(storage_ && storage_->graph != nullptr,
                "Cannot instantiate a non-created graph");
            PARM_CHECK(cudaGraphInstantiate(
                &instance_, storage_->graph, NULL, NULL, 0));
            instantiated_ = true;
        }
    }

    /** @brief Destroy the instance */
    void destroyInstance_()
    {
        if (instantiated_) {
            PARM_CHECK(cudaGraphExecDestroy(instance_));
            instantiated_ = false;
        }
    }

    /* Data members. ``storage_`` is declared first so it is destroyed
     * last (after ``destroyInstance_`` runs in ``~Launcher``). */
    std::shared_ptr<Graph::Storage> storage_;
    cudaGraphExec_t instance_     = nullptr;
    bool instantiated_            = false;
    cudaStream_t stream_          = 0;

    /** @brief Snapshot of ``Graph::kernelNodes_`` at ``launcher()``
     *  call time. Subsequent ``addNode`` calls on the producing Graph
     *  do not affect this list. Handles inside reference internals
     *  of ``storage_->graph``. */
    std::vector<KernelNodeRecord> kernelNodes_;
};

/* Inline definition of Graph::launcher(); requires the full Launcher
 * definition above. */
inline Launcher Graph::launcher() const
{
    return Launcher(stream_, storage_, kernelNodes_);
}

} // namespace graph
} // namespace util
} // namespace parm

#endif
