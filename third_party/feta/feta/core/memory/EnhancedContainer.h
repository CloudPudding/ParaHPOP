#pragma once

#include "feta/core/memory/Container.h"

namespace feta {
namespace core {
namespace memory {

/**
 * @brief Pair an interface container with an optional workable (device) container.
 *
 * Only the `Interface` container is allocated on construction.
 * The `Workable` container is built lazily on the first call to `upload()`.
 * Use `Interface = CPU` / `Workable = CUDA_DEVICE` for the standard host↔GPU pattern.
 *
 * @tparam DataT      Scalar element type.
 * @tparam Interface  Device for the primary (host-accessible) container.
 * @tparam Workable   Device for the secondary (e.g. GPU) container; use `NONE` for host-only.
 * @tparam VectorDim  Compile-time vector dimension; 1 for scalar arrays.
 */
template<typename DataT, Device Interface, Device Workable, idx_t VectorDim = 1>
class EnhancedContainer {
public:
    /** @brief Expose the types */
    using InterfaceT = Container<DataT, Interface, VectorDim>;
    using WorkableT  = Container<DataT, Workable, VectorDim>;
    using StreamT    = typename StreamSwitcher<Interface>::T;
    /** @brief Expose the vector dimension */
    static constexpr idx_t VecDims = VectorDim;

    /**
     * @brief Construct a non-owning `EnhancedContainer` that borrows its data from `other`.
     *
     * @param other  Source container whose pointer will be borrowed.
     * @return New `EnhancedContainer` with a borrowed interface and an unallocated workable.
     */
    template<typename OtherContainerT>
    static EnhancedContainer borrow(const OtherContainerT& other)
    {
        InterfaceT interface = InterfaceT::template borrow<OtherContainerT>(
            other);
        return EnhancedContainer(std::move(interface));
    }

    /** @brief Default constructor */
    EnhancedContainer(const bool& flexible = false)
        : interface_{ flexible }
        , workable_{ flexible }
    {
        validateTypes();
    }

    /** @brief Construct by size and initial value */
    EnhancedContainer(const idx_t& sz,
        const DataT& initialValue = Default<DataT>(),
        const bool& flexible      = false)
        : interface_{ sz, initialValue, flexible }
        , workable_{ flexible }
    {
        validateTypes();
    }

    /** @brief Copy constructor */
    EnhancedContainer(const EnhancedContainer& other)
        : interface_{ other.interface_ }
        , workable_{ other.workable_ }
    {
        validateTypes();
    }

    /** @brief Move constructor */
    EnhancedContainer(EnhancedContainer&& other)
        : interface_{ std::move(other.interface_) }
        , workable_{ std::move(other.workable_) }
    {
        validateTypes();
    }

    /** @brief Construct by moving an already-built interface container. */
    EnhancedContainer(InterfaceT&& interface)
        : interface_{ std::move(interface) }
        , workable_{ interface.isFlexible() }
    {
        validateTypes();
    }

    /** @brief Copy assignment operator */
    EnhancedContainer& operator=(const EnhancedContainer& other)
    {
        interface_ = other.interface_;
        workable_  = other.workable_;
        return *this;
    }

    /** @brief Move assignment operator */
    EnhancedContainer& operator=(EnhancedContainer&& other)
    {
        interface_ = std::move(other.interface_);
        workable_  = std::move(other.workable_);
        return *this;
    }

    /** @brief Return a reference to the interface container. */
    InterfaceT& interface() { return interface_; }
    const InterfaceT& interface() const { return interface_; }

    /** @brief Return a reference to the workable container. */
    WorkableT& workable()
    {
        static_assert(!isNone<Workable>, "Workable is None");
        return workable_;
    }
    const WorkableT& workable() const
    {
        static_assert(!isNone<Workable>, "Workable is None");
        return workable_;
    }

    /**
     * @brief Copy interface data to the workable container, using an async stream if supported.
     *
     * Rebuilds the workable container if its size does not match the interface.
     * Falls back to a blocking copy when the device pair does not support async transfers.
     *
     * @param stream  CUDA stream to use for async transfers (ignored on CPU builds).
     */
    void upload([[maybe_unused]] const StreamT& stream = 0)
    {
        /* Do nothing if workable is none */
        if constexpr (VecDims != 0 && !isNone<Workable>) {
            /* If interface is flexible, prevent the crash */
            if (!interface_.isFlexible())
                FETA_ASSERT(interface_.isValid() && interface_.size() != 0,
                    std::string("Interface data is not allocated! isValid=")
                        + (interface_.isValid() ? "true" : "false")
                        + ", size=" + std::to_string(interface_.size()));
            /* Check whether this pair supports asynchronous data exchange*/
            constexpr bool canBeAsync = hasAsynchronousCopySupport<Interface>
                && hasAsynchronousCopySupport<Workable>;
            /* If the pair can run async support*/
            if constexpr (canBeAsync) {
                /* rebuild workable if required */
                if (workable_.size() != size()) {
                    rebuildWorkable_();
                }
                /* Do nothing if size() == 0 */
                if (interface_.size() == 0)
                    return;
                /* Do asynchronous copy otherwise */
                using CopyT = Copy<Interface, Workable>;
                CopyT::template async<DataT>(
                    interface_.data(), workable_.data(), totalSize(), stream);
            } else {
                blockingUpload();
            }
        }
        /** TODO: add further cases here (e.g. CUDA Managed memory) */
    }
    /**
     * @brief Copy interface data to the workable container synchronously.
     *
     * Rebuilds the workable container if its size does not match the interface.
     */
    void blockingUpload()
    {
        /* Do nothing if workable is none */
        if constexpr (VecDims != 0 && !isNone<Workable>) {
            /* If interface is flexible, prevent the crash */
            if (!interface_.isFlexible())
                FETA_ASSERT(interface_.isValid() && interface_.size() != 0,
                    "Interface data is not allocated!");
            /* rebuild workable if required */
            if (workable_.size() != size()) {
                rebuildWorkable_();
            }
            /* Copy the data */
            /* Do nothing if size() == 0 */
            if (interface_.size() == 0)
                return;
            /* Do the synchronous copy otherwise */
            using CopyT = Copy<Interface, Workable>;
            CopyT::template blocking<DataT>(
                interface_.data(), workable_.data(), totalSize());
        }
        /** TODO: add further cases here (e.g. CUDA Managed memory) */
    }

    /**
     * @brief Copy workable data back to the interface container, using an async stream if supported.
     *
     * @param stream  CUDA stream to use for async transfers (ignored on CPU builds).
     */
    void download([[maybe_unused]] const StreamT& stream = 0)
    {
        /* Do nothing if workable is none */
        if constexpr (VecDims != 0 && !isNone<Workable>) {
            /* If interface is flexible, prevent the crash */
            if (!interface_.isFlexible()) {
                FETA_ASSERT(interface_.isValid() && interface_.size() != 0,
                    "Interface data is not allocated!");
                FETA_ASSERT(workable_.isValid() && workable_.size() != 0,
                    "Workable data is not allocated!");
                FETA_ASSERT(workable_.size() == interface_.size(),
                    "Workable and Interface are not compatible!");
            }
            /* Check whether this pair supports asynchronous data exchange*/
            constexpr bool canBeAsync = hasAsynchronousCopySupport<Interface>
                && hasAsynchronousCopySupport<Workable>;
            /* If the pair can run async support*/
            if constexpr (canBeAsync) {
                /* Do nothing if size() == 0 */
                if (interface_.size() == 0)
                    return;
                /* Do asynchronous copy otherwise*/
                using CopyT = Copy<Workable, Interface>;
                CopyT::template async<DataT>(
                    workable_.data(), interface_.data(), totalSize(), stream);
            } else {
                blockingUpload();
            }
        }
        /** TODO: add further cases here (e.g. CUDA Managed memory) */
    }
    /** @brief Copy workable data back to the interface container synchronously. */
    void blockingDownload()
    {
        /* Do nothing if workable is none */
        if constexpr (VecDims != 0 && !isNone<Workable>) {
            /* If interface is flexible, prevent the crash */
            if (!interface_.isFlexible()) {
                FETA_ASSERT(interface_.isValid() && interface_.size() != 0,
                    "Interface data is not allocated!");
                FETA_ASSERT(workable_.isValid() && workable_.size() != 0,
                    "Workable data is not allocated!");
                FETA_ASSERT(workable_.size() == interface_.size(),
                    "Workable and Interface are not compatible!");
            }
            /* Do nothing if size() == 0 */
            if (interface_.size() == 0)
                return;
            /* Do the synchronous copy otherwise */
            using CopyT = Copy<Workable, Interface>;
            CopyT::template blocking<DataT>(
                workable_.data(), interface_.data(), totalSize());
        }
        /** TODO: add further cases here (e.g. CUDA Managed memory) */
    }

    /** @brief Return the number of elements in the array. */
    inline idx_t size() const { return interface_.size(); }

    /** @brief Return the total number of scalar elements (`size() * VecDims`). */
    inline idx_t totalSize() const { return interface_.totalSize(); }

    /** @brief Return a pointer to the interface data. */
    inline DataT* data() const { return interface_.data(); }

    /** @brief Destroy the workable container, freeing device memory. */
    void clearWorkable()
    {
        if constexpr (VecDims != 0 && !isNone<Workable>) {
            workable_ = std::move(WorkableT(interface_.isFlexible()));
        }
    }

    /** @brief Return `true` if the container is flexible (allows null data pointer). */
    inline bool isFlexible() const { return interface_.isFlexible(); }

protected:
    /** @brief Assert at compile time that `Interface` and `Workable` form a valid pairing. */
    static constexpr inline void validateTypes()
    {
        /* interface type must be self standing (i.e. directly accessible
         * with almost standard C++) */
        static_assert(
            isSelfStanding<Interface>, "Interface type must be Self-standing");

        /* if workable is not none, we need to assess the type compatibility
         */
        if constexpr (!isNone<Workable>) {
            static_assert(areCompatible<Interface, Workable>,
                "Incompatible interface and workable container types");
        }
    }

    /** @brief Reallocate the workable container to match the current interface size. */
    inline void rebuildWorkable_()
    {
        if constexpr (VecDims != 0 && !isNone<Workable>) {
            workable_ = std::move(WorkableT(
                interface_.size(), Default<DataT>(), interface_.isFlexible()));
        }
    }

    InterfaceT interface_;
    WorkableT workable_;
};

} // namespace memory
} // namespace core
} // namespace feta