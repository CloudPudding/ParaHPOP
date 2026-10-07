#pragma once

#include "feta/core/memory/Allocator.h"
#include "feta/core/memory/Copy.h"

namespace feta {
namespace core {
namespace memory {

/** @brief Switcher for texture data type */
template<bool HasTextureSupport>
struct TextureTypeSwitcher {
    // Dummy type for non-texture-supporting devices
    using T = unsigned long long;
    static T Default() { return 0; }
};

#ifdef FETA_CPU_ONLY

/** @brief Dummy wrapper for texture object interface */
class CudaTextureObject {
public:
    /** @brief Default constructor */
    CudaTextureObject() = default;

    /** @brief Copy constructor is forbidden */
    CudaTextureObject(CudaTextureObject& other)       = delete;
    CudaTextureObject(const CudaTextureObject& other) = delete;

    /** @brief Move constructor is allowed */
    CudaTextureObject(CudaTextureObject&& other)
        : tex_{ std::exchange(other.tex_, 0) }
    {
    }

    /** @brief Construct from the given pointer and size - does nothing */
    template<typename DataT>
    CudaTextureObject(DataT* ptr, const idx_t& size)
    {
    }

    /** @brief Copy assignment is forbidden */
    CudaTextureObject& operator=(CudaTextureObject& other)       = delete;
    CudaTextureObject& operator=(const CudaTextureObject& other) = delete;

    /** @brief Move assignment is allowed */
    CudaTextureObject& operator=(CudaTextureObject&& other)
    {
        if (this == &other)
            return *this;
        tex_ = std::exchange(other.tex_, 0);
        return *this;
    }
    /** @brief Expose whether the current texture object is valid */
    bool isValid() const { return true; }

    /** @brief Expose the native texture object */
    unsigned int native() const { return tex_; }

protected:
    unsigned int tex_ = 0;
};

#else

/** @brief RAII wrapper for cuda texture */
class CudaTextureObject {
public:
    /** @brief Default constructor */
    CudaTextureObject() = default;

    /** @brief Copy constructor is forbidden */
    CudaTextureObject(CudaTextureObject& other)       = delete;
    CudaTextureObject(const CudaTextureObject& other) = delete;

    /** @brief Move constructor is allowed */
    CudaTextureObject(CudaTextureObject&& other)
        : tex_{ std::exchange(other.tex_, 0) }
        , textureCreated_{ std::exchange(other.textureCreated_, false) }
    {
    }

    /** @brief Construct from the given pointer and size */
    template<typename DataT>
    CudaTextureObject(DataT* ptr, const idx_t& size)
    {
        if (ptr != nullptr && size != 0) {
            using AllocatorT = Allocator<Device::CUDA_DEVICE>;
            AllocatorT::template bindToTexture<DataT, cudaTextureObject_t>(
                ptr, tex_, size);
            textureCreated_ = true;
        }
    }

    /** @brief Copy assignment is forbidden */
    CudaTextureObject& operator=(CudaTextureObject& other)       = delete;
    CudaTextureObject& operator=(const CudaTextureObject& other) = delete;

    /** @brief Move assignment is allowed */
    CudaTextureObject& operator=(CudaTextureObject&& other)
    {
        if (this == &other)
            return *this;
        tex_            = std::exchange(other.tex_, 0);
        textureCreated_ = std::exchange(other.textureCreated_, false);
        return *this;
    }

    /** @brief Expose the native texture object */
    cudaTextureObject_t native() const { return tex_; }

    /** @brief Expose whether the current texture object is valid */
    bool isValid() const { return textureCreated_; }

    /** @brief Destructor */
    ~CudaTextureObject() { destroy_(); }

protected:
    /** @brief Destroy the texture object if required */
    void destroy_()
    {
        if (textureCreated_) {
            FETA_CHECK(cudaDestroyTextureObject(tex_));
            textureCreated_ = false;
        }
    }

    cudaTextureObject_t tex_ = 0;
    bool textureCreated_     = false;
};
#endif

/* Specialize texture type switcher */
template<>
struct TextureTypeSwitcher<true> {
    using T = CudaTextureObject;
    static T Default() { return T(); }
};

/**
 * @brief Generic data container that manages allocation and deallocation.
 *
 * Generalises scalar arrays (default `VectorDim = 1`) and SoA vector arrays.
 * Ownership, device target, and optional texture binding are controlled via
 * template parameters.
 *
 * @tparam DataT      Scalar element type.
 * @tparam Target_    Target memory device (see `Device` enum).
 * @tparam VectorDim  Compile-time vector dimension; 1 for scalar arrays.
 */
template<typename DataT, Device Target_, idx_t VectorDim = 1>
class Container {
    /* Allow other Container types to access private members for conversion */
    template<typename, Device, idx_t>
    friend class Container;

    using Self       = Container;
    using AllocatorT = Allocator<Target_>;

public:
    /** @brief Target memory device for this container. */
    static constexpr Device Target = Target_;
    /** @brief Indicate whether this device supports texture objects. */
    static constexpr bool HasTexture = hasTextureSupport<Target>;
    static constexpr bool IsNone     = isNone<Target>;
    using TexT = typename TextureTypeSwitcher<HasTexture>::T;
    /** @brief Compile-time vector dimension (1 for scalar containers). */
    static constexpr idx_t VecDims = VectorDim;

    /**
     * @brief Construct a non-owning container that borrows its pointer from
     * `other`.
     *
     * For `CUDA_HOST` containers borrowing from a `CPU` container, this also
     * manages pinned-memory registration so that async transfers are possible.
     *
     * @tparam OtherContainerT  Type of the source container; must satisfy
     * `areBorrowable`.
     * @param  other            Source container whose pointer will be borrowed.
     * @return Non-owning `Container` wrapping `other`'s data.
     */
    template<typename OtherContainerT>
    static Container borrow(const OtherContainerT& other)
    {
        /* Assert that the types are compatible for borrowing */
        static_assert(areBorrowable<Target, OtherContainerT::Target>,
            "Incompatible container types for borrowing");
        /* Create an empty container */
        Container out;

        if constexpr (VecDims == 0)
            return out;

        /* Set the internal members manually */
        out.ptr_  = const_cast<DataT*>(other.ptr_);
        out.size_ = other.size_;
        /* Override borrowing flags */
        out.isBorrowed_ = true;
        /* Now we manage the registration. We register ONLY if we are a
         * CUDA_HOST container. If other is already registered, then we do not
         * manage it. If other is not registered, but we are a CUDA_HOST
         * container, then we do manage it */
        if constexpr (Target == Device::CUDA_HOST
            && OtherContainerT::Target == Device::CPU) {
            /* The compile-time condition to register is true. Now we proceed to
             * check whether 1) other is already borrowed and 2) out is already
             * registered */
            if (!other.isBorrowed_ && !other.isRegistered_) {
                out.manageRegistration_ = true;
                /* Only override the const-ness to mark the registration */
                const_cast<OtherContainerT&>(other).isRegistered_ = true;
            }
        }
        /* Call Malloc (results in no-op or completes the registration, if
         * required) */
        out.malloc_();
        return out;
    }

    /** @brief Default constructor doesn't allocate anything */
    Container(const bool& flexible = false)
        : flexible_{ VecDims == 0 || flexible }
    {
    }

    /** @brief Construct with allocation and initial value setting */
    Container(const idx_t& sz, const DataT& initialValue = Default<DataT>(),
        const bool& flexible = false)
        : flexible_{ VecDims == 0 || flexible }
    {
        /* set the size */
        size_ = sz;
        malloc_();
        if (sz > 0 && initialValue != Default<DataT>())
            fill(initialValue);
    }

    /** @brief Copy constructor - allocate and copy only if other is allocated
     * This preserves the "unallocated but sized" state when cloning an
     * object whose workable/device memory hasn't been built yet.
     */
    Container(const Container& other)
        : Container{ other.size(), Default<DataT>(), other.flexible_ }
    {
        if (other.size() > 0)
            *this = other;
    }

    /** @brief Move constructor */
    Container(Container&& other) noexcept
        : allocated_{ std::exchange(other.allocated_, false) }
        , ptr_{ std::exchange(other.ptr_, nullptr) }
        , size_{ std::exchange(other.size_, 0) }
        , tex_{ std::move(other.tex_) }
        , flexible_{ std::exchange(other.flexible_, false) }
        , isBorrowed_{ std::exchange(other.isBorrowed_, false) }
        , manageRegistration_{ std::exchange(other.manageRegistration_, false) }
    {
    }

    /** @brief Converting Move Constructor (Cross-Device) */
    template<Device OtherTarget>
    Container(Container<DataT, OtherTarget, VectorDim>&& other)
        : flexible_{ VecDims == 0 || other.flexible_ }
    {
        if (VecDims != 0 && other.allocated_) {
            /* 1. Allocate our own memory */
            size_ = other.size_;
            malloc_();

            /* 2. Copy data from other to us (Blocking, as other dies) */
            using CopyT = Copy<OtherTarget, Target>;
            CopyT::blocking(other.ptr_, this->ptr_, this->totalSize());

            /* 3. Free 'other' memory explicitly */
            other.free_();
        }
    }

    /** @brief Copy assignment copies the data from other*/
    Container& operator=(const Container& other)
    {
        /* do nothing if none */
        if constexpr (!IsNone) {
            if (this == &other)
                return *this;
            /* ensure same size */
            FETA_ASSERT(
                this->size_ == other.size_, "Incompatible sizes for data copy");
            /* Do nothing if size is equal to 0 */
            if (this->size_ == 0 && other.size_ == 0)
                return *this;
            /* If we are not allocated and other is not allocated, prevent the
             * crash */
            if (this->isValid() || other.isValid()) {
                FETA_ASSERT(this->isValid(), "Invalid 'this' for data copy");
                FETA_ASSERT(other.isValid(), "Invalid 'other' for data copy");
                using CopyT = Copy<Target, Target>;
                CopyT::blocking(other.data(), this->data(), this->totalSize());
                if constexpr (HasTexture) {
                    if (other.tex_.isValid()) {
                        this->bindToTexture();
                    }
                }
            }
        }
        return *this;
    }

    /** @brief Move assignment */
    Container& operator=(Container&& other) noexcept
    {
        /* free ourselves before taking over */
        if (this == &other)
            return *this;

        free_();
        allocated_          = std::exchange(other.allocated_, false);
        ptr_                = std::exchange(other.ptr_, nullptr);
        size_               = std::exchange(other.size_, 0);
        tex_                = std::move(other.tex_);
        flexible_           = std::exchange(other.flexible_, false);
        isBorrowed_         = std::exchange(other.isBorrowed_, false);
        manageRegistration_ = std::exchange(other.manageRegistration_, false);

        return *this;
    }

    /**
     * @brief Fill the array with the given value.
     *
     * No-op when the target memory type is not fillable via `std::fill`
     * (e.g. `CUDA_DEVICE`).
     *
     * @param initialValue  Value written to every element.
     */
    void fill(const DataT& initialValue)
    {
        if constexpr (VecDims != 0 && isFillable<Target>) {
            if (size_ > 0) {
                std::fill(ptr_, ptr_ + size_, initialValue);
            }
        }
    }

    /** @brief Return `true` if the container has allocated (or borrowed)
     * memory. */
    inline bool isValid() const { return allocated_; }

    /** @brief Return `true` if the container is flexible (allows null data
     * pointer). */
    inline bool isFlexible() const { return flexible_; }

    /** @brief Return a pointer to the raw data at the given vector dimension
     * offset. */
    template<idx_t dim = 0>
    inline DataT* data() const
    {
        static_assert(dim < VecDims, "Dimension out of bounds");
        if (!flexible_)
            FETA_ASSERT(allocated_, "This container is not allocated");
        if constexpr (dim == 0)
            return ptr_;
        else
            return ptr_ ? ptr_ + dim * size_ : nullptr;
    }

    /** @brief Return the number of vectors (or scalars) in the container. */
    inline idx_t size() const { return size_; }

    /** @brief Return the total number of scalar elements (`size() * VecDims`).
     */
    inline idx_t totalSize() const { return size_ * VecDims; }

    /** @brief Bind device data to a CUDA texture object if texture support is
     * enabled. */
    void bindToTexture()
    {
        if constexpr (VecDims != 0 && HasTexture) {
            if (!tex_.isValid()) {
                /* Bind using the total number of scalar elements. For vector
                 * arrays the allocated memory is VecDims * size_, so pass the
                 * total size to the texture resource descriptor to avoid
                 * fetching out-of-range memory. */
                tex_ = std::move(
                    TexT(ptr_, static_cast<idx_t>(VecDims * size_)));
            }
        }
    }

    /** @brief Return a reference to the texture object handle. */
    TexT& tex() { return tex_; }
    const TexT& tex() const { return tex_; }

    /** @brief Destructor */
    ~Container() { free_(); }

private:
    /** @brief Allocate backing memory (no-op if already allocated or size is
     * zero). */
    void malloc_()
    {
        if constexpr (VecDims != 0) {
            if (!allocated_ && size_ != 0) {
                AllocatorT::template malloc<DataT>(
                    &ptr_, VecDims * size_, isBorrowed_, manageRegistration_);
                allocated_ = true;
            }
        }
    }

    /** @brief Free backing memory (no-op for borrowed or unallocated
     * containers). */
    void free_()
    {
        if constexpr (VecDims != 0) {
            if (allocated_) {
                AllocatorT::template free<DataT>(
                    ptr_, VecDims * size_, isBorrowed_, manageRegistration_);
                allocated_ = false;
            }
        }
    }

    /* Data members */
    bool allocated_ = false;
    DataT* ptr_     = nullptr;
    idx_t size_     = 0;
    TexT tex_       = TextureTypeSwitcher<HasTexture>::Default();

    /* Flexible containers can expose nullptr, deferring the validity assertion
     * later on in the code. This can be useful if some containers to be used
     * are 1) optional 2) well-guarded inside if conditions that check their
     * activation BEFORE using the data. USE WITH CAUTION! They can directly
     * expose nullptr and result in difficult error tracing and debugging.
     * Default is false (for safety) */
    bool flexible_ = VecDims == 0;

    /* Borrowed marker. This flags controls whether allocation and deallocations
     * follow a borrowing prinicple (i.e. resulting in either no-op or
     * decorative registrations e.g. CUDA pinned memory) */
    bool isBorrowed_ = false;

    /**
     * @brief Flag indicating this container manages pinned-memory registration.
     *
     * `true` only for `CUDA_HOST` containers that borrow from a plain `CPU`
     * container and where the source was not already registered.
     */
    bool manageRegistration_ = false;

    /** @brief Flag indicating this container's memory has already been
     * registered as pinned. */
    bool isRegistered_ = false;
};

} // namespace memory
} // namespace core
} // namespace feta