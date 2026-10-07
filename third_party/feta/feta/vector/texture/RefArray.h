#pragma once

#include "feta/scalar/texture/Array.h"
#include "feta/typedefs.h"
#include "feta/vector/Item.h"
#include "feta/vector/OnesNT.h"
#include "feta/vector/ZerosNT.h"
#include "feta/vector/expr/Expression.h"

namespace feta {
namespace vector {
namespace texture {
namespace detail {

/** @brief Minimal handle - also a vector expression */
template<typename DataT, dims_t VectorDim, bool work_, bool MaybeVolatile = false>
struct Handle
    : public expr::Expression<
          Handle<DataT, VectorDim, work_, MaybeVolatile>, DataT> {
    /** @brief Volatile is allowed only with work */
    static constexpr bool Volatile = MaybeVolatile && work_;
    using RDataT =
        typename helper::DataReturn<DataT, false, Volatile>::T;
    /** @brief The pointer type */
    using PointerT =
        typename scalar::texture::detail::Pointer<DataT, false, Volatile>::T;
    /** @brief Self */
    using Self = Handle<DataT, VectorDim, work_, MaybeVolatile>;
    /** @brief Core handle type (scalar) */
    using CoreT =
        scalar::texture::detail::Handle<DataT, work_, MaybeVolatile>;

    /** @brief Texture is only global reference */
    static constexpr bool work = work_;

    /** @brief Number of components in the vectors */
    static constexpr dims_t VecDims = VectorDim;

    /** @brief This type is a leaf in the expression template trees */
    static constexpr bool isLeaf    = true;
    static constexpr bool isWritable = true;
    static constexpr expr::ExprKind kind = expr::ExprKind::Leaf;
    static constexpr dims_t depth        = 0;
    static constexpr idx_t arraySize     = 0;

    /** @brief The inner pointer (or tex reference) */
    PointerT ptr =
        scalar::texture::detail::Pointer<DataT, false, Volatile>::Null;
    /** @brief The dimension offset */
    idx_t dOffset = 0;

    /** @brief Access the given data */
    template<idx_t dim>
    DEVICEHOST()
    inline decltype(auto) get(const SampleIndex& i) const
    {
        if constexpr (work) {
            return get<dim>(i.work());
        } else {
            return get<dim>(i.global());
        }
    }
    template<idx_t dim>
    DEVICEHOST()
    inline decltype(auto) get(const idx_t& i) const
    {
/* Return by reference for shared memory, by value for global __ldg*/
#ifdef __CUDA_ARCH__
        /* Shared memory - return by reference to avoid register pressure */
        if constexpr (helper::HasLdg<DataT>::value && !work) {
            return __ldg(ptr + i + dOffset * dim);
        } else {
            return ptr[i + dOffset * dim];
        }
#else
        return ptr[i + dOffset * dim];
#endif
    }
    template<idx_t dim>
    DEVICEHOST()
    inline RDataT get(const SampleIndex& i)
    {
        if constexpr (work) {
            return get<dim>(i.work());
        } else {
            return get<dim>(i.global());
        }
    }
    template<idx_t dim>
    DEVICEHOST()
    inline RDataT get(const idx_t& i)
    {
        return ptr[i + dOffset * dim];
    }

    /** @brief Return the data pointed to */
    DEVICEHOST() inline PointerT data() const { return ptr; }

    /** @brief Return the dimension offset */
    DEVICEHOST() inline idx_t dimOffset() const { return dOffset; }

    /** @brief Return a scalar handle of the given component */
    template<idx_t dim>
    DEVICEHOST()
    inline CoreT component() const
    {
        return { ptr + dOffset * dim };
    }

    /** @brief Extract a subset vector array.  Preserves volatility. */
    template<idx_t start, idx_t HowMany>
    DEVICEHOST()
    inline Handle<DataT, HowMany, work, MaybeVolatile> subset() const
    {
        static_assert(
            start + HowMany <= VectorDim, "Invalid subset parameters");
        Handle<DataT, HowMany, work, MaybeVolatile> h;
        h.ptr     = ptr + dOffset * start;
        h.dOffset = dOffset;
        return h;
    }
};

/** @brief Handle Switcher */
template<typename DataT, dims_t VectorDim, bool work, bool UseTexture,
    bool MaybeVolatile = false>
struct HandleSwitcher {
    using T = Handle<DataT, VectorDim, work, MaybeVolatile>;
};

#ifndef FETA_CPU_ONLY

/** @brief Texture handle - also a vector expression */
template<typename DataT, dims_t VectorDim>
struct TextureHandle
    : public expr::Expression<TextureHandle<DataT, VectorDim>, DataT> {
    /** @brief The pointer type */
    using PointerT = typename scalar::texture::detail::Pointer<DataT, true>::T;
    /** @brief Self */
    using Self = TextureHandle<DataT, VectorDim>;
    /** @brief Core handle type (scalar) */
    using CoreT = scalar::texture::detail::TextureHandle<DataT>;

    /** @brief Texture is only global reference */
    static constexpr bool work = false;

    /** @brief Number of components in the vectors */
    static constexpr dims_t VecDims = VectorDim;

    /** @brief This type is a leaf in the expression template trees */
    static constexpr bool isLeaf    = true;
    static constexpr bool isWritable = true;
    static constexpr expr::ExprKind kind = expr::ExprKind::Leaf;
    static constexpr dims_t depth        = 0;
    static constexpr idx_t arraySize     = 0;

    /** @brief The inner pointer (or tex reference) */
    PointerT ptr = scalar::texture::detail::Pointer<DataT, true>::Null;
    /** @brief The offset within the data */
    idx_t texOffset = 0;
    /** @brief The dimension offset */
    idx_t dimOffset = 0;

    /** @brief Access the given data */
    template<idx_t dim>
    DEVICEHOST()
    inline DataT get(const SampleIndex& i) const
    {
        return get<dim>(i.global());
    }
    template<idx_t dim>
    DEVICEHOST()
    inline DataT get(const idx_t& i) const
    {
#ifdef __CUDA_ARCH__
        return scalar::texture::detail::texFetch<DataT>(
            i + texOffset + dim * dimOffset, ptr);
#else
        return ptr[i + texOffset + dim * dimOffset];
#endif
    }

    /** @brief Return a scalar handle of the given component */
    template<idx_t dim>
    DEVICEHOST()
    inline CoreT component() const
    {
        return { ptr, texOffset + dim * dimOffset };
    }

    /** @brief Extract a subset vector array */
    template<idx_t start, idx_t HowMany>
    DEVICEHOST()
    inline TextureHandle<DataT, HowMany> subset() const
    {
        static_assert(
            start + HowMany <= VectorDim, "Invalid subset parameters");
        TextureHandle<DataT, HowMany> h;
        h.ptr       = ptr;
        h.texOffset = texOffset + dimOffset * start;
        h.dimOffset = dimOffset;
        return h;
    }
};

/** @brief Texture specialisation ignores ``MaybeVolatile`` — textures are
 * read-only by construction and cannot be work-references, so volatile
 * never takes effect on them. */
template<typename DataT, dims_t VectorDim, bool work, bool MaybeVolatile>
struct HandleSwitcher<DataT, VectorDim, work, true, MaybeVolatile> {
    using T = TextureHandle<DataT, VectorDim>;
};
#endif
/**
 * @brief Non-owning refrence to vector::texture::Array. See feta.cuh for
 * the reference implementation
 *
 */
template<typename DataT_, dims_t VectorDim, bool work_, bool UseTexture,
    bool MaybeVolatile = false>
class RefArray
    : public expr::Expression<
          RefArray<DataT_, VectorDim, work_, UseTexture, MaybeVolatile>,
          DataT_> {
public:
    /** @brief Volatile is allowed only with work */
    static constexpr bool Volatile = MaybeVolatile && work_;

private:
    using RefScalarArray_ = scalar::texture::detail::RefArray<DataT_, work_,
        UseTexture, MaybeVolatile>;
    /* Data Return type */
    using Self = RefArray;

public:
    /** @brief The data return type */
    using RDataT =
        typename helper::DataReturn<DataT_, UseTexture, Volatile>::T;
/** @brief Expose the texture and stream types */
#ifndef FETA_CPU_ONLY
    using TexT    = cudaTextureObject_t;
    using StreamT = cudaStream_t;
#else
    using TexT    = int;
    using StreamT = int;
#endif

    /** @brief Expose the data type */
    using DataT = DataT_;

    /** @brief Expose work vs global reference */
    static constexpr bool work = work_;

    /** @brief The buffer type */
    using BufT = buffer::vector::Array<DataT, VectorDim>;

    /** @brief Datatype of the vector elements */
    using ComponentT = DataT;

    /** @brief Expose the array item type */
    using ItemT = Item<DataT, VectorDim>;

    /** @brief Number of components in the vectors */
    static constexpr dims_t VecDims = VectorDim;

    /** @brief This type is a leaf in the expression template trees */
    static constexpr bool isLeaf     = true;
    static constexpr bool isWritable = true;

    /** @brief Expression tree annotations. */
    static constexpr expr::ExprKind kind = expr::ExprKind::Leaf;
    static constexpr dims_t depth        = 0;
    static constexpr idx_t arraySize     = 0; ///< Runtime-sized

    /** @brief A vector expression where each component is 1. */
    using Ones = OnesNT<DataT, VectorDim>;

    /** @brief A vector expression where each component is 0. */
    using Zeros = ZerosNT<DataT, VectorDim>;

    /** @brief Expose the core data type */
    using CoreT = RefScalarArray_;

    /** @brief Expose the view types */
    using ViewT      = expr::View<Self>;
    using ConstViewT = expr::ConstView<Self>;

    /** @brief Expose the buffer info type */
    using BufInfoT = typename BufT::InfoT;

    /** @brief Expose the handle type */
    using HandleT =
        typename HandleSwitcher<DataT, VectorDim, work, UseTexture,
            MaybeVolatile>::T;

    /** @brief Returns the number of vectors in this array. */
    DEVICEHOST() idx_t size() const { return nVecs_; }

    /**
     * @brief Get given component of the i-th vector
     *
     * @tparam dim Dimension (component) of the vector to return.
     * @param index Index of the vector to return from.
     */
    template<idx_t dim>
    DEVICEHOST()
    inline decltype(auto) get(const SampleIndex& index) const
    {
        if constexpr (work) {
            return get<dim>(index.work());
        } else {
            return get<dim>(index.global());
        }
    }
    template<idx_t dim>
    DEVICEHOST()
    inline decltype(auto) get(const idx_t& idx) const
    {
        static_assert(dim < VectorDim, "Invalid vector dimension access");
#ifdef __CUDA_ARCH__
        FETA_GPU_ASSERT(idx < nVecs_, ::feta::err::OUT_OF_RANGE_VECTOR);
        if constexpr (UseTexture && !work) {
            return scalar::texture::detail::texFetch<DataT>(
                idx + texOffset_ + dim * dimOffset_, tex_);
        }
        /* Shared memory - return by reference to avoid register pressure */
        /* Global memory - use __ldg which returns by value if available */
        else if constexpr (helper::HasLdg<DataT>::value && !work) {
            return __ldg(data_ + dimOffset_ * dim + idx);
        } else {
            return data_[dimOffset_ * dim + idx];
        }
#else
        FETA_ASSERT(idx < nVecs_, "RefVectorArray: out of bounds access");
        return data_[dimOffset_ * dim + idx];
#endif
    }
    template<idx_t dim>
    DEVICEHOST()
    inline RDataT get(const SampleIndex& index)
    {
        static_assert(dim < VectorDim, "Invalid vector dimension access");
        const idx_t idx = work ? index.work() : index.global();
#ifdef __CUDA_ARCH__
        FETA_GPU_ASSERT(idx < nVecs_, ::feta::err::OUT_OF_RANGE_VECTOR);
        static_assert(!UseTexture,
            "Non-const access not supported for texture references");
#else
        FETA_ASSERT(idx < nVecs_, "RefVectorArray: out of bounds access");
#endif
        return data_[dimOffset_ * dim + idx];
    }
    template<idx_t dim>
    DEVICEHOST()
    inline RDataT get(const idx_t& idx)
    {
        static_assert(dim < VectorDim, "Invalid vector dimension access");
#ifdef __CUDA_ARCH__
        FETA_GPU_ASSERT(idx < nVecs_, ::feta::err::OUT_OF_RANGE_VECTOR);
        static_assert(!UseTexture,
            "Non-const access not supported for texture references");
#else
        FETA_ASSERT(idx < nVecs_, "RefVectorArray: out of bounds access");
#endif
        return data_[dimOffset_ * dim + idx];
    }

    /** @brief Return a normalized version of this vector */
    DEVICEHOST() inline ItemT normalize(const SampleIndex& index) const
    {
        return this->rNorm(index) * (*this)[index];
    }
    /** @brief Return the quaternion reciprocal of this vector */
    DEVICEHOST()
    inline ItemT quatReciprocal(const SampleIndex& index) const
    {
        return this->rSquaredNorm(index) * this->quatConj()[index];
    }

    /** @brief Check whether a given vector holds finite values */
    DEVICEHOST() bool isValid(const SampleIndex& i)
    {
        return expr::reduce::isFinite<VectorDim,
            RefArray<DataT, VectorDim, work, UseTexture>>::eval(*this, i);
    }

    /** @brief Return a data handler */
    DEVICEHOST() inline HandleT handle() const
    {
        HandleT h;
#ifdef __CUDA_ARCH__
        if constexpr (UseTexture && !work) {
            h.ptr       = tex_;
            h.texOffset = texOffset_;
        } else {
            h.ptr = data_;
        }
#else
        h.ptr = data_;
#endif
        h.dOffset = dimOffset_;
        return h;
    }

    /** @brief The (volatility-aware) pointer type for ``data_``. */
    using PointerT =
        typename scalar::texture::detail::Pointer<DataT, false, Volatile>::T;

    /** @brief Return a pointer to the underlying data. */
    DEVICEHOST() PointerT data() const { return data_; }

    /**
     * @brief Returns a pointer to the element following the last element in
     * the vector array.
     */
    DEVICEHOST() PointerT end() const
    {
        return data_ + (VecDims - 1) * dimOffset_ + nVecs_;
    }

    /**
     * @brief Minimum size (in elements of type `DataT`) required for
     * allocation on an arbitrary buffer, if the object were to be
     * constructed for `arrayLen` elements.
     */
    DEVICEHOST() static size_t bufSize(const size_t& arrayLen)
    {
        return VecDims * arrayLen;
    }

    /**
     * @brief Minimum size (in bytes) required for allocation on an
     * arbitrary buffer, if the object were to be constructed for `arrayLen`
     * elements.
     */
    DEVICEHOST() static size_t bufBytes(const size_t& arrayLen)
    {
        return bufSize(arrayLen) * sizeof(DataT);
    }

    /** @brief Return a core scalar array with all the components */
    DEVICEHOST() CoreT core() const
    {
        CoreT c;
        c.data_      = data_;
        c.size_      = VectorDim * dimOffset_;
        c.tex_       = tex_;
        c.texOffset_ = texOffset_;
        return c;
    }

    /**
     * @brief Returns a work array reference of the given size which uses
     * the given buffer.
     *
     * @param buf Workspace for the work reference. Must be at least
     * `bufSize(size)` elements long.
     * @param size Size of the work array to initialize.
     */
    DEVICEHOST()
    static RefArray<DataT, VectorDim, true, false> workRef(
        DataT* const buf, const idx_t& size)
    {
        RefArray<DataT, VectorDim, true, false> r;
        r.data_      = buf;
        r.nVecs_     = size;
        r.dimOffset_ = size;
        r.tex_       = 0;
        r.texOffset_ = 0;
        return r;
    }

    /**
     * @brief Returns a volatile work-reference of the given size.
     *
     * Same as ``workRef`` except the resulting ref's inner pointer is
     * ``volatile DataT*``.  Intended for shared-memory shmem-scratch
     * patterns where the consumer needs to defeat compiler caching /
     * stack-backed mirrors (see ``[[sh-phasor-shmem-failed]]``).
     */
    DEVICEHOST()
    static RefArray<DataT, VectorDim, true, false, true> volatileWorkRef(
        DataT* const buf, const idx_t& size)
    {
        RefArray<DataT, VectorDim, true, false, true> r;
        r.data_      = buf;
        r.nVecs_     = size;
        r.dimOffset_ = size;
        r.tex_       = 0;
        r.texOffset_ = 0;
        return r;
    }

    /** @brief Return a RefTextureScalarArray for the given component */
    template<dims_t dim>
    DEVICEHOST()
    inline CoreT component() const
    {
        const idx_t offset = dim * dimOffset_;
        return { data_ + offset, nVecs_, tex_, texOffset_ + offset };
    }

    /** @brief Extract a subset vector array.  Volatility is preserved
     * across the sub-view so a ``VolatileRef::subset<...>()`` stays
     * volatile. */
    template<idx_t start, idx_t HowMany>
    DEVICEHOST()
    inline RefArray<DataT, HowMany, work, UseTexture, MaybeVolatile> subset()
        const
    {
        static_assert(
            start + HowMany <= VectorDim, "Invalid subset parameters");
        RefArray<DataT, HowMany, work, UseTexture, MaybeVolatile> r;
        r.data_      = data_ + start * dimOffset_;
        r.nVecs_     = nVecs_;
        r.dimOffset_ = dimOffset_;
        r.tex_       = tex_;
        r.texOffset_ = texOffset_ + start * dimOffset_;
        return r;
    }

    /** @brief Create a new global reference to this vector array */
    DEVICEHOST() RefArray clone() const { return *this; }

    /** @brief Generic fetch from other array */
    void fetch(const RefArray& other, const StreamT& stream = 0)
    {
        core().fetch(other.core(), stream);
    }

#ifndef FETA_CPU_ONLY

    /** @brief Async memcpy from host to the given device array */
    void upload(RefArray& other, const StreamT& stream = 0) const
    {
        CoreT otherCore = other.core();
        core().upload(otherCore, stream);
    }

    /** @brief Async memcpy from device to this data */
    void download(const RefArray& other, const StreamT& stream = 0)
    {
        core().download(other.core(), stream);
    }

#endif

    /** @brief Read the data from buffer */
    void fromBuffer(const BufT& buf)
    {
        FETA_ASSERT(
            buf.size() == nVecs_, "Incompatible buffer and this array sizes");
        std::copy(buf.data(), buf.data() + VectorDim * nVecs_, data_);
    }

    /** @brief Read the data from the buffer info */
    void fromBuffer(const BufInfoT& i)
    {
        FETA_ASSERT(i.size() == nVecs_,
            "Incompatible feta::buffer::Info and this array sizes");
        std::copy(i.ptr, i.ptr + VectorDim * nVecs_, data_);
    }

    /** @brief Read the data from buffer slice */
    void fromBuffer(const typename BufT::SliceT& buf)
    {
        FETA_ASSERT(
            buf.size() == nVecs_, "Incompatible buffer and this array sizes");
        for (idx_t i = 0; i < nVecs_; i++)
            (*this)[i] = buf[i];
    }

    /** @brief Write the data to buffer (return) */
    BufT toBuffer() const
    {
        BufT buf(this->size());
        toBuffer(buf);
        return buf;
    }

    /** @brief Write the data to buffer (void) */
    void toBuffer(BufT& buf) const
    {
        FETA_ASSERT(
            buf.size() == nVecs_, "Incompatible buffer and this array sizes");
        std::copy(data(), data() + VectorDim * nVecs_, buf.data());
    }

    /** @brief Write the data to buffer (slice) */
    void toBuffer(typename BufT::SliceT& buf) const
    {
        FETA_ASSERT(
            buf.size() == nVecs_, "Incompatible buffer and this array sizes");
        for (idx_t i = 0; i < nVecs_; i++)
            buf[i] = (*this)[i];
    }

    /** @brief Return whether this is a host reference */
    bool isHostRef() const { return core().isHostRef(); }

    /** @brief Return the first element of this array */
    DEVICEHOST() decltype(auto) front() { return (*this)[0]; }
    DEVICEHOST() decltype(auto) front() const { return (*this)[0]; }

    /** @brief Return the last element of this array */
    DEVICEHOST() decltype(auto) back() { return (*this)[nVecs_ - 1]; }
    DEVICEHOST() decltype(auto) back() const { return (*this)[nVecs_ - 1]; }

    /** @brief Expose the dimension offset */
    DEVICEHOST() idx_t dimOffset() const { return dimOffset_; }

    /** @brief Public data for PODification.
     *
     * ``data_`` is volatility-aware via ``PointerT`` — when MaybeVolatile
     * is on and ``work_=true``, it is ``volatile DataT*`` so reads/writes
     * through it are not CSE'd or stack-mirrored. */
    PointerT data_   = nullptr;
    idx_t nVecs_     = 0;
    idx_t dimOffset_ = 0;
    TexT tex_        = 0;
    idx_t texOffset_ = 0;

    /** @brief Return the appropriate index (global or work) from `i` */
    DEVICEHOST() inline idx_t index_(const SampleIndex& i) const
    {
        if constexpr (work)
            return i.work();
        else
            return i.global();
    }
};

} // namespace detail
} // namespace texture
} // namespace vector
} // namespace feta