#pragma once

#include "parm/typedefs.h"

namespace parm {
namespace util {

/** @brief Forward declare reference abstract */
template<bool work, idx_t IntSize, idx_t BoolSize, idx_t RealSize,
    bool MaybeVolatile = false>
class RefMultiContainer;

/** @brief Forward declarations */
template<bool work, idx_t IntSize, idx_t BoolSize, idx_t RealSize,
    bool MaybeVolatile = false>
struct View;
template<bool work, idx_t IntSize, idx_t BoolSize, idx_t RealSize,
    bool MaybeVolatile = false>
struct ConstView;

/** @brief Item view */
template<bool work, idx_t IntSize, idx_t BoolSize, idx_t RealSize,
    bool MaybeVolatile>
struct ConstView {
    /* the reference data type */
    using RefT = RefMultiContainer<work, IntSize, BoolSize, RealSize,
        MaybeVolatile>;

    /** @brief Data members made public for PODification */
    const SampleIndex idx_;
    const RefT& ref_;
};

/** @brief Item view */
template<bool work, idx_t IntSize, idx_t BoolSize, idx_t RealSize,
    bool MaybeVolatile>
struct View {
    /* the reference data type */
    using RefT = RefMultiContainer<work, IntSize, BoolSize, RealSize,
        MaybeVolatile>;

    /** @brief Data members made public for PODification */
    const SampleIndex idx_;
    RefT& ref_;
};

/**
 * @brief Non-owning reference to `cudaj::MetaData`
 *
 */
template<bool work, idx_t IntSize, idx_t BoolSize, idx_t RealSize,
    bool MaybeVolatile>
class RefMultiContainer {
public:
    /** @brief Volatile is allowed only with work */
    static constexpr bool Volatile = MaybeVolatile && work;

protected:
    using Self = RefMultiContainer<work, IntSize, BoolSize, RealSize,
        MaybeVolatile>;
    /* Single vector types */
    using VecInt  = feta::vector::Item<mInt_t, IntSize>;
    using VecBool = feta::vector::Item<bool, BoolSize>;
    using VecReal = feta::vector::Item<Real, RealSize>;

    /* Friend other reference */
    friend class RefMultiContainer<!work, IntSize, BoolSize, RealSize,
        MaybeVolatile>;

public:
    /** @brief Item view type */
    using ViewT      = View<work, IntSize, BoolSize, RealSize, MaybeVolatile>;
    using ConstViewT
        = ConstView<work, IntSize, BoolSize, RealSize, MaybeVolatile>;
    /** @brief Vector array types */
    using RefVecIntArrayT = typename feta::vector::Array<mInt_t,
        IntSize>::template Ref<work, MaybeVolatile>;
    using RefVecBoolArrayT = typename feta::vector::Array<bool,
        BoolSize>::template Ref<work, MaybeVolatile>;
    using RefVecRealArrayT = typename feta::vector::Array<Real,
        RealSize>::template Ref<work, MaybeVolatile>;
    /** @brief Handle types */
    using VecIntHandleT  = typename RefVecIntArrayT::HandleT;
    using VecBoolHandleT = typename RefVecBoolArrayT::HandleT;
    using VecRealHandleT = typename RefVecRealArrayT::HandleT;
    /** @brief Stream type, mirroring FETA's mode-aware alias.
     *  ``cudaStream_t`` in CUDA builds, ``int`` in CPU-only builds. */
    using StreamT = typename RefVecIntArrayT::StreamT;

    /** @brief Component types */
    using IntT  = typename RefVecIntArrayT::ComponentT;
    using BoolT = typename RefVecBoolArrayT::ComponentT;
    using RealT = typename RefVecRealArrayT::ComponentT;
    /** @brief Shared Memory data type (for heterogeneous initialisation and
     * subsequent type casting )*/
    using SharedMemDataT    = IntT;
    using EndSharedMemDataT = RealT;

    /** @brief Factory method to construct from Reference arrays */
    DEVICEHOST()
    static RefMultiContainer make(const RefVecIntArrayT& intMembers,
        const RefVecBoolArrayT& boolMembers,
        const RefVecRealArrayT& realMembers)
    {
        return RefMultiContainer{ intMembers.handle(), boolMembers.handle(),
            realMembers.handle(), intMembers.size() };
    }


    /**
     * @brief Return the number of samples in the corresponding
     * `cudaj::model::samples::MetaData`
     *
     * @return Nbodies
     *
     */
    DEVICEHOST() idx_t size() const { return size_; }

    /**
     * @brief Get the int member Handle (PODified for kernel args).
     *
     * Use ``intMembersRef`` instead when you need the full GRef (e.g.
     * for host-side composite copies via ``RefArray::fetch``).
     */
    DEVICEHOST()
    VecIntHandleT intMembers() const { return intMembers_; }

    /** @brief Get the bool member Handle (PODified for kernel args). */
    DEVICEHOST()
    VecBoolHandleT boolMembers() const { return boolMembers_; }

    /**
     * @brief Get the real member Handle (PODified for kernel args).
     */
    DEVICEHOST()
    VecRealHandleT realMembers() const { return realMembers_; }

    /** @brief Reconstruct the int member GRef from the stored Handle and
     *  ``size_``.
     *
     *  ``RefMultiContainer`` is built by ``MultiContainer::deviceRef``
     *  via ``GRef::make(intMembers.deviceRef(), …)`` which discards the
     *  full GRef and keeps only the Handle (POD for kernel args). This
     *  accessor pieces a GRef back together so host-side composite
     *  operations (``fetch``, ``copyFrom``) can drive the auto-
     *  dispatching ``feta::vector::RefArray`` API directly. Vector
     *  member arrays here are non-texture (``UseTexture=false``), so
     *  ``Handle::ptr`` is a plain ``DataT*`` and ``Handle::dOffset`` is
     *  the per-component stride — exactly the data the GRef needs.
     */
    DEVICEHOST() RefVecIntArrayT intMembersRef() const
    {
        RefVecIntArrayT r;
        r.data_      = intMembers_.ptr;
        r.nVecs_     = size_;
        r.dimOffset_ = intMembers_.dOffset;
        r.tex_       = 0;
        r.texOffset_ = 0;
        return r;
    }
    /** @brief Reconstruct the bool member GRef. See ``intMembersRef``. */
    DEVICEHOST() RefVecBoolArrayT boolMembersRef() const
    {
        RefVecBoolArrayT r;
        r.data_      = boolMembers_.ptr;
        r.nVecs_     = size_;
        r.dimOffset_ = boolMembers_.dOffset;
        r.tex_       = 0;
        r.texOffset_ = 0;
        return r;
    }
    /** @brief Reconstruct the real member GRef. See ``intMembersRef``. */
    DEVICEHOST() RefVecRealArrayT realMembersRef() const
    {
        RefVecRealArrayT r;
        r.data_      = realMembers_.ptr;
        r.nVecs_     = size_;
        r.dimOffset_ = realMembers_.dOffset;
        r.tex_       = 0;
        r.texOffset_ = 0;
        return r;
    }

    /** @brief Async copy from another ``RefMultiContainer`` (composite
     *  ref-to-ref fetch).
     *
     *  Drives FETA's auto-dispatching ``RefArray::fetch`` via the
     *  reconstructed GRefs (``*MembersRef``). Each member array's
     *  fetch chooses H2H/H2D/D2H/D2D at runtime from the pointer
     *  locations. Host-only — composing fetches at the ref level on
     *  the device side is not a meaningful operation.
     */
    void fetch(const RefMultiContainer& other, const StreamT& stream = 0)
    {
        intMembersRef().fetch(other.intMembersRef(), stream);
        boolMembersRef().fetch(other.boolMembersRef(), stream);
        realMembersRef().fetch(other.realMembersRef(), stream);
    }

    /** @brief Per-slot composite copy.
     *
     *  Copies sample slot ``src_i`` of ``src`` into slot ``dst_i`` of
     *  ``*this`` for every member array (int, bool, real). Drives
     *  ``feta::vector::expr::View::copyFrom`` per-array, which
     *  internally materializes through ``Item`` to dodge the
     *  implicitly-deleted ``View<E>::operator=(const View<E>&)``.
     *  Generated SASS matches what the per-slot ``Item`` pattern in
     *  ``compaction.h::permute`` produced — register-resident slot
     *  data, single SoA write per component.
     *
     *  Marked DEVICEHOST so kernels can call it; on host this just
     *  reads/writes regular memory.
     */
    DEVICEHOST() void copyFrom(const SampleIndex& dst_i,
        const RefMultiContainer& src, const SampleIndex& src_i)
    {
        auto dstInt = intMembersRef();
        auto srcInt = src.intMembersRef();
        dstInt[dst_i].copyFrom(srcInt[src_i]);

        auto dstBool = boolMembersRef();
        auto srcBool = src.boolMembersRef();
        dstBool[dst_i].copyFrom(srcBool[src_i]);

        auto dstReal = realMembersRef();
        auto srcReal = src.realMembersRef();
        dstReal[dst_i].copyFrom(srcReal[src_i]);
    }

    /**
     * @brief Retrieve an integer metadata item
     *
     * @tparam intItemCount
     * @param SampleIdx
     * @return intMember
     *
     */
    /**
     * @brief Retrieve a metadata item from one of the typed member buckets.
     *
     * Each bucket (int / bool / real) exposes the same four accessors — a
     * mutable and a const reference for the work view, and a by-value getter
     * for the const view — over both a ``SampleIndex`` and a raw ``idx_t``.
     * The twelve one-line forwarders are stamped from a single macro so they
     * cannot drift.
     */
#define PARM_REFMC_GET(NAME, TYPE, MEMBER)                                     \
    template<idx_t item>                                                       \
    DEVICEHOST() TYPE& NAME(const SampleIndex& idx)                            \
    {                                                                          \
        return MEMBER.template get<item>(idx);                                 \
    }                                                                          \
    template<idx_t item>                                                       \
    DEVICEHOST() TYPE& NAME(const idx_t idx)                                   \
    {                                                                          \
        return MEMBER.template get<item>(idx);                                 \
    }                                                                          \
    template<idx_t item>                                                       \
    DEVICEHOST() TYPE NAME(const SampleIndex& idx) const                       \
    {                                                                          \
        return MEMBER.template get<item>(idx);                                 \
    }                                                                          \
    template<idx_t item>                                                       \
    DEVICEHOST() TYPE NAME(const idx_t idx) const                              \
    {                                                                          \
        return MEMBER.template get<item>(idx);                                 \
    }

    PARM_REFMC_GET(getInt, mInt_t, intMembers_)
    PARM_REFMC_GET(getBool, bool, boolMembers_)
    PARM_REFMC_GET(getReal, Real, realMembers_)

#undef PARM_REFMC_GET

    /**
     * @brief Return a `feta::vector::Item` vector containing the integer
     * metadata of a given sample
     *
     * @param SampleIndex
     * @return IntMetadataVector
     *
     */
    DEVICEHOST()
    VecInt getIntVec(const SampleIndex& idx)
    {
        VecInt out = intMembers_[idx];
        return out;
    }
    DEVICEHOST()
    VecInt getIntVec(const idx_t idx)
    {
        VecInt out = intMembers_[idx];
        return out;
    }

    /**
     * @brief Assign a `feta::vector::Item` vector containing the integer
     * metadata to a given sample
     *
     * @param SampleIndex
     * @param IntVec
     *
     */
    DEVICEHOST() void assignIntVec(const SampleIndex& idx, const VecInt& vec)
    {
        intMembers_[idx] = vec;
    }
    DEVICEHOST() void assignIntVec(const idx_t idx, const VecInt& vec)
    {
        intMembers_[idx] = vec;
    }

    /** @brief Return a `feta::vector::Item` vector containing the boolean
     * metadata of a given sample
     *
     * @param SampleIndex
     * @return BoolMetadataVector
     *
     */
    DEVICEHOST()
    VecBool getBoolVec(const SampleIndex& idx)
    {
        VecBool out = boolMembers_[idx];
        return out;
    }
    DEVICEHOST()
    VecBool getBoolVec(const idx_t idx)
    {
        VecBool out = boolMembers_[idx];
        return out;
    }

    /**
     * @brief Assign a `feta::vector::Item` vector containing the boolean
     * metadata to a given sample
     *
     * @param SampleIndex
     * @param BoolVec
     *
     */
    DEVICEHOST() void assignBoolVec(const SampleIndex& idx, const VecBool& vec)
    {
        boolMembers_[idx] = vec;
    }
    DEVICEHOST() void assignBoolVec(const idx_t idx, const VecBool& vec)
    {
        boolMembers_[idx] = vec;
    }

    /**
     * @brief Return a `feta::vector::Item` vector containing the real metadata
     * of a given sample
     *
     * @param SampleIndex
     * @return RealMetadataVector
     *
     */
    DEVICEHOST()
    VecReal getRealVec(const SampleIndex& idx)
    {
        VecReal out = realMembers_[idx];
        return out;
    }
    DEVICEHOST()
    VecReal getRealVec(const idx_t idx)
    {
        VecReal out = realMembers_[idx];
        return out;
    }

    /**
     * @brief Assign a `feta::vector::Item` vector containing the real metadata
     * to a given sample
     *
     * @param SampleIndex
     * @param RealVec
     *
     */
    DEVICEHOST() void assignRealVec(const SampleIndex& idx, const VecReal& vec)
    {
        realMembers_[idx] = vec;
    }
    DEVICEHOST() void assignRealVec(const idx_t idx, const VecReal& vec)
    {
        realMembers_[idx] = vec;
    }

    /** @brief Get the item view */
    DEVICEHOST()
    inline ViewT operator[](const SampleIndex& idx)
    {
        assertInBounds(idx);
        return ViewT(idx, *this);
    }
    DEVICEHOST()
    inline ConstViewT operator[](const SampleIndex& idx) const
    {
        assertInBounds(idx);
        return ConstViewT(idx, *this);
    }

    /**
     * @brief Create new global reference for this object
     *
     */
    DEVICEHOST() RefMultiContainer clone() const { return *this; }

    /** @brief Assert that the given sample index is inbounds */
    DEVICEHOST() void assertInBounds(const SampleIndex& idx) const
    {
        assertInBounds(work ? idx.work() : idx.global());
    }
    DEVICEHOST() void assertInBounds(const idx_t idx) const
    {
#ifdef __CUDA_ARCH__
        FETA_GPU_ASSERT(idx < size_, feta::err::OUT_OF_RANGE_VECTOR);
#else
        FETA_ASSERT(idx < size_, "RefMultiContainer: out of bounds access");
#endif
    }

    /** @brief Data members made public for PODification */
    VecIntHandleT intMembers_;
    VecBoolHandleT boolMembers_;
    VecRealHandleT realMembers_;
    /* the size */
    idx_t size_;
};

} // namespace util
} // namespace parm