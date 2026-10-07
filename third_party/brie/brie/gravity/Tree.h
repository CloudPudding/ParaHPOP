#pragma once

#include "brie/gravity/RefTree.h"

namespace brie {
namespace gravity {

using namespace schema;

/** @brief Owning gravity tree */
class Tree {
    using Self = Tree;

public:
#ifndef BRIE_CPU_ONLY
    using StreamT = cudaStream_t;
#else
    using StreamT = int;
#endif

    /** @brief Reference type.  ``MaybeVolatile`` is accepted (ignored)
     * for uniform 2-arg shape across containers — gravity::Tree is a
     * read-only metadata holder and not yet volatile-capable. */
    template<bool work, bool MaybeVolatile = false>
    using Ref = RefTree<work>;
    /** @brief Global reference type */
    using GRef = Ref<false>;

    /** @brief Factory method to construct from a given set of active bodies
     * TODO: Address whether to include a user-tunable barycentre vs actual body
     * selection policy. At the moment, SSB is enforced for Sun as COI only.
     */
    static Tree fromActiveBodies(
        const NaifIdArray& activeBodies, const bool& SunToSSB = true);

    /** @brief Default constructor is forbidden */
    Tree() = delete;

    /** @brief Copy constructor is forbidden */
    Tree(Tree& other) = delete;

    /** @brief Move constructor from data members */
    Tree(NaifIdArray&& bodies, BoolVecArray&& bools, IntVecArray&& ints)
        : bodies_{ std::move(bodies) }
        , bools_{ std::move(bools) }
        , ints_{ std::move(ints) }
    {
    }

    /** @brief Move constructor */
    Tree(Tree&& other)
        : bodies_{ std::move(other.bodies_) }
        , bools_{ std::move(other.bools_) }
        , ints_{ std::move(other.ints_) }
    {
    }

    /** @brief Move assignment operator */
    Tree& operator=(Tree&& other)
    {
        this->bodies_ = std::move(other.bodies_);
        this->bools_  = std::move(other.bools_);
        this->ints_   = std::move(other.ints_);
        return *this;
    }

#ifndef BRIE_CPU_ONLY
    /** @brief Copy the gravity tree data from host to device */
    void upload(const StreamT& stream = 0)
    {
        bodies_.upload(stream);
        bools_.upload(stream);
        ints_.upload(stream);
    }

    /** @brief Copy the gravity tree data from device to host */
    void download(const StreamT& stream = 0)
    {
        bodies_.download(stream);
        bools_.download(stream);
        ints_.download(stream);
    }

    /** @brief Clear data from the GPU */
    void clearDevice()
    {
        bodies_.clearDevice();
        bools_.clearDevice();
        ints_.clearDevice();
    }
#endif

    /** @brief Expose bodies */
    NaifIdArray& bodies() { return bodies_; }
    const NaifIdArray& bodies() const { return bodies_; }

    /** @brief Expose boolean metadata */
    BoolVecArray& boolmeta() { return bools_; }
    const BoolVecArray& boolmeta() const { return bools_; }

    /** @brief Expose integer metadata */
    IntVecArray& intmeta() { return ints_; }
    const IntVecArray& intmeta() const { return ints_; }

    /** @brief Return host reference */
    GRef hostRef() const
    {
        return GRef::make(bodies_.hostRef(), bools_.hostRef(), ints_.hostRef());
    }
    GRef ref() const { return hostRef(); }

#ifndef BRIE_CPU_ONLY
    /** @brief Return device reference */
    GRef deviceRef() const
    {
        return GRef::make(
            bodies_.deviceRef(), bools_.deviceRef(), ints_.deviceRef());
    }
#endif

    /** @brief Clone this object. */
    Tree clone() const
    {
        return Tree(std::move(bodies_.clone()), std::move(bools_.clone()),
            std::move(ints_.clone()));
    }

protected:
    /** @brief Gravity tree creation templated static method */
    template<typename SystemInfo>
    static void addToTree_(NaifIdArray::GRef& bref,
        const NaifIdArray::GRef& aref, BoolVecArray::GRef& boolRef,
        IntVecArray::GRef& intRef, idx_t& idx, const NaifIdArray::GRef& bodies,
        const NaifIdArray::GRef& barycentres, const bool& SunToSSB = true)
    {
        /* Add the barycentre and/or the main body of this system */
        bref.addItem(aref, SystemInfo::bary(), idx, SunToSSB);
        bref.addItem(aref, SystemInfo::main(), idx, SunToSSB);

        /* Now we find the (possible) parent node of this system. Search
         * main body first, to prevent bug related to SSB = 0 (which is the
         * default value in arrays as well) */
        idx_t pIdx = bref.indexOf(SystemInfo::main());
        if (pIdx == bref.size()) {
            /* main body not found. serach for the bary */
            pIdx = bref.indexOf(SystemInfo::bary());
        }
        if (pIdx < bref.size()) {
            /* This system has a parent node */
            /* how many children of the parent node we have */
            if constexpr (SystemInfo::hasBarycentres)
                intRef.get<CHILDRENSIZE>(pIdx)
                    += aref.countContained(barycentres);
            if constexpr (SystemInfo::hasBodies)
                intRef.get<CHILDRENSIZE>(pIdx) += aref.countContained(bodies);
            /* If the parent has children, proceed with the ordered filling
             * and the computation of the offsets */
            if (intRef.get<CHILDRENSIZE>(pIdx) > 0) {
                boolRef.get<HASCHILDREN>(pIdx) = true;
                /* The current idx is the offset for the parent node */
                intRef.get<CHILDRENOFFSET>(pIdx) = idx;
                /* we store the current idx: it will serve to fill the
                 * parent offset for all the children */
                idx_t startIdx = idx;
                /* Collect in the sorted array */
                if constexpr (SystemInfo::hasBarycentres)
                    bref.collectSorted(aref, barycentres, idx, SunToSSB);
                if constexpr (SystemInfo::hasBodies)
                    bref.collectSorted(aref, bodies, idx, SunToSSB);
                /* All the collected bodies have a parent */
                for (idx_t i = startIdx; i < idx; i++) {
                    intRef.get<PARENTOFFSET>(i) = pIdx;
                    boolRef.get<HASPARENT>(i)   = true;
                }
            }
        } else {
            /* The barycentre or the main body of this system are not contained
             * in the set of active bodies, which means that the root node is
             * among the children in this system. */
            /* We then just collect the bodies, the respective sub-system(s)
             * will possibly update the corresponding parent node(s)*/
            if constexpr (SystemInfo::hasBarycentres)
                bref.collectSorted(aref, barycentres, idx, SunToSSB);
            if constexpr (SystemInfo::hasBodies)
                bref.collectSorted(aref, bodies, idx, SunToSSB);
        }
    }

    /** @brief Recursion to create the Tree, checking all the available systems
     */
    template<Systems SYS>
    static void recurseAddToTree_(NaifIdArray::GRef bref,
        const NaifIdArray::GRef aref, BoolVecArray::GRef boolRef,
        IntVecArray::GRef intRef, idx_t& idx, const bool SunToSSB = true)
    {
        /* previous recursion step */
        if constexpr (SYS >= 1)
            Self::recurseAddToTree_<static_cast<Systems>(SYS - 1)>(
                bref, aref, boolRef, intRef, idx, SunToSSB);
        /* this recursion step */
        using SInfo = typename SystemSwitcher<SYS>::Info;
        /* Use context to deallocate bodies and barycentres */
        {
            NaifIdArray bodies      = SInfo::bodies();
            NaifIdArray barycentres = SInfo::barycentres();
            Self::addToTree_<SInfo>(bref, aref, boolRef, intRef, idx,
                bodies.hostRef(), barycentres.hostRef(), SunToSSB);
        }
    }

    /* Data members */
    NaifIdArray bodies_;
    BoolVecArray bools_;
    IntVecArray ints_;
};

} // namespace gravity
} // namespace brie