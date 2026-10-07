#pragma once

#include "brie/typedefs.h"
#include "brie/util.h"

namespace brie {
namespace core {

/** @brief A body traverser - helps the fast lookup table traversal to find
 * common centers s*/
class BodyLocator {
public:
    /** @brief Check whether this and other have a common center */
    DEVICEHOST() inline bool sharesCenterWith(const BodyLocator& other) const
    {
        return centerPos_ == other.centerPos_ || centerPos_ == other.pos_
            || pos_ == other.centerPos_ || pos_ == other.pos_;
    }

    /** @brief Return the common center position */
    DEVICEHOST() inline idx_t commonCenter(const BodyLocator& other) const
    {
#ifdef __CUDA_ARCH__
        BRIE_GPU_ASSERT(sharesCenterWith(other), err::NO_SHARED_CENTER);
#else
        BRIE_ASSERT(sharesCenterWith(other),
            "No shared center found between target and center bodies");
#endif
        if (centerPos_ == other.centerPos_ || centerPos_ == other.pos_) {
            return centerPos_;
        } else if (pos_ == other.centerPos_ || pos_ == other.pos_) {
            return pos_;
        } else {
            /* This should never happen due to the assertion, but we need it to
             * silence compiler warnings */
            return pos_;
        }
    }

    /** @brief Operator != */
    DEVICEHOST() bool operator!=(const BodyLocator& other) const
    {
        return this->id_ != other.id_ || this->pos_ != other.pos_
            || this->centerPos_ != other.centerPos_;
    }

    /* Data members made public for PODification */

    /** @brief the body naif id */
    NaifId id_ = 0;
    /** @brief the body position in the lookup table */
    idx_t pos_ = 0;
    /** @brief The position of the body's center in the lookup table */
    idx_t centerPos_ = 0;
};

/** @brief Chain walkers - stores the minimally relevant information to walk the
 * chain */
struct ChainWalker {
    const idx_t start_ = 0;
    const idx_t steps_ = 0;
};
struct WalkerPair {
    const ChainWalker target_;
    const ChainWalker center_;
};

/** @brief The full traverser */
class RefTraverser : public feta::scalar::Array<BodyLocator>::GRef {
    using ParentT = typename feta::scalar::Array<BodyLocator>::GRef;

public:
    /** @brief Factory method to construct from parent */
    static RefTraverser make(const ParentT& parent)
    {
        return RefTraverser{ parent };
    }

    /** @brief Get the body locator for the given Naif ID */
    DEVICEHOST() inline BodyLocator& at(const NaifId& naifId) const
    {
        idx_t pos = this->size(); // default to size() to indicate not found
        for (idx_t i = 0; i < this->size(); i++) {
            if ((*this)[i].id_ == naifId) {
                pos = i;
                break;
            }
        }
#ifdef __CUDA_ARCH__
        BRIE_GPU_ASSERT(pos != this->size(), err::BODY_NOT_IN_TRAVERSER);
#else
        BRIE_ASSERT(pos != this->size(), "Body not found in traverser");
#endif
        return (*this)[pos];
    }

    /** @brief Find the highest possible common center position for the given
     * target-center pair */
    DEVICEHOST()
    idx_t commonCenterPos(
        const NaifId& targetNaifID, const NaifId& centerNaifID) const
    {
        /* We leverage pos == size() at an outer level as a marker for the Solar
         * System Barycenter */
        idx_t targetPos
            = targetNaifID == 0 ? this->size() : at(targetNaifID).pos_;
        idx_t centerPos
            = centerNaifID == 0 ? this->size() : at(centerNaifID).pos_;
        return commonCenterPos(targetPos, centerPos);
    }

    /** @brief Find the highest possible common center position for the given
     * target-center pair */
    DEVICEHOST()
    idx_t commonCenterPos(idx_t tc, idx_t cc) const
    {
        /* Solar System Barycenter (SSB) case uses size() as marker */
        for (;;) {
            /* exact common center case */
            if (tc == cc)
                return tc;
            /* if either is the SSB return it as common center */
            if (tc == this->size() || cc == this->size())
                return this->size();
            /* Run a new step otherwise */
            const BodyLocator& a = (*this)[tc];
            const BodyLocator& b = (*this)[cc];
            /* Check if there is a cross-common center */
            if (a.sharesCenterWith(b))
                return a.commonCenter(b);
            /* Continue to the next step */
            tc = a.centerPos_;
            cc = b.centerPos_;
        }
        /* finally return (prevent the compiler from complaining) */
        return this->size();
    }

    /** @brief Construct the chain walker for the given start positions */
    DEVICEHOST()
    WalkerPair makeWalkers(const idx_t& tc, const idx_t& cc) const
    {
        /* we extract the common center first */
        const idx_t endpoint = commonCenterPos(tc, cc);
        /* and we now proceed to find the required steps */
        idx_t ts = tc, cs = cc, tSteps = 0, cSteps = 0;
        for (;;) {
            if (ts == endpoint)
                break;
            ts = (*this)[ts].centerPos_;
            tSteps++;
        }
        for (;;) {
            if (cs == endpoint)
                break;
            cs = (*this)[cs].centerPos_;
            cSteps++;
        }
        return { { tc, tSteps }, { cc, cSteps } };
    }

    /** @brief Construct the chain walker for the given start and end naif id */
    DEVICEHOST()
    WalkerPair makeWalkers(const NaifId& target, const NaifId& center) const
    {
        /* we extract the common center first */
        const idx_t tc = target == 0 ? this->size() : at(target).pos_;
        const idx_t cc = center == 0 ? this->size() : at(center).pos_;
        return makeWalkers(tc, cc);
    }
};

/** @brief The Traverser */
class Traverser : public feta::scalar::Array<BodyLocator> {
    using ParentT = feta::scalar::Array<BodyLocator>;

public:
    /** @brief Expose the reference type.  ``MaybeVolatile`` is accepted
     * (ignored) for uniform 2-arg shape across containers — Traverser's
     * RefTraverser is work-invariant and not yet volatile-capable. */
    template<bool work, bool MaybeVolatile = false>
    using Ref = RefTraverser;
    /** @brief Expose the global reference type */
    using GRef = Ref<false>;

    /** @brief Factory method to construct a flexible item */
    static Traverser flexible() { return Traverser(ParentT::flexible()); }

    /** @brief Construct from size */
    Traverser(const idx_t& size)
        : ParentT{ size, BodyLocator{} }
    {
    }

    /** @brief Construct from parent type */
    Traverser(ParentT&& parent)
        : ParentT{ std::move(parent) }
    {
    }

    /** @brief Copy constructor is forbidden */
    Traverser(Traverser& other)       = delete;
    Traverser(const Traverser& other) = delete;

    /** @brief Move constructor is allowed */
    Traverser(Traverser&& other)
        : ParentT{ std::move(other) }
    {
    }

    /** @brief Copy assignment is forbidden */
    Traverser& operator=(Traverser& other)       = delete;
    Traverser& operator=(const Traverser& other) = delete;

    /** @brief Move assignment is allowed */
    Traverser& operator=(Traverser&& other)
    {
        ParentT::operator=(std::move(other));
        return *this;
    }


    /** @brief return a host reference */
    GRef hostRef() const { return GRef::make(ParentT::hostRef()); }
    GRef ref() const { return hostRef(); }

#ifndef BRIE_CPU_ONLY
    /** @brief return a device reference */
    GRef deviceRef() const { return GRef::make(ParentT::deviceRef()); }
#endif
};

} // namespace core
} // namespace brie