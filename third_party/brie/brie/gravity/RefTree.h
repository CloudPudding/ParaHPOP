#pragma once

#include "brie/gravity/Node.h"
#include "brie/gravity/schema.h"
#include "brie/typedefs.h"
#include "brie/util.h"

namespace brie {
namespace gravity {

using namespace schema;

/** @brief Gravity tree boolean metadata */
enum BoolMeta {
    HASPARENT,
    HASCHILDREN,

    /* Leave the following item as last -- it only acts as enum size */
    BOOLSIZE,

};

/** @brief Gravity tree integer metadata */
enum IntMeta {
    PARENTOFFSET,   /* Where we can find the possible parent of this node */
    CHILDRENOFFSET, /* Where we can find the first child of this node */
    CHILDRENSIZE,   /* How many children this node has */

    /* Leave the following item as last -- it only acts as enum size */
    INTSIZE,

};

/* Array types */
using BoolVecArray = feta::vector::Array<bool, BOOLSIZE>;
using IntVecArray  = feta::vector::Array<idx_t, INTSIZE>;

/* Vector types */
using BoolVec = feta::vector::Item<bool, BOOLSIZE>;
using IntVec  = feta::vector::Item<idx_t, INTSIZE>;

/** @brief Non-owning reference to the gravity tree */
template<bool work>
class RefTree {
    using Self            = RefTree<work>;
    using RefNaifIdArray  = NaifIdArray::Ref<work>;
    using RefBoolVecArray = BoolVecArray::Ref<work>;
    using RefIntVecArray  = IntVecArray::Ref<work>;

public:
    /** @brief Expose Node type */
    using NodeT = Node<Self>;

    /** @brief Factory method to construct from data members */
    DEVICEHOST()
    static RefTree make(const RefNaifIdArray& bodies,
        const RefBoolVecArray& bools, const RefIntVecArray& ints)
    {
        return { bodies, bools, ints };
    }


    /** @brief Expose bodies */
    DEVICEHOST() RefNaifIdArray bodies() const { return bodies_; }

    /** @brief Expose boolean metadata */
    DEVICEHOST() RefBoolVecArray boolmeta() const { return bools_; }

    /** @brief Expose integer metadata */
    DEVICEHOST() RefIntVecArray intmeta() const { return ints_; }

    /** @brief Return the ID of the given node index */
    DEVICEHOST() NaifId ID(const SampleIndex idx) const { return bodies_[idx]; }

    /** @brief Check if the given node has any parent */
    DEVICEHOST() bool hasParent(const SampleIndex& idx) const
    {
        return bools_.template get<HASPARENT>(idx);
    }
    DEVICEHOST() bool hasParent(const idx_t& idx) const
    {
        return bools_.template get<HASPARENT>(idx);
    }

    /** @brief Check if the given node has any parent */
    DEVICEHOST() bool hasChildren(const SampleIndex& idx) const
    {
        return bools_.template get<HASCHILDREN>(idx);
    }
    DEVICEHOST() bool hasChildren(const idx_t& idx) const
    {
        return bools_.template get<HASCHILDREN>(idx);
    }

    /** @brief Get parent offset for the given node index */
    DEVICEHOST() idx_t parentOffset(const SampleIndex& idx) const
    {
        return ints_.template get<PARENTOFFSET>(idx);
    }
    DEVICEHOST() idx_t parentOffset(const idx_t& idx) const
    {
        return ints_.template get<PARENTOFFSET>(idx);
    }

    /** @brief Get children offset for the given node index */
    DEVICEHOST() idx_t childrenOffset(const SampleIndex& idx) const
    {
        return ints_.template get<CHILDRENOFFSET>(idx);
    }
    DEVICEHOST() idx_t childrenOffset(const idx_t& idx) const
    {
        return ints_.template get<CHILDRENOFFSET>(idx);
    }

    /** @brief Get children size for the given node index */
    DEVICEHOST() idx_t childrenSize(const SampleIndex& idx) const
    {
        return ints_.template get<CHILDRENSIZE>(idx);
    }
    DEVICEHOST() idx_t childrenSize(const idx_t& idx) const
    {
        return ints_.template get<CHILDRENSIZE>(idx);
    }

    /** @brief Extract the node for the given index */
    DEVICEHOST() NodeT node(const SampleIndex& idx) const
    {
        return node(*this, work ? idx.work() : idx.global());
    }
    DEVICEHOST() NodeT node(const idx_t& idx) const
    {
        return NodeT(*this, idx);
    }

    /** @brief Create new global reference from this object */
    DEVICEHOST() RefTree clone() const { return *this; }

    /** @brief Data members made public for PODification */
    RefNaifIdArray bodies_;
    RefBoolVecArray bools_;
    RefIntVecArray ints_;
};

} // namespace gravity
} // namespace brie