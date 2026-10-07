#pragma once

#include "brie/core/NaifIdArray.h"
#include "brie/typedefs.h"
#include "brie/util.h"

namespace brie {
namespace gravity {

using core::NaifIdArray;

/** @brief Individual gravity node. Contains a Reference Tree and the
 * offset to the body currently pointed */
template<typename TreeT>
class Node {
public:
    /** @brief Factory method to construct from data members */
    DEVICEHOST()
    static Node make(const TreeT& tree, const idx_t& idx)
    {
        return { tree, idx };
    }

    /** @brief Factory method to construct from Tree and Node Naif ID */
    DEVICEHOST()
    static Node make(const TreeT& tree, const NaifId& coi)
    {
        return { tree, tree.bodies().indexOf(coi) };
    }

    /** @brief Check if this node has a parent */
    DEVICEHOST() bool hasParent() const { return tree_.hasParent(nodeIdx_); }

    /** @brief Check if this node has children */
    DEVICEHOST() bool hasChildren() const
    {
        return tree_.hasChildren(nodeIdx_);
    }

    /** @brief Get the body ID for this node, according to the barycenter
     * preference */
    DEVICEHOST() NaifId ID() const { return tree_.bodies()[nodeIdx_]; }

    /** Expose tree */
    DEVICEHOST() TreeT tree() const { return tree_; }
    DEVICEHOST() idx_t idx() const { return nodeIdx_.global(); }

    /** @brief Get the parent ID for this node, according to its barycenter
     * preference. It returns this id if this node has no parent */
    DEVICEHOST() NaifId parentID() const
    {
        SampleIndex i = SampleIndex::make(tree_.parentOffset(nodeIdx_));
        return tree_.bodies()[i];
    }

    /** @brief Get the a global reference NaifIdArray of all the children of
     * this body */
    DEVICEHOST() NaifIdArray::GRef children() const
    {
        NaifId* data = nullptr;
        if (this->hasChildren())
            data = tree_.bodies().data() + tree_.childrenOffset(nodeIdx_);

        NaifIdArray::GRef out;
        out.data_ = data;
        out.size_ = tree_.childrenSize(nodeIdx_);
        return out;
    }

    /** @brief Create the parent node */
    DEVICEHOST() Node parentNode() const
    {
        return Node::make(tree_, tree_.parentOffset(nodeIdx_));
    }

    /** @brief Create the given child node */
    DEVICEHOST() Node childNode(const idx_t childNum) const
    {
        return Node::make(tree_, tree_.childrenOffset(nodeIdx_) + childNum);
    }

    /* Data members - made public for PODification */
    TreeT tree_;
    SampleIndex nodeIdx_;
};

} // namespace gravity
} // namespace brie