#include "brie/gravity/Tree.h"

namespace brie {
namespace gravity {

/** @brief Factory method to construct from a given set of active bodies
 * TODO: Address whether to include a user-tunable barycentre vs actual body
 * selection policy. At the moment, SSB is enforced for Sun as COI only.
 */
Tree Tree::fromActiveBodies(
    const NaifIdArray& activeBodies, const bool& SunToSSB)
{
    NaifIdArray::GRef aref = activeBodies.hostRef();

    /* validate the active body list:
        - No double bodies
        - No invalid bodies
        - Either the bodies or their barycentres
     */
    aref.validateActive();

    /* we allocate the arrays for the data members and get all the
     * required references */
    NaifIdArray bodies(activeBodies.size());
    BoolVecArray bools(activeBodies.size());
    IntVecArray ints(activeBodies.size());
    NaifIdArray::GRef bref     = bodies.hostRef();
    BoolVecArray::GRef boolRef = bools.hostRef();
    IntVecArray::GRef intRef   = ints.hostRef();

    /* We collect the active bodies sorted with the gravity schema */
    idx_t idx                  = 0;
    constexpr Systems FirstSYS = static_cast<Systems>(SYSTEMSSIZE - 1);
    Self::recurseAddToTree_<FirstSYS>(
        bref, aref, boolRef, intRef, idx, SunToSSB);

    /* Finally return the full gravity tree */
    return Self(std::move(bodies), std::move(bools), std::move(ints));
}

} // namespace gravity
} // namespace brie