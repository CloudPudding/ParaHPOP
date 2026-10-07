#pragma once

#include "brie/typedefs.h"
#include "brie/util/throw.h"

namespace brie {
namespace core {

using nlohmann::json;

/** @brief Reference Naif ID Array */
template<bool work, bool MaybeVolatile = false>
class RefNaifIdArray
    : public feta::scalar::Array<NaifId>::template Ref<work, MaybeVolatile> {
    using ParentT =
        typename feta::scalar::Array<NaifId>::template Ref<work, MaybeVolatile>;
    using Self    = RefNaifIdArray<work, MaybeVolatile>;

public:
    /** @brief Volatile is allowed only with work */
    static constexpr bool Volatile = MaybeVolatile && work;

    /** @brief Factory method to construct from parent type */
    DEVICEHOST() static RefNaifIdArray make(const ParentT& other)
    {
        RefNaifIdArray out;
        static_cast<ParentT&>(out) = other;
        return out;
    }

    /** @brief Factory method to construct from pointer and size */
    DEVICEHOST()
    static RefNaifIdArray make(NaifId* const data, const idx_t& size)
    {
        RefNaifIdArray out;
        static_cast<ParentT&>(out) = ParentT{ data, size };
        return out;
    }

    /** @brief Return the array index corresponding to the given item */
    DEVICEHOST() idx_t indexOf(const NaifId& id) const
    {
        idx_t out = this->size();
        for (idx_t i = 0; i < this->size(); i++) {
            if ((*this)[i] == id) {
                out = i;
                break;
            }
        }

        return out;
    }

    /** @brief Check if the given body array contains the given naif id. Check
     * up to the specified limit (default case checks the full array) */
    DEVICEHOST() bool contains(const NaifId& id, const idx_t& lim = 0) const
    {
        idx_t loopLim = this->size();
        if (lim != 0)
            loopLim = lim;
        bool out = false;
        for (idx_t i = 0; i < loopLim; i++) {
            if ((*this)[i] == id) {
                out = true;
                break;
            }
        }

        return out;
    }

    /** @brief Add the given body to this array, if it's contained in `from`
    and
     * is not already present */
    DEVICEHOST()
    void addItem(const Self& from, const NaifId& item, idx_t& idx,
        const bool& SunToSSB = true)
    {
        if (from.contains(item)) {
            /* this body should be added to `to`, if not present yet */
            if (!this->contains(item, idx)) {
                /* Override Sun (10) with SSB, if flag requires so */
                if (item == 10 && SunToSSB)
                    (*this)[idx] = 0;
                else
                    (*this)[idx] = item;
                idx += 1;
            }
        }
    }

    /** @brief If a body in the Given Full list appears in the active bodies to
     * collect (from), place it in the given array (to) */
    DEVICEHOST()
    void collectSorted(const Self& from, const Self& orderedList, idx_t& idx,
        const bool& SunToSSB = true)
    {
        /* Only include bodies if the `to` array is not full */
        if (idx < this->size()) {
            for (idx_t i = 0; i < orderedList.size(); i++) {
                this->addItem(from, orderedList[i], idx, SunToSSB);
            }
        }
    }

    /** @brief Count how many elements of the given id are contained in `items`
     */
    DEVICEHOST() idx_t countContainedID(const NaifId& id) const
    {
        idx_t out = 0;
        for (idx_t i = 0; i < this->size(); i++)
            if ((*this)[i] == id)
                out += 1;
        return out;
    }

    /** @brief Count how many elements of `fullList` are contained in `items` */
    DEVICEHOST() idx_t countContained(const Self& fullList) const
    {
        idx_t out = 0;
        for (idx_t i = 0; i < fullList.size(); i++)
            if (this->contains(fullList[i]))
                out += 1;
        return out;
    }

    /** @brief The basic validation - bodies can be defined only once, and must
     * not contain 0 (the SSB) */
    void validate() const
    {
        /* SSB CANNOT be among the active bodies */
        if (this->contains(0))
            BRIE_THROW(std::runtime_error,
                "Solar System Barycentre is not a valid active body.");

        /* Bodies can be defined only once */
        for (idx_t i = 1; i < this->size(); i++)
            if (this->countContainedID((*this)[i]) > 1)
                BRIE_THROW(std::runtime_error,
                    "Multiple definitions of the same body.");
    }

    /** @brief Validate the list of active bodies */
    void validateActive() const
    {

        /* Do the simple validation first */
        this->validate();

        /* Either the barycenter or the single bodies of the respective system
         * are present among the active bodies, NOT BOTH. */
        for (idx_t i = 0; i < this->size(); i++) {
            /* The Sun has already been checked above */
            if ((*this)[i] == 10)
                continue;
            else {
                /* bodies are characterized by values higher than the
                 * Sun */
                if ((*this)[i] > 10) {
                    /* this is a body. If its a main body, we need to check that
                     * the barycenter of its system is NOT among the active
                     * bodies.
                     */
                    NaifId testBary = (*this)[i] / 100;
                    /** TODO: add body and bary in the error message */
                    if (this->contains(testBary)) {
                        BRIE_THROW(std::runtime_error,
                            "Please select either the main body or its system "
                            "barycenter");
                    }
                } else {
                    /* this is a barycenter. We need to check that NONE of the
                     * bodies of its system is present among the active bodies
                     */
                    for (idx_t j = i + 1; j < this->size(); j++) {
                        NaifId testBary = (*this)[j] / 100;
                        /** TODO: add body and bary in the error message
                         */
                        if ((*this)[i] == testBary) {
                            BRIE_THROW(std::runtime_error,
                                "Please select either body or its barycenter");
                        }
                    }
                }
            }
        }
    }
};

/** @brief Owning Naif ID Array */
class NaifIdArray : public feta::scalar::Array<NaifId> {
    using ParentT = feta::scalar::Array<NaifId>;

public:
    /** @brief Reference types */
    template<bool work, bool MaybeVolatile = false>
    using Ref = RefNaifIdArray<work, MaybeVolatile>;
    /** @brief Global reference type */
    using GRef = Ref<false>;
    /** @brief Work reference type */
    using WRef = Ref<true>;
    /** @brief Volatile work reference */
    using VolatileRef = Ref<true, true>;

    /** @brief Inherit constructors */
    NaifIdArray(ParentT&& other)
        : ParentT{ std::move(other) }
    {
    }

    /** @brief Create from `std::vector` of NaifIds */
    NaifIdArray(const std::vector<NaifId>& vec)
        : ParentT{ static_cast<idx_t>(vec.size()) }
    {
        GRef ref = this->hostRef();
        for (idx_t i = 0; i < vec.size(); i++) {
            ref[i] = vec[i];
        }
    }

    /** @brief Return a host reference */
    GRef hostRef() const { return GRef::make(ParentT::hostRef()); }
    GRef ref() const { return hostRef(); }

#ifndef BRIE_CPU_ONLY
    /** @brief Return a device reference */
    GRef deviceRef() const { return GRef::make(ParentT::deviceRef()); }
#endif
};

} // namespace core
} // namespace brie