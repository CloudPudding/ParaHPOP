#pragma once

#include "brie/core.h"

namespace brie {
namespace gravity {
namespace schema {
namespace detail {

using core::NaifIdArray;

/** @brief Generic, abstract gravity information manager */
template<typename MapsT, typename ChildT>
struct Info {
private:
    using Self = ChildT;

public:
    /** @brief Expose bodies enum class*/
    using BodiesT = typename MapsT::BodiesT;
    /** @brief Expose barycentre enum class */
    using BarysT = typename MapsT::BarysT;

    /** @brief How many bodies */
    static constexpr idx_t BodySize = static_cast<idx_t>(BodiesT::SIZE);
    /** @brief How many barycentres */
    static constexpr idx_t BarySize = static_cast<idx_t>(BarysT::SIZE);

    /** @brief Whether this system has barycentres */
    static constexpr bool hasBarycentres = Self::BarySize > 0;
    /** @brief Whether this system has bodies */
    static constexpr bool hasBodies = Self::BodySize > 0;

    /** @brief Return the Main body ID */
    static NaifId main() { return Self::main_; }
    /** @brief Return the Barycentre ID */
    static NaifId bary() { return Self::bary_; }

    /** @brief Create a NaifIdArray containing the system bodies */
    static NaifIdArray bodies()
    {

        if constexpr (Self::hasBodies) {
            NaifIdArray b(Self::BodySize);
            NaifIdArray::GRef bref = b.hostRef();
            for (idx_t i = 0; i < Self::BodySize; i++)
                bref[i] = MapsT::ebo2ibo().at(static_cast<BodiesT>(i));
            return b;
        } else
            return NaifIdArray(1);
    }

    /** @brief Create a NaifIdArray containing the system barycentres */
    static NaifIdArray barycentres()
    {

        if constexpr (Self::hasBarycentres) {
            NaifIdArray b(Self::BarySize);
            NaifIdArray::GRef bref = b.hostRef();
            for (idx_t i = 0; i < Self::BarySize; i++)
                bref[i] = MapsT::eba2iba().at(static_cast<BarysT>(i));
            return b;
        } else
            return NaifIdArray(1);
    }

    /** @brief Check if the given item is found among the bodies */
    template<typename T>
    static bool bodiesContain(T item)
    {
        return Self::bodyparse_(item) != BodiesT::INVALID;
    }

    /** @brief Check if the given item is found among the barycentres */
    template<typename T>
    static bool barycentresContain(T item)
    {
        return Self::baryparse_(item) != BarysT::INVALID;
    }

    /** @brief Check if the given item is found in this system */
    template<typename T>
    static bool contains(T item)
    {
        return Self::bodiesContain(item) || Self::barycentresContain(item);
    }

    /** @brief Return the Naif ID of the given item */
    static NaifId getNaifId(NaifId item)
    {
        if (!Self::contains(item)) {
            std::stringstream ss;
            ss << "Naif Id (" << item << ") not in this system";
            std::string s = ss.str();
            BRIE_THROW(std::runtime_error, s.c_str());
        }

        return item;
    }

    /** @brief Return the Naif ID of the given item */
    static NaifId getNaifId(std::string item)
    {
        if (Self::bodiesContain(item))
            return MapsT::ebo2ibo().at(Self::bodyparse_(item));
        else if (Self::barycentresContain(item))
            return MapsT::eba2iba().at(Self::baryparse_(item));
        else {
            std::stringstream ss;
            ss << "Body (" << item << ") not in this system";
            std::string s = ss.str();
            BRIE_THROW(std::runtime_error, s.c_str());
        }
    }

    /** @brief Return the name of the given item */
    static std::string getName(std::string item)
    {
        if (!Self::contains(item)) {
            std::stringstream ss;
            ss << "Body (" << item << ") not in this system";
            std::string s = ss.str();
            BRIE_THROW(std::runtime_error, s.c_str());
        }

        return item;
    }

    /** @brief Return the name of the given item */
    static std::string getName(NaifId item)
    {
        if (Self::bodiesContain(item))
            return MapsT::ebo2sbo().at(Self::bodyparse_(item));
        else if (Self::barycentresContain(item))
            return MapsT::eba2sba().at(Self::baryparse_(item));
        else {
            std::stringstream ss;
            ss << "Body (" << item << ") not in this system";
            std::string s = ss.str();
            BRIE_THROW(std::runtime_error, s.c_str());
        }
    }

protected:
    /** @brief Parse the corresponding Map to look for the given body (Naif
     * Id)
     */
    static BodiesT bodyparse_([[maybe_unused]] NaifId item)
    {
        if constexpr (Self::hasBodies) {
            /* find the corresponding iterator */
            auto m        = MapsT::ibo2ebo();
            auto iterator = m.find(item);
            if (iterator != m.end())
                return iterator->second;
            /* no valid source was parsed */
            return BodiesT::INVALID;
        }
        return BodiesT::INVALID;
    }

    /** @brief Parse the corresponding Map to look for the given body (string)
     */
    static BodiesT bodyparse_([[maybe_unused]] std::string item)
    {
        if constexpr (Self::hasBodies) {
            /* find the corresponding iterator */
            auto m        = MapsT::sbo2ebo();
            auto iterator = m.find(item);
            if (iterator != m.end())
                return iterator->second;
            /* no valid source was parsed */
            return BodiesT::INVALID;
        }
        return BodiesT::INVALID;
    }

    /** @brief Parse the corresponding map to look for the given bary (Naif Id)
     */
    static BarysT baryparse_([[maybe_unused]] NaifId item)
    {
        if constexpr (Self::hasBarycentres) {
            /* find the corresponding iterator */
            auto m        = MapsT::iba2eba();
            auto iterator = m.find(item);
            if (iterator != m.end())
                return iterator->second;
            /* no valid source was parsed */
            return BarysT::INVALID;
        }
        return BarysT::INVALID;
    }

    /** @brief Parse the corresponding map to look for the given bary (string)
     */
    static BarysT baryparse_([[maybe_unused]] std::string item)
    {
        if constexpr (Self::hasBarycentres) {
            /* find the corresponding iterator */
            auto m        = MapsT::sba2eba();
            auto iterator = m.find(item);
            if (iterator != m.end())
                return iterator->second;
            /* no valid source was parsed */
            return BarysT::INVALID;
        }
        return BarysT::INVALID;
    }

    /** @brief The main body of this system */
    static constexpr NaifId main_ = -199;
    /** @brief The bary of this system */
    static constexpr NaifId bary_ = -1;
};

} // namespace detail
} // namespace schema
} // namespace gravity
} // namespace brie