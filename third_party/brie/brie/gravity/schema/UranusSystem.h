#pragma once

#include "brie/gravity/schema/detail/Info.h"

namespace brie {
namespace gravity {
namespace schema {
namespace uranus {

/** @brief Counters for the bodies in the system */
enum class Bodies {
    ARIEL,
    UMBRIEL,
    TITANIA,
    OBERON,
    MIRANDA,
    CORDELIA,
    OPHELIA,
    BIANCA,
    CRESSIDA,
    DESDEMONA,
    JULIET,
    PORTIA,
    ROSALIND,
    BELINDA,
    PUCK,
    CALIBAN,
    SYCORAX,
    PROSPERO,
    SETEBOS,
    STEPHANO,
    TRINCULO,
    FRANCISCO,
    MARGARET,
    FERDINAND,
    PERDITA,
    MAB,
    CUPID,
    /* SIZE */
    SIZE,
    /* invalid item */
    INVALID
};

/** @brief Counters for the barycentres in the System */
enum class Barys {
    /* SIZE */
    SIZE,
    /* invalid item */
    INVALID
};

/** @brief Body Maps */
struct MapsT {

    /** @brief Enum class for Bodies */
    using BodiesT = Bodies;
    /** @brief Enum class for Barys */
    using BarysT = Barys;

    /** @brief Body Naif ID to enum map */
    using BodyIdToEnumT = std::map<NaifId, BodiesT>;
    /** @brief Barycentre Naif ID to enum map */
    using BaryIdToEnumT = std::map<NaifId, BarysT>;
    /** @brief Body enum to Naif ID map */
    using EnumToBodyIdT = std::map<BodiesT, NaifId>;
    /** @brief Barycentre enum to Naif ID map */
    using EnumToBaryIdT = std::map<BarysT, NaifId>;
    /** @brief Body name string to enum map */
    using BodyStringToEnumT = std::map<std::string, BodiesT>;
    /** @brief Barycentre name string to enum map */
    using BaryStringToEnumT = std::map<std::string, BarysT>;
    /** @brief Body enum to name string map */
    using EnumToBodyStringT = std::map<BodiesT, std::string>;
    /** @brief Barycentre enum to name string map */
    using EnumToBaryStringT = std::map<BarysT, std::string>;

    /** @brief Maps */
    static BodyIdToEnumT ibo2ebo()
    {
        return { { 701, Bodies::ARIEL }, { 702, Bodies::UMBRIEL },
            { 703, Bodies::TITANIA }, { 704, Bodies::OBERON },
            { 705, Bodies::MIRANDA }, { 706, Bodies::CORDELIA },
            { 707, Bodies::OPHELIA }, { 708, Bodies::BIANCA },
            { 709, Bodies::CRESSIDA }, { 710, Bodies::DESDEMONA },
            { 711, Bodies::JULIET }, { 712, Bodies::PORTIA },
            { 713, Bodies::ROSALIND }, { 714, Bodies::BELINDA },
            { 715, Bodies::PUCK }, { 716, Bodies::CALIBAN },
            { 717, Bodies::SYCORAX }, { 718, Bodies::PROSPERO },
            { 719, Bodies::SETEBOS }, { 720, Bodies::STEPHANO },
            { 721, Bodies::TRINCULO }, { 722, Bodies::FRANCISCO },
            { 723, Bodies::MARGARET }, { 724, Bodies::FERDINAND },
            { 725, Bodies::PERDITA }, { 726, Bodies::MAB },
            { 727, Bodies::CUPID } };
    }
    static BaryIdToEnumT iba2eba() { return {}; }
    static EnumToBodyIdT ebo2ibo()
    {
        return { { Bodies::ARIEL, 701 }, { Bodies::UMBRIEL, 702 },
            { Bodies::TITANIA, 703 }, { Bodies::OBERON, 704 },
            { Bodies::MIRANDA, 705 }, { Bodies::CORDELIA, 706 },
            { Bodies::OPHELIA, 707 }, { Bodies::BIANCA, 708 },
            { Bodies::CRESSIDA, 709 }, { Bodies::DESDEMONA, 710 },
            { Bodies::JULIET, 711 }, { Bodies::PORTIA, 712 },
            { Bodies::ROSALIND, 713 }, { Bodies::BELINDA, 714 },
            { Bodies::PUCK, 715 }, { Bodies::CALIBAN, 716 },
            { Bodies::SYCORAX, 717 }, { Bodies::PROSPERO, 718 },
            { Bodies::SETEBOS, 719 }, { Bodies::STEPHANO, 720 },
            { Bodies::TRINCULO, 721 }, { Bodies::FRANCISCO, 722 },
            { Bodies::MARGARET, 723 }, { Bodies::FERDINAND, 724 },
            { Bodies::PERDITA, 725 }, { Bodies::MAB, 726 },
            { Bodies::CUPID, 727 } };
    }
    static EnumToBaryIdT eba2iba() { return {}; }
    static BodyStringToEnumT sbo2ebo()
    {
        return { { "ariel", Bodies::ARIEL }, { "umbriel", Bodies::UMBRIEL },
            { "titania", Bodies::TITANIA }, { "oberon", Bodies::OBERON },
            { "miranda", Bodies::MIRANDA }, { "cordelia", Bodies::CORDELIA },
            { "ophelia", Bodies::OPHELIA }, { "bianca", Bodies::BIANCA },
            { "cressida", Bodies::CRESSIDA },
            { "desdemona", Bodies::DESDEMONA }, { "juliet", Bodies::JULIET },
            { "portia", Bodies::PORTIA }, { "rosalind", Bodies::ROSALIND },
            { "belinda", Bodies::BELINDA }, { "puck", Bodies::PUCK },
            { "caliban", Bodies::CALIBAN }, { "sycorax", Bodies::SYCORAX },
            { "prospero", Bodies::PROSPERO }, { "setebos", Bodies::SETEBOS },
            { "stephano", Bodies::STEPHANO }, { "trinculo", Bodies::TRINCULO },
            { "francisco", Bodies::FRANCISCO },
            { "margaret", Bodies::MARGARET },
            { "ferdinand", Bodies::FERDINAND }, { "perdita", Bodies::PERDITA },
            { "mab", Bodies::MAB }, { "cupid", Bodies::CUPID },
            { "701", Bodies::ARIEL }, { "702", Bodies::UMBRIEL },
            { "703", Bodies::TITANIA }, { "704", Bodies::OBERON },
            { "705", Bodies::MIRANDA }, { "706", Bodies::CORDELIA },
            { "707", Bodies::OPHELIA }, { "708", Bodies::BIANCA },
            { "709", Bodies::CRESSIDA }, { "710", Bodies::DESDEMONA },
            { "711", Bodies::JULIET }, { "712", Bodies::PORTIA },
            { "713", Bodies::ROSALIND }, { "714", Bodies::BELINDA },
            { "715", Bodies::PUCK }, { "716", Bodies::CALIBAN },
            { "717", Bodies::SYCORAX }, { "718", Bodies::PROSPERO },
            { "719", Bodies::SETEBOS }, { "720", Bodies::STEPHANO },
            { "721", Bodies::TRINCULO }, { "722", Bodies::FRANCISCO },
            { "723", Bodies::MARGARET }, { "724", Bodies::FERDINAND },
            { "725", Bodies::PERDITA }, { "726", Bodies::MAB },
            { "727", Bodies::CUPID } };
    }
    static BaryStringToEnumT sba2eba() { return {}; }
    static EnumToBodyStringT ebo2sbo()
    {
        return { { Bodies::ARIEL, "Ariel" }, { Bodies::UMBRIEL, "Umbriel" },
            { Bodies::TITANIA, "Titania" }, { Bodies::OBERON, "Oberon" },
            { Bodies::MIRANDA, "Miranda" }, { Bodies::CORDELIA, "Cordelia" },
            { Bodies::OPHELIA, "Ophelia" }, { Bodies::BIANCA, "Bianca" },
            { Bodies::CRESSIDA, "Cressida" },
            { Bodies::DESDEMONA, "Desdemona" }, { Bodies::JULIET, "Juliet" },
            { Bodies::PORTIA, "Portia" }, { Bodies::ROSALIND, "Rosalind" },
            { Bodies::BELINDA, "Belinda" }, { Bodies::PUCK, "Puck" },
            { Bodies::CALIBAN, "Caliban" }, { Bodies::SYCORAX, "Sycorax" },
            { Bodies::PROSPERO, "Prospero" }, { Bodies::SETEBOS, "Setebos" },
            { Bodies::STEPHANO, "Stephano" }, { Bodies::TRINCULO, "Trinculo" },
            { Bodies::FRANCISCO, "Francisco" },
            { Bodies::MARGARET, "Margaret" },
            { Bodies::FERDINAND, "Ferdinand" }, { Bodies::PERDITA, "Perdita" },
            { Bodies::MAB, "Mab" }, { Bodies::CUPID, "Cupid" } };
    }
    static EnumToBaryStringT eba2sba() { return {}; }
};

/** @brief Information structure for this system */
struct Info : public detail::Info<MapsT, Info> {
    friend struct detail::Info<MapsT, Info>;

protected:
    /** @brief The main body of this system */
    static constexpr NaifId main_ = 799;
    /** @brief The bary of this system */
    static constexpr NaifId bary_ = 7;
};

} // namespace uranus
} // namespace schema
} // namespace gravity
} // namespace brie