#pragma once

#include "brie/gravity/schema/detail/Info.h"

namespace brie {
namespace gravity {
namespace schema {
namespace jupiter {

/** @brief Counters for the bodies in the system */
enum class Bodies {
    IO,
    EUROPA,
    GANYMEDE,
    CALLISTO,
    AMALTHEA,
    HIMALIA,
    ELARA,
    PASIPHAE,
    SINOPE,
    LYSITHEA,
    CARME,
    ANANKE,
    LEDA,
    THEBE,
    ADRASTEA,
    METIS,
    CALLIRRHOE,
    THEMISTO,
    MEGACLITE,
    TAYGETE,
    CHALDENE,
    HARPALYKE,
    KALYKE,
    IOCASTE,
    ERINOME,
    ISONOE,
    PRAXIDIKE,
    AUTONOE,
    THYONE,
    HERMIPPE,
    AITNE,
    EURYDOME,
    EUANTHE,
    EUPORIE,
    ORTHOSIE,
    SPONDE,
    KALE,
    PASITHEE,
    HEGEMONE,
    MNEME,
    AOEDE,
    THELXINOE,
    ARCHE,
    KALLICHORE,
    HELIKE,
    CARPO,
    EUKELADE,
    CYLLENE,
    KORE,
    HERSE,
    DIA,
    /* SIZE */
    SIZE,
    /* invalid item */
    INVALID
};

/** @brief Counters for the barycentres in the system */
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
        return { { 501, Bodies::IO }, { 502, Bodies::EUROPA },
            { 503, Bodies::GANYMEDE }, { 504, Bodies::CALLISTO },
            { 505, Bodies::AMALTHEA }, { 506, Bodies::HIMALIA },
            { 507, Bodies::ELARA }, { 508, Bodies::PASIPHAE },
            { 509, Bodies::SINOPE }, { 510, Bodies::LYSITHEA },
            { 511, Bodies::CARME }, { 512, Bodies::ANANKE },
            { 513, Bodies::LEDA }, { 514, Bodies::THEBE },
            { 515, Bodies::ADRASTEA }, { 516, Bodies::METIS },
            { 517, Bodies::CALLIRRHOE }, { 518, Bodies::THEMISTO },
            { 519, Bodies::MEGACLITE }, { 520, Bodies::TAYGETE },
            { 521, Bodies::CHALDENE }, { 522, Bodies::HARPALYKE },
            { 523, Bodies::KALYKE }, { 524, Bodies::IOCASTE },
            { 525, Bodies::ERINOME }, { 526, Bodies::ISONOE },
            { 527, Bodies::PRAXIDIKE }, { 528, Bodies::AUTONOE },
            { 529, Bodies::THYONE }, { 530, Bodies::HERMIPPE },
            { 531, Bodies::AITNE }, { 532, Bodies::EURYDOME },
            { 533, Bodies::EUANTHE }, { 534, Bodies::EUPORIE },
            { 535, Bodies::ORTHOSIE }, { 536, Bodies::SPONDE },
            { 537, Bodies::KALE }, { 538, Bodies::PASITHEE },
            { 539, Bodies::HEGEMONE }, { 540, Bodies::MNEME },
            { 541, Bodies::AOEDE }, { 542, Bodies::THELXINOE },
            { 543, Bodies::ARCHE }, { 544, Bodies::KALLICHORE },
            { 545, Bodies::HELIKE }, { 546, Bodies::CARPO },
            { 547, Bodies::EUKELADE }, { 548, Bodies::CYLLENE },
            { 549, Bodies::KORE }, { 550, Bodies::HERSE },
            { 553, Bodies::DIA } };
    }
    static BaryIdToEnumT iba2eba() { return {}; }
    static EnumToBodyIdT ebo2ibo()
    {
        return { { Bodies::IO, 501 }, { Bodies::EUROPA, 502 },
            { Bodies::GANYMEDE, 503 }, { Bodies::CALLISTO, 504 },
            { Bodies::AMALTHEA, 505 }, { Bodies::HIMALIA, 506 },
            { Bodies::ELARA, 507 }, { Bodies::PASIPHAE, 508 },
            { Bodies::SINOPE, 509 }, { Bodies::LYSITHEA, 510 },
            { Bodies::CARME, 511 }, { Bodies::ANANKE, 512 },
            { Bodies::LEDA, 513 }, { Bodies::THEBE, 514 },
            { Bodies::ADRASTEA, 515 }, { Bodies::METIS, 516 },
            { Bodies::CALLIRRHOE, 517 }, { Bodies::THEMISTO, 518 },
            { Bodies::MEGACLITE, 519 }, { Bodies::TAYGETE, 520 },
            { Bodies::CHALDENE, 521 }, { Bodies::HARPALYKE, 522 },
            { Bodies::KALYKE, 523 }, { Bodies::IOCASTE, 524 },
            { Bodies::ERINOME, 525 }, { Bodies::ISONOE, 526 },
            { Bodies::PRAXIDIKE, 527 }, { Bodies::AUTONOE, 528 },
            { Bodies::THYONE, 529 }, { Bodies::HERMIPPE, 530 },
            { Bodies::AITNE, 531 }, { Bodies::EURYDOME, 532 },
            { Bodies::EUANTHE, 533 }, { Bodies::EUPORIE, 534 },
            { Bodies::ORTHOSIE, 535 }, { Bodies::SPONDE, 536 },
            { Bodies::KALE, 537 }, { Bodies::PASITHEE, 538 },
            { Bodies::HEGEMONE, 539 }, { Bodies::MNEME, 540 },
            { Bodies::AOEDE, 541 }, { Bodies::THELXINOE, 542 },
            { Bodies::ARCHE, 543 }, { Bodies::KALLICHORE, 544 },
            { Bodies::HELIKE, 545 }, { Bodies::CARPO, 546 },
            { Bodies::EUKELADE, 547 }, { Bodies::CYLLENE, 548 },
            { Bodies::KORE, 549 }, { Bodies::HERSE, 550 },
            { Bodies::DIA, 553 } };
    }
    static EnumToBaryIdT eba2iba() { return {}; }
    static BodyStringToEnumT sbo2ebo()
    {
        return { { "io", Bodies::IO }, { "europa", Bodies::EUROPA },
            { "ganymede", Bodies::GANYMEDE }, { "callisto", Bodies::CALLISTO },
            { "amalthea", Bodies::AMALTHEA }, { "himalia", Bodies::HIMALIA },
            { "elara", Bodies::ELARA }, { "pasiphae", Bodies::PASIPHAE },
            { "sinope", Bodies::SINOPE }, { "lysithea", Bodies::LYSITHEA },
            { "carme", Bodies::CARME }, { "ananke", Bodies::ANANKE },
            { "leda", Bodies::LEDA }, { "thebe", Bodies::THEBE },
            { "adrastea", Bodies::ADRASTEA }, { "metis", Bodies::METIS },
            { "callirrhoe", Bodies::CALLIRRHOE },
            { "themisto", Bodies::THEMISTO },
            { "megaclite", Bodies::MEGACLITE }, { "taygete", Bodies::TAYGETE },
            { "chaldene", Bodies::CHALDENE },
            { "harpalyke", Bodies::HARPALYKE }, { "kalyke", Bodies::KALYKE },
            { "iocaste", Bodies::IOCASTE }, { "erinome", Bodies::ERINOME },
            { "isonoe", Bodies::ISONOE }, { "praxidike", Bodies::PRAXIDIKE },
            { "autonoe", Bodies::AUTONOE }, { "thyone", Bodies::THYONE },
            { "hermippe", Bodies::HERMIPPE }, { "aitne", Bodies::AITNE },
            { "eurydome", Bodies::EURYDOME }, { "euanthe", Bodies::EUANTHE },
            { "euporie", Bodies::EUPORIE }, { "orthosie", Bodies::ORTHOSIE },
            { "sponde", Bodies::SPONDE }, { "kale", Bodies::KALE },
            { "pasithee", Bodies::PASITHEE }, { "hegemone", Bodies::HEGEMONE },
            { "mneme", Bodies::MNEME }, { "aoede", Bodies::AOEDE },
            { "thelxinoe", Bodies::THELXINOE }, { "arche", Bodies::ARCHE },
            { "kallichore", Bodies::KALLICHORE }, { "helike", Bodies::HELIKE },
            { "carpo", Bodies::CARPO }, { "eukelade", Bodies::EUKELADE },
            { "cyllene", Bodies::CYLLENE }, { "kore", Bodies::KORE },
            { "herse", Bodies::HERSE }, { "dia", Bodies::DIA },
            { "501", Bodies::IO }, { "502", Bodies::EUROPA },
            { "503", Bodies::GANYMEDE }, { "504", Bodies::CALLISTO },
            { "505", Bodies::AMALTHEA }, { "506", Bodies::HIMALIA },
            { "507", Bodies::ELARA }, { "508", Bodies::PASIPHAE },
            { "509", Bodies::SINOPE }, { "510", Bodies::LYSITHEA },
            { "511", Bodies::CARME }, { "512", Bodies::ANANKE },
            { "513", Bodies::LEDA }, { "514", Bodies::THEBE },
            { "515", Bodies::ADRASTEA }, { "516", Bodies::METIS },
            { "517", Bodies::CALLIRRHOE }, { "518", Bodies::THEMISTO },
            { "519", Bodies::MEGACLITE }, { "520", Bodies::TAYGETE },
            { "521", Bodies::CHALDENE }, { "522", Bodies::HARPALYKE },
            { "523", Bodies::KALYKE }, { "524", Bodies::IOCASTE },
            { "525", Bodies::ERINOME }, { "526", Bodies::ISONOE },
            { "527", Bodies::PRAXIDIKE }, { "528", Bodies::AUTONOE },
            { "529", Bodies::THYONE }, { "530", Bodies::HERMIPPE },
            { "531", Bodies::AITNE }, { "532", Bodies::EURYDOME },
            { "533", Bodies::EUANTHE }, { "534", Bodies::EUPORIE },
            { "535", Bodies::ORTHOSIE }, { "536", Bodies::SPONDE },
            { "537", Bodies::KALE }, { "538", Bodies::PASITHEE },
            { "539", Bodies::HEGEMONE }, { "540", Bodies::MNEME },
            { "541", Bodies::AOEDE }, { "542", Bodies::THELXINOE },
            { "543", Bodies::ARCHE }, { "544", Bodies::KALLICHORE },
            { "545", Bodies::HELIKE }, { "546", Bodies::CARPO },
            { "547", Bodies::EUKELADE }, { "548", Bodies::CYLLENE },
            { "549", Bodies::KORE }, { "550", Bodies::HERSE },
            { "553", Bodies::DIA } };
    }
    static BaryStringToEnumT sba2eba() { return {}; }
    static EnumToBodyStringT ebo2sbo()
    {
        return { { Bodies::IO, "Io" }, { Bodies::EUROPA, "Europa" },
            { Bodies::GANYMEDE, "Ganymede" }, { Bodies::CALLISTO, "Callisto" },
            { Bodies::AMALTHEA, "Amalthea" }, { Bodies::HIMALIA, "Himalia" },
            { Bodies::ELARA, "Elara" }, { Bodies::PASIPHAE, "Pasiphae" },
            { Bodies::SINOPE, "Sinope" }, { Bodies::LYSITHEA, "Lysithea" },
            { Bodies::CARME, "Carme" }, { Bodies::ANANKE, "Ananke" },
            { Bodies::LEDA, "Leda" }, { Bodies::THEBE, "Thebe" },
            { Bodies::ADRASTEA, "Adrastea" }, { Bodies::METIS, "Metis" },
            { Bodies::CALLIRRHOE, "Callirrhoe" },
            { Bodies::THEMISTO, "Themisto" },
            { Bodies::MEGACLITE, "Megaclite" }, { Bodies::TAYGETE, "Taygete" },
            { Bodies::CHALDENE, "Chaldene" },
            { Bodies::HARPALYKE, "Harpalyke" }, { Bodies::KALYKE, "Kalyke" },
            { Bodies::IOCASTE, "Iocaste" }, { Bodies::ERINOME, "Erinome" },
            { Bodies::ISONOE, "Isonoe" }, { Bodies::PRAXIDIKE, "Praxidike" },
            { Bodies::AUTONOE, "Autonoe" }, { Bodies::THYONE, "Thyone" },
            { Bodies::HERMIPPE, "Hermippe" }, { Bodies::AITNE, "Aitne" },
            { Bodies::EURYDOME, "Eurydome" }, { Bodies::EUANTHE, "Euanthe" },
            { Bodies::EUPORIE, "Euporie" }, { Bodies::ORTHOSIE, "Orthosie" },
            { Bodies::SPONDE, "Sponde" }, { Bodies::KALE, "Kale" },
            { Bodies::PASITHEE, "Pasithee" }, { Bodies::HEGEMONE, "Hegemone" },
            { Bodies::MNEME, "Mneme" }, { Bodies::AOEDE, "Aoede" },
            { Bodies::THELXINOE, "Thelxinoe" }, { Bodies::ARCHE, "Arche" },
            { Bodies::KALLICHORE, "Kallichore" }, { Bodies::HELIKE, "Helike" },
            { Bodies::CARPO, "Carpo" }, { Bodies::EUKELADE, "Eukelade" },
            { Bodies::CYLLENE, "Cyllene" }, { Bodies::KORE, "Kore" },
            { Bodies::HERSE, "Herse" }, { Bodies::DIA, "Dia" } };
    }
    static EnumToBaryStringT eba2sba() { return {}; }
};

/** @brief Information structure for this system */
struct Info : public detail::Info<MapsT, Info> {
    friend struct detail::Info<MapsT, Info>;

protected:
    /** @brief The main body of this system */
    static constexpr NaifId main_ = 599;
    /** @brief The bary of this system */
    static constexpr NaifId bary_ = 5;
};

} // namespace jupiter
} // namespace schema
} // namespace gravity
} // namespace brie