#pragma once

#include "brie/gravity/schema/detail/Info.h"

namespace brie {
namespace gravity {
namespace schema {
namespace saturn {

/** @brief Counters for the bodies in the system */
enum class Bodies {
    MIMAS,
    ENCELADUS,
    TETHYS,
    DIONE,
    RHEA,
    TITAN,
    HYPERION,
    IAPETUS,
    PHOEBE,
    JANUS,
    EPIMETHEUS,
    HELENE,
    TELESTO,
    CALYPSO,
    ATLAS,
    PROMETHEUS,
    PANDORA,
    PAN,
    YMIR,
    PAALIAQ,
    TARVOS,
    IJIRAQ,
    SUTTUNGR,
    KIVIUQ,
    MUNDILFARI,
    ALBIORIX,
    SKATHI,
    ERRIAPUS,
    SIARNAQ,
    THRYMR,
    NARVI,
    METHONE,
    PALLENE,
    POLYDEUCES,
    DAPHNIS,
    AEGIR,
    BEBHIONN,
    BERGELMIR,
    BESTLA,
    FARBAUTI,
    FENRIR,
    FORNJOT,
    HATI,
    HYRROKKIN,
    KARI,
    LOGE,
    SKOLL,
    SURTUR,
    ANTHE,
    JARNSAXA,
    GREIP,
    TARQEQ,
    AEGAEON,
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
        return { { 601, Bodies::MIMAS }, { 602, Bodies::ENCELADUS },
            { 603, Bodies::TETHYS }, { 604, Bodies::DIONE },
            { 605, Bodies::RHEA }, { 606, Bodies::TITAN },
            { 607, Bodies::HYPERION }, { 608, Bodies::IAPETUS },
            { 609, Bodies::PHOEBE }, { 610, Bodies::JANUS },
            { 611, Bodies::EPIMETHEUS }, { 612, Bodies::HELENE },
            { 613, Bodies::TELESTO }, { 614, Bodies::CALYPSO },
            { 615, Bodies::ATLAS }, { 616, Bodies::PROMETHEUS },
            { 617, Bodies::PANDORA }, { 618, Bodies::PAN },
            { 619, Bodies::YMIR }, { 620, Bodies::PAALIAQ },
            { 621, Bodies::TARVOS }, { 622, Bodies::IJIRAQ },
            { 623, Bodies::SUTTUNGR }, { 624, Bodies::KIVIUQ },
            { 625, Bodies::MUNDILFARI }, { 626, Bodies::ALBIORIX },
            { 627, Bodies::SKATHI }, { 628, Bodies::ERRIAPUS },
            { 629, Bodies::SIARNAQ }, { 630, Bodies::THRYMR },
            { 631, Bodies::NARVI }, { 632, Bodies::METHONE },
            { 633, Bodies::PALLENE }, { 634, Bodies::POLYDEUCES },
            { 635, Bodies::DAPHNIS }, { 636, Bodies::AEGIR },
            { 637, Bodies::BEBHIONN }, { 638, Bodies::BERGELMIR },
            { 639, Bodies::BESTLA }, { 640, Bodies::FARBAUTI },
            { 641, Bodies::FENRIR }, { 642, Bodies::FORNJOT },
            { 643, Bodies::HATI }, { 644, Bodies::HYRROKKIN },
            { 645, Bodies::KARI }, { 646, Bodies::LOGE },
            { 647, Bodies::SKOLL }, { 648, Bodies::SURTUR },
            { 649, Bodies::ANTHE }, { 650, Bodies::JARNSAXA },
            { 651, Bodies::GREIP }, { 652, Bodies::TARQEQ },
            { 653, Bodies::AEGAEON } };
    }
    static BaryIdToEnumT iba2eba() { return {}; }
    static EnumToBodyIdT ebo2ibo()
    {
        return { { Bodies::MIMAS, 601 }, { Bodies::ENCELADUS, 602 },
            { Bodies::TETHYS, 603 }, { Bodies::DIONE, 604 },
            { Bodies::RHEA, 605 }, { Bodies::TITAN, 606 },
            { Bodies::HYPERION, 607 }, { Bodies::IAPETUS, 608 },
            { Bodies::PHOEBE, 609 }, { Bodies::JANUS, 610 },
            { Bodies::EPIMETHEUS, 611 }, { Bodies::HELENE, 612 },
            { Bodies::TELESTO, 613 }, { Bodies::CALYPSO, 614 },
            { Bodies::ATLAS, 615 }, { Bodies::PROMETHEUS, 616 },
            { Bodies::PANDORA, 617 }, { Bodies::PAN, 618 },
            { Bodies::YMIR, 619 }, { Bodies::PAALIAQ, 620 },
            { Bodies::TARVOS, 621 }, { Bodies::IJIRAQ, 622 },
            { Bodies::SUTTUNGR, 623 }, { Bodies::KIVIUQ, 624 },
            { Bodies::MUNDILFARI, 625 }, { Bodies::ALBIORIX, 626 },
            { Bodies::SKATHI, 627 }, { Bodies::ERRIAPUS, 628 },
            { Bodies::SIARNAQ, 629 }, { Bodies::THRYMR, 630 },
            { Bodies::NARVI, 631 }, { Bodies::METHONE, 632 },
            { Bodies::PALLENE, 633 }, { Bodies::POLYDEUCES, 634 },
            { Bodies::DAPHNIS, 635 }, { Bodies::AEGIR, 636 },
            { Bodies::BEBHIONN, 637 }, { Bodies::BERGELMIR, 638 },
            { Bodies::BESTLA, 639 }, { Bodies::FARBAUTI, 640 },
            { Bodies::FENRIR, 641 }, { Bodies::FORNJOT, 642 },
            { Bodies::HATI, 643 }, { Bodies::HYRROKKIN, 644 },
            { Bodies::KARI, 645 }, { Bodies::LOGE, 646 },
            { Bodies::SKOLL, 647 }, { Bodies::SURTUR, 648 },
            { Bodies::ANTHE, 649 }, { Bodies::JARNSAXA, 650 },
            { Bodies::GREIP, 651 }, { Bodies::TARQEQ, 652 },
            { Bodies::AEGAEON, 653 } };
    }
    static EnumToBaryIdT eba2iba() { return {}; }
    static BodyStringToEnumT sbo2ebo()
    {
        return { { "mimas", Bodies::MIMAS }, { "enceladus", Bodies::ENCELADUS },
            { "tethys", Bodies::TETHYS }, { "dione", Bodies::DIONE },
            { "rhea", Bodies::RHEA }, { "titan", Bodies::TITAN },
            { "hyperion", Bodies::HYPERION }, { "iapetus", Bodies::IAPETUS },
            { "phoebe", Bodies::PHOEBE }, { "janus", Bodies::JANUS },
            { "epimetheus", Bodies::EPIMETHEUS }, { "helene", Bodies::HELENE },
            { "telesto", Bodies::TELESTO }, { "calypso", Bodies::CALYPSO },
            { "atlas", Bodies::ATLAS }, { "prometheus", Bodies::PROMETHEUS },
            { "pandora", Bodies::PANDORA }, { "pan", Bodies::PAN },
            { "ymir", Bodies::YMIR }, { "paaliaq", Bodies::PAALIAQ },
            { "tarvos", Bodies::TARVOS }, { "ijiraq", Bodies::IJIRAQ },
            { "suttungr", Bodies::SUTTUNGR }, { "kiviuq", Bodies::KIVIUQ },
            { "mundilfari", Bodies::MUNDILFARI },
            { "albiorix", Bodies::ALBIORIX }, { "skathi", Bodies::SKATHI },
            { "erriapus", Bodies::ERRIAPUS }, { "siarnaq", Bodies::SIARNAQ },
            { "thrymr", Bodies::THRYMR }, { "narvi", Bodies::NARVI },
            { "methone", Bodies::METHONE }, { "pallene", Bodies::PALLENE },
            { "polydeuces", Bodies::POLYDEUCES },
            { "daphnis", Bodies::DAPHNIS }, { "aegir", Bodies::AEGIR },
            { "bebhionn", Bodies::BEBHIONN },
            { "bergelmir", Bodies::BERGELMIR }, { "bestla", Bodies::BESTLA },
            { "farbauti", Bodies::FARBAUTI }, { "fenrir", Bodies::FENRIR },
            { "fornjot", Bodies::FORNJOT }, { "hati", Bodies::HATI },
            { "hyrrokkin", Bodies::HYRROKKIN }, { "kari", Bodies::KARI },
            { "loge", Bodies::LOGE }, { "skoll", Bodies::SKOLL },
            { "surtur", Bodies::SURTUR }, { "anthe", Bodies::ANTHE },
            { "jarnsaxa", Bodies::JARNSAXA }, { "greip", Bodies::GREIP },
            { "tarqeq", Bodies::TARQEQ }, { "aegaeon", Bodies::AEGAEON },
            { "601", Bodies::MIMAS }, { "602", Bodies::ENCELADUS },
            { "603", Bodies::TETHYS }, { "604", Bodies::DIONE },
            { "605", Bodies::RHEA }, { "606", Bodies::TITAN },
            { "607", Bodies::HYPERION }, { "608", Bodies::IAPETUS },
            { "609", Bodies::PHOEBE }, { "610", Bodies::JANUS },
            { "611", Bodies::EPIMETHEUS }, { "612", Bodies::HELENE },
            { "613", Bodies::TELESTO }, { "614", Bodies::CALYPSO },
            { "615", Bodies::ATLAS }, { "616", Bodies::PROMETHEUS },
            { "617", Bodies::PANDORA }, { "618", Bodies::PAN },
            { "619", Bodies::YMIR }, { "620", Bodies::PAALIAQ },
            { "621", Bodies::TARVOS }, { "622", Bodies::IJIRAQ },
            { "623", Bodies::SUTTUNGR }, { "624", Bodies::KIVIUQ },
            { "625", Bodies::MUNDILFARI }, { "626", Bodies::ALBIORIX },
            { "627", Bodies::SKATHI }, { "628", Bodies::ERRIAPUS },
            { "629", Bodies::SIARNAQ }, { "630", Bodies::THRYMR },
            { "631", Bodies::NARVI }, { "632", Bodies::METHONE },
            { "633", Bodies::PALLENE }, { "634", Bodies::POLYDEUCES },
            { "635", Bodies::DAPHNIS }, { "636", Bodies::AEGIR },
            { "637", Bodies::BEBHIONN }, { "638", Bodies::BERGELMIR },
            { "639", Bodies::BESTLA }, { "640", Bodies::FARBAUTI },
            { "641", Bodies::FENRIR }, { "642", Bodies::FORNJOT },
            { "643", Bodies::HATI }, { "644", Bodies::HYRROKKIN },
            { "645", Bodies::KARI }, { "646", Bodies::LOGE },
            { "647", Bodies::SKOLL }, { "648", Bodies::SURTUR },
            { "649", Bodies::ANTHE }, { "650", Bodies::JARNSAXA },
            { "651", Bodies::GREIP }, { "652", Bodies::TARQEQ },
            { "653", Bodies::AEGAEON } };
    }
    static BaryStringToEnumT sba2eba() { return {}; }
    static EnumToBodyStringT ebo2sbo()
    {
        return { { Bodies::MIMAS, "Mimas" }, { Bodies::ENCELADUS, "Enceladus" },
            { Bodies::TETHYS, "Tethys" }, { Bodies::DIONE, "Dione" },
            { Bodies::RHEA, "Rhea" }, { Bodies::TITAN, "Titan" },
            { Bodies::HYPERION, "Hyperion" }, { Bodies::IAPETUS, "Iapetus" },
            { Bodies::PHOEBE, "Phoebe" }, { Bodies::JANUS, "Janus" },
            { Bodies::EPIMETHEUS, "Epimetheus" }, { Bodies::HELENE, "Helene" },
            { Bodies::TELESTO, "Telesto" }, { Bodies::CALYPSO, "Calypso" },
            { Bodies::ATLAS, "Atlas" }, { Bodies::PROMETHEUS, "Prometheus" },
            { Bodies::PANDORA, "Pandora" }, { Bodies::PAN, "Pan" },
            { Bodies::YMIR, "Ymir" }, { Bodies::PAALIAQ, "Paaliaq" },
            { Bodies::TARVOS, "Tarvos" }, { Bodies::IJIRAQ, "Ijiraq" },
            { Bodies::SUTTUNGR, "Suttungr" }, { Bodies::KIVIUQ, "Kiviuq" },
            { Bodies::MUNDILFARI, "Mundilfari" },
            { Bodies::ALBIORIX, "Albiorix" }, { Bodies::SKATHI, "Skathi" },
            { Bodies::ERRIAPUS, "Erriapus" }, { Bodies::SIARNAQ, "Siarnaq" },
            { Bodies::THRYMR, "Thrymr" }, { Bodies::NARVI, "Narvi" },
            { Bodies::METHONE, "Methone" }, { Bodies::PALLENE, "Pallene" },
            { Bodies::POLYDEUCES, "Polydeuces" },
            { Bodies::DAPHNIS, "Daphnis" }, { Bodies::AEGIR, "Aegir" },
            { Bodies::BEBHIONN, "Bebhionn" },
            { Bodies::BERGELMIR, "Bergelmir" }, { Bodies::BESTLA, "Bestla" },
            { Bodies::FARBAUTI, "Farbauti" }, { Bodies::FENRIR, "Fenrir" },
            { Bodies::FORNJOT, "Fornjot" }, { Bodies::HATI, "Hati" },
            { Bodies::HYRROKKIN, "Hyrrokkin" }, { Bodies::KARI, "Kari" },
            { Bodies::LOGE, "Loge" }, { Bodies::SKOLL, "Skoll" },
            { Bodies::SURTUR, "Surtur" }, { Bodies::ANTHE, "Anthe" },
            { Bodies::JARNSAXA, "Jarnsaxa" }, { Bodies::GREIP, "Greip" },
            { Bodies::TARQEQ, "Tarqeq" }, { Bodies::AEGAEON, "Aegaeon" } };
    }
    static EnumToBaryStringT eba2sba() { return {}; }
};

/** @brief Information structure for this system */
struct Info : public detail::Info<MapsT, Info> {
    friend struct detail::Info<MapsT, Info>;

protected:
    /** @brief The main body of this system */
    static constexpr NaifId main_ = 699;
    /** @brief The bary of this system */
    static constexpr NaifId bary_ = 6;
};

} // namespace saturn
} // namespace schema
} // namespace gravity
} // namespace brie