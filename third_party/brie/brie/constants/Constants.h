#pragma once

#include "brie/constants/ConstantsJson.h"
#include "brie/constants/RefConstants.h"
#include "brie/util.h"

#include <optional>

namespace brie {
/**
 * @brief Memory manager for universal and per-body physical constants.
 *
 * It is used to import constants from JSON files. Imported values can be read
 * via `hostRef`/`deviceRef`.
 */
class Constants {
public:
#ifndef BRIE_CPU_ONLY
    using StreamT = cudaStream_t;
#else
    using StreamT = int;
#endif
    /** @brief Generic work or global reference type.  ``MaybeVolatile``
     * is accepted (ignored) — RefConstants is work-invariant. */
    template<bool work, bool MaybeVolatile = false>
    using Ref = detail::RefConstants;
    /** @brief Global reference type */
    using GRef = detail::RefConstants;
    /** @brief Work reference type */
    using WRef = detail::RefConstants;
    /** @brief Volatile work reference (alias for symmetry — no actual
     * volatile pointer involved since RefConstants is work-invariant). */
    using VolatileRef = detail::RefConstants;

    /** @brief Factory method. Return an empty object */
    static Constants empty()
    {
        feta::scalar::Array<NaifId> ids(0);
        feta::scalar::Array<Real> vals(0);
        return Constants(std::move(ids), std::move(vals));
    }

    /** @brief Factory method. Read constants from a JSON file. */
    static Constants fromFile(const std::string& file,
        const util::Paths& paths = util::paths::brieDefaultPath());

    /** @brief Factory method. Read constants from multiple JSON files. */
    static Constants fromFiles(const std::vector<std::string>& files,
        const util::Paths& paths = util::paths::brieDefaultPath());

    /**
     * @brief Factory method. Return a new Constants combining @p base with the
     * per-body @p extra constants.
     *
     * Universal constants (AU, CLIGHT) are taken from @p uniOverride when it is
     * set, otherwise from @p base. Supplying @p uniOverride is the programmatic
     * way to edit the universal constants, or to supply them when @p base is
     * `empty()` (a fileless declaration). A NAIF id present in both @p base and
     * @p extra must carry identical physical values — the @p extra entry is then
     * a no-op; a @b conflicting entry for the same id throws. Ids absent from
     * @p base are appended. This is the programmatic injection path used to
     * register bodies declared outside the constants files (e.g. massless points
     * carrying a radius) without editing those files.
     *
     * @throws `std::runtime_error` on a conflicting redefinition of an existing
     * body id.
     */
    static Constants merged(const Constants& base,
        const std::vector<detail::BodyConstants>& extra,
        const std::optional<detail::UniversalConstants>& uniOverride
        = std::nullopt)
    {
        detail::UniversalConstants uni{ 0.0, 0.0 };
        std::vector<detail::BodyConstants> bodies;

        /* Read the base only when it actually holds data: a default/empty
         * Constants has unallocated arrays whose data() would throw.
         * (`values_` always carries the universal constants slots once any
         * file is loaded.) Merging into `empty()` is supported and yields a
         * constants set of only the extra bodies. */
        if (base.values_.size() >= detail::UNI_CONSTANT_COUNT) {
            detail::RefConstants ref = base.hostRef();
            uni.au                   = ref.au();
            uni.clight               = ref.cLight();
            const idx_t nBase        = base.bodyIds_.size();
            bodies.reserve(static_cast<size_t>(nBase) + extra.size());
            for (idx_t i = 0; i < nBase; i++) {
                detail::BodyConstants b;
                b.naifId = ref.bodyIds_[i];
                b.gm     = ref.bodyDataView_.template get<detail::GM>(i);
                b.r      = ref.bodyDataView_.template get<detail::R>(i);
                b.aMean  = ref.bodyDataView_.template get<detail::AMEAN>(i);
                b.j2     = ref.bodyDataView_.template get<detail::J2>(i);
                b.soi    = ref.bodyDataView_.template get<detail::SOI>(i);
                bodies.push_back(b);
            }
        }

        /* Explicit universals win over the base (or the zero default for an
         * empty base): this is the editable-universals / fileless path. */
        if (uniOverride)
            uni = *uniOverride;

        for (const detail::BodyConstants& e : extra) {
            idx_t found = static_cast<idx_t>(bodies.size());
            for (idx_t i = 0; i < static_cast<idx_t>(bodies.size()); i++) {
                if (bodies[i].naifId == e.naifId) {
                    found = i;
                    break;
                }
            }
            if (found < static_cast<idx_t>(bodies.size())) {
                const detail::BodyConstants& b = bodies[found];
                const bool identical = b.gm == e.gm && b.r == e.r
                    && b.aMean == e.aMean && b.j2 == e.j2 && b.soi == e.soi;
                if (!identical) {
                    std::stringstream ss;
                    ss << "Constants::merged: conflicting physical constants "
                          "for NAIF id "
                       << e.naifId
                       << " (already defined with different values)";
                    BRIE_THROW(std::runtime_error, ss.str());
                }
                /* identical re-declaration -> no-op */
            } else {
                bodies.push_back(e);
            }
        }

        return Constants(uni, bodies);
    }

    /** @brief Constructor is deleted. Use factory methods instead. */
    Constants() = delete;

    /** @brief Move constructor */
    Constants(Constants&& other)
        : bodyIds_{ std::move(other.bodyIds_) }
        , values_{ std::move(other.values_) }
    {
    }

    /** @brief Move constructor from data members */
    Constants(
        feta::scalar::Array<NaifId>&& ids, feta::scalar::Array<Real>&& values)
        : bodyIds_{ std::move(ids) }
        , values_{ std::move(values) }
    {
    }

    /** @brief Copy assignment is forbidden */
    Constants& operator=(Constants& other)       = delete;
    Constants& operator=(const Constants& other) = delete;

    /** @brief Move assignment */
    Constants& operator=(Constants&& other)
    {
        bodyIds_ = std::move(other.bodyIds_);
        values_  = std::move(other.values_);
        return *this;
    }

#ifndef BRIE_CPU_ONLY
    /**
     * @brief Async copy data from host to device.
     *
     */
    void upload(const StreamT& stream = 0)
    {
        bodyIds_.upload(stream);
        values_.upload(stream);
    }

    /**
     * @brief Async copy data from device to host
     *
     */
    void download(const StreamT& stream = 0)
    {
        bodyIds_.download(stream);
        values_.download(stream);
    }

    /** @brief Clear data from the GPU */
    void clearDevice()
    {
        bodyIds_.clearDevice();
        values_.clearDevice();
    }
#endif

    /** @brief Return a RefPhysConst pointing to the host data. A
     * universals-only set has a size-0 (unallocated) bodyIds_; substitute an
     * empty ref (size 0 -> body lookups correctly find nothing). */
    GRef hostRef() const
    {
        using BIdRef = feta::scalar::Array<NaifId>::GRef;
        return GRef::make(
            bodyIds_.size() == 0 ? BIdRef{} : bodyIds_.hostRef(),
            values_.hostRef());
    }
    GRef ref() const { return hostRef(); }

#ifndef BRIE_CPU_ONLY
    /** @brief Return a RefPhysConst pointing to the device data. */
    GRef deviceRef() const
    {
        using BIdRef = feta::scalar::Array<NaifId>::GRef;
        return GRef::make(
            bodyIds_.size() == 0 ? BIdRef{} : bodyIds_.deviceRef(),
            values_.deviceRef());
    }
#endif

    /** @brief Clone this object. */
    Constants clone() const
    {
        return Constants(
            std::move(bodyIds_.clone()), std::move(values_.clone()));
    }

private:
    /** @brief Private constructor */
    Constants(const detail::UniversalConstants& uni,
        const std::vector<detail::BodyConstants>& bodies)
        : bodyIds_{ static_cast<idx_t>(bodies.size()) }
        , values_{ detail::UNI_CONSTANT_COUNT
            + static_cast<idx_t>(bodies.size()) * detail::BODY_CONSTANT_COUNT }
    {
        feta::scalar::Array<Real>::GRef vref = values_.hostRef();
        vref[detail::AU]                     = uni.au;
        vref[detail::CLIGHT]                 = uni.clight;

        const idx_t nBodies = bodyIds_.size();
        /* A universals-only set has a size-0 bodyIds_ array, which is
         * unallocated — hostRef() would throw. Nothing more to write. */
        if (nBodies == 0)
            return;

        feta::scalar::Array<NaifId>::GRef bref = bodyIds_.hostRef();
        const auto ucc                         = detail::UNI_CONSTANT_COUNT;
        using BIT           = detail::RefBodyConstants::InnerT;
        BIT bodyData;
        {
            bodyData.data_      = values_.getHostData() + ucc;
            bodyData.nVecs_     = nBodies;
            bodyData.dimOffset_ = nBodies;
        }
        for (idx_t i = 0; i < bodies.size(); i++) {
            bref[i]                        = bodies[i].naifId;
            bodyData.get<detail::R>(i)     = bodies[i].r;
            bodyData.get<detail::GM>(i)    = bodies[i].gm;
            bodyData.get<detail::AMEAN>(i) = bodies[i].aMean;
            bodyData.get<detail::J2>(i)    = bodies[i].j2;
            bodyData.get<detail::SOI>(i)   = bodies[i].soi;
        }
    }

    feta::scalar::Array<NaifId> bodyIds_;
    feta::scalar::Array<Real> values_;
};


} // namespace brie