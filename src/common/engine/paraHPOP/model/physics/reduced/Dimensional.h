/* Dimensional version of the physical model, that does not include events and
 * interactions */
#pragma once

#include "paraHPOP/model/physics/reduced/RefDimensional.h"
#include "interface/config/Model.h"

namespace paraHPOP {
namespace model {
namespace physics {
namespace reduced {

/**
 * @brief The Dimensional Physics
 *
 */
class Dimensional {
    using Self       = Dimensional;
    using InterfaceT = interface::config::Model;

public:
    /** @brief member types */
    using EnvT = environment::Env;
    using AccT = accelerations::Accelerations;
    /** @brief reference types */
    template<bool work, bool MaybeVolatile = false>
    using Ref = RefDimensional<work, MaybeVolatile>;
    /** @brief global reference type */
    using GRef = Ref<false>;
    /** @brief work reference type */
    using WRef = Ref<true>;
    /** @brief Volatile work reference */
    using VolatileRef = Ref<true, true>;
    /** @brief states type */
    using StatesT    = samples::states::cartesian::States;
    using EpochsT    = feta::scalar::Array<StatesT::ComponentT>;
    using BoolArrayT = feta::scalar::Array<bool>;

    /** @brief Constexpr to mark if this physical model is non dimensional */
    static constexpr bool IsNondimensional = false;

    /** @brief Default constructor is forbidden */
    Dimensional() = delete;

    /** @brief Copy constructor is forbidden */
    Dimensional(Dimensional& other) = delete;

    /** @brief Construct from Interface */
    Dimensional(const InterfaceT& icfg)
        : env_{ icfg.environment() }
        , accs_{ icfg.accelerations() }
    {
        /* Assert that orientations object is defined if either oblateness or
         * Spherical Harmonics is active */
        namespace accsrc = interface::config::model::accelerations;
        const auto bref  = env_.bodies().hostRef();

        if (accs_.template anyActive<accsrc::SHAPE>()
            || accs_.template anyActive<accsrc::OBLATENESS>()) {
            PARAHPOP_ASSERT(!icfg.environment().orientations().files().empty(),
                "Active spherical harmonics and/or J2 requires defined "
                "body-fixed orientations, and no `.brot` (pck-like) file was found.");
        }

        for (idx_t i = 0; i < bref.size(); i++) {
            /* if any acceleration source is active, body must be found in
             * ephemeris */
            if (accs_.anyActive(i)) {
                bool found
                    = env_.ephemeris().metadata().hostRef().hasBody(bref[i]);
                if (!found) {
                    std::stringstream ss;
                    ss << "Active accelerations for "
                       << brie::gravity::Parser::parsedName(bref[i]) << " ("
                       << bref[i]
                       << ") require ephemeris data for the body, but it "
                          "was not found among the ephemeris metadata.";
                    PARAHPOP_THROW(std::runtime_error, ss.str().c_str());
                }
                /* Active gravity reads constants().body(id).gm() every step;
                 * a missing entry is a silent out-of-bounds read on device
                 * (release builds elide the device-side throw).  Require the
                 * entry here — presence only, since a massless point carries
                 * gm == 0. */
                try {
                    (void)env_.constants().hostRef().body(bref[i]);
                } catch (const std::exception&) {
                    std::stringstream cs;
                    cs << "Active accelerations for "
                       << brie::gravity::Parser::parsedName(bref[i]) << " ("
                       << bref[i]
                       << ") require a GM/constants entry for the body, but "
                          "none was found in the constants set. Provide a "
                          "constants file, or declare the body's gm inline.";
                    PARAHPOP_THROW(std::runtime_error, cs.str().c_str());
                }
            }
            /* if shape or oblateness is active, body must have defined
             * orientations */
            if (accs_.template active<accsrc::SHAPE>(i)
                || accs_.template active<accsrc::OBLATENESS>(i)) {
                /* Validate the resolved rotation target (auto-best or
                 * explicit), not the bare body id — so an env that loads
                 * only a body's high-precision frame (e.g. ITRF93 target
                 * 3000) passes while still catching genuinely missing data. */
                const NaifId rotTarget = env_.orientationTarget(i);
                bool found
                    = env_.rotations().metadata().hostRef().hasBody(rotTarget);
                if (!found) {
                    std::stringstream ss;
                    ss << "Active spherical harmonics and/or J2 for "
                       << brie::gravity::Parser::parsedName(bref[i]) << " ("
                       << bref[i] << ") rotate under orientation target "
                       << rotTarget
                       << ", which has no body-fixed orientation data among "
                          "the orientations metadata.";
                    PARAHPOP_THROW(std::runtime_error, ss.str().c_str());
                }
            }
        }

        /* Occultation prerequisites: when an occulting body can shadow
         * an active radiation source, the dual-cone geometry needs a
         * physical radius for the Sun and for every occulter.  Checked
         * here at model build (never inside kernels) so a missing
         * constants entry fails with an actionable message. */
        if (accs_.template anyActive<accsrc::RADIATION>()) {
            constexpr NaifId SUN_NAIF_ID = 10;
            const auto cref              = env_.constants().hostRef();
            for (idx_t i = 0; i < bref.size(); i++) {
                if (!accs_.template active<accsrc::OCCULTING>(i)
                    || bref[i] == SUN_NAIF_ID)
                    continue;
                PARAHPOP_ASSERT(cref.body(bref[i]).r() > 0,
                    "Occulting body "
                        + brie::gravity::Parser::parsedName(bref[i])
                        + " has no physical radius in the constants set; "
                          "the dual-cone occultation geometry requires one.");
                PARAHPOP_ASSERT(cref.body(SUN_NAIF_ID).r() > 0,
                    "Occultation with an active radiation source requires "
                    "the Sun's physical radius in the constants set.");
            }
        }
    }

    /** @brief Move constructor from data members without non-dimensionalization
     */
    Dimensional(EnvT&& env, AccT&& accs)
        : env_{ std::move(env) }
        , accs_{ std::move(accs) }
    {
    }

    /** @brief Move constructor */
    Dimensional(Dimensional&& other)
        : env_{ std::move(other.env_) }
        , accs_{ std::move(other.accs_) }
    {
    }

    /** @brief Move assignment operator */
    Dimensional& operator=(Dimensional&& other)
    {
        this->env_  = std::move(other.env_);
        this->accs_ = std::move(other.accs_);
        return *this;
    }

    /** @brief Async memcpy from host to device */
    void upload(const cudaStream_t& stream = 0)
    {
        env_.upload(stream);
        accs_.upload(stream);
    }

    /** @brief Async memcpy from device to host */
    void download(const cudaStream_t& stream = 0)
    {
        env_.download(stream);
        accs_.download(stream);
    }

    /** @brief Clear data from the GPU */
    void clearDevice()
    {
        env_.clearDevice();
        accs_.clearDevice();
    }

    /** @brief Return host reference */
    GRef hostRef() const { return GRef::make(env_.hostRef(), accs_.hostRef()); }

    /** @brief Return device reference */
    GRef deviceRef() const
    {
        return GRef::make(env_.deviceRef(), accs_.deviceRef());
    }

    /** @brief Expose environment */
    EnvT& env() { return env_; }
    const EnvT& env() const { return env_; }

    /** @brief Expose accelerations */
    AccT& accs() { return accs_; }
    const AccT& accs() const { return accs_; }

    /** @brief Estimated per-sample working-set bytes for the host
     *  integration step.  Drives the L2-aligned tile size used by
     *  every host kernel in a step.
     *
     *  Components:
     *  - Native eph cache (Vec6R) for every body unit in the
     *    ephemeris traverser
     *  - Rotation cache (Vec4R) for every active SH/J2 body
     *  - States + dStates: 2 × Vec6R
     *  - Acceleration scratch + per-sample handle reads (~48 B) */
    inline idx_t hostBytesPerSample() const
    {
        namespace accsrc = interface::config::model::accelerations;
        const auto bref  = env_.bodies().hostRef();
        idx_t nRotBodies = 0;
        for (idx_t i = 0; i < bref.size(); ++i) {
            if (accs_.template active<accsrc::SHAPE>(i)
                || accs_.template active<accsrc::OBLATENESS>(i))
                ++nRotBodies;
        }
        const idx_t nNative = env_.ephemeris().nBodyUnits();
        return nNative * idx_t{ sizeof(Vec6R) }
            + nRotBodies * idx_t{ sizeof(feta::vector::Item<Real, 4>) }
            + idx_t{ 2 } * idx_t{ sizeof(Vec6R) }
            + idx_t{ 48 };
    }

    /** @brief Run the evaluation */
    template<typename MetadataT>
    void eval(StatesT::GRef& dStates, const StatesT::GRef& states,
        const typename EpochsT::GRef& epochs,
        const typename BoolArrayT::GRef::HandleT& terminated,
        const MetadataT& metadata, const cudaStream_t& stream) const;
    template<idx_t stage, typename MetadataT>
    void eval(parm::util::graph::Graph& graph, StatesT::GRef& dStates,
        const StatesT::GRef& states, const typename EpochsT::GRef& epochs,
        const typename BoolArrayT::GRef::HandleT& terminated,
        const MetadataT& metadata, const cudaStream_t& stream = 0) const;

protected:
    EnvT env_;
    AccT accs_;
};

} // namespace reduced
} // namespace physics
} // namespace model
} // namespace paraHPOP