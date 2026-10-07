#pragma once

#include "paraHPOP/IntegrationTableau.h"
#include "paraHPOP/model/accelerations/AccumulateBodies.h"
#include "paraHPOP/model/accelerations/Drag.h"
#include "paraHPOP/model/accelerations/PrivateAccelerations.h"
#include "paraHPOP/model/environment/eclipse/Occultation.h"
#include "paraHPOP/model/physics/reduced/Dimensional.h"
#include "parm/util/graph/ComputeBlocks.h"

namespace paraHPOP {
namespace model {
namespace physics {
namespace reduced {

// Shared launch-size policy.
using parm::util::graph::computeBlocks;

// The final stage has the endpoint epoch, enabling inter-step cache reuse.
inline constexpr idx_t kFinalRkStage = IntegrationTableau::nStages() - 1;
static_assert(
    IntegrationTableau::c<IntegrationTableau::nStages() - 2>() == Real(1.0),
    "Environment-cache reuse requires the final stage epoch to be t + h");

/** @brief Capture one stream-launched kernel as a graph node and return its
 *  index.  Folds the ``begin → launch → addNode(end, deps) → lastNode``
 *  idiom that every per-body acceleration sub-chain in the graph
 *  ``Dimensional::eval`` repeats.  @p launch is a thunk issuing exactly one
 *  ``<<<>>>`` launch on the captured @p stream; @p deps is the node's
 *  dependency list (a single ``{ anchor }`` brace-init or a multi-dependency
 *  ``std::vector``).  Node topology and launch parameters are identical to
 *  the hand-rolled form — purely structural. */
template<typename LaunchFn>
inline idx_t captureNode_(parm::util::graph::Graph& graph,
    parm::util::graph::StreamCapturer& capturer, LaunchFn&& launch,
    const std::vector<idx_t>& deps)
{
    capturer.begin();
    launch();
    graph.addNode(capturer.end(), deps);
    return graph.lastNode();
}

template<typename MetadataT>
void Dimensional::eval(StatesT::GRef& dStates, const StatesT::GRef& states,
    const typename EpochsT::GRef& epochs,
    const typename BoolArrayT::GRef::HandleT& terminated,
    const MetadataT& metadata, const cudaStream_t& stream) const
{
    // Keep one device implementation. Standalone calls refresh the variable
    // cache; the fixed-step propagator captures/reuses the graph directly.
    parm::util::graph::Graph graph;
    graph.stream(stream);
    parm::util::graph::StreamCapturer capturer(stream);
    idx_t blocks, threads;
    computeBlocks(dStates.size(), blocks, threads,
        accelerations::kernel::AccumulateBodiesLaunch::maxBlockSize);
    captureNode_(graph, capturer, [&] {
        accelerations::kernel::zeroBodyAccScratch<<<blocks, threads, 0, stream>>>(
            dStates.template subset<3, 3>(), terminated);
    }, {});
    this->template eval<1>(graph, dStates, states, epochs, terminated, metadata, stream);
    auto launcher = graph.launcher();
    launcher.launch();
    PARM_CHECK(cudaStreamSynchronize(stream));
}

template<idx_t stage, typename MetadataT>
void Dimensional::eval(parm::util::graph::Graph& graph, StatesT::GRef& dStates,
    const StatesT::GRef& states, const typename EpochsT::GRef& epochs,
    const typename BoolArrayT::GRef::HandleT& terminated,
    const MetadataT& metadata, const cudaStream_t& stream) const
{
    /* dStates tail is the accelerations */
    using StateArrT = feta::vector::Array<Real, 6>::GRef;
    using VecArrT   = feta::vector::Array<Real, 3>::GRef;

    /* Acceleration array */
    VecArrT accarray = dStates.template subset<3, 3>();

    /* Position sub-component from the states */
    VecArrT pos = states.template subset<0, 3>();

    /* Velocity sub-component from the states (drag needs v_sc) */
    VecArrT vel = states.template subset<3, 3>();

    /* Environment reference - to extract graph assemble-time const values
     */
    EnvT::GRef eref = env_.hostRef();

    /* Core evaluation */
    NaifIdArray::GRef bodies = env_.bodies().hostRef();

    /* Per-body sub-chains fan out from the input barrier; their
     * terminals join in a single addEmptyNode(bodyTerminals) below. */
    const idx_t initialNode = graph.lastNode();
    std::vector<idx_t> bodyTerminals;
    bodyTerminals.reserve(bodies.size() + 1);

    // Kernel launch/math options are retained; topology is always per-body.
    const auto options = accelerations::kernel::forceKernelOptions(dStates.size());

    parm::util::graph::StreamCapturer capturer(stream);
    namespace acckernels = paraHPOP::model::accelerations::kernel;
    const auto correctionModel = env_.deviceRef().earthCorrections();
    const auto correctionFlags = env_.earthCorrections().flags;
    const VecArrT solidSI = env_.correctionScratchSI(0);
    const VecArrT oceanSI = env_.correctionScratchSI(1);
    const VecArrT relativitySI = env_.correctionScratchSI(2);
    if (correctionFlags.relativity) {
        idx_t blocks, threads;
        computeBlocks(dStates.size(), blocks, threads,
            acckernels::EarthCorrectionsLaunch::maxBlockSize);
        bodyTerminals.push_back(captureNode_(graph, capturer, [&] {
            acckernels::relativisticCorrection<<<blocks, threads, 0, stream>>>(
                relativitySI, pos, vel, terminated);
        }, { initialNode }));
    }

    // Stage zero reuses the previous endpoint cache. Subsequent stages
    // refresh force-model dependencies; the last stage prepares all slots
    // that will be promoted to the next step's initial cache.
    idx_t cacheBarrier = initialNode;
    if constexpr (stage != 0) {
        const idx_t nNative = env_.ephNativeVariable().nBodyUnits();
        std::vector<idx_t> tier1Terminals;
        tier1Terminals.reserve(nNative);

        for (idx_t b = 0; b < nNative; b++) {
            if constexpr (stage != kFinalRkStage) {
                // Skip slots not read by force evaluation at intermediate stages.
                if (env_.nativeSlotUnused(b))
                    continue;
            }
            tier1Terminals.push_back(captureNode_(graph, capturer,
                [&] { env_.ephNativeVariable().fillBody(b, epochs, stream); },
                { initialNode }));
        }

        if (tier1Terminals.empty())
            tier1Terminals.push_back(initialNode);
        graph.addEmptyNode(tier1Terminals);
        cacheBarrier = graph.lastNode();
    }

    namespace accsrc     = paraHPOP::model::accelerations::src;
    /* Per-body block geometry for the zero / reduce kernels. */
    idx_t accBlocks, accBlockSize;
    computeBlocks(dStates.size(), accBlocks, accBlockSize,
        acckernels::AccumulateBodiesLaunch::maxBlockSize);

    /* Occultation prelude: SRP attenuation is scene-level N→1 — every
     * `occulting`-flagged body (the Sun excluded) attenuates the single
     * Sun radiation source into ONE combined per-sample factor,
     * precomputed by the occultation node below and consumed by the
     * `srpShadow` launch in the Sun's chain.  Independent of the
     * eclipse *event* (1:1:1), which never reads the occulting flag. */
    constexpr NaifId SUN_NAIF_ID = 10;
    idx_t radiationIdx           = bodies.size();
    std::vector<idx_t> occulters;
    std::vector<char> isOcculter(bodies.size(), 0);
    for (idx_t i = 0; i < bodies.size(); i++) {
        if (accs_.template active<accsrc::RADIATION>(i))
            radiationIdx = i; /* the Sun — radiation is validated Sun-only */
        if (accs_.template active<accsrc::OCCULTING>(i)
            && bodies[i] != SUN_NAIF_ID) {
            occulters.push_back(i);
            isOcculter[i] = 1;
        }
    }
    const bool anyShadow = !occulters.empty() && radiationIdx < bodies.size();

    /* Tier 2 (stage > 0): COI resolve from the native cache for every
     * body that needs a resolved slot this stage — force-active bodies
     * plus (when shadowing is on) occulters that may carry no force of
     * their own.  Hoisted out of the force loop so the occultation node
     * can depend on the Sun's and every occulter's resolve regardless
     * of body iteration order.  Per-body topology is unchanged (same
     * kernel, same `{ cacheBarrier }` anchor); only the emission order
     * moved out of the loop. */
    std::vector<idx_t> resolveNode(bodies.size(), cacheBarrier);
    if constexpr (stage != 0) {
        idx_t nBlocks, blockSize;
        computeBlocks(dStates.size(), nBlocks, blockSize,
            environment::kernel::EnvKernelLaunch::maxBlockSize);
        for (idx_t i = 0; i < bodies.size(); i++) {
            if (!accs_.anyForceActive(i) && !(anyShadow && isOcculter[i])
                && !env_.needsTidePosition(i))
                continue;
            /* Atmosphere-active bodies use the fused state kernel (one
             * Dim=6 native-cache pass writing both ephVariable and
             * velVariable); pos-only bodies the lighter pos-only
             * resolve. */
            resolveNode[i] = captureNode_(graph, capturer, [&] {
                if (accs_.template active<accsrc::ATMOSPHERE>(i)) {
                    environment::kernel::coiResolveStateKernel<<<nBlocks,
                        blockSize, 0, stream>>>(env_.ephVariable(i),
                        env_.velVariable(i), env_.ephNativeVariable().ref(),
                        bodies[i], metadata.own().cois(), epochs.handle(),
                        terminated);
                } else {
                    environment::kernel::coiResolveKernel<<<nBlocks, blockSize,
                        0, stream>>>(env_.ephVariable(i),
                        env_.ephNativeVariable().ref(), bodies[i],
                        metadata.own().cois(), epochs.handle(), terminated);
                }
            }, { cacheBarrier });
        }
    }

    /* Occultation node: writes the combined per-sample factor cache
     * (stage-0: occPinned from the pinned resolved slots, valid by
     * bootstrap + post-step pin-copy; stage-k: occVariable from the
     * hoisted Tier-2 resolves above).  Consumed only by the srpShadow
     * node, which lists it as a dependency — the node is never a graph
     * leaf, preserving the eval-subgraph closure invariant. */
    idx_t occNode = cacheBarrier;
    if (anyShadow) {
        PARAHPOP_ASSERT(static_cast<idx_t>(occulters.size())
                <= environment::eclipse::MAXOCCULTERS,
            "Too many occulting bodies: at most "
                + std::to_string(environment::eclipse::MAXOCCULTERS)
                + " bodies may carry the 'occulting' feature.");
        environment::eclipse::OcculterPack pack;
        pack.n = static_cast<idx_t>(occulters.size());
        std::vector<idx_t> occDeps;
        occDeps.reserve(occulters.size() + 1);
        for (idx_t k = 0; k < pack.n; k++) {
            const idx_t oi = occulters[k];
            pack.eph[k]
                = (stage == 0) ? env_.ephPinned(oi) : env_.ephVariable(oi);
            pack.r[k] = eref.constants().body(bodies[oi]).r();
            if constexpr (stage != 0)
                occDeps.push_back(resolveNode[oi]);
        }
        const Real rSun  = eref.constants().body(SUN_NAIF_ID).r();
        VecArrT sunCache = (stage == 0) ? env_.ephPinned(radiationIdx)
                                        : env_.ephVariable(radiationIdx);
        if constexpr (stage == 0)
            occDeps.push_back(cacheBarrier); /* == input anchor at stage 0 */
        else
            occDeps.push_back(resolveNode[radiationIdx]);

        idx_t nBlocks, blockSize;
        computeBlocks(dStates.size(), nBlocks, blockSize,
            environment::eclipse::kernel::OccultationLaunch::maxBlockSize);
        occNode = captureNode_(graph, capturer, [&] {
            environment::eclipse::kernel::occultationFactors<<<nBlocks,
                blockSize, 0, stream>>>(
                (stage == 0) ? env_.occPinned() : env_.occVariable(), pos,
                sunCache, pack, terminated, rSun);
        }, occDeps);
    }

    // Earth orientation is shared by drag, SH and tides. Tide-only
    // configurations also need it, independently of Earth's force chain.
    idx_t earthIdx = bodies.size(), sunIdx = bodies.size(), moonIdx = bodies.size();
    for (idx_t i = 0; i < bodies.size(); ++i) {
        if (bodies[i] == NaifId{399}) earthIdx = i;
        if (bodies[i] == NaifId{10}) sunIdx = i;
        if (bodies[i] == NaifId{301}) moonIdx = i;
    }
    feta::vector::Array<Real, 4>::GRef earthRotation;
    idx_t earthRotationNode = cacheBarrier;
    if (correctionFlags.tidesActive()) {
        PARAHPOP_ASSERT(earthIdx < bodies.size(), "Tides require an Earth branch");
        earthRotation = stage == 0 ? env_.quatPinned(earthIdx) : env_.quatVariable(earthIdx);
        if constexpr (stage != 0) {
            idx_t blocks, threads;
            computeBlocks(dStates.size(), blocks, threads,
                environment::kernel::EnvKernelLaunch::maxBlockSize);
            earthRotationNode = captureNode_(graph, capturer, [&] {
                environment::kernel::cacheRotation<<<blocks, threads, 0, stream>>>(
                    earthRotation, epochs.handle(), terminated,
                    env_.deviceRef().orientations(), env_.orientationTarget(earthIdx));
            }, { cacheBarrier });
        }
    }
    auto appendEarthTides = [&](idx_t predecessor) -> idx_t {
        idx_t blocks, threads;
        computeBlocks(dStates.size(), blocks, threads,
            acckernels::EarthCorrectionsLaunch::maxBlockSize);
        auto uniqueDeps = [](std::vector<idx_t> deps) {
            std::sort(deps.begin(), deps.end());
            deps.erase(std::unique(deps.begin(), deps.end()), deps.end());
            return deps;
        };
        if (correctionFlags.solidEarthTides) {
            PARAHPOP_ASSERT(sunIdx < bodies.size() && moonIdx < bodies.size(),
                "Solid tides require prepared Sun and Moon positions");
            const VecArrT sun = stage == 0 ? env_.ephPinned(sunIdx) : env_.ephVariable(sunIdx);
            const VecArrT moon = stage == 0 ? env_.ephPinned(moonIdx) : env_.ephVariable(moonIdx);
            const auto deps = uniqueDeps({predecessor, earthRotationNode,
                resolveNode[sunIdx], resolveNode[moonIdx]});
            predecessor = captureNode_(graph, capturer, [&] {
                acckernels::solidEarthTides<<<blocks, threads, 0, stream>>>(
                    solidSI, pos, epochs.handle(), terminated, correctionModel,
                    earthRotation, sun, moon);
            }, deps);
        }
        if (correctionFlags.oceanTides) {
            const auto deps = uniqueDeps({predecessor, earthRotationNode});
            predecessor = captureNode_(graph, capturer, [&] {
                acckernels::oceanTides<<<blocks, threads, 0, stream>>>(
                    oceanSI, pos, epochs.handle(), terminated, correctionModel, earthRotation);
            }, deps);
        }
        return predecessor;
    };

    for (idx_t i = 0; i < bodies.size(); i++) {
        /* OCCULTING alone emits no force chain — occulters were handled
         * by the hoisted Tier-2 + occultation node above. */
        if (!accs_.anyForceActive(i)) {
            if (bodies[i] == NaifId{399} && correctionFlags.tidesActive())
                bodyTerminals.push_back(appendEarthTides(cacheBarrier));
            continue;
        }

        NaifId body = bodies[i];

        /* Per-body chain entry: stage 0 anchors at the input barrier;
         * stages > 0 at this body's hoisted Tier-2 resolve node
         * (`resolveNode` defaults to `cacheBarrier`, which IS the
         * stage-0 input anchor). */
        idx_t lastNode = resolveNode[i];

        /* Per-body partial-acceleration scratch (Dim=3, device-only).
         * Each acceleration kernel below `+=`-accumulates into this
         * slot; intra-body kernels are graph-edge-serialised via
         * `lastNode`, so the writes are race-free without atomics.
         * A deterministic reduce kernel sums across bodies in fixed
         * body order after the cross-body barrier. */
        VecArrT bodyAcc = env_.bodyAccScratch(i);

        /* Zero this body's scratch before per-body kernels accumulate
         * into it. Anchored at `cacheBarrier` so it can run in parallel
         * with the body's `coiResolveKernel` (Tier 2) — both are quick
         * and write to disjoint slots. */
        const idx_t zeroNode = captureNode_(graph, capturer, [&] {
            acckernels::zeroBodyAccScratch<<<accBlocks, accBlockSize, 0,
                stream>>>(bodyAcc, terminated);
        }, { cacheBarrier });

        /* Stage 0 reads ephPinned_[i] (bootstrap + post-step pin-copy
         * keep it valid); stages 1..n read ephVariable_[i], resolved by
         * the hoisted Tier-2 loop above (this body's `resolveNode[i]`
         * is the chain anchor). */
        VecArrT ephcache = (stage == 0) ? env_.ephPinned(i)
                                        : env_.ephVariable(i);

        idx_t nBlocks, blockSize;
        computeBlocks(dStates.size(), nBlocks, blockSize,
            environment::kernel::EnvKernelLaunch::maxBlockSize);

        /* Stage 0 short-circuit: pinned already populated. */
        if constexpr (stage != 0) {
            /* Tier 2c: ω(epoch, body) cache for drag.  Orientations-
             * driven (no native-cache or COI-chain dependency); peer of
             * cacheRotation.  Only emitted for atmosphere-active bodies.
             * Chained after this body's Tier-2 resolve (`lastNode`),
             * exactly as before the hoist. */
            if (accs_.template active<accsrc::ATMOSPHERE>(i)) {
                lastNode = captureNode_(graph, capturer, [&] {
                    environment::kernel::omegaResolveKernel<<<nBlocks, blockSize,
                        0, stream>>>(env_.omegaVariable(i),
                        env_.deviceRef().orientations(),
                        env_.orientationTarget(i), epochs.handle(), terminated);
                }, { lastNode });
            }
        }

        /* The first acceleration kernel in this body's chain must
         * depend on BOTH the post-coiResolve `lastNode` (it reads
         * `ephcache`) AND the zero kernel (it `+=`-writes
         * `bodyAcc`).  The remainder of the chain only needs to
         * depend on the previous kernel because (a) `ephcache` is
         * static across all per-body kernels and (b) `bodyAcc` is
         * the slot they're chaining writes against. */
        bool firstAccelLaunched = false;
        auto chainAccelDeps = [&](){
            if (!firstAccelLaunched) {
                firstAccelLaunched = true;
                return std::vector<idx_t>{ lastNode, zeroNode };
            }
            return std::vector<idx_t>{ lastNode };
        };

        computeBlocks(dStates.size(), nBlocks, blockSize,
            acckernels::PointGravityLaunch::maxBlockSize);

        /* Point mass gravity */
        if (accs_.template active<accsrc::POINTGRAVITY>(i)) {
            auto deps = chainAccelDeps();
            lastNode  = captureNode_(graph, capturer, [&] {
                acckernels::pointGravity<<<nBlocks, blockSize, 0, stream>>>(
                    bodyAcc, pos, ephcache, metadata.own().cois(), terminated,
                    bodies[i], eref.constants().body(bodies[i]).gm());
            }, deps);
        }

        /* Radiation pressure — occultation-scaled when any occulting
         * body shadows it (`srpShadow` + the occultation node as an
         * extra dependency); the no-shadow configuration emits the
         * untouched `srp` node with unchanged dependencies. */
        if (accs_.template active<accsrc::RADIATION>(i)) {
            computeBlocks(dStates.size(), nBlocks, blockSize,
                acckernels::SRPLaunch::maxBlockSize);
            std::vector<idx_t> deps = chainAccelDeps();
            if (anyShadow)
                deps.push_back(occNode);
            lastNode = captureNode_(graph, capturer, [&] {
                if (anyShadow) {
                    acckernels::srpShadow<<<nBlocks, blockSize, 0, stream>>>(
                        bodyAcc, pos, ephcache, metadata.own().mass(),
                        metadata.own().area(), metadata.own().cr(), terminated,
                        eref.constants().au(),
                        (stage == 0) ? env_.occPinned() : env_.occVariable());
                } else {
                    acckernels::srp<<<nBlocks, blockSize, 0, stream>>>(bodyAcc,
                        pos, ephcache, metadata.own().mass(),
                        metadata.own().area(), metadata.own().cr(), terminated,
                        eref.constants().au());
                }
            }, deps);
        }

        /* Rotation cache: stage 0 reads quatPinned_[i] (bootstrap +
         * post-step pin); stages 1..n fill+read quatVariable_[i].
         * Anchored at `cacheBarrier` so it runs in parallel with
         * gravity/SRP; SH/J2/drag pick up `rotNode` as a dependency.
         * Atmosphere bodies need this too — drag reads the quat to
         * extract pole for geodetic altitude on oblate bodies. */
        feta::vector::Array<Real, 4>::GRef rotcache;
        idx_t rotNode = lastNode; /* tracked separately so SH/J2/drag
                                   * can pick it up alongside `lastNode` */
        if (body == NaifId{399} && correctionFlags.tidesActive()) {
            rotcache = earthRotation;
            rotNode = earthRotationNode;
        } else if (accs_.template active<accsrc::SHAPE>(i)
            || accs_.template active<accsrc::OBLATENESS>(i)
            || accs_.template active<accsrc::ATMOSPHERE>(i)) {
            rotcache = (stage == 0) ? env_.quatPinned(i)
                                    : env_.quatVariable(i);

            if constexpr (stage != 0) {
                computeBlocks(dStates.size(), nBlocks, blockSize,
                    environment::kernel::EnvKernelLaunch::maxBlockSize);
                rotNode = captureNode_(graph, capturer, [&] {
                    environment::kernel::cacheRotation<<<nBlocks, blockSize, 0,
                        stream>>>(rotcache, epochs.handle(), terminated,
                        env_.deviceRef().orientations(),
                        env_.orientationTarget(i));
                }, { cacheBarrier });
            }
        }

        /* Atmospheric drag (body property "atmosphere" activates the
         * drag-force kernel).  velcache (stage-0: velPinned; stage>0:
         * velVariable filled by the Tier-2 coiResolveStateKernel above)
         * carries the body's COI-relative velocity.  omegacache (same
         * stage-0/stage>0 split) carries ω(epoch_i, body) pre-computed
         * by the Tier-2c omegaResolveKernel.  quatcache carries the
         * body-fixed→inertial rotation (filled by cacheRotation above);
         * drag applies it to ẑ_body to recover the polar axis for
         * geodetic altitude evaluation. */
        if (accs_.template active<accsrc::ATMOSPHERE>(i)) {
            VecArrT velcache = (stage == 0) ? env_.velPinned(i)
                                            : env_.velVariable(i);
            VecArrT omegacache = (stage == 0) ? env_.omegaPinned(i)
                                              : env_.omegaVariable(i);
            std::vector<idx_t> deps = chainAccelDeps();
            deps.push_back(rotNode);
            if (env_.isNrlmsise00(i)) {
                namespace nrlk = environment::atmosphere::kernel;
                computeBlocks(dStates.size(), nBlocks, blockSize,
                    nrlk::NrlPreprocessLaunch::maxBlockSize);
                const idx_t inputNode = captureNode_(graph, capturer, [&] {
                    nrlk::prepareNrlmsise00<<<nBlocks, blockSize, 0, stream>>>(
                        env_.nrlInputs(i), pos, ephcache, rotcache,
                        epochs.handle(), terminated, env_.nrlWeather(i),
                        env_.nrlSettings(i),
                        eref.constants().body(body).r(), env_.flattening(i));
                }, deps);
                computeBlocks(dStates.size(), nBlocks, blockSize,
                    nrlk::NrlDensityLaunch::maxBlockSize);
                if (options.nrlThreads) {
                    blockSize = options.nrlThreads;
                    nBlocks = (dStates.size() + blockSize - 1) / blockSize;
                }
                const idx_t densityNode = captureNode_(graph, capturer, [&] {
                    nrlk::evaluateNrlmsise00<<<nBlocks, blockSize, 0, stream>>>(
                        env_.nrlInputs(i), env_.nrlDensity(i), terminated,
                        env_.nrlSettings(i).densityScale);
                }, { inputNode });
                computeBlocks(dStates.size(), nBlocks, blockSize,
                    acckernels::DragLaunch::maxBlockSize);
                lastNode = captureNode_(graph, capturer, [&] {
                    acckernels::dragNrlmsise00<<<nBlocks, blockSize, 0,
                        stream>>>(bodyAcc, pos, vel, ephcache, velcache,
                        omegacache, env_.nrlDensity(i), metadata.own().mass(),
                        metadata.own().area(), metadata.own().cd(), terminated);
                }, { densityNode });
            } else {
                computeBlocks(dStates.size(), nBlocks, blockSize,
                    acckernels::DragLaunch::maxBlockSize);
                /* Exponential model retains its single captured drag node. */
                lastNode = captureNode_(graph, capturer, [&] {
                if (env_.nAtmosphereSegments(i) == 1) {
                    acckernels::dragExp<<<nBlocks, blockSize, 0, stream>>>(
                        bodyAcc, pos, vel, ephcache, velcache, omegacache,
                        rotcache, metadata.own().mass(), metadata.own().area(),
                        metadata.own().cd(), terminated,
                        eref.constants().body(body).r(), env_.flattening(i),
                        env_.expBlock(i));
                } else {
                    acckernels::dragPiecewise<<<nBlocks, blockSize, 0,
                        stream>>>(bodyAcc, pos, vel, ephcache, velcache,
                        omegacache, rotcache, metadata.own().mass(),
                        metadata.own().area(), metadata.own().cd(), terminated,
                        eref.constants().body(body).r(), env_.flattening(i),
                        env_.atmosphere(i));
                }
                }, deps);
            }
        }

        /* Spherical harmonics */
        if (accs_.template active<accsrc::SHAPE>(i)) {
            computeBlocks(dStates.size(), nBlocks, blockSize,
                acckernels::SphericalHarmonicsLaunch::maxBlockSize);
            if (options.shThreads) {
                blockSize = options.shThreads;
                nBlocks = (dStates.size() + blockSize - 1) / blockSize;
            }
            std::vector<idx_t> deps = chainAccelDeps();
            deps.push_back(rotNode);
            lastNode = captureNode_(graph, capturer, [&] {
                acckernels::sphericalHarmonics<<<nBlocks, blockSize,
                    acckernels::SphericalHarmonicsLaunch::sharedBytes(blockSize),
                    stream>>>(bodyAcc, pos, ephcache, rotcache,
                    env_.shCoeffs(i).deviceRef(), terminated,
                    eref.constants().body(body).soi(), options.shRadialCache,
                    options.shCorrectedDivision);
            }, deps);
        }
        /* Else-if: full SH overrides simple oblateness. */
        else if (accs_.template active<accsrc::OBLATENESS>(i)) {
            auto bodyconstants = eref.constants().body(body);
            computeBlocks(dStates.size(), nBlocks, blockSize,
                acckernels::J2Launch::maxBlockSize);
            std::vector<idx_t> deps = chainAccelDeps();
            deps.push_back(rotNode);
            lastNode = captureNode_(graph, capturer, [&] {
                acckernels::j2<<<nBlocks, blockSize, 0, stream>>>(bodyAcc, pos,
                    ephcache, rotcache, terminated, bodyconstants.gm(),
                    bodyconstants.j2(), bodyconstants.r(), bodyconstants.soi());
            }, deps);
        }

        if (body == NaifId{399} && correctionFlags.tidesActive())
            lastNode = appendEarthTides(lastNode);

        /* Per-body terminal: `lastNode` already terminates the
         * intra-body accel chain; no addEmptyNode merge needed. */
        bodyTerminals.push_back(lastNode);
    }

    /* Cross-body barrier — sync point between parallel per-body
     * chains and the deterministic reduce that follows. */
    // Close auxiliary cache work even for a force-free configuration.
    bodyTerminals.push_back(cacheBarrier);
    graph.addEmptyNode(bodyTerminals);
    idx_t reduceAnchor = graph.lastNode();

    /* Deterministic reduce: one kernel per active body, chained
     * serially in fixed body-iteration order, each `+=`-ing that
     * body's scratch into `accarray`.  `accarray[i]` has a single
     * writer per launch, so no atomics are needed; cross-launch
     * order is fixed by the graph edges, so the FP rounding is
     * bit-identical across replays — same semantics as the host
     * RHS sequential body loop in PrivateAccelerations.h::eval_. */
    for (idx_t i = 0; i < bodies.size(); i++) {
        if (!accs_.anyForceActive(i))
            continue;
        reduceAnchor = captureNode_(graph, capturer, [&] {
            acckernels::reduceBodyAccScratch<<<accBlocks, accBlockSize, 0,
                stream>>>(accarray, env_.bodyAccScratch(i), terminated);
        }, { reduceAnchor });
    }
    // One writer combines the three independent SI outputs in the same
    // arithmetic order as the CPU implementation, after the body sum.
    if (correctionFlags.any()) {
        idx_t blocks, threads;
        computeBlocks(dStates.size(), blocks, threads,
            acckernels::EarthCorrectionsLaunch::maxBlockSize);
        captureNode_(graph, capturer, [&] {
            acckernels::accumulateEarthCorrections<<<blocks, threads, 0, stream>>>(
                accarray, solidSI, oceanSI, relativitySI, terminated, correctionFlags);
        }, { reduceAnchor });
    }
}

} // namespace reduced
} // namespace physics
} // namespace model
} // namespace paraHPOP
