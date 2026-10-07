#pragma once

namespace feta {

/* Template helpers to distinguish among Global, Work, and Texture related
 * patterns */
namespace helper {

/**
 * @brief Boolean to distinguish single and double precision
 *
 */
template<typename DataT>
struct DoublePrecision {
    static const bool T = false;
};
template<>
struct DoublePrecision<double> {
    static const bool T = true;
};

/**
 * @brief Texture fetching helper for consistent type casting
 *
 */
template<typename DataT>
struct Fetch {
    using T                           = int;
    static const bool needsConversion = false;
};
template<>
struct Fetch<long int> {
    using T                           = idx_t;
    static const bool needsConversion = false;
};
template<>
struct Fetch<float> {
    using T                           = int;
    static const bool needsConversion = true;
};
template<>
struct Fetch<double> {
#ifndef FETA_CPU_ONLY
    using T = int2;
#endif
    static const bool needsConversion = true;
};

/**
 * @brief Data return type helper - reference returning is not allowed with
 * Texture memory (Texture is read-only).
 *
 * The optional `Volatile` parameter selects a `volatile DataT&` writable
 * reference when on; this is used by work-references whose backing storage
 * is in shared memory (and must defeat compiler caching / stack-backed
 * mirrors).  Volatile is only meaningful for non-texture refs — there is
 * no `<DataT, true, true>` specialisation by design.
 */
template<typename DataT, bool UseTexture, bool Volatile = false>
struct DataReturn {
    using T = DataT&;
};
template<typename DataT>
struct DataReturn<DataT, true, false> {
    using T = DataT;
};
template<typename DataT>
struct DataReturn<DataT, false, true> {
    using T = volatile DataT&;
};

/* Check if type T supports __ldg */
template<typename T, typename = void>
struct HasLdg {
    static constexpr bool value = false;
};

#ifdef __CUDA_ARCH__
template<typename T>
struct HasLdg<T, std::void_t<decltype(__ldg(std::declval<const T*>()))>> {
    static constexpr bool value = std::is_trivially_copyable_v<T>;
};
#endif

} // namespace helper
} // namespace feta