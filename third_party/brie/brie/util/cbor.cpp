#include "brie/util/cbor.h"

#include "brie/util/throw.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <algorithm>
#include <nlohmann/json.hpp>

namespace brie {
namespace cbor {

namespace detail {

/* Minimal CBOR cursor for pass 1: reads length headers, skips unknown values,
 * and stops at the `data` array head — never traverses the payload. */
struct Cursor {
    const unsigned char* it;
    const unsigned char* end;
};

static inline uint64_t readArg_(Cursor& c, uint8_t addInfo)
{
    if (addInfo < 24)
        return addInfo;
    if (addInfo == 24) {
        BRIE_ASSERT(c.it < c.end, "CBOR truncated at uint8 arg");
        uint8_t v = *c.it++;
        return v;
    }
    if (addInfo == 25) {
        BRIE_ASSERT(c.it + 2 <= c.end, "CBOR truncated at uint16 arg");
        uint16_t v;
        std::memcpy(&v, c.it, 2);
        c.it += 2;
        return __builtin_bswap16(v);
    }
    if (addInfo == 26) {
        BRIE_ASSERT(c.it + 4 <= c.end, "CBOR truncated at uint32 arg");
        uint32_t v;
        std::memcpy(&v, c.it, 4);
        c.it += 4;
        return __builtin_bswap32(v);
    }
    if (addInfo == 27) {
        BRIE_ASSERT(c.it + 8 <= c.end, "CBOR truncated at uint64 arg");
        uint64_t v;
        std::memcpy(&v, c.it, 8);
        c.it += 8;
        return __builtin_bswap64(v);
    }
    if (addInfo == 31)
        return 0; // indefinite-length sentinel (handled by caller)
    BRIE_THROW(std::runtime_error, "Unsupported CBOR additional info");
}

static inline std::string_view readTextStr_(Cursor& c)
{
    BRIE_ASSERT(c.it < c.end, "CBOR truncated at string head");
    uint8_t initial = *c.it++;
    uint8_t major   = initial >> 5;
    BRIE_ASSERT(major == 3, "CBOR: expected text string");
    uint64_t len = readArg_(c, initial & 0x1f);
    BRIE_ASSERT(c.it + len <= c.end, "CBOR truncated text string body");
    std::string_view sv(reinterpret_cast<const char*>(c.it), len);
    c.it += len;
    return sv;
}

/* Forward decl of the value-skipper. */
static void skipValue_(Cursor& c);

static void skipValue_(Cursor& c)
{
    BRIE_ASSERT(c.it < c.end, "CBOR truncated at value head");
    uint8_t initial = *c.it++;
    uint8_t major   = initial >> 5;
    uint8_t addInfo = initial & 0x1f;

    switch (major) {
    case 0: // unsigned
    case 1: // negative
        readArg_(c, addInfo);
        return;
    case 2:   // byte string
    case 3: { // text string
        if (addInfo == 31) {
            // indefinite-length: read chunks until break
            while (true) {
                BRIE_ASSERT(c.it < c.end, "CBOR truncated indef str");
                if (*c.it == 0xff) {
                    c.it++;
                    break;
                }
                skipValue_(c);
            }
            return;
        }
        uint64_t len = readArg_(c, addInfo);
        BRIE_ASSERT(c.it + len <= c.end, "CBOR truncated str body");
        c.it += len;
        return;
    }
    case 4: { // array
        if (addInfo == 31) {
            while (true) {
                BRIE_ASSERT(c.it < c.end, "CBOR truncated indef arr");
                if (*c.it == 0xff) {
                    c.it++;
                    break;
                }
                skipValue_(c);
            }
            return;
        }
        uint64_t n = readArg_(c, addInfo);
        for (uint64_t i = 0; i < n; ++i)
            skipValue_(c);
        return;
    }
    case 5: { // map
        if (addInfo == 31) {
            while (true) {
                BRIE_ASSERT(c.it < c.end, "CBOR truncated indef map");
                if (*c.it == 0xff) {
                    c.it++;
                    break;
                }
                skipValue_(c); // key
                skipValue_(c); // value
            }
            return;
        }
        uint64_t n = readArg_(c, addInfo);
        for (uint64_t i = 0; i < n; ++i) {
            skipValue_(c); // key
            skipValue_(c); // value
        }
        return;
    }
    case 6: { // tag
        readArg_(c, addInfo);
        skipValue_(c); // tagged content
        return;
    }
    case 7: { // simple/float
        if (addInfo < 24)
            return; // simple value, no payload
        if (addInfo == 24) {
            BRIE_ASSERT(c.it < c.end, "CBOR truncated simple-1");
            c.it++;
            return;
        }
        if (addInfo == 25) {
            BRIE_ASSERT(c.it + 2 <= c.end, "CBOR truncated f16");
            c.it += 2;
            return;
        }
        if (addInfo == 26) {
            BRIE_ASSERT(c.it + 4 <= c.end, "CBOR truncated f32");
            c.it += 4;
            return;
        }
        if (addInfo == 27) {
            BRIE_ASSERT(c.it + 8 <= c.end, "CBOR truncated f64");
            c.it += 8;
            return;
        }
        BRIE_THROW(std::runtime_error, "CBOR: unsupported simple/float info");
    }
    }
}

static inline int readIntScalar_(Cursor& c)
{
    BRIE_ASSERT(c.it < c.end, "CBOR truncated at int head");
    uint8_t initial = *c.it++;
    uint8_t major   = initial >> 5;
    uint8_t addInfo = initial & 0x1f;
    if (major == 0)
        return static_cast<int>(readArg_(c, addInfo));
    if (major == 1)
        return -1 - static_cast<int>(readArg_(c, addInfo));
    BRIE_THROW(std::runtime_error, "CBOR: expected integer scalar");
}

/**
 * @brief Read an array head and return its element count. Cursor lands on the
 * first element. Caller must consume or skip those elements.
 */
static inline uint64_t readArrayHead_(Cursor& c)
{
    BRIE_ASSERT(c.it < c.end, "CBOR truncated at array head");
    uint8_t initial = *c.it++;
    uint8_t major   = initial >> 5;
    uint8_t addInfo = initial & 0x1f;
    BRIE_ASSERT(major == 4, "CBOR: expected array");
    BRIE_ASSERT(addInfo != 31,
        "CBOR: indefinite-length top-level arrays not supported by peek");
    return readArg_(c, addInfo);
}

/** @brief Recover layout + array lengths without traversing array bodies. */
static EphFileSizes peekSizes_(const std::vector<unsigned char>& bytes)
{
    EphFileSizes out{};
    Cursor c{ bytes.data(), bytes.data() + bytes.size() };

    /* Unwrap any single-element array wrappers (legacy writer artifact). */
    while (c.it < c.end && (*c.it >> 5) == 4) {
        // peek without consuming: temporarily inspect
        Cursor probe = c;
        uint8_t initial = *probe.it++;
        uint8_t addInfo = initial & 0x1f;
        if (addInfo == 31)
            break; // indefinite — bail
        uint64_t n = readArg_(probe, addInfo);
        if (n != 1)
            break;
        c = probe; // commit unwrap
    }

    /* Now expect a map. */
    BRIE_ASSERT(c.it < c.end, "CBOR truncated at top-level");
    uint8_t initial = *c.it++;
    BRIE_ASSERT((initial >> 5) == 5, "CBOR: expected top-level map");
    uint8_t addInfo = initial & 0x1f;
    BRIE_ASSERT(
        addInfo != 31, "CBOR: indefinite top-level maps not supported");
    uint64_t topPairs = readArg_(c, addInfo);

    for (uint64_t i = 0; i < topPairs; ++i) {
        std::string_view key = readTextStr_(c);
        if (key == "layout") {
            out.layout = readIntScalar_(c);
        } else if (key == "core") {
            /* layout was set first when present; if not, infer below. */
            if (out.layout == 1) {
                /* Skip the entire `core` value — caller falls back to DOM. */
                skipValue_(c);
                continue;
            }
            /* Layout 2: expect `core` to be a map with `metadata` and `data` */
            BRIE_ASSERT(
                c.it < c.end, "CBOR truncated at core head");
            uint8_t coreInit = *c.it++;
            BRIE_ASSERT(
                (coreInit >> 5) == 5, "CBOR: expected core to be a map");
            uint8_t coreInfo = coreInit & 0x1f;
            BRIE_ASSERT(coreInfo != 31, "CBOR: indef core map unsupported");
            uint64_t corePairs = readArg_(c, coreInfo);

            for (uint64_t j = 0; j < corePairs; ++j) {
                std::string_view ck = readTextStr_(c);
                if (ck == "metadata") {
                    BRIE_ASSERT(
                        c.it < c.end, "CBOR truncated at metadata");
                    uint8_t mInit = *c.it++;
                    BRIE_ASSERT((mInit >> 5) == 5,
                        "CBOR: metadata must be a map");
                    uint8_t mInfo = mInit & 0x1f;
                    BRIE_ASSERT(mInfo != 31,
                        "CBOR: indef metadata map unsupported");
                    uint64_t mPairs = readArg_(c, mInfo);
                    for (uint64_t k = 0; k < mPairs; ++k) {
                        std::string_view mk = readTextStr_(c);
                        if (mk == "nBodyUnits") {
                            out.nBodyUnits = readIntScalar_(c);
                        } else if (mk == "intMetadata") {
                            uint64_t n = readArrayHead_(c);
                            out.intMetadataLen = static_cast<size_t>(n);
                            for (uint64_t e = 0; e < n; ++e)
                                skipValue_(c);
                        } else if (mk == "doubleMetadata") {
                            uint64_t n = readArrayHead_(c);
                            out.doubleMetadataLen
                                = static_cast<size_t>(n);
                            for (uint64_t e = 0; e < n; ++e)
                                skipValue_(c);
                        } else {
                            skipValue_(c);
                        }
                    }
                } else if (ck == "data") {
                    /* O(1): read just the length header, do NOT traverse. */
                    uint64_t n      = readArrayHead_(c);
                    out.dataLen     = static_cast<size_t>(n);
                    /* Stop here — `data` is the last needed key. */
                    return out;
                } else {
                    skipValue_(c);
                }
            }
        } else {
            /* unknown top-level key (e.g. version) — skip */
            skipValue_(c);
        }
    }

    return out;
}

} // namespace detail

/* ------------------------------------------------------------------------- */
/*  Public API                                                                */
/* ------------------------------------------------------------------------- */

EphFileBuffer peekEphFile(
    const std::string& fileName, const util::Paths& paths)
{
    util::path filePath = paths.requireFile(fileName);
    std::ifstream ifs(filePath, std::ios::binary | std::ios::ate);
    BRIE_ASSERT(ifs.is_open(),
        std::string("Failed to open: ") + filePath.string());
    const std::streamsize fileSize = ifs.tellg();
    ifs.seekg(0);
    EphFileBuffer fb;
    fb.bytes.resize(static_cast<size_t>(fileSize));
    if (fileSize > 0) {
        ifs.read(reinterpret_cast<char*>(fb.bytes.data()), fileSize);
        BRIE_ASSERT(ifs.good(),
            std::string("Failed to read: ") + filePath.string());
    }
    fb.sizes = detail::peekSizes_(fb.bytes);
    return fb;
}

void loadEphLayout2(const EphFileBuffer& fb,
    int*  intMetaDst,    std::size_t intMetaCap,
    Real* doubleMetaDst, std::size_t doubleMetaCap,
    Real* dataDst,       std::size_t dataCap)
{
    BRIE_ASSERT(fb.sizes.layout == 2,
        "loadEphLayout2 called on non-Layout-2 file");
    BRIE_ASSERT(intMetaCap == fb.sizes.intMetadataLen,
        "intMeta capacity mismatch with peek result");
    BRIE_ASSERT(doubleMetaCap == fb.sizes.doubleMetadataLen,
        "doubleMeta capacity mismatch with peek result");
    BRIE_ASSERT(dataCap == fb.sizes.dataLen,
        "data capacity mismatch with peek result");
    BRIE_ASSERT(
        doubleMetaCap == 0 || doubleMetaDst != nullptr,
        "doubleMetaDst is null but doubleMetadataLen > 0");

    nlohmann::json j = nlohmann::json::from_cbor(fb.bytes);
    while (j.is_array() && j.size() == 1)
        j = j[0];

    BRIE_ASSERT(j.at("layout").get<int>() == 2,
        "Layout mismatch after deserialization (expected 2)");
    const auto& metadata = j.at("core").at("metadata");
    const auto& ints = metadata.at("intMetadata");
    const auto& doubles = metadata.at("doubleMetadata");
    const auto& data = j.at("core").at("data");

    BRIE_ASSERT(ints.size() == intMetaCap,
        "intMetadata length mismatch after deserialization");
    BRIE_ASSERT(doubles.size() == doubleMetaCap,
        "doubleMetadata length mismatch after deserialization");
    BRIE_ASSERT(data.size() == dataCap,
        "data length mismatch after deserialization");

    std::copy(ints.begin(), ints.end(), intMetaDst);
    if (doubleMetaCap != 0)
        std::copy(doubles.begin(), doubles.end(), doubleMetaDst);
    std::copy(data.begin(), data.end(), dataDst);
}

} // namespace cbor
} // namespace brie
