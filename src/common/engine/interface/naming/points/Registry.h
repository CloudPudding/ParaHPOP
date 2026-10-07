#pragma once

#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "interface/naming/bands.h"
#include "interface/naming/normalize.h"
#include "interface/util/err.h"

namespace interface {
namespace naming {
namespace points {

/** @brief A registered point: a unique id <-> canonical name mapping plus
 * aliases, optionally enriched with canonical libration defaults (catalog
 * Sun-EMB L-points only). The registry stores naming only — evaluation
 * semantics attach per-environment via `environment::Points`. */
struct Entry {
    /** @brief Auto-assignment sentinel: ids below never collide with it */
    static constexpr NaifId AUTO = 0;

    NaifId id = AUTO;
    std::string name;
    std::vector<std::string> aliases;

    /* Canonical libration defaults — set for catalog entries only, so a
     * per-environment `{"libration": {}}` declaration can inherit them */
    bool hasLibrationDefaults = false;
    NaifId primary            = 0;
    NaifId secondary          = 0;
    int which                 = 0;
};

/** @brief Process-global, mutex-guarded point-name registry. Fed by each
 * `Environment::points()`: identical re-registration is a no-op, conflicting
 * redefinition throws, the registry never shrinks. All lookups happen at
 * configuration time, never on the propagation hot path.
 *
 * Pre-seeded (names-only) with the Sun-EMB Lagrange points carried by the
 * official NAIF L*.bsp kernels: 391=SEL1, 392=SEL2, 394=SEL4, 395=SEL5.
 * No L3 kernel exists, so 393 is deliberately not seeded. */
class Registry {
    using Self = Registry;

public:
    /** @brief Access the process-global instance.
     *
     * Defined OUT-OF-LINE in `paraHPOP/paraHPOP/util/naming.cu` (compiled
     * into `libparaHPOP_models.so`; Stage-1 test binaries compile the TU
     * directly, like the log machinery in `paraHPOP/util/log.cu`). A
     * header-local static would be duplicated per Python extension
     * module — nanobind `.so`s are loaded RTLD_LOCAL with hidden
     * visibility, so a point registered through `_environment` would be
     * invisible to `_samples`. Anchoring the definition in the one shared
     * models library every module links keeps the registry truly
     * process-global. */
    static Registry& instance();

    /* Process-global singleton: no copies, no moves */
    Registry(const Registry&)            = delete;
    Registry& operator=(const Registry&) = delete;

    /** @brief Register the given entry, returning its (possibly
     * auto-assigned) id. Identical re-registration is a no-op that returns
     * the existing id; new aliases for an existing entry are merged in;
     * any conflicting mapping throws. */
    NaifId add(const Entry& entry)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return this->add_(entry);
    }

    /** @brief Non-throwing name lookup (canonical name or alias, normalized
     * comparison). Writes `id` only on success. */
    bool tryResolve(const std::string& name, NaifId& id) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = byName_.find(normalized(name));
        if (it == byName_.end())
            return false;
        id = it->second;
        return true;
    }

    /** @brief Whether the given id is registered */
    bool contains(const NaifId& id) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return byId_.find(id) != byId_.end();
    }

    /** @brief Canonical name of the given registered id (throws if unknown) */
    std::string nameOf(const NaifId& id) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = byId_.find(id);
        PARAHPOP_ASSERT(it != byId_.end(),
            "Id " + std::to_string(id) + " is not a registered point");
        return it->second.name;
    }

    /** @brief Copy of the entry registered under the given id, if any */
    std::optional<Entry> find(const NaifId& id) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = byId_.find(id);
        if (it == byId_.end())
            return std::nullopt;
        return it->second;
    }

    /** @brief Copy of the entry matching the given name/alias, if any */
    std::optional<Entry> find(const std::string& name) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto nit = byName_.find(normalized(name));
        if (nit == byName_.end())
            return std::nullopt;
        return byId_.at(nit->second);
    }

    /** @brief Number of registered points (catalog included) */
    idx_t size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<idx_t>(byId_.size());
    }

    /** @brief Reset to the freshly-seeded catalog state. Test isolation
     * only — production code never shrinks the registry. */
    void clearForTesting()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        byId_.clear();
        byName_.clear();
        nextCustomId_ = bands::custompoint::BASE;
        this->seedCatalog_();
    }

private:
    Registry() { this->seedCatalog_(); }

    /** @brief Unlocked implementation of add() (also used by seeding) */
    NaifId add_(const Entry& entry)
    {
        PARAHPOP_ASSERT(!entry.name.empty(), "Point names must not be empty");

        const std::string key = normalized(entry.name);

        /* The name namespace must stay unambiguous against the resolution
         * chain: NAIF bodies and the Sample_<idx> pattern take precedence,
         * so a point registered under such a name would be shadowed. */
        this->assertNotShadowed_(entry.name);
        for (const auto& alias : entry.aliases)
            this->assertNotShadowed_(alias);

        /* Locate any existing registration under this name */
        auto nit = byName_.find(key);

        NaifId id = entry.id;
        if (nit != byName_.end()) {
            /* Existing name: explicit id (if any) must agree */
            PARAHPOP_ASSERT(id == Entry::AUTO || id == nit->second,
                "Point '" + entry.name + "' is already registered with id "
                    + std::to_string(nit->second)
                    + "; conflicting redefinition with id "
                    + std::to_string(id));
            id = nit->second;
        } else if (id == Entry::AUTO) {
            /* New name, auto id: assign sequentially from the reserved
             * custom-point band */
            PARAHPOP_ASSERT(nextCustomId_ < bands::custompoint::END,
                "Custom-point id band exhausted");
            id = nextCustomId_++;
        } else {
            /* New name, explicit id: must not collide with reserved bands,
             * NAIF bodies, or an id already registered under another name */
            PARAHPOP_ASSERT(!bands::inReservedBands(id),
                "Point '" + entry.name + "': explicit id " + std::to_string(id)
                    + " lies in a reserved band");
            NaifId parsed = 0;
            PARAHPOP_ASSERT(!brie::gravity::Parser::tryParsedNaifId(id, parsed),
                "Point '" + entry.name + "': id " + std::to_string(id)
                    + " is a valid NAIF body ('"
                    + brie::gravity::Parser::parsedName(id)
                    + "'); bodies cannot be re-registered as points");
            auto iit = byId_.find(id);
            PARAHPOP_ASSERT(iit == byId_.end(),
                "Id " + std::to_string(id)
                    + " is already registered as point '"
                    + ((iit != byId_.end()) ? iit->second.name : "")
                    + "'; conflicting redefinition as '" + entry.name + "'");
        }

        /* Merge with the existing entry or insert a fresh one */
        auto iit = byId_.find(id);
        if (iit != byId_.end()) {
            Entry& existing = iit->second;
            PARAHPOP_ASSERT(normalized(existing.name) == key,
                "Id " + std::to_string(id) + " is registered as '"
                    + existing.name + "'; conflicting redefinition as '"
                    + entry.name + "'");
            this->mergeLibrationDefaults_(existing, entry);
            for (const auto& alias : entry.aliases)
                this->addAlias_(existing, alias);
        } else {
            Entry fresh = entry;
            fresh.id    = id;
            byId_[id]   = std::move(fresh);
            this->mapName_(entry.name, id);
            for (const auto& alias : entry.aliases)
                this->mapName_(alias, id);
        }
        return id;
    }

    /** @brief Map a normalized name onto an id, throwing on cross-id clash */
    void mapName_(const std::string& name, const NaifId& id)
    {
        const std::string key = normalized(name);
        auto it               = byName_.find(key);
        if (it != byName_.end()) {
            PARAHPOP_ASSERT(it->second == id,
                "Point name/alias '" + name + "' is already registered for '"
                    + byId_.at(it->second).name + "' (id "
                    + std::to_string(it->second) + ")");
            return;
        }
        byName_[key] = id;
    }

    /** @brief Merge a new alias into an existing entry */
    void addAlias_(Entry& existing, const std::string& alias)
    {
        const std::string key = normalized(alias);
        if (byName_.find(key) != byName_.end()) {
            this->mapName_(alias, existing.id); /* validates the clash */
            return;
        }
        byName_[key] = existing.id;
        existing.aliases.push_back(alias);
    }

    /** @brief Merge libration defaults, rejecting conflicting redefinition */
    void mergeLibrationDefaults_(Entry& existing, const Entry& incoming)
    {
        if (!incoming.hasLibrationDefaults)
            return;
        if (existing.hasLibrationDefaults) {
            PARAHPOP_ASSERT(existing.primary == incoming.primary
                    && existing.secondary == incoming.secondary
                    && existing.which == incoming.which,
                "Point '" + existing.name
                    + "': conflicting libration defaults redefinition");
            return;
        }
        existing.hasLibrationDefaults = true;
        existing.primary              = incoming.primary;
        existing.secondary            = incoming.secondary;
        existing.which                = incoming.which;
    }

    /** @brief Reject names the resolution chain would shadow (NAIF bodies
     * and the structural Sample_<idx> pattern outrank registry lookups) */
    void assertNotShadowed_(const std::string& name) const
    {
        NaifId parsed = 0;
        PARAHPOP_ASSERT(!brie::gravity::Parser::tryParsedNaifId(name, parsed),
            "Point name/alias '" + name
                + "' is a valid NAIF body name; bodies cannot be "
                  "re-registered as points");

        /* Reject the reserved sample pattern: normalized "sample" + digits */
        const std::string key = normalized(name);
        constexpr char prefix[] = "sample";
        constexpr size_t plen   = sizeof(prefix) - 1;
        if (key.size() > plen && key.compare(0, plen, prefix) == 0) {
            bool alldigits = true;
            for (size_t i = plen; i < key.size(); i++)
                alldigits &= (key[i] >= '0' && key[i] <= '9');
            PARAHPOP_ASSERT(!alldigits,
                "Point name/alias '" + name
                    + "' matches the reserved Sample_<idx> pattern");
        }
    }

    /** @brief Seed the names-only L-point catalog (NAIF L*.bsp ids) with
     * canonical Sun-EMB libration defaults attached for the empty-dict
     * declaration idiom (`{"libration": {}}`). */
    void seedCatalog_()
    {
        constexpr NaifId Sun = 10;
        constexpr NaifId EMB = 3;

        auto seed = [this](const NaifId& id, const int& which) {
            const std::string n = std::to_string(which);
            Entry e;
            e.id      = id;
            e.name    = "SEL" + n;
            e.aliases = { "SEML" + n, "Sun Earth L" + n,
                "Sun Earth Lagrange " + n, "Sun Earth Libration " + n };
            e.hasLibrationDefaults = true;
            e.primary              = Sun;
            e.secondary            = EMB;
            e.which                = which;
            this->add_(e);
        };
        seed(391, 1);
        seed(392, 2);
        seed(394, 4);
        seed(395, 5);
    }

    mutable std::mutex mutex_;
    std::map<NaifId, Entry> byId_;
    std::map<std::string, NaifId> byName_;
    NaifId nextCustomId_ = bands::custompoint::BASE;
};

} // namespace points
} // namespace naming
} // namespace interface
