#pragma once

#include "interface/util/FileAdder.h"
#include "interface/util/TimeConversion.h"
#include "interface/util/collection.h"
#include "interface/util/err.h"
namespace interface {
namespace util {

using brie::util::Paths;

/**
 * @brief Default PATH for paraHPOP. Includes env{BRIE_PATH} and
 * env{PARAHPOP_PATH}.
 */
struct paths {
    static Paths paraHPOPDefaultPath()
    {
        Paths p = brie::util::paths::brieDefaultPath();
        p.addEnv("PARAHPOP_PATH");
        return p;
    }
};

} // namespace util
} // namespace interface