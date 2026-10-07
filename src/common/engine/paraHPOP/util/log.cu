#include "interface/util/err/log.hpp"
#include <parm/util/log.h>

/** @brief Out-of-line definitions that ensure the writes to currentLogLevel
 *  go through a symbol with default DSO visibility, so that callers in other
 *  shared libraries (e.g. Python extension modules) hit the single copy of
 *  each variable that lives in libparaHPOP_models.so rather than their own
 *  inlined/private copy. */

namespace paraHPOP {
namespace util {

/** @brief Out-of-line level check + print. Lives in libparaHPOP_models.so so it
 *  always reads the single shared currentLogLevel regardless of which DSO the
 *  caller was compiled into. */
void log_impl(const Level& level, const std::string& message)
{
    if (level > currentLogLevel)
        return;
    std::printf("%s", message.c_str());
}

} // namespace util
} // namespace paraHPOP

extern "C" {

void paraHPOP_setLogLevel(int level)
{
    paraHPOP::util::setLogLevel(static_cast<paraHPOP::util::Level>(level));
}

void parm_setLogLevel(int level)
{
    parm::util::setLogLevel(static_cast<parm::util::Level>(level));
}

} // extern "C"
