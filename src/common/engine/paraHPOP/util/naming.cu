/* Out-of-line anchors of the process-global naming registries (points and
 * orientation frames).
 *
 * Compiled into libparaHPOP_models.so (and directly into the Stage-1
 * interface test binary, which does not link the models library — same
 * arrangement as the log machinery in log.cu). Every DSO in the process,
 * including the per-module Python extension libraries loaded RTLD_LOCAL,
 * resolves Registry::instance() against this single definition, so the
 * registries are truly process-global. A header-inline static local would
 * be duplicated per extension module instead. */

#include "interface/naming/orientations/Registry.h"
#include "interface/naming/points/Registry.h"

namespace interface {
namespace naming {

namespace points {

Registry& Registry::instance()
{
    static Registry registry;
    return registry;
}

} // namespace points

namespace orientations {

Registry& Registry::instance()
{
    static Registry registry;
    return registry;
}

} // namespace orientations

} // namespace naming
} // namespace interface
