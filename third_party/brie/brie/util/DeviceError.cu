#include "brie/util/DeviceError.h"

#ifdef FETA_INTO_SHARED_LIBRARY
namespace feta {
namespace err {

DEVICE() DeviceFlag deviceErrorFlag;

} // namespace err
} // namespace feta
#endif

#ifdef PARM_INTO_SHARED_LIBRARY
namespace parm {
namespace err {

DEVICE() DeviceFlag deviceErrorFlag;

} // namespace err
} // namespace parm
#endif

#ifdef BRIE_INTO_SHARED_LIBRARY
namespace brie {
namespace err {

DEVICE() DeviceFlag deviceErrorFlag;

} // namespace err
} // namespace brie
#endif

namespace brie {
namespace err {

// Host-side wrappers: keep cudaMemcpyTo/FromSymbol calls in the same
// CUDA module/DSO that defines 'deviceErrorFlag' to avoid cross-DSO
// device symbol lookups when used from Python extension modules.
void resetDeviceErrors()
{
    ::feta::err::detail::resetDeviceErrors(deviceErrorFlag);
}

void checkDeviceErrors(const char* file, int line, const char* func)
{
    ::feta::err::detail::checkDeviceErrors(
        deviceErrorFlag, ::brie::err::errorMessages, file, line, func);
}
} // namespace err
} // namespace brie