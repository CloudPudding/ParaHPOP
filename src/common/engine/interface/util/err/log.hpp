#pragma once

// CUDA headers define  which interferes with libstdc++'s use of
// `__attribute(())`. In order to avoid compilation error,
// temporarily unset  when we include affected libstdc++ header.
// This issue only affects clang/clangd

#ifdef __clang__
#pragma push_macro("")
#undef
#endif
#include <string>
#ifdef __clang__
#pragma pop_macro("")
#endif

/** Log message at ERROR severity. Doesn't throw and exception, use PARAHPOP_THROW
 * for that. */
#define PARAHPOP_ERROR(MSG, ...)                                                  \
    paraHPOP::util::log_fn(                                                       \
        paraHPOP::util::Level::ERROR, std::string(__func__), MSG, ##__VA_ARGS__)

/** Log message at WARNING severity */
#define PARAHPOP_WARN(MSG, ...)                                                   \
    paraHPOP::util::log_fn(                                                       \
        paraHPOP::util::Level::WARN, std::string(__func__), MSG, ##__VA_ARGS__)

/** Log message at INFO severity */
#define PARAHPOP_INFO(MSG, ...)                                                   \
    paraHPOP::util::log_fn(                                                       \
        paraHPOP::util::Level::INFO, std::string(__func__), MSG, ##__VA_ARGS__)

/** Log message at DEBUG severity */
#define PARAHPOP_DEBUG(MSG, ...)                                                  \
    paraHPOP::util::log_fn(                                                       \
        paraHPOP::util::Level::DEBUG, std::string(__func__), MSG, ##__VA_ARGS__)

/** Log message at TRACE severity */
#define PARAHPOP_TRACE(MSG, ...)                                                  \
    paraHPOP::util::log_fn(                                                       \
        paraHPOP::util::Level::TRACE, std::string(__func__), MSG, ##__VA_ARGS__)


namespace paraHPOP {
namespace util {

enum Level : int {
    OFF   = 0,
    ERROR = 1,
    WARN  = 2,
    INFO  = 3,
    DEBUG = 4,
    TRACE = 5,
    ALL   = 6
};

inline Level currentLogLevel = INFO;

[[maybe_unused]] inline void setLogLevel(const Level& level)
{
    currentLogLevel = level;
}

[[maybe_unused]] inline void disableLogging()
{
    setLogLevel(OFF);
}

/** @brief Non-template, out-of-line function that owns the currentLogLevel
 *  check and the actual print. Defined in paraHPOP/util/log.cu so that it lives
 *  in libparaHPOP_models.so and always reads the single shared currentLogLevel. */
void log_impl(const Level& level, const std::string& message);

[[maybe_unused]] static std::string getLevelTag(const Level& level)
{
    switch (level) {
    case TRACE:
        return "* [\x1B[96mTRACE\x1B[0m]";
    case DEBUG:
        return "* [\x1B[94mDEBUG\x1B[0m]";
    case WARN:
        return "* [\x1B[93mWARN\x1B[0m ]";
    case ERROR:
        return "* [\x1B[91mERROR\x1B[0m]";
    case INFO:
    default:
        return "* [\x1B[92mINFO\x1B[0m ]";
    }
}

template<typename... Args>
void log(const Level& level, const std::string& msg, Args... args)
{
    // Format the message in the caller's TU, then hand off to the out-of-line
    // function that does the level check against the single shared variable.
    char buf[1024];
#if __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
#endif
    std::snprintf(buf, sizeof(buf), msg.c_str(), args...);
#if __GNUC__
#pragma GCC diagnostic pop
#endif
    log_impl(level, " " + getLevelTag(level) + " " + std::string(buf) + "\n");
}

template<typename... Args>
void log_fn(const Level& level, const std::string& func, const std::string& msg,
    Args... args)
{
    log(level, "[" + func + "] " + msg, args...);
}

} // namespace util
} // namespace paraHPOP
