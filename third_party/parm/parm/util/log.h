#pragma once

// CUDA headers define NOINLINE() which interferes with libstdc++'s use of
// `__attribute((NOINLINE()))`. In order to avoid compilation error,
// temporarily unset NOINLINE() when we include affected libstdc++ header.
// This issue only affects clang/clangd

#ifdef __clang__
#pragma push_macro("NOINLINE()")
#undef NOINLINE()
#endif
#include <string>
#ifdef __clang__
#pragma pop_macro("NOINLINE()")
#endif

/** @brief Log a message at ERROR severity (does not throw; use ``PARM_THROW`` for that). */
#define PARM_ERROR(MSG, ...)                                                   \
    parm::util::log_fn(                                                        \
        parm::util::Level::ERROR, std::string(__func__), MSG, ##__VA_ARGS__)

/** @brief Log a message at WARNING severity. */
#define PARM_WARN(MSG, ...)                                                    \
    parm::util::log_fn(                                                        \
        parm::util::Level::WARN, std::string(__func__), MSG, ##__VA_ARGS__)

/** @brief Log a message at INFO severity. */
#define PARM_INFO(MSG, ...)                                                    \
    parm::util::log_fn(                                                        \
        parm::util::Level::INFO, std::string(__func__), MSG, ##__VA_ARGS__)

/** @brief Log a message at DEBUG severity. */
#define PARM_DEBUG(MSG, ...)                                                   \
    parm::util::log_fn(                                                        \
        parm::util::Level::DEBUG, std::string(__func__), MSG, ##__VA_ARGS__)

/** @brief Log a message at TRACE severity. */
#define PARM_TRACE(MSG, ...)                                                   \
    parm::util::log_fn(                                                        \
        parm::util::Level::TRACE, std::string(__func__), MSG, ##__VA_ARGS__)


namespace parm {
namespace util {

/** @brief Logging severity levels, ordered from least to most verbose. */
enum Level : int {
    OFF   = 0, ///< Suppress all log output.
    ERROR = 1, ///< Error conditions; does not throw — use ``PARM_THROW`` for that.
    WARN  = 2, ///< Warnings: unexpected but recoverable situations.
    INFO  = 3, ///< Informational messages (default level).
    DEBUG = 4, ///< Detailed diagnostic output.
    TRACE = 5, ///< Per-step trace; very verbose.
    ALL   = 6  ///< Alias for TRACE; enables everything.
};

/** @brief Active log level; messages above this level are suppressed. */
inline Level currentLogLevel = INFO;

/**
 * @brief Set the active log level.
 *
 * @param level  Messages with severity above ``level`` are suppressed.
 */
[[maybe_unused]] inline void setLogLevel(const Level& level)
{
    currentLogLevel = level;
}

/** @brief Disable all log output (equivalent to ``setLogLevel(OFF)``). */
[[maybe_unused]] inline void disableLogging()
{
    setLogLevel(OFF);
}

/**
 * @brief Return an ANSI-coloured log-level tag string for terminal output.
 *
 * @param level  The severity level to render.
 * @return Bracketed, coloured tag string, e.g. `"* [INFO ]"`.
 */
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

/**
 * @brief Log a formatted message at the given severity level.
 *
 * The message is suppressed if `level > currentLogLevel`.
 * Formatting follows `printf` conventions; the format string is `msg`.
 *
 * @tparam Args  Variadic pack of `printf`-compatible argument types.
 * @param level  Severity level of the message.
 * @param msg    `printf`-style format string.
 * @param args   Format arguments.
 */
template<typename... Args>
void log(const Level& level, const std::string& msg, Args... args)
{
    if (level > currentLogLevel)
        return;

    std::string message = " " + getLevelTag(level) + " " + msg + "\n";
    // Yes, we're blatantly suppressing a security warning here
    // This is only an issue if the format string `msg` is determined at
    // runtime, which is not usually the case (since this function is typically
    // called via the PARM_INFO etc macros)
    // But more importantly, this is not a security-sensitive piece of software
#if __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
#endif
    std::printf(message.c_str(), args...);
#if __GNUC__
#pragma GCC diagnostic pop
#endif
}

/**
 * @brief Log a formatted message prefixed with a function name.
 *
 * Prepends `[func]` to `msg` and delegates to `log()`.
 * Used internally by the `PARM_INFO`, `PARM_DEBUG`, … macros.
 *
 * @tparam Args  Variadic pack of `printf`-compatible argument types.
 * @param level  Severity level of the message.
 * @param func   Name of the calling function (typically `__func__`).
 * @param msg    `printf`-style format string.
 * @param args   Format arguments.
 */
template<typename... Args>
void log_fn(const Level& level, const std::string& func, const std::string& msg,
    Args... args)
{
    std::string message = "[" + func + "] " + msg;
    log(level, message, args...);
}

} // namespace util
} // namespace parm
