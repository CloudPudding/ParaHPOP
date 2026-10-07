#pragma once

// CUDA headers define __noinline__ which interferes with libstdc++'s use of
// `__attribute((__noinline__))`. In order to avoid compilation error,
// temporarily unset __noinline__ when we include affected libstdc++ headers.
// Only applies when compiling with clang(d).
// See https://github.com/llvm/llvm-project/issues/62939#issuecomment-1563455451

#ifdef __clang__
#pragma push_macro("__noinline__")
#undef __noinline__
#endif
#include <stdexcept>
#ifdef __clang__
#pragma pop_macro("__noinline__")
#endif

#include <string>


/** Throw exception */
#define FETA_THROW(EXCEPTION_TYPE, MESSAGE)                                    \
    ::feta::err::detail::Throw<EXCEPTION_TYPE>(                                \
        __FILE__, __LINE__, static_cast<const char*>(__func__))(MESSAGE);

/** Throw exception if condition not met */
#define FETA_ASSERT(CONDITION, MESSAGE)                                        \
    if (!(CONDITION)) {                                                        \
        ::feta::err::detail::Throw<std::runtime_error>(                        \
            __FILE__, __LINE__, static_cast<const char*>(__func__))(           \
            "Condition check failed: " #CONDITION ". Help message: "           \
            + std::string(MESSAGE));                                           \
    }

namespace feta {
namespace err {

namespace detail {

/**
 * @brief Throw an error with a detailed message including user provided
 * message.
 *
 * @tparam ErrorType  Type of error to throw, defaults to std::runtime_error.
 * @tparam Params     Additional parameter pack for Errortype
 */
template<typename ErrorType = std::runtime_error, typename... Params>
class Throw {

    using LineType = std::decay<decltype(__LINE__)>::type;

public:
    /**
     * @brief Construct a new Throw object
     *
     * @param[in] file File source of throw
     * @param[in] line Line source of throw
     * @param[in] func Function source of throw
     */
    Throw(const std::string& file, LineType line, const std::string& func)
        : file_{ file }
        , line_{ line }
        , func_{ func }
    {
    }

    /**
     * @brief Generate throw message and throw
     *
     * @param[in] message User provided message
     * @param[in] params  Additional parameter pack for Errortype
     */
    [[noreturn]] void operator()(const std::string& message, Params... params)
    {
        std::string msg("\nException raised: ");
        msg += "\nWhat : ";
        msg += message;
        msg += "\nFunc : ";
        msg += func_;
        msg += "\nSrc  : ";
        msg += file_;
        msg += "\nLine : ";
        msg += std::to_string(line_);
        msg += "\n";
        throw ErrorType(msg, params...);
    }

private:
    const std::string file_;
    const LineType line_;
    const std::string func_;
};

} // namespace detail

} // namespace err
} // namespace feta
