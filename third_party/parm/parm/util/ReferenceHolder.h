/** @brief Templated holder - class to hold both host and device references of a
 * given object */
#pragma once

#include "parm/typedefs.h"
#include "parm/util/throw.h"

namespace parm {
namespace util {

/** @brief Reference holder: it can hold the reference to any object, provided
 * that a corresponding Global Reference is (`::GRef` or `::Ref<false>`) is
 * available */
template<typename MyClass>
class ReferenceHolder {
    using Self = ReferenceHolder;

public:
    /** @brief expose the reference type.  ``MaybeVolatile`` is
     * forwarded to ``MyClass::Ref<work, MaybeVolatile>`` — this
     * works as long as ``MyClass`` itself exposes the 2-arg form
     * (all parm/brie/cudaj fully-paired types do post-this-change). */
    template<bool work, bool MaybeVolatile = false>
    using Ref  = typename MyClass::template Ref<work, MaybeVolatile>;
    using GRef = Ref<false>;
    using WRef = Ref<true>;
    /** @brief Volatile work reference */
    using VolatileRef = Ref<true, true>;
    /** @brief Expose underlying class */
    using T = MyClass;

    /** @brief Default constructor is forbidden */
    ReferenceHolder() = delete;

    /** @brief Copy constructor is forbidden */
    ReferenceHolder(ReferenceHolder& other)       = delete;
    ReferenceHolder(const ReferenceHolder& other) = delete;

    /** @brief Host constructor only, with a reference of the object to be held
     */
    ReferenceHolder(const MyClass& myclass)
        : ptr_{ &myclass }
    {
    }

    /** @brief Construct from pointer */
    ReferenceHolder(const MyClass* ptr)
        : ptr_{ ptr }
    {
    }

    /** @brief Move constructor */
    ReferenceHolder(ReferenceHolder&& other)
        : ptr_{ std::exchange(other.ptr_, nullptr) }
    {
    }

    /** @brief Copy assignment is forbidden */
    ReferenceHolder& operator=(ReferenceHolder& other)       = delete;
    ReferenceHolder& operator=(const ReferenceHolder& other) = delete;

    /** @brief Move assignment is allowed */
    ReferenceHolder& operator=(ReferenceHolder&& other)
    {
        ptr_ = std::exchange(other.ptr_, nullptr);
        return *this;
    }

    /** @brief Value-return host reference */
    GRef hostRef() const
    {
        PARM_ASSERT(ptr_, "Invalid inner class reference");
        return ptr_->hostRef();
    }
    GRef ref() const { return hostRef(); }

#ifndef PARM_CPU_ONLY

    /** @brief Value-return device reference */
    GRef deviceRef() const
    {
        PARM_ASSERT(ptr_, "Invalid inner class reference");
        return ptr_->deviceRef();
    }

    /** @brief Async memcpy from host to device */
    void upload(const cudaStream_t& stream = 0)
    {
        const_cast<MyClass*>(ptr_)->upload(stream);
    }

    /** @brief Async memcpy from device to host */
    void download(const cudaStream_t& stream = 0)
    {
        const_cast<MyClass*>(ptr_)->download(stream);
    }

#endif

    /** @brief Clone the reference holder */
    ReferenceHolder clone() const { return ReferenceHolder(ptr_); }

    /** @brief Return whether the reference points to a valid object */
    bool isValid() const { return ptr_ != nullptr; }

    /** @brief expose the size of the underlying data */
    idx_t size() const
    {
        if (!isValid())
            return 0;
        else
            return ptr_->size();
    }

    /** @brief Arrow operator - give access to the underlined class */
    const MyClass* operator->() const
    {
        PARM_ASSERT(ptr_ != nullptr, "Invalid inner class reference");
        return ptr_;
    }

protected:
    const MyClass* ptr_ = nullptr;
};

} // namespace util
} // namespace parm