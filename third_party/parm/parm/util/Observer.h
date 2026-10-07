/** @brief Templated observer for observer pattern - used to track move
 * semantics with custom references */
#pragma once

#include "parm/typedefs.h"
#include "parm/util/throw.h"

namespace parm {
namespace util {

/** @brief Move semantics observer */
template<typename T>
class Observer {
public:
    /** @brief Default construction */
    Observer() = default;

    /** @brief Construct by directly observing an object */
    Observer(T& obj) { observe(obj); }

    /** @brief Construct with the given pointer to object (if not nullptr) */
    Observer(T* obj)
    {
        if (obj != nullptr)
            observe(*obj);
    }

    /** @brief Copy construction is forbidden */
    Observer(Observer& other)       = delete;
    Observer(const Observer& other) = delete;

    /** @brief Move construction is allowed */
    Observer(Observer&& other) { *this = std::move(other); }

    /** @brief Copy assignment is forbidden */
    Observer& operator=(Observer& other)       = delete;
    Observer& operator=(const Observer& other) = delete;

    /** @brief Move assignment is allowed */
    Observer& operator=(Observer&& other)
    {
        ref_ = other.ref_;
        updateRegistration_(std::move(other));
        other.ref_ = nullptr;
        return *this;
    }

    /** @brief Observe */
    void observe(T& obj)
    {
        if (ref_ != &obj)
            obj.registerObserver(this);
    }

    /** @brief Update internal refernece on the move */
    void onNotify(T* newobj) { ref_ = newobj; }

    /** @brief Expose the log */
    T* operator->()
    {
        assertValidObj_();
        return ref_;
    }
    const T* operator->() const
    {
        assertValidObj_();
        return ref_;
    }

    /** @brief Destructor calls de-registration */
    ~Observer()
    {
        if (ref_ != nullptr)
            ref_->deregisterObserver(this);
    }

    /** @brief Create a clone of this observer */
    Observer clone() const { return Observer(const_cast<T&>(ref_)); }

private:
    /** @brief run registration and de-registration on move */
    void updateRegistration_(Observer&& other)
    {
        if (other.ref_ != nullptr) {
            ref_->deregisterObserver(&other);
            ref_->registerObserver(this);
        }
    }

    /** @brief Assert the obj validity */
    void assertValidObj_() const
    {
        FETA_ASSERT(ref_ != nullptr, "Invalid object reference in Observer");
    }

    T* ref_ = nullptr;
};

} // namespace util
} // namespace parm