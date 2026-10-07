#pragma once

#include "paraHPOP/model/physics/full/detail/RefBase.h"
#include "interface/config/Model.h"

namespace paraHPOP {
namespace model {
namespace physics {
namespace full {
namespace detail {

template<typename ReducedT_>
class Base {
    using InterfaceT = interface::config::Model;

public:
    
    static constexpr idx_t ORDER = 2;
    
    static constexpr idx_t Dim = 3;
    
    using ReducedT = ReducedT_;
    
    using AccT = accelerations::Accelerations;
    
    using EnvT = environment::Env;
    
    using StatesT = typename ReducedT::StatesT;

    template<bool work, bool MaybeVolatile = false>
    using Ref = RefBase<ReducedT, work, MaybeVolatile>;
    
    using GRef = Ref<false>;
    
    using WRef = Ref<true>;
    
    using VolatileRef = Ref<true, true>;

    Base() = delete;

    Base(Base& other) = delete;

    Base(const InterfaceT& icfg)
        : reduced_{ icfg }
    {
    }

    Base(ReducedT&& reduced)
        : reduced_{ std::move(reduced) }
    {
    }

    Base(Base&& other)
        : reduced_{ std::move(other.reduced_) }
    {
    }

    Base& operator=(Base&& other)
    {
        this->reduced_  = std::move(other.reduced_);
        return *this;
    }

    void upload(const cudaStream_t& stream = 0)
    {
        reduced_.upload(stream);
    }

    void download(const cudaStream_t& stream = 0)
    {
        reduced_.download(stream);
    }

    void clearDevice()
    {
        reduced_.clearDevice();
    }

    GRef hostRef() const
    {
        return GRef::make(reduced_.hostRef());
    }

    GRef deviceRef() const
    {
        return GRef::make(reduced_.deviceRef());
    }

    ReducedT& reduced() { return reduced_; }
    const ReducedT& reduced() const { return reduced_; }

    EnvT& env() { return reduced_.env(); }
    const EnvT& env() const { return reduced_.env(); }

    AccT& accs() { return reduced_.accs(); }
    const AccT& accs() const { return reduced_.accs(); }

protected:
    ReducedT reduced_;
};

} // namespace detail
} // namespace full
} // namespace physics
} // namespace model
} // namespace paraHPOP