#pragma once
#include "interface/config/model/Accelerations.h"
#include "interface/config/model/Environment.h"
namespace interface::config {
class Model {
public:
    using AccT = model::Accelerations;
    using EnvT = model::Environment;
    explicit Model(const json& j) : env_(j.at("environment")) {
        for (auto& item : j.items())
            if (item.key() != "environment")
                throw std::runtime_error("Unsupported model field: " + item.key());
        refresh();
    }
    EnvT& environment() { return env_; }
    const EnvT& environment() const { return env_; }
    const AccT& accelerations() const { return accs_; }
    const AccT& accelerations() { refresh(); return accs_; }
    Model& refresh() {
        accs_ = AccT(env_.bodies());
        env_.bodies().clearDirty();
        return *this;
    }
    json to_json() const { return {{"environment",env_.to_json()}}; }
private:
    AccT accs_;
    EnvT env_;
};
}
