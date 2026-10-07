// ParaHPOP standalone runner. Modified/added 2026-10-07.
// The numerical engine and upstream notices are preserved separately.
#include <paraHPOP/driver.h>
#include <paraHPOP/run.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

namespace paraHPOP {
namespace fs = std::filesystem;
using Json = nlohmann::json;

static Json read_json(const fs::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("Cannot read file: " + path.string());
    Json value; stream >> value; return value;
}

static void check_keys(const Json& value, std::initializer_list<const char*> allowed,
                       const char* context) {
    if (!value.is_object()) throw std::runtime_error(std::string(context) + " must be an object.");
    for (const auto& item : value.items()) {
        bool found = false;
        for (const auto* key : allowed) found |= item.key() == key;
        if (!found) throw std::runtime_error(std::string("Unsupported ") + context + " option: " + item.key());
    }
}

// Resolve only configuration fields documented as paths. Relative paths are
// relative to the configuration file, never to an old development directory.
static void resolve_paths(Json& value, const fs::path& base, const std::string& key = "") {
    const bool path_field = key == "file" || key == "spaceWeatherFile" || key == "eopFile"
        || key == "constants" || key == "ephemeris" || key == "orientations";
    if (value.is_string() && path_field) {
        fs::path path(value.get<std::string>());
        if (path.is_relative()) path = base / path;
        path = fs::absolute(path).lexically_normal();
        if (!fs::is_regular_file(path)) throw std::runtime_error("Required data file not found: " + path.string());
        value = path.string();
    } else if (value.is_object()) {
        for (auto& item : value.items()) resolve_paths(item.value(), base, item.key());
    } else if (value.is_array()) {
        for (auto& item : value) resolve_paths(item, base, key);
    }
}

int run(int argc, char** argv, const char* backend) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: " << argv[0] << " CONFIG.json TARGETS.json OUTPUT_DIRECTORY\n"
                      << "Backend: " << backend << "; integrator: RKF78 "
                      << "fixed step" << "\n";
            return 0;
        }
        if (argc != 4) throw std::runtime_error("Expected CONFIG.json TARGETS.json OUTPUT_DIRECTORY (or --help).");
        fs::path config_path = fs::absolute(argv[1]);
        fs::path input_path = fs::absolute(argv[2]);
        fs::path output_path = fs::absolute(argv[3]);
        for (const auto* filename : {"final_states.csv", "summary.json", "configuration_resolved.json"})
            if (fs::exists(output_path / filename))
                throw std::runtime_error("Output exists; use a new directory: " + output_path.string());
        Json config = read_json(config_path);
        check_keys(config, {"run", "execution", "model"}, "configuration");
        check_keys(config.at("run"), {"duration_seconds"}, "run");
        check_keys(config.at("execution"), {"maxThreads", "integration"}, "execution");
        check_keys(config.at("execution").at("integration"), {"initialStepSize", "maxNumSteps"}, "integration");
        const int max_steps = config.at("execution").at("integration").at("maxNumSteps").get<int>();
        const int threads = config.at("execution").value("maxThreads", 1);
        if (max_steps <= 0 || threads <= 0) throw std::runtime_error("maxNumSteps and maxThreads must be positive integers.");
        const double duration = config.at("run").at("duration_seconds").get<double>();
        config.erase("run");
        if (!(duration > 0) || !std::isfinite(duration)) throw std::runtime_error("duration_seconds must be finite and positive.");
        config["execution"]["on"] = backend;
        const double step = config.at("execution").at("integration").at("initialStepSize").get<double>();
        if (!(step > 0) || !std::isfinite(step)) throw std::runtime_error("initialStepSize must be finite and positive.");
        resolve_paths(config, config_path.parent_path());
        Json input = read_json(input_path);
        const auto& targets = input.at("targets");
        if (!targets.is_array() || targets.empty()) throw std::runtime_error("targets must be a nonempty array.");
        if (targets.size() > static_cast<std::size_t>(std::numeric_limits<feta::idx_t>::max()))
            throw std::runtime_error("Too many targets for the engine index type.");
        const feta::idx_t target_count = static_cast<feta::idx_t>(targets.size());
        interface::samples::Collection samples(target_count);
        std::set<int> ids;
        double earliest = std::numeric_limits<double>::infinity();
        double latest = -earliest;
        for (feta::idx_t i = 0; i < target_count; ++i) {
            const auto& target = targets.at(i);
            const int id = target.at("id").get<int>();
            if (!ids.insert(id).second) throw std::runtime_error("Target IDs must be unique.");
            const auto state = target.at("state_km_km_s").get<std::array<double,6>>();
            for (double x : state) if (!std::isfinite(x)) throw std::runtime_error("State contains a nonfinite value.");
            const double epoch = target.at("epoch_mjd2000_tdb").get<double>();
            const double mass = target.at("mass_kg").get<double>();
            const double area = target.at("drag_area_m2").get<double>();
            const double srp_area = target.at("srp_area_m2").get<double>();
            const double cd = target.at("cd").get<double>();
            const double cr = target.at("cr").get<double>();
            if (!std::isfinite(epoch) || !std::isfinite(mass) || !std::isfinite(area) ||
                !std::isfinite(srp_area) || !std::isfinite(cd) || !std::isfinite(cr) ||
                mass <= 0 || area <= 0 || srp_area < 0 || cd < 0 || cr < 0)
                throw std::runtime_error("Invalid target epoch or physical parameter.");
            samples.ids()[i] = id;
            samples.states()[i] = state;
            samples.epochs()[i] = epoch;
            samples.centres()[i] = 399;
            samples.mass()[i] = mass;
            samples.area()[i] = area * 1e-6; // m^2 -> km^2
            samples.cd()[i] = cd;
            samples.cr()[i] = cr * srp_area / area;
            earliest = std::min(earliest, epoch);
            latest = std::max(latest, epoch + duration / 86400.0);
        }
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
        if (std::string(backend) == "device") {
            int count = 0;
            const auto status = cudaGetDeviceCount(&count);
            if (status != cudaSuccess || count == 0)
                throw std::runtime_error("No usable CUDA device: " + std::string(cudaGetErrorString(status)));
        }
        interface::config::Model cfg(config.at("model"));
        const auto environment_start = std::chrono::steady_clock::now();
        cfg.environment().load();
        cfg.environment().assertEpochCoverage(earliest, latest, "paraHPOP propagation");
        const auto integration_start = std::chrono::steady_clock::now();
        const auto final = paraHPOP::run(cfg, samples, duration, step,
            max_steps, threads, std::string(backend)=="device");
        if (std::string(backend) == "device") {
            const auto status = cudaDeviceSynchronize();
            if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        }
        const auto integration_end = std::chrono::steady_clock::now();
        if (static_cast<std::size_t>(final.size()) != targets.size()) throw std::runtime_error("Final target count mismatch.");
        double max_epoch_error_s = 0;
        for (feta::idx_t i = 0; i < target_count; ++i) {
            if (!final.status().endOfSimulation()[i])
                throw std::runtime_error("A target stopped without reaching the requested end of simulation.");
            const double expected = targets.at(i).at("epoch_mjd2000_tdb").get<double>() + duration / 86400.0;
            const double error_s = std::abs(final.epochs()[i] - expected) * 86400.0;
            if (!std::isfinite(error_s) || error_s > 1e-4) throw std::runtime_error("A target did not reach its requested end epoch.");
            max_epoch_error_s = std::max(max_epoch_error_s, error_s);
            for (int j = 0; j < 6; ++j)
                if (!std::isfinite(static_cast<double>(final.states()[i][j]))) throw std::runtime_error("Nonfinite final state.");
        }
        fs::create_directories(output_path);
        std::ofstream csv(output_path / "final_states.csv");
        csv << "id,epoch_mjd2000_tdb,x_km,y_km,z_km,vx_km_s,vy_km_s,vz_km_s\n" << std::setprecision(17);
        for (feta::idx_t i = 0; i < target_count; ++i) {
            csv << final.ids()[i] << ',' << final.epochs()[i];
            for (int j = 0; j < 6; ++j) csv << ',' << final.states()[i][j];
            csv << '\n';
        }
        csv.close();
        if (!csv) throw std::runtime_error("Failed to write final_states.csv.");
        Json summary = {{"backend", backend}, {"targets", targets.size()},
            {"integrator", "RKF78 fixed step"},
            {"initial_step_seconds", step}, {"duration_seconds", duration},
            {"environment_seconds", std::chrono::duration<double>(integration_start-environment_start).count()},
            {"propagation_seconds", std::chrono::duration<double>(integration_end-integration_start).count()},
            {"max_end_epoch_error_seconds", max_epoch_error_s}, {"validation_passed", true}};
        std::ofstream resolved_stream(output_path / "configuration_resolved.json");
        resolved_stream << config.dump(2) << '\n';
        resolved_stream.close();
        if (!resolved_stream) throw std::runtime_error("Failed to write resolved configuration.");
        std::ofstream summary_stream(output_path / "summary.json");
        summary_stream << summary.dump(2) << '\n';
        summary_stream.close();
        if (!summary_stream) throw std::runtime_error("Failed to write summary.");
        std::cout << summary.dump(2) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ParaHPOP: " << error.what() << '\n';
        return 1;
    }
}
}
