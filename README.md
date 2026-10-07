# ParaHPOP

A GPU-Accelerated High Precision Orbit Propagator for Large Object Populations

ParaHPOP provides CPU and GPU backends for propagating object states in an Earth-centered inertial frame.

## Features

- Double-precision computation with a fixed-step RKF78 integrator, using 13 stages and eighth-order state-update weights. The final step is shortened to match the remaining propagation duration.
- Earth central gravity, spherical-harmonic gravity, NRLMSISE-00 atmospheric drag, solar radiation pressure, lunar, solar, and planetary third-body gravity, solid Earth tides, ocean tides, and the Earth Schwarzschild relativistic correction.

The GPU backend processes independent objects in batches, repeatedly executes a single integration step through CUDA Graph, and reuses ephemeris and Earth orientation data at matching epochs in adjacent integration steps. See [Architecture](docs/architecture.md) for stage-level dependencies and [Force Models](docs/force_models.md) for model configuration and scope.

## Build

Build on Linux or in WSL2 on Windows. Requirements are CMake 3.22 or later, a C++ compiler, the CUDA Toolkit, OpenMP, and nlohmann_json 3.10.5 or later. The project uses C++23 and CUDA C++20 compilation settings; CUDA Toolkit 12.5 is recommended. Python 3 is used for optional comparison and validation scripts.

The CPU and GPU executables share a model library that contains CUDA code. The CPU executable therefore also requires the CUDA runtime, although its orbit propagation computations run on the CPU.

Run from the project root:

```bash
bash scripts/build.sh
```

The default outputs are `build/bin/parahpop_cpu`, `build/bin/parahpop_gpu`, and `build/lib/libparahpop_engine.so`.

Build options:

- `CUDA_ARCHITECTURES`: GPU compute capability, defaulting to 75. For example: `CUDA_ARCHITECTURES=86 bash scripts/build.sh`.
- `CUDACXX`: path to the CUDA compiler, such as `/usr/local/cuda-12.5/bin/nvcc`.
- `BUILD_JOBS`: number of parallel build jobs, defaulting to 2.
- `BUILD_DIR`: build directory, defaulting to `build/`. Set this variable when running examples from a custom build directory as well.

The default host compilation target is `x86-64-v3`. For CPUs that do not support this instruction set, configure `PARAHPOP_CPU_ARCH` in CMake; an empty string uses the compiler's default target.

From Windows PowerShell, run `./build.ps1`. The script uses the `Ubuntu-22.04` WSL distribution by default; use `-Distribution` to select another installed distribution. The resulting executables run on Linux/WSL.

## Configuration and Input

Run configurations and object inputs use JSON. Both backends use the same configuration format. The run configuration specifies the propagation duration, fixed step size, maximum number of integration steps, CPU thread count, and force models.

Objects are supplied in a `targets` array. Each object must have a unique `id` and may have its own initial state, epoch, mass, effective drag and solar radiation pressure areas, and coefficients. Objects in the same batch share the force-model configuration.

States use Earth-centered, ICRF-aligned inertial axes, with position in km and velocity in km/s. Epochs use MJD2000/TDB: days since 2000-01-01 00:00:00 TDB. See [Configuration](docs/configuration.md) for field definitions.

## Run the ENVISAT Example

Before running, prepare ephemerides, gravity-field coefficients, Earth orientation data, and space weather data as described in [Environment Data](docs/configuration.md#environment-data). Data paths are resolved relative to the configuration file's directory. Git ignores `data/local/`, so compatible data files must be obtained separately after downloading the source.

The example starts from an initial state on January 18, 2011, and propagates for 600 s with a fixed step size of 60 s. If you change the initial epoch or propagation duration, ensure that the environment data cover all required epochs.

Run from the project root:

```bash
bash example/run.sh cpu full output/envisat_cpu
bash example/run.sh gpu full output/envisat_gpu
python3 example/compare.py output/envisat_cpu output/envisat_gpu
```

Replace `full` with `central` to use only Earth central gravity. If the output directory is omitted, the script creates a timestamped directory. The program refuses to overwrite existing result files. See the [ENVISAT Example](example/README.md) for example parameters.

From PowerShell, run `./example/run.ps1 -Backend cpu` or `./example/run.ps1 -Backend gpu`.

To use your own configuration and object files, invoke the executable directly:

```bash
build/bin/parahpop_cpu CONFIG.json TARGETS.json OUTPUT_DIRECTORY
build/bin/parahpop_gpu CONFIG.json TARGETS.json OUTPUT_DIRECTORY
```

The program writes each object's final state, not a trajectory at every integration step. Output files include:

- `final_states.csv`: object ID, final epoch, position, and velocity, using the same units as the input.
- `summary.json`: object count, execution backend, timing, and final-epoch checks.
- `configuration_resolved.json`: the resolved run configuration.

CPU/GPU comparisons assess implementation consistency; they do not establish absolute accuracy against the true orbit or precise ephemerides.

## Documentation

- [Configuration](docs/configuration.md): run parameters, object inputs, environment data, and command-line usage.
- [Architecture](docs/architecture.md): overall workflow, RKF78 stages, force-model dependencies, and source-code entry points.
- [Force Models](docs/force_models.md): model inputs, tidal and relativistic correction settings, scope, and references.

Run `python3 scripts/validate.py` to check short-duration propagation, final-step shortening, object batches, and invalid-input handling. Results are written to a new `output/validation-*` directory.