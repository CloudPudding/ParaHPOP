# Architecture

## Overall Workflow

The CPU reads the configuration, object states, and environment data, and prepares the required memory. The CPU backend performs propagation using C++; the GPU backend transfers data to device memory and constructs and instantiates a graph for one RKF78 integration step.

During propagation, the GPU repeatedly executes the integration graph and updates object states. CPU task submission can overlap with GPU execution. Once propagation finishes, the final states are transferred to the host and saved.

## Integration Step

Each RKF78 integration step has 13 stages, executed in the order required by the stage coefficients. Each stage constructs a temporary state and evaluates its state derivative. After all stages are complete, eighth-order weights are used to update the state and epoch.

Objects are processed independently. Each object uses its own epoch, state, and physical parameters. The final step is shortened to match the remaining duration and reach the specified final epoch.

Ephemeris and Earth orientation data that depend only on the epoch can be reused at matching epochs in adjacent integration steps. Stage states and state derivatives are still evaluated separately.

## Force Evaluation Within a Stage

- Once the stage state is prepared, the relativistic correction, ephemeris evaluation, and position-derivative assignment can be scheduled independently.
- After ephemeris evaluation, data are prepared for the Earth, Sun, Moon, and other body branches. Each branch executes according to its own data dependencies.
- The Earth branch evaluates central gravity, atmospheric drag, spherical-harmonic gravity, solid Earth tides, and ocean tides in sequence. Disabled models are skipped.
- Solid Earth tides also require the Sun and Moon positions. These positions must be prepared even when solar and lunar third-body gravity are disabled.
- Solar gravity and the sunlight fraction can be scheduled independently. Solar radiation pressure waits for both the sunlight result and the solar gravity node to complete. The latter dependency prevents concurrent writes to the same acceleration buffer; the radiation pressure formula does not depend on the solar gravity result.
- The Moon and other body branches evaluate their respective third-body gravity contributions.
- Once all branches finish, acceleration contributions are summed in a fixed order and combined with the position derivative to form the complete stage derivative.

Nodes without mutual dependencies can be scheduled concurrently. Actual overlap depends on available GPU resources and workload size.

## Source-Code Entry Points

| Path | Purpose |
| --- | --- |
| `src/cpu/`, `src/gpu/` | CPU and GPU command-line entry points |
| `src/common/driver.cpp` | Configuration loading, input/output, and execution control |
| `src/common/engine/paraHPOP/propagators/host/` | CPU integration backend |
| `src/common/engine/paraHPOP/propagators/device/` | GPU integration backend |
| `src/common/engine/paraHPOP/model/` | Force models and environment-data evaluation |