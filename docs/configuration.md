# Configuration

The program propagates forward in an Earth-centered inertial frame using fixed-step RKF78. The CPU and GPU backends share the same configuration format.

## Run Parameters

| Field | Description |
| --- | --- |
| `run.duration_seconds` | Positive propagation duration, in s |
| `execution.integration.initialStepSize` | Positive fixed step size, in s; the final step is shortened to match the remaining duration |
| `execution.integration.maxNumSteps` | Maximum number of integration steps; must be sufficient to complete the requested duration |
| `execution.maxThreads` | CPU thread count; a positive integer, defaulting to 1 |
| `model.environment` | Celestial bodies, force models, and environment-data configuration |

Models are configured through `model.environment.bodies` and `earthCorrections`. See [envisat_full.json](../example/config/envisat_full.json) for a complete example and [Force Models](force_models.md) for model scope.

## Object Inputs

The input file contains a `targets` array. Each object must have a unique `id`. Objects share the force-model configuration but may have different initial states, epochs, and physical parameters.

| Field | Description |
| --- | --- |
| `epoch_mjd2000_tdb` | Days since 2000-01-01 00:00:00 TDB |
| `state_km_km_s` | `[x,y,z,vx,vy,vz]` in Earth-centered, ICRF-aligned inertial axes; position in km and velocity in km/s |
| `mass_kg` | Object mass, in kg |
| `drag_area_m2`, `srp_area_m2` | Effective atmospheric drag and solar radiation pressure areas, in m² |
| `cd`, `cr` | Dimensionless drag and solar radiation pressure coefficients |

See [envisat.json](../example/input/envisat.json) for an input example.

## Environment Data

Data paths are resolved relative to the configuration file's directory. The full example requires the following files:

| Path within the project | Contents |
| --- | --- |
| `data/constants.json` | Celestial-body and physical constants |
| `data/local/ephemeris/de440s.brie` | Celestial-body ephemerides |
| `data/local/gravity/ggm03c_360_sha.tab` | GGM03C Earth gravity-field coefficients |
| `data/local/earth/SW-All.txt` | Solar and geomagnetic activity data for the atmosphere model |
| `data/local/earth/EOP-All.txt` | Earth orientation parameters |
| `data/local/earth/itrf_iers2000_20110110_20110202.brot2` | Preprocessed Earth orientation data |

Git excludes `data/local/`. The [manifest.json](../data/manifest.json) file lists the sizes and SHA-256 checksums of the example data. Obtain these files or generate data in compatible formats; downloading the source alone is not sufficient to run the full example. Renaming a raw DE440 ephemeris file does not convert it to `.brie`, and Earth orientation text files cannot directly replace `.brot2` files.

Environment data must cover all epochs required by all objects. The example Earth orientation file covers January 10 through February 2, 2011. When changing object epochs or propagation duration, prepare data for the new interval and update the configured coverage ranges accordingly.

## Command-Line Usage and Output

From the project root, run `build/bin/parahpop_cpu CONFIG.json TARGETS.json OUTPUT_DIRECTORY`. For GPU execution, use `build/bin/parahpop_gpu`.

The program writes final states, not trajectories at every integration step. See the [project README](../README.md#run-the-envisat-example) for output files and units. Use an output directory that does not contain existing result files.