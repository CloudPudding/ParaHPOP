# ENVISAT Example

The input file is [input/envisat.json](input/envisat.json). The initial UTC epoch is 2011-01-18T00:00:26Z, corresponding to MJD2000/TDB 4035.0010669489266.

The object mass is 7835.953 kg, the drag area is 55.64 m², the solar radiation pressure area is 88.4 m², the drag coefficient is 2.7, and the solar radiation pressure coefficient is 1.0. The program uses the two effective areas for their respective force calculations.

The example propagates for 600 s with a fixed step size of 60 s and writes the final state:

- [config/envisat_full.json](config/envisat_full.json): enables the full set of [force models](../docs/force_models.md).
- [config/envisat_central.json](config/envisat_central.json): enables only Earth central gravity.

Before running, [build the project](../README.md#build) and prepare the [environment data](../docs/configuration.md#environment-data). Both configurations require the ephemeris and constants files referenced in their configurations.

Run from the project root:

```bash
bash example/run.sh cpu full output/envisat_cpu
bash example/run.sh gpu full output/envisat_gpu
python3 example/compare.py output/envisat_cpu output/envisat_gpu
```

The output directory must not contain existing result files. If the output directory is omitted, the run script creates a timestamped directory.

By default, the comparison script requires the final position difference to be no greater than 1 mm and the velocity difference to be no greater than 1e-6 m/s. This checks CPU/GPU consistency for the short example; it does not establish absolute accuracy against precise ephemerides.