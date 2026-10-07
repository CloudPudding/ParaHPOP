# Force Models

The models evaluate acceleration contributions from each object's Earth-centered inertial position, velocity, epoch, and physical parameters. The CPU and GPU backends use the same model configuration.

## Models and Inputs

| Model | Main inputs and configuration |
| --- | --- |
| Earth central gravity | Earth-centered position and Earth's gravitational parameter |
| Spherical-harmonic gravity | Earth gravity-field coefficients, Earth orientation, and truncation degree and order; the full example uses GGM03C 70×70 |
| Atmospheric drag | NRLMSISE-00 atmospheric density, solar and geomagnetic activity, relative velocity, mass, drag area, and drag coefficient |
| Solar radiation pressure | Sun position, sunlight fraction, mass, effective area, and radiation pressure coefficient |
| Third-body gravity | Ephemerides and gravitational parameters for the Sun, Moon, and enabled planetary systems |
| Solid Earth tides | Sun and Moon positions, Earth orientation, and the tide-system convention |
| Ocean tides | FES2004 coefficients, epoch, and Earth orientation parameters |
| Relativistic correction | Earth-centered position and velocity, Earth's gravitational parameter, and the speed of light |

The sunlight fraction accounts for the effect of occultation on solar radiation pressure. The full example includes the Sun, Moon, and the system barycenters of Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune, and Pluto as third bodies.

## Tidal and Relativistic Correction Settings

Set the following fields within `model.environment`:

```json
"earthCorrections": {
  "model": "iers2010_fes2004",
  "solidEarthTides": true,
  "oceanTides": true,
  "relativity": true,
  "tideSystem": "zero_tide",
  "eopFile": "../../data/local/earth/EOP-All.txt"
}
```

The path above assumes that the configuration file is in `example/config/`; adjust it for other locations.

- Solid Earth tides include degree-2 and degree-3 direct terms, induced degree-4 terms, frequency-dependent corrections, and the solid pole tide. The mean pole follows the IERS 2010 convention.
- Ocean tides use 18 FES2004 constituents and 402 coefficient records, truncated to degree and order 6, plus the degree-2, order-1 ocean pole tide. Minor-constituent admittance expansion is not included.
- The relativistic model includes only the Earth Schwarzschild correction, not Lense–Thirring or de Sitter terms.
- `tideSystem` must match the static gravity field. The full example uses `zero_tide`; use `tide_free` only for a tide-free gravity field. This setting does not convert the static gravity-field coefficients.

## Scope and Usage Notes

Environment data must cover the object epochs and propagation intervals. Select the step size, gravity-field truncation, and physical parameters according to the task's accuracy requirements. CPU/GPU consistency does not establish absolute accuracy against the true orbit.

The tidal gradient calculation requires a nonzero object position that does not lie exactly on the terrestrial z-axis. The source-data phase warning for the long-period FES2004 constituents Om1/Om2 also applies to this model.

Object inputs and outputs use km and km/s. Internal tidal and relativistic interfaces use m and m/s, with unit conversion performed when acceleration contributions are combined.

## References

- [IERS Conventions 2010](https://iers-conventions.obspm.fr/conventions/content/tn36.pdf), Chapters 6, 7, and 10.
- [IERS Chapter 6 clarifications](https://iers-conventions.obspm.fr/content/chapter6/icc6.pdf) and [coefficient corrections and phase notes](https://iers-conventions.obspm.fr/chapter6.php).
- [FES2004 Stokes coefficients](https://iers-conventions.obspm.fr/content/chapter6/additional_info/tidemodels/fes2004_Cnm-Snm.dat).
- [GGM03 model notes](https://download.csr.utexas.edu/pub/grace/GGM03/GGM03_Notes.pdf).