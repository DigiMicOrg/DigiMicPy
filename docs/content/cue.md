# DigiMicPy carbon-use-efficiency recipe

The [platform CUE workflow](https://digimic.org/workflows/carbon-use-efficiency/)
owns the definitions, reference-environment choices, interpretation, and
reporting guidance. DigiMicPy does not currently expose a CUE helper; the
calculation uses public parameter arrays and a solver endpoint.

```python
import numpy as np

C_eq = result.y[:parameters.n_consumers, -1]
R0 = np.asarray(reference_resources, dtype=float)
survivors = C_eq > 1e-5

eta = 1.0 - parameters.leakage.sum(axis=2)
uptake0 = np.sum(parameters.uptake * R0[None, :], axis=1)
retained0 = np.sum(parameters.uptake * eta * R0[None, :], axis=1)

valid = uptake0 > 0.0
species_cue = np.full(parameters.n_consumers, np.nan)
species_cue[valid] = (
    retained0[valid] - parameters.mortality[valid]
) / uptake0[valid]

included = survivors & valid
if not np.any(included):
    raise ValueError("No surviving consumers have positive reference uptake")

biomass_cue = np.average(
    species_cue[included],
    weights=C_eq[included],
)
flux_cue = np.average(
    species_cue[included],
    weights=C_eq[included] * uptake0[included],
)
```

This recipe uses maintenance-adjusted species CUE. Remove the mortality term for
the gross convention. Use the same `R0`, survivor threshold, and zero-uptake
rule across every community being compared.
