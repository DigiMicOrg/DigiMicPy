# DigiMicPy resource-flux recipe

The [platform resource-flux workflow](https://digimic.org/workflows/resource-flux/)
defines the quantities and reporting conventions. DigiMicPy does not currently
provide a resource-flux helper; calculate fluxes explicitly from a state and the
validated parameter arrays.

```python
import numpy as np

state = result.y[:, -1]
C = state[:parameters.n_consumers]
R = state[parameters.n_consumers:]

uptake_flux = C[:, None] * parameters.uptake * R[None, :]
eta = 1.0 - parameters.leakage.sum(axis=2)
retained_flux = uptake_flux * eta
leakage_flux = (
    C[:, None, None]
    * parameters.uptake[:, :, None]
    * R[None, :, None]
    * parameters.leakage
)
maintenance_flux = parameters.mortality * C

community_uptake = uptake_flux.sum()
community_retained = retained_flux.sum()
community_leakage = leakage_flux.sum()
community_maintenance = maintenance_flux.sum()

direct_cue = (
    community_retained / community_uptake
    if community_uptake > 0.0
    else np.nan
)
```

The calculation describes the supplied state. Check the derivative norm before
calling it an equilibrium flux. For cross-community prediction, evaluate
potential uptake using a common reference resource vector rather than each
community's post-assembly resource state.
