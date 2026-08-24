# Simulation diagnostics

DigiMicPy returns SciPy solver results and exposes the pure `micrm_rhs`
right-hand side. These interfaces support endpoint checks, perturbation
experiments, and numerical differentiation without a separate analysis API.

## Check an endpoint

The final recorded time is not automatically an equilibrium. Calculate the
derivative at the endpoint and compare its largest absolute value with a
tolerance appropriate to the model units:

```python
import numpy as np
from digimicpy import micrm_rhs

state = result.y[:, -1]
residual = np.max(np.abs(micrm_rhs(result.t[-1], state, parameters)))
```

Also inspect `result.success`, `result.message`, minimum state values, and
sensitivity to the integration interval and solver tolerances.

## Perturb and reintegrate

```python
perturbed = state.copy()
perturbed[0] *= 0.5
perturbed[parameters.n_consumers] += 0.5

post = solve_micrm(
    parameters,
    perturbed,
    (0.0, 25.0),
    t_eval=np.linspace(0.0, 25.0, 150),
)
if not post.success:
    raise RuntimeError(post.message)
```

Compare recovery only after confirming that the reference endpoint was close
to equilibrium.

## Stability calculations

{doc}`micrm_stability` shows a package-specific finite-difference recipe using
`micrm_rhs`. DigiMicPy does not currently provide Jacobian, effective GLV,
stability, reactivity, or feasibility helpers. The scientific definitions and
reporting guidance are maintained in the
[platform stability workflow](https://digimic.org/workflows/stability/).
