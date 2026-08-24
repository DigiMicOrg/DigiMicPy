# Numerical Jacobian recipe

The [platform stability workflow](https://digimicorg.github.io/workflows/stability/)
defines equilibrium checks, local stability, reactivity, feasibility, and
reporting requirements. DigiMicPy has no public Jacobian or stability helper,
but its pure `micrm_rhs` can be differentiated numerically.

```python
import numpy as np

from digimicpy import micrm_rhs


def numerical_jacobian(parameters, state, relative_step=1e-6):
    state = np.asarray(state, dtype=float)
    jacobian = np.empty((state.size, state.size))
    steps = relative_step * np.maximum(1.0, np.abs(state))

    for column, step in enumerate(steps):
        offset = np.zeros_like(state)
        offset[column] = step
        jacobian[:, column] = (
            micrm_rhs(0.0, state + offset, parameters)
            - micrm_rhs(0.0, state - offset, parameters)
        ) / (2.0 * step)

    return jacobian


state_equilibrium = result.y[:, -1]
residual = np.max(
    np.abs(micrm_rhs(result.t[-1], state_equilibrium, parameters))
)
J = numerical_jacobian(parameters, state_equilibrium)
eigenvalues = np.linalg.eigvals(J)
leading_real_part = np.max(eigenvalues.real)
reactivity = np.max(np.linalg.eigvalsh((J + J.T) / 2.0))
```

Confirm that `residual` is sufficiently small before interpreting the spectrum.
Repeat with smaller and larger finite-difference steps, and report the step rule,
solver tolerances, endpoint residual, and state variables retained in the
Jacobian.
