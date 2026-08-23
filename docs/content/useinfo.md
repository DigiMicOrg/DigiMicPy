---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Basic usage

This executable example generates a modular community, constructs validated
parameters, integrates MiCRM, and plots the trajectories.

## 1. Generate community parameters

```{code-cell} ipython3
import numpy as np

from digimicpy import (
    MiCRMParameters,
    generate_l_tensor,
    modular_uptake,
    solve_micrm,
)

rng = np.random.default_rng(42)
n_consumers = 10
n_resources = 5
total_leakage = 0.1

uptake = modular_uptake(
    n_consumers,
    n_resources,
    n_modules=2,
    specialization_ratio=10.0,
    rng=rng,
)
leakage = generate_l_tensor(
    n_consumers,
    n_resources,
    n_modules=2,
    specialization_ratio=10.0,
    total_leakage=total_leakage,
    rng=rng,
)

parameters = MiCRMParameters(
    uptake=uptake,
    mortality=np.full(n_consumers, 0.2),
    resource_supply=np.full(n_resources, 0.5),
    resource_decay=np.full(n_resources, 0.5),
    leakage=leakage,
    leakage_fraction=np.full(n_resources, total_leakage),
)

parameters.uptake.shape, parameters.leakage.shape
```

## 2. Run the simulation

```{code-cell} ipython3
initial_state = np.concatenate(
    [np.full(n_consumers, 0.01), np.ones(n_resources)]
)
t_span = (0.0, 50.0)
t_eval = np.linspace(*t_span, 300)

result = solve_micrm(
    parameters,
    initial_state,
    t_span,
    t_eval=t_eval,
)
if not result.success:
    raise RuntimeError(result.message)

result.success, result.message
```

## 3. Plot trajectories

```{code-cell} ipython3
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 5))
for index in range(parameters.n_consumers):
    ax.plot(result.t, result.y[index], label=f"Consumer {index + 1}")
for index in range(parameters.n_resources):
    ax.plot(
        result.t,
        result.y[parameters.n_consumers + index],
        label=f"Resource {index + 1}",
        linestyle="--",
    )

ax.set(xlabel="Time", ylabel="Abundance", title="MiCRM dynamics")
ax.legend(ncol=3, fontsize=8)
fig.tight_layout()
```

## 4. Check the final state

```{code-cell} ipython3
from digimicpy import micrm_rhs

final_state = result.y[:, -1]
final_consumers = final_state[:parameters.n_consumers]
final_resources = final_state[parameters.n_consumers:]
derivative_norm = np.max(np.abs(micrm_rhs(result.t[-1], final_state, parameters)))

print("Final consumer biomasses:", np.round(final_consumers, 4))
print("Final resource abundances:", np.round(final_resources, 4))
print("Maximum absolute derivative:", f"{derivative_norm:.3e}")
```

The final recorded time is not automatically an equilibrium. Increase the
integration interval or use an equilibrium event if the derivative norm is not
small enough for the intended analysis.
