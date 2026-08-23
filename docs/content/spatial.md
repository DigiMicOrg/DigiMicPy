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

# Spatial patches

DigiMicPy can couple several MiCRM patches through consumer migration and
resource diffusion. Each patch has its own `MiCRMParameters`, while transport
acts between matching state variables.

## Assumptions

The current implementation deliberately requires:

- the same number and ordering of consumer identities in every patch;
- the same number and ordering of resource identities in every patch;
- nonnegative, symmetric connectivity with a zero diagonal;
- consumer and resource diffusion coefficients shared across patches;
- equal-volume patches when states are concentrations, or states interpreted as
  total quantities.

These conditions make undirected transport conserve every consumer and resource
across the full landscape. Directed networks, heterogeneous identity sets, and
patch-specific diffusion require additional flux conventions and are not yet
implemented.

## Connectivity

For patch coordinates $x_k$, `distance_connectivity` creates:

$$
A_{kj}=\exp(-\lambda ||x_k-x_j||), \qquad A_{kk}=0.
$$

`decay_rate` $\lambda$ has inverse units of the supplied coordinates. Larger
values make coupling fall off more quickly with distance.

```{code-cell} ipython3
from digimicpy import distance_connectivity

connectivity = distance_connectivity(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
    decay_rate=1.0,
)
connectivity
```

## Conservative transport

For any matching consumer or resource state $X_k$ and diffusion coefficient
$d$, transport is:

$$
\left.\frac{dX_k}{dt}\right|_{\mathrm{transport}}
=d\sum_j A_{kj}(X_j-X_k).
$$

Symmetric $A$ makes the transport derivatives sum to zero. Diffusion may be a
scalar or a vector with one coefficient per consumer or resource.

## Fixed-temperature landscape example

The same reference community can be evaluated at different fixed patch
temperatures and then coupled:

```python
patch_parameters = [
    temperature_adjusted_parameters(reference, 288.15, 283.15, **traits),
    temperature_adjusted_parameters(reference, 298.15, 283.15, **traits),
]
result = solve_spatial_micrm(
    patch_parameters,
    initial_state,
    (0.0, 20.0),
    connectivity=connectivity,
    consumer_diffusion=0.01,
    resource_diffusion=0.05,
)
```

See the
[`thermal_spatial_simulation.py` example](https://github.com/DigiMicOrg/DigiMicPy/blob/main/examples/thermal_spatial_simulation.py)
for a complete, tested program.
