# DigiMicPy

[![CI](https://github.com/DigiMicOrg/DigiMicPy/actions/workflows/ci.yml/badge.svg)](https://github.com/DigiMicOrg/DigiMicPy/actions/workflows/ci.yml)

DigiMicPy is a Python package for constructing and simulating microbial
consumer-resource models (MiCRM). It provides a validated model core,
reproducible modular parameter generators, fixed-temperature trait scaling,
and conservative coupling between spatial patches.

> **Project status:** DigiMicPy is an early, source-distributed release. The
> functions listed below are implemented and tested. The documentation also
> contains theory for planned analysis workflows; those pages are explicitly
> marked when no package helper exists yet.

## Implemented features

- validated `MiCRMParameters` objects with copied, read-only arrays;
- reproducible modular uptake and leakage generation using
  `numpy.random.Generator`;
- vectorized MiCRM derivatives and a thin SciPy `solve_ivp` wrapper;
- modified Sharpe-Schoolfield thermal performance curves and scaling of
  reference-temperature uptake and mortality;
- distance-decay patch connectivity and mass-conserving consumer/resource
  transport across undirected patch networks;
- runnable single-patch and thermal-spatial examples.

## Installation

DigiMicPy is not yet published on PyPI. Install it from a clone of this
repository:

```bash
git clone https://github.com/DigiMicOrg/DigiMicPy.git
cd DigiMicPy
python -m pip install .
```

Install plotting support for the examples with:

```bash
python -m pip install ".[examples]"
```

DigiMicPy requires Python 3.11 or newer.

## Quick start

```python
import numpy as np

from digimicpy import (
    MiCRMParameters,
    generate_l_tensor,
    modular_uptake,
    solve_micrm,
)

rng = np.random.default_rng(42)
n_consumers = 6
n_resources = 4
total_leakage = 0.1

uptake = modular_uptake(
    n_consumers,
    n_resources,
    n_modules=2,
    specialization_ratio=8.0,
    rng=rng,
)
leakage = generate_l_tensor(
    n_consumers,
    n_resources,
    n_modules=2,
    specialization_ratio=8.0,
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

initial_state = np.concatenate(
    [np.full(n_consumers, 0.01), np.ones(n_resources)]
)
result = solve_micrm(
    parameters,
    initial_state,
    (0.0, 20.0),
    t_eval=np.linspace(0.0, 20.0, 101),
)
if not result.success:
    raise RuntimeError(result.message)
```

The state vector always stores all consumers first and all resources second.
See [`examples/basic_simulation.py`](examples/basic_simulation.py) for plotting
and [`examples/thermal_spatial_simulation.py`](examples/thermal_spatial_simulation.py)
for fixed-temperature patches connected by diffusion.

## Model conventions

For `N` consumers and `M` resources:

| Parameter | Shape | Meaning |
|---|---:|---|
| `uptake` | `(N, M)` | Consumer uptake rates or preferences |
| `mortality` | `(N,)` | Consumer maintenance or mortality rates |
| `resource_supply` | `(M,)` | External resource input rates |
| `resource_decay` | `(M,)` | Resource loss or washout rates |
| `leakage` | `(N, M, M)` | Consumed-resource to by-product fractions |
| `leakage_fraction` | stored as `(N, M)` | Total leaked fraction for each uptake channel; `(M,)` inputs are broadcast |
| `consumer_ids` | optional `(N,)` | Unique consumer labels, required for consumer diffusion |
| `resource_ids` | optional `(M,)` | Unique resource labels, required for resource diffusion |

Every leakage row must sum to the corresponding `leakage_fraction`, and all
model parameters and initial states must be finite and nonnegative.

Spatial simulations require explicit identifiers for every transported consumer
or resource, with identical ordered identifiers in every patch. Connectivity
must be undirected. Diffusion rates are common across patches but may differ by
consumer or resource. The transport equations conserve each state variable for
equal-volume patches (or when states represent total quantities rather than
concentrations).

## Public API

| Area | Public functions and types |
|---|---|
| Core model | `MiCRMParameters`, `micrm_rhs`, `solve_micrm` |
| Parameter generation | `modular_uptake`, `modular_leakage`, `generate_l_tensor` |
| Temperature | `BOLTZMANN_CONSTANT`, `thermal_performance`, `thermal_scaling_factor`, `temperature_adjusted_parameters` |
| Space | `distance_connectivity`, `spatial_micrm_rhs`, `solve_spatial_micrm` |

Random thermal-trait generation, time-varying temperature, directed transport,
patch-specific diffusion, and heterogeneous species sets are not yet part of
the public API.

## Documentation

The project documentation is available at [digimic.org](https://digimic.org).
It includes the core theory, executable usage, temperature and spatial
assumptions, and clearly marked theory-only analysis workflows.

## Development

```bash
python -m pip install -e ".[dev,examples,docs]"
python -m pytest
python examples/basic_simulation.py
python examples/thermal_spatial_simulation.py
jupyter-book build docs --warningiserror
python -m build
```

The test suite targets Python 3.11 and 3.14 in CI. Please open an issue before
starting a large new model extension so its equations, units, and compatibility
with the core state layout can be agreed first.

## Citation and license

There is not yet an archival DigiMicPy software release. Until one is published,
cite this repository and the scientific sources for the model formulation used
in your analysis. The software is distributed under the [MIT License](LICENSE).
