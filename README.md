# DigiMicPy

DigiMicPy is a Python package for constructing and simulating microbial
consumer-resource models. It provides validated MiCRM parameters, reproducible
modular uptake and leakage generators, a pure right-hand side, and a SciPy
integration wrapper. It also supports fixed-temperature trait scaling and
conservative transport between labeled spatial patches.

## Install from source

DigiMicPy requires Python 3.11 or later.

```bash
git clone https://github.com/DigiMicOrg/DigiMicPy.git
cd DigiMicPy
python -m pip install -e .
```

Install optional dependencies when working on examples, documentation, or the
test suite:

```bash
python -m pip install -e ".[examples,docs,test]"
```

## Minimal simulation

```python
import numpy as np

from digimicpy import MiCRMParameters, generate_l_tensor, modular_uptake, solve_micrm

rng = np.random.default_rng(42)
n_consumers = 6
n_resources = 4
leakage_fraction = 0.1

parameters = MiCRMParameters(
    uptake=modular_uptake(
        n_consumers, n_resources, 2, 8.0, rng=rng
    ),
    mortality=np.full(n_consumers, 0.2),
    resource_supply=np.full(n_resources, 0.5),
    resource_decay=np.full(n_resources, 0.5),
    leakage=generate_l_tensor(
        n_consumers,
        n_resources,
        2,
        8.0,
        leakage_fraction,
        rng=rng,
    ),
    leakage_fraction=np.full(n_resources, leakage_fraction),
)

initial_state = np.concatenate(
    [np.full(n_consumers, 0.01), np.ones(n_resources)]
)
result = solve_micrm(parameters, initial_state, (0.0, 20.0))
if not result.success:
    raise RuntimeError(result.message)
```

See [`examples/thermal_spatial_simulation.py`](examples/thermal_spatial_simulation.py)
for coupled patches with different fixed temperatures.

## Documentation

- [DigiMicPy package documentation](https://digimic.org/)
- [DigiMic platform documentation](https://digimicorg.github.io/) for the
  project vision, shared workflows, package directory, training, team, funding,
  and general support

## Development

```bash
python -m pip install -e ".[dev,examples,docs]"
python -m pytest
python examples/thermal_spatial_simulation.py
jupyter-book build docs --warningiserror
python -m build
```

Report package bugs and feature requests in the
[DigiMicPy issue tracker](https://github.com/DigiMicOrg/DigiMicPy/issues).
