# Public API

The top-level `digimicpy` namespace exposes the supported interface below.
Validation, assumptions, and examples are documented on the topic pages and in
the API docstrings.

## Core model

| Symbol | Purpose |
|---|---|
| `MiCRMParameters(...)` | Validate and store one MiCRM parameter set |
| `micrm_rhs(time, state, parameters)` | Evaluate the autonomous MiCRM derivative |
| `solve_micrm(parameters, initial_state, t_span, **options)` | Integrate one MiCRM system with SciPy |

## Synthetic parameter generation

| Symbol | Purpose |
|---|---|
| `modular_uptake(...)` | Generate row-normalised modular uptake preferences |
| `modular_leakage(...)` | Generate one modular by-product matrix |
| `generate_l_tensor(...)` | Generate one by-product matrix per consumer |

All random generators accept `rng=np.random.default_rng(seed)`. Omitting `rng`
creates an independent generator rather than using NumPy's global random state.
