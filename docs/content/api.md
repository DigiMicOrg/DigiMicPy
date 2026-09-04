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

## Fixed-temperature traits

| Symbol | Purpose |
|---|---|
| `BOLTZMANN_CONSTANT` | Boltzmann constant in electronvolts per kelvin |
| `thermal_performance(...)` | Evaluate the modified Sharpe-Schoolfield curve |
| `thermal_scaling_factor(...)` | Calculate performance relative to a reference temperature |
| `temperature_adjusted_parameters(...)` | Scale uptake and mortality in a new parameter object |

Temperatures must be absolute. With the default Boltzmann constant, activation
and deactivation energies are in electronvolts and temperature is in kelvin.

## Spatial patches

| Symbol | Purpose |
|---|---|
| `SpatialPatch(parameters, volume=1.0, name=None)` | Define one labeled local community and its volume |
| `SpatialLayout(patches)` | Compile heterogeneous state slices, ID maps and packing helpers |
| `distance_connectivity(...)` | Build symmetric exponential distance-decay weights |
| `spatial_micrm_rhs(...)` | Evaluate local dynamics plus conservative transport |
| `solve_spatial_micrm(...)` | Integrate a fixed network of MiCRM patches |

`SpatialLayout.pack_state(...)` and `unpack_state(...)` convert between a flat
solver state and heterogeneous per-patch arrays. Transport uses explicit IDs,
symmetric conductance and volume-aware paired flux. Scalar and ID-keyed
diffusion support different local identity sets; positional vectors remain
available only when every patch has identical ordered IDs.
