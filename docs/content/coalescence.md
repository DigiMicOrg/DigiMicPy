# DigiMicPy coalescence recipe

The [platform coalescence workflow](https://digimicorg.github.io/workflows/coalescence/)
defines the scientific assumptions, resource mapping, metrics, interpretation,
and reporting requirements. This page only shows how to combine compatible
DigiMicPy parameter objects with the current public API.

```{important}
DigiMicPy does not provide a coalescence helper. The following is an explicit
recipe. Both parents must use the same ordered resources, supply, and decay
parameters. More general resource mappings must be constructed and validated by
the caller.
```

## Assemble and combine two parents

```python
import numpy as np

from digimicpy import MiCRMParameters, solve_micrm

result1 = solve_micrm(params1, initial_state1, (0.0, 100.0))
result2 = solve_micrm(params2, initial_state2, (0.0, 100.0))
if not result1.success or not result2.success:
    raise RuntimeError("A parental simulation failed")

C1_eq = result1.y[:params1.n_consumers, -1]
C2_eq = result2.y[:params2.n_consumers, -1]

params3 = MiCRMParameters(
    uptake=np.vstack([params1.uptake, params2.uptake]),
    mortality=np.concatenate([params1.mortality, params2.mortality]),
    resource_supply=params1.resource_supply,
    resource_decay=params1.resource_decay,
    leakage=np.concatenate([params1.leakage, params2.leakage], axis=0),
    leakage_fraction=np.concatenate(
        [params1.leakage_fraction, params2.leakage_fraction], axis=0
    ),
)

# Choose and report this post-mixing resource state explicitly.
initial_state3 = np.concatenate([C1_eq, C2_eq, R0_mix])
result3 = solve_micrm(params3, initial_state3, (0.0, 100.0))
if not result3.success:
    raise RuntimeError(result3.message)
```

Before interpreting the merge, calculate endpoint derivative norms for both
parents and the combined system. Keep the parent-origin index ranges so survivor
counts and biomass contributions can be reported without guessing from the
combined ordering.

For partially overlapping resources, first construct a stable union of resource
identifiers and embed each parent's uptake and leakage arrays into that common
ordering. Do not stack arrays whose columns merely have the same length.
