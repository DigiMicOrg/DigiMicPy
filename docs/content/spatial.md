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

DigiMicPy can couple local MiCRM communities through consumer migration and
resource diffusion. A landscape may contain different numbers and identities
of consumers and resources in each patch. Transport follows biological IDs,
not local row or column positions.

## Local communities and the regional pool

`SpatialPatch` combines one local `MiCRMParameters` object with a positive
volume and an optional name. Both `consumer_ids` and `resource_ids` must be
explicit, unique, hashable and stable (so `None` and non-reflexive values such
as `NaN` are rejected). The union of those local IDs is the regional pool, but
an ID need not occur in every patch.

`SpatialLayout` compiles the patches into:

- patch, consumer and resource state slices;
- `(patch_index, consumer_id)` and `(patch_index, resource_id)` flat indices;
- shared consumer and resource IDs for every patch pair; and
- patch volumes and the total flat state length.

Patch names are descriptive labels. Integer patch positions and explicit
biological IDs define the state mapping.

```{code-cell} ipython3
import numpy as np

from digimicpy import MiCRMParameters, SpatialLayout, SpatialPatch


def inert_parameters(consumer_ids, resource_ids):
    """Make a labeled community with zero local dynamics for illustration."""
    n_consumers = len(consumer_ids)
    n_resources = len(resource_ids)
    return MiCRMParameters(
        uptake=np.zeros((n_consumers, n_resources)),
        mortality=np.zeros(n_consumers),
        resource_supply=np.zeros(n_resources),
        resource_decay=np.zeros(n_resources),
        leakage=np.zeros((n_consumers, n_resources, n_resources)),
        leakage_fraction=np.zeros(n_resources),
        consumer_ids=consumer_ids,
        resource_ids=resource_ids,
    )


layout = SpatialLayout(
    [
        SpatialPatch(
            inert_parameters(["consumer-a", "generalist"], ["carbon"]),
            volume=2.0,
            name="nearshore",
        ),
        SpatialPatch(
            inert_parameters(
                ["generalist", "consumer-b"],
                ["nitrogen", "carbon"],
            ),
            volume=3.0,
            name="offshore",
        ),
    ]
)

layout.patch_slices, layout.consumer_indices[(1, "generalist")]
```

## Structural absence, zero abundance and colonisation

Structural absence and zero abundance are different model states:

- If an ID is absent from a destination patch's parameter object, that route
  is closed for the identity. DigiMicPy does not invent immigrant parameters
  or add ODE variables dynamically.
- A potential colonist must already have an ID, valid local parameters and a
  state variable in the destination. Its initial abundance may be zero or low;
  migration can then increase it.

This fixed-dimensional approach makes local membership explicit at model
construction and keeps numerical integration predictable.

## Packing heterogeneous states

State order is patch-major. Each local block contains consumers followed by
resources in that patch's declared ID order. `pack_state` validates and joins
ragged per-patch arrays; `unpack_state` reverses the operation.

```{code-cell} ipython3
patch_states = [
    # nearshore: consumer-a, generalist, carbon
    np.array([0.2, 0.1, 1.0]),
    # offshore: generalist, consumer-b, nitrogen, carbon
    np.array([0.0, 0.3, 0.8, 1.2]),
]
flat_state = layout.pack_state(patch_states)
round_trip = layout.unpack_state(flat_state)

flat_state, round_trip
```

A flat vector of length `layout.state_size` is always accepted. The original
rectangular `(n_patches, block_size)` form is also accepted when all patch
blocks have equal size.

## Connectivity and conservative transport

For patch coordinates $x_i$, `distance_connectivity` creates a symmetric
zero-diagonal conductance matrix:

$$
g_{ij}=\exp(-\lambda ||x_i-x_j||), \qquad g_{ii}=0.
$$

```{code-cell} ipython3
from digimicpy import distance_connectivity

connectivity = distance_connectivity([[0.0], [1.0]], decay_rate=0.5)
connectivity
```

For identity $s$ represented in both connected patches $i$ and $j$, the
undirected pair flux is

$$
F_{ij,s}=g_{ij}d_s(X_{i,s}-X_{j,s}).
$$

States are concentrations. With patch volumes $V_i$ and $V_j$, transport adds

$$
\dot X_{i,s} \mathrel{-}= F_{ij,s}/V_i,
\qquad
\dot X_{j,s} \mathrel{+}= F_{ij,s}/V_j.
$$

One paired flux therefore conserves total modelled amount:

$$
\frac{d}{dt}\sum_i V_iX_{i,s}=0.
$$

Local MiCRM dynamics remain concentration derivatives and are evaluated
independently with each patch's local parameters. Volume affects only spatial
transport.

## Diffusion rates and identity alignment

Consumer and resource diffusion may be supplied as:

- one nonnegative scalar applied to every represented identity;
- an ID-keyed mapping such as `{"generalist": 0.02}`; omitted known IDs have
  zero diffusion; or
- a positional vector only when every patch has identical IDs in identical
  order.

Use ID-keyed mappings for heterogeneous or reordered communities. DigiMicPy
rejects an ambiguous positional vector instead of guessing its alignment, and
rejects unknown IDs, negative rates and nonfinite rates.

```{code-cell} ipython3
from digimicpy import spatial_micrm_rhs

derivative = spatial_micrm_rhs(
    0.0,
    patch_states,
    layout,
    connectivity,
    consumer_diffusion={"generalist": 0.02},
    resource_diffusion={"carbon": 0.05},
)
layout.unpack_state(derivative)
```

Connectivity must remain finite, nonnegative, symmetric and zero-diagonal.
Directed source-to-destination transfer is a separate process and is not
represented by this matrix.

## Backwards compatibility

`spatial_micrm_rhs` and `solve_spatial_micrm` still accept the original
sequence of `MiCRMParameters`. Such calls imply unit patch volumes. Equal-size
patches with identical ordered IDs retain scalar and positional-vector
diffusion and the same numerical transport results introduced in PR #4.
Passing a `SpatialLayout` compiles ragged slices and identity mappings once for
repeated RHS evaluation by the solver.

See
[`heterogeneous_spatial_simulation.py`](https://github.com/DigiMicOrg/DigiMicPy/blob/main/examples/heterogeneous_spatial_simulation.py)
for a deterministic executable landscape with different local pools, reordered
resources, unequal volumes and an initially absent-in-abundance potential
colonist. The existing
[`thermal_spatial_simulation.py`](https://github.com/DigiMicOrg/DigiMicPy/blob/main/examples/thermal_spatial_simulation.py)
continues to demonstrate the homogeneous fixed-temperature API.

## Design provenance

The named, patch-local community design adapts Chan Li's `patch_settings`,
`generate_patch_landscape` and `dCdt_Rdt_spatial` prototype from
[`chan-branch`](https://github.com/DigiMicOrg/DigiMicPy/blob/8f43332d2832c402a0ded07ebb551312a692e35e/docs/content/spatial_and_temperature.ipynb),
principally commit
[`cb773ae`](https://github.com/DigiMicOrg/DigiMicPy/commit/cb773aead39d5e3fce1f516277fe9fdee260a09e).
The production layout preserves independent local communities and patch-major
state order while replacing fixed patch-0 block indexing and positional
matching with validated slices, IDs and conservative volume-aware flux. Chan's
[scientific review of PR #4](https://github.com/DigiMicOrg/DigiMicPy/pull/4#issuecomment-5468189697)
motivates the local-versus-regional distinction. Directed transport and
temperature-linked community composition remain separate follow-up phases.
