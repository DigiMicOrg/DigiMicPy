"""Spatial coupling for microbial consumer-resource model patches."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from .micrm import micrm_rhs
from .model import MiCRMParameters


FloatArray = NDArray[np.float64]
DiffusionInput = ArrayLike | Mapping[Hashable, float]


def _real_array(name: str, values: ArrayLike, *, copy: bool = True) -> FloatArray:
    try:
        raw = np.asanyarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a rectangular numeric array") from error
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain only real values")
    try:
        array = np.array(values, dtype=float, copy=copy)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a rectangular numeric array") from error
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _finite_scalar(name: str, value: object, *, positive: bool = False) -> float:
    try:
        raw = np.asanyarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite scalar") from error
    if raw.shape != () or np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a finite scalar")
    try:
        scalar = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite scalar") from error
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


@dataclass(frozen=True, slots=True, eq=False)
class SpatialPatch:
    """One local MiCRM community in a spatial landscape.

    Parameters
    ----------
    parameters
        Local model parameters with explicit consumer and resource identifiers.
    volume
        Positive patch volume. State variables are interpreted as
        concentrations, so volume scales transport derivatives but not local
        MiCRM dynamics.
    name
        Optional human-readable patch label. Integer patch positions, not
        names, are used as compiled layout keys.
    """

    parameters: MiCRMParameters
    volume: float = 1.0
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, MiCRMParameters):
            raise TypeError("parameters must be a MiCRMParameters instance")
        if self.parameters.consumer_ids is None:
            raise ValueError("SpatialPatch parameters must define consumer_ids")
        if self.parameters.resource_ids is None:
            raise ValueError("SpatialPatch parameters must define resource_ids")
        for name, identifiers in (
            ("consumer_ids", self.parameters.consumer_ids),
            ("resource_ids", self.parameters.resource_ids),
        ):
            assert identifiers is not None
            for identifier in identifiers:
                if identifier is None:
                    raise ValueError(f"SpatialPatch {name} must not contain None")
                try:
                    is_reflexive = bool(identifier == identifier)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"SpatialPatch {name} must contain stable identifiers"
                    ) from error
                if not is_reflexive:
                    raise ValueError(
                        f"SpatialPatch {name} must contain stable identifiers"
                    )

        object.__setattr__(
            self,
            "volume",
            _finite_scalar("volume", self.volume, positive=True),
        )
        if self.name is not None:
            if not isinstance(self.name, str):
                raise TypeError("name must be a string or None")
            if not self.name.strip():
                raise ValueError("name must not be empty")


@dataclass(frozen=True, slots=True, init=False, eq=False)
class SpatialLayout:
    """Compiled state layout for patches with heterogeneous communities.

    State blocks are patch-major. Within each patch, consumers precede
    resources in the order declared by that patch's :class:`MiCRMParameters`.
    The compiled mappings ensure transport is matched by biological identifier
    rather than by local array position.
    """

    patches: tuple[SpatialPatch, ...]
    patch_slices: tuple[slice, ...]
    consumer_slices: tuple[slice, ...]
    resource_slices: tuple[slice, ...]
    consumer_indices: Mapping[tuple[int, Hashable], int]
    resource_indices: Mapping[tuple[int, Hashable], int]
    shared_consumer_ids: Mapping[tuple[int, int], tuple[Hashable, ...]]
    shared_resource_ids: Mapping[tuple[int, int], tuple[Hashable, ...]]
    consumer_ids: tuple[Hashable, ...]
    resource_ids: tuple[Hashable, ...]
    volumes: tuple[float, ...]
    state_size: int

    def __init__(self, patches: Sequence[SpatialPatch]) -> None:
        patch_tuple = tuple(patches)
        if not patch_tuple:
            raise ValueError("patches must contain at least one SpatialPatch")
        if not all(isinstance(patch, SpatialPatch) for patch in patch_tuple):
            raise TypeError("every patch must be a SpatialPatch instance")

        patch_slices: list[slice] = []
        consumer_slices: list[slice] = []
        resource_slices: list[slice] = []
        consumer_indices: dict[tuple[int, Hashable], int] = {}
        resource_indices: dict[tuple[int, Hashable], int] = {}
        regional_consumers: list[Hashable] = []
        regional_resources: list[Hashable] = []
        seen_consumers: set[Hashable] = set()
        seen_resources: set[Hashable] = set()

        offset = 0
        for patch_index, patch in enumerate(patch_tuple):
            parameters = patch.parameters
            consumer_ids = parameters.consumer_ids
            resource_ids = parameters.resource_ids
            # SpatialPatch validation guarantees these are present.
            assert consumer_ids is not None
            assert resource_ids is not None

            consumer_start = offset
            resource_start = consumer_start + parameters.n_consumers
            patch_stop = resource_start + parameters.n_resources
            patch_slices.append(slice(consumer_start, patch_stop))
            consumer_slices.append(slice(consumer_start, resource_start))
            resource_slices.append(slice(resource_start, patch_stop))

            for local_index, identifier in enumerate(consumer_ids):
                consumer_indices[(patch_index, identifier)] = (
                    consumer_start + local_index
                )
                if identifier not in seen_consumers:
                    seen_consumers.add(identifier)
                    regional_consumers.append(identifier)
            for local_index, identifier in enumerate(resource_ids):
                resource_indices[(patch_index, identifier)] = resource_start + local_index
                if identifier not in seen_resources:
                    seen_resources.add(identifier)
                    regional_resources.append(identifier)
            offset = patch_stop

        shared_consumers: dict[tuple[int, int], tuple[Hashable, ...]] = {}
        shared_resources: dict[tuple[int, int], tuple[Hashable, ...]] = {}
        for first in range(len(patch_tuple)):
            first_consumer_ids = patch_tuple[first].parameters.consumer_ids
            first_resource_ids = patch_tuple[first].parameters.resource_ids
            assert first_consumer_ids is not None
            assert first_resource_ids is not None
            for second in range(first + 1, len(patch_tuple)):
                second_consumer_ids = set(
                    patch_tuple[second].parameters.consumer_ids or ()
                )
                second_resource_ids = set(
                    patch_tuple[second].parameters.resource_ids or ()
                )
                shared_consumers[(first, second)] = tuple(
                    identifier
                    for identifier in first_consumer_ids
                    if identifier in second_consumer_ids
                )
                shared_resources[(first, second)] = tuple(
                    identifier
                    for identifier in first_resource_ids
                    if identifier in second_resource_ids
                )

        object.__setattr__(self, "patches", patch_tuple)
        object.__setattr__(self, "patch_slices", tuple(patch_slices))
        object.__setattr__(self, "consumer_slices", tuple(consumer_slices))
        object.__setattr__(self, "resource_slices", tuple(resource_slices))
        object.__setattr__(
            self,
            "consumer_indices",
            MappingProxyType(consumer_indices),
        )
        object.__setattr__(
            self,
            "resource_indices",
            MappingProxyType(resource_indices),
        )
        object.__setattr__(
            self,
            "shared_consumer_ids",
            MappingProxyType(shared_consumers),
        )
        object.__setattr__(
            self,
            "shared_resource_ids",
            MappingProxyType(shared_resources),
        )
        object.__setattr__(self, "consumer_ids", tuple(regional_consumers))
        object.__setattr__(self, "resource_ids", tuple(regional_resources))
        object.__setattr__(
            self,
            "volumes",
            tuple(patch.volume for patch in patch_tuple),
        )
        object.__setattr__(self, "state_size", offset)

    @property
    def n_patches(self) -> int:
        """Number of local communities in the layout."""

        return len(self.patches)

    @property
    def parameters(self) -> tuple[MiCRMParameters, ...]:
        """Local parameter sets in patch order."""

        return tuple(patch.parameters for patch in self.patches)

    def pack_state(self, patch_states: ArrayLike | Sequence[ArrayLike]) -> FloatArray:
        """Validate and flatten a spatial state into compiled patch order.

        Heterogeneous states may be supplied as one one-dimensional array per
        patch. A flat vector is always accepted, and a rectangular matrix is
        accepted when every patch block has the same size.
        """

        return _state_array(patch_states, self)

    def unpack_state(self, state: ArrayLike) -> tuple[FloatArray, ...]:
        """Return independent per-patch arrays from a packed spatial state."""

        flat_state = _state_array(state, self)
        return tuple(flat_state[patch_slice].copy() for patch_slice in self.patch_slices)


@dataclass(frozen=True, slots=True)
class _LegacyIdentity:
    """Private placeholder used only by the backwards-compatible wrapper."""

    patch: int
    kind: str
    position: int


def distance_connectivity(
    patch_positions: ArrayLike,
    *,
    decay_rate: float = 1.0,
) -> FloatArray:
    """Create symmetric, zero-diagonal connectivity from patch coordinates."""

    positions = _real_array("patch_positions", patch_positions)
    if positions.ndim == 1:
        positions = positions[:, np.newaxis]
    elif positions.ndim != 2:
        raise ValueError("patch_positions must be a one- or two-dimensional array")
    if positions.shape[0] == 0 or positions.shape[1] == 0:
        raise ValueError("patch_positions must describe at least one patch and dimension")

    decay = _finite_scalar("decay_rate", decay_rate)
    if decay < 0.0:
        raise ValueError("decay_rate must be nonnegative")

    displacement = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.linalg.norm(displacement, axis=2)
    connectivity = np.exp(-decay * distances)
    np.fill_diagonal(connectivity, 0.0)
    return connectivity


def _connectivity_array(connectivity: ArrayLike, n_patches: int) -> FloatArray:
    matrix = _real_array("connectivity", connectivity)
    expected_shape = (n_patches, n_patches)
    if matrix.shape != expected_shape:
        raise ValueError(
            f"connectivity must have shape {expected_shape}, got {matrix.shape}"
        )
    if np.any(matrix < 0.0):
        raise ValueError("connectivity must contain only nonnegative values")
    if not np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("connectivity diagonal must be zero")
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-12):
        raise ValueError("connectivity must be symmetric to conserve transported amount")
    # Remove tolerated floating-point asymmetry so paired fluxes use one value.
    symmetric = matrix / 2.0 + matrix.T / 2.0
    np.fill_diagonal(symmetric, 0.0)
    return symmetric


def _diffusion_has_positive(name: str, diffusion: DiffusionInput) -> bool:
    if isinstance(diffusion, Mapping):
        return any(
            _finite_scalar(f"{name}[{identifier!r}]", value) > 0.0
            for identifier, value in diffusion.items()
        )
    values = _real_array(name, diffusion)
    return bool(np.any(values > 0.0))


def _with_legacy_ids(
    parameters: MiCRMParameters,
    patch_index: int,
) -> MiCRMParameters:
    consumer_ids = parameters.consumer_ids
    if consumer_ids is None:
        consumer_ids = tuple(
            _LegacyIdentity(patch_index, "consumer", position)
            for position in range(parameters.n_consumers)
        )
    resource_ids = parameters.resource_ids
    if resource_ids is None:
        resource_ids = tuple(
            _LegacyIdentity(patch_index, "resource", position)
            for position in range(parameters.n_resources)
        )
    if (
        consumer_ids is parameters.consumer_ids
        and resource_ids is parameters.resource_ids
    ):
        return parameters
    return MiCRMParameters(
        uptake=parameters.uptake,
        mortality=parameters.mortality,
        resource_supply=parameters.resource_supply,
        resource_decay=parameters.resource_decay,
        leakage=parameters.leakage,
        leakage_fraction=parameters.leakage_fraction,
        consumer_ids=consumer_ids,
        resource_ids=resource_ids,
    )


def _coerce_layout(
    patches: SpatialLayout | Sequence[SpatialPatch] | Sequence[MiCRMParameters],
    consumer_diffusion: DiffusionInput,
    resource_diffusion: DiffusionInput,
) -> SpatialLayout:
    if isinstance(patches, SpatialLayout):
        return patches

    patch_items = tuple(patches)
    if not patch_items:
        raise ValueError("patch_parameters must contain at least one patch")
    if all(isinstance(patch, SpatialPatch) for patch in patch_items):
        return SpatialLayout(patch_items)
    if not all(isinstance(item, MiCRMParameters) for item in patch_items):
        raise TypeError(
            "patch_parameters must be a SpatialLayout or a sequence containing "
            "only SpatialPatch or only MiCRMParameters instances"
        )

    parameters = tuple(patch_items)
    if _diffusion_has_positive("consumer_diffusion", consumer_diffusion) and any(
        item.consumer_ids is None for item in parameters
    ):
        raise ValueError(
            "consumer_ids must be provided for every patch when consumers diffuse"
        )
    if _diffusion_has_positive("resource_diffusion", resource_diffusion) and any(
        item.resource_ids is None for item in parameters
    ):
        raise ValueError(
            "resource_ids must be provided for every patch when resources diffuse"
        )

    # PR #4 allowed unlabeled parameter sequences when the corresponding rate
    # was zero. Patch-local sentinels retain that no-transport compatibility
    # without exposing positional identity as biological identity.
    return SpatialLayout(
        tuple(
            SpatialPatch(_with_legacy_ids(item, index))
            for index, item in enumerate(parameters)
        )
    )


def _normalise_diffusion(
    name: str,
    diffusion: DiffusionInput,
    layout: SpatialLayout,
    *,
    kind: str,
) -> Mapping[Hashable, float]:
    regional_ids = (
        layout.consumer_ids if kind == "consumer" else layout.resource_ids
    )
    regional_set = set(regional_ids)

    if isinstance(diffusion, Mapping):
        rates = dict.fromkeys(regional_ids, 0.0)
        for identifier, raw_rate in diffusion.items():
            if identifier not in regional_set:
                raise ValueError(f"{name} contains unknown {kind} ID {identifier!r}")
            rate = _finite_scalar(f"{name}[{identifier!r}]", raw_rate)
            if rate < 0.0:
                raise ValueError(f"{name} must contain only nonnegative values")
            rates[identifier] = rate
        return MappingProxyType(rates)

    values = _real_array(name, diffusion)
    if values.shape == () or (values.ndim == 1 and values.size == 1):
        rate = float(values.reshape(-1)[0])
        if rate < 0.0:
            raise ValueError(f"{name} must contain only nonnegative values")
        return MappingProxyType(dict.fromkeys(regional_ids, rate))
    if values.ndim != 1:
        raise ValueError(f"{name} must be scalar, one-dimensional, or ID-keyed")
    if np.any(values < 0.0):
        raise ValueError(f"{name} must contain only nonnegative values")
    attribute = "consumer_ids" if kind == "consumer" else "resource_ids"
    ordered_ids = getattr(layout.patches[0].parameters, attribute)
    assert ordered_ids is not None
    if any(
        getattr(patch.parameters, attribute) != ordered_ids
        for patch in layout.patches[1:]
    ):
        raise ValueError(
            f"positional {name} is ambiguous when ordered {attribute} differ; "
            "provide a scalar or ID-keyed mapping"
        )
    try:
        vector = np.broadcast_to(values, (len(ordered_ids),)).copy()
    except ValueError as error:
        raise ValueError(
            f"{name} must be scalar or broadcast to ({len(ordered_ids)},)"
        ) from error
    return MappingProxyType(
        {
            identifier: float(rate)
            for identifier, rate in zip(ordered_ids, vector, strict=True)
        }
    )


def _state_array(
    state: ArrayLike | Sequence[ArrayLike],
    layout: SpatialLayout,
    *,
    require_nonnegative: bool = False,
) -> FloatArray:
    array: FloatArray | None = None
    try:
        raw = np.asanyarray(state)
    except (TypeError, ValueError):
        raw = None

    if raw is not None:
        if np.iscomplexobj(raw):
            raise ValueError("state must contain only real values")
        try:
            candidate = np.array(state, dtype=float, copy=True)
        except (TypeError, ValueError):
            candidate = None
        if candidate is not None:
            if candidate.shape == (layout.state_size,):
                array = candidate
            block_sizes = tuple(
                patch_slice.stop - patch_slice.start
                for patch_slice in layout.patch_slices
            )
            if len(set(block_sizes)) == 1:
                matrix_shape = (layout.n_patches, block_sizes[0])
                if array is None and candidate.shape == matrix_shape:
                    array = candidate.reshape(-1)

    if array is None:
        if isinstance(state, (str, bytes)) or not isinstance(state, Sequence):
            raise ValueError(
                f"state must be a flat vector of length {layout.state_size} "
                "or a sequence of per-patch arrays"
            )
        patch_states = tuple(state)
        if len(patch_states) != layout.n_patches:
            raise ValueError(
                f"state must contain {layout.n_patches} patch arrays, "
                f"got {len(patch_states)}"
            )
        validated: list[FloatArray] = []
        for patch_index, (patch_state, patch_slice) in enumerate(
            zip(patch_states, layout.patch_slices, strict=True)
        ):
            local_state = _real_array(f"state[{patch_index}]", patch_state)
            expected_shape = (patch_slice.stop - patch_slice.start,)
            if local_state.shape != expected_shape:
                raise ValueError(
                    f"state[{patch_index}] must have shape {expected_shape}, "
                    f"got {local_state.shape}"
                )
            validated.append(local_state)
        array = np.concatenate(validated)

    if not np.all(np.isfinite(array)):
        raise ValueError("state must contain only finite values")
    if require_nonnegative and np.any(array < 0.0):
        raise ValueError("initial state must contain only nonnegative values")
    return array


def _validated_spatial_rhs(
    time: float,
    state: FloatArray,
    layout: SpatialLayout,
    connectivity: FloatArray,
    consumer_diffusion: Mapping[Hashable, float],
    resource_diffusion: Mapping[Hashable, float],
) -> FloatArray:
    derivative = np.empty(layout.state_size, dtype=float)
    for patch, patch_slice in zip(layout.patches, layout.patch_slices, strict=True):
        derivative[patch_slice] = micrm_rhs(
            time,
            state[patch_slice],
            patch.parameters,
        )

    for patch_pair, shared_ids in layout.shared_consumer_ids.items():
        first, second = patch_pair
        conductance = connectivity[first, second]
        if conductance == 0.0:
            continue
        first_volume = layout.volumes[first]
        second_volume = layout.volumes[second]
        for identifier in shared_ids:
            rate = consumer_diffusion[identifier]
            if rate == 0.0:
                continue
            first_index = layout.consumer_indices[(first, identifier)]
            second_index = layout.consumer_indices[(second, identifier)]
            flux = conductance * rate * (
                state[first_index] - state[second_index]
            )
            derivative[first_index] -= flux / first_volume
            derivative[second_index] += flux / second_volume

    for patch_pair, shared_ids in layout.shared_resource_ids.items():
        first, second = patch_pair
        conductance = connectivity[first, second]
        if conductance == 0.0:
            continue
        first_volume = layout.volumes[first]
        second_volume = layout.volumes[second]
        for identifier in shared_ids:
            rate = resource_diffusion[identifier]
            if rate == 0.0:
                continue
            first_index = layout.resource_indices[(first, identifier)]
            second_index = layout.resource_indices[(second, identifier)]
            flux = conductance * rate * (
                state[first_index] - state[second_index]
            )
            derivative[first_index] -= flux / first_volume
            derivative[second_index] += flux / second_volume

    return derivative


def _validated_inputs(
    patch_parameters: (
        SpatialLayout | Sequence[SpatialPatch] | Sequence[MiCRMParameters]
    ),
    connectivity: ArrayLike,
    consumer_diffusion: DiffusionInput,
    resource_diffusion: DiffusionInput,
) -> tuple[
    SpatialLayout,
    FloatArray,
    Mapping[Hashable, float],
    Mapping[Hashable, float],
]:
    layout = _coerce_layout(
        patch_parameters,
        consumer_diffusion,
        resource_diffusion,
    )
    matrix = _connectivity_array(connectivity, layout.n_patches)
    consumer_rates = _normalise_diffusion(
        "consumer_diffusion",
        consumer_diffusion,
        layout,
        kind="consumer",
    )
    resource_rates = _normalise_diffusion(
        "resource_diffusion",
        resource_diffusion,
        layout,
        kind="resource",
    )
    return layout, matrix, consumer_rates, resource_rates


def spatial_micrm_rhs(
    time: float,
    state: ArrayLike | Sequence[ArrayLike],
    patch_parameters: (
        SpatialLayout | Sequence[SpatialPatch] | Sequence[MiCRMParameters]
    ),
    connectivity: ArrayLike,
    *,
    consumer_diffusion: DiffusionInput = 0.0,
    resource_diffusion: DiffusionInput = 0.0,
) -> FloatArray:
    """Evaluate local MiCRM dynamics plus conservative patch transport.

    ``patch_parameters`` may be a compiled :class:`SpatialLayout`, a sequence
    of :class:`SpatialPatch` objects, or the homogeneous sequence of
    :class:`MiCRMParameters` accepted by the original API. Heterogeneous
    transport is matched by explicit IDs. Only identities represented at both
    endpoints can move; a structurally absent identity is never created.

    For a shared identity, symmetric conductance ``g`` and diffusion rate ``d``
    define ``F = g * d * (X_i - X_j)``. Concentration derivatives receive
    ``-F / V_i`` and ``+F / V_j``, conserving the volume-weighted amount.
    """

    layout, matrix, consumer_rates, resource_rates = _validated_inputs(
        patch_parameters,
        connectivity,
        consumer_diffusion,
        resource_diffusion,
    )
    state_array = _state_array(state, layout)
    return _validated_spatial_rhs(
        time,
        state_array,
        layout,
        matrix,
        consumer_rates,
        resource_rates,
    )


def solve_spatial_micrm(
    patch_parameters: (
        SpatialLayout | Sequence[SpatialPatch] | Sequence[MiCRMParameters]
    ),
    initial_state: ArrayLike | Sequence[ArrayLike],
    t_span: tuple[float, float],
    *,
    connectivity: ArrayLike,
    consumer_diffusion: DiffusionInput = 0.0,
    resource_diffusion: DiffusionInput = 0.0,
    t_eval: ArrayLike | None = None,
    **solver_options: Any,
):
    """Integrate coupled patch dynamics with :func:`scipy.integrate.solve_ivp`."""

    if "args" in solver_options:
        raise ValueError("args is managed internally by solve_spatial_micrm")
    if solver_options.get("vectorized", False):
        raise ValueError("solve_spatial_micrm does not support vectorized=True")

    layout, matrix, consumer_rates, resource_rates = _validated_inputs(
        patch_parameters,
        connectivity,
        consumer_diffusion,
        resource_diffusion,
    )
    state_array = _state_array(
        initial_state,
        layout,
        require_nonnegative=True,
    )
    evaluation_times = None if t_eval is None else np.asarray(t_eval, dtype=float)

    def model_rhs(time: float, state: FloatArray) -> FloatArray:
        return _validated_spatial_rhs(
            time,
            state,
            layout,
            matrix,
            consumer_rates,
            resource_rates,
        )

    return solve_ivp(
        model_rhs,
        t_span,
        state_array,
        t_eval=evaluation_times,
        **solver_options,
    )
