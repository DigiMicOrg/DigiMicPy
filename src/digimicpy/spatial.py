"""Spatial coupling for microbial consumer-resource model patches."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from .micrm import micrm_rhs
from .model import MiCRMParameters


FloatArray = NDArray[np.float64]


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

    decay = float(decay_rate)
    if not np.isfinite(decay) or decay < 0.0:
        raise ValueError("decay_rate must be finite and nonnegative")

    displacement = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.linalg.norm(displacement, axis=2)
    connectivity = np.exp(-decay * distances)
    np.fill_diagonal(connectivity, 0.0)
    return connectivity


def _patch_parameters(
    patch_parameters: Sequence[MiCRMParameters],
) -> tuple[MiCRMParameters, ...]:
    parameters = tuple(patch_parameters)
    if not parameters:
        raise ValueError("patch_parameters must contain at least one patch")
    if not all(isinstance(item, MiCRMParameters) for item in parameters):
        raise TypeError("every patch parameter set must be a MiCRMParameters instance")

    expected_dimensions = (
        parameters[0].n_consumers,
        parameters[0].n_resources,
    )
    for index, item in enumerate(parameters[1:], start=1):
        dimensions = (item.n_consumers, item.n_resources)
        if dimensions != expected_dimensions:
            raise ValueError(
                "all patches must share consumer and resource dimensions; "
                f"patch {index} has {dimensions}, expected {expected_dimensions}"
            )
    return parameters


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
        raise ValueError("connectivity must be symmetric to conserve transported mass")
    # Remove tolerated floating-point asymmetry so conservation is exact.
    symmetric = matrix / 2.0 + matrix.T / 2.0
    np.fill_diagonal(symmetric, 0.0)
    return symmetric


def _diffusion_vector(name: str, diffusion: ArrayLike, size: int) -> FloatArray:
    values = _real_array(name, diffusion)
    try:
        vector = np.broadcast_to(values, (size,)).copy()
    except ValueError as error:
        raise ValueError(f"{name} must be scalar or broadcast to ({size},)") from error
    if np.any(vector < 0.0):
        raise ValueError(f"{name} must contain only nonnegative values")
    return vector


def _require_matching_transport_ids(
    parameters: tuple[MiCRMParameters, ...],
    name: str,
    diffusion: FloatArray,
) -> None:
    if not np.any(diffusion > 0.0):
        return
    expected = getattr(parameters[0], name)
    if expected is None:
        raise ValueError(
            f"{name} must be provided for every patch when its variables diffuse"
        )
    for index, item in enumerate(parameters[1:], start=1):
        identifiers = getattr(item, name)
        if identifiers != expected:
            raise ValueError(
                f"all patches must use identical ordered {name}; "
                f"patch {index} does not match patch 0"
            )


def _state_array(
    state: ArrayLike,
    n_patches: int,
    block_size: int,
    *,
    require_nonnegative: bool = False,
) -> FloatArray:
    array = _real_array("state", state)
    matrix_shape = (n_patches, block_size)
    flat_shape = (n_patches * block_size,)
    if array.shape == matrix_shape:
        array = array.reshape(flat_shape)
    elif array.shape != flat_shape:
        raise ValueError(
            f"state must have shape {flat_shape} or {matrix_shape}, got {array.shape}"
        )
    if require_nonnegative and np.any(array < 0.0):
        raise ValueError("initial state must contain only nonnegative values")
    return array


def _validated_spatial_rhs(
    time: float,
    state: FloatArray,
    parameters: tuple[MiCRMParameters, ...],
    connectivity: FloatArray,
    consumer_diffusion: FloatArray,
    resource_diffusion: FloatArray,
) -> FloatArray:
    n_patches = len(parameters)
    n_consumers = parameters[0].n_consumers
    n_resources = parameters[0].n_resources
    block_size = n_consumers + n_resources
    patch_states = state.reshape((n_patches, block_size))
    derivative = np.stack(
        [
            micrm_rhs(time, patch_state, patch_parameter)
            for patch_state, patch_parameter in zip(patch_states, parameters)
        ]
    )

    degree = connectivity.sum(axis=1, keepdims=True)
    consumers = patch_states[:, :n_consumers]
    resources = patch_states[:, n_consumers:]
    derivative[:, :n_consumers] += (
        connectivity @ consumers - degree * consumers
    ) * consumer_diffusion
    derivative[:, n_consumers:] += (
        connectivity @ resources - degree * resources
    ) * resource_diffusion
    return derivative.reshape(-1)


def _validated_inputs(
    patch_parameters: Sequence[MiCRMParameters],
    connectivity: ArrayLike,
    consumer_diffusion: ArrayLike,
    resource_diffusion: ArrayLike,
) -> tuple[
    tuple[MiCRMParameters, ...],
    FloatArray,
    FloatArray,
    FloatArray,
]:
    parameters = _patch_parameters(patch_parameters)
    matrix = _connectivity_array(connectivity, len(parameters))
    consumer_rates = _diffusion_vector(
        "consumer_diffusion",
        consumer_diffusion,
        parameters[0].n_consumers,
    )
    resource_rates = _diffusion_vector(
        "resource_diffusion",
        resource_diffusion,
        parameters[0].n_resources,
    )
    _require_matching_transport_ids(parameters, "consumer_ids", consumer_rates)
    _require_matching_transport_ids(parameters, "resource_ids", resource_rates)
    return parameters, matrix, consumer_rates, resource_rates


def spatial_micrm_rhs(
    time: float,
    state: ArrayLike,
    patch_parameters: Sequence[MiCRMParameters],
    connectivity: ArrayLike,
    *,
    consumer_diffusion: ArrayLike = 0.0,
    resource_diffusion: ArrayLike = 0.0,
) -> FloatArray:
    """Evaluate MiCRM dynamics plus conservative transport between patches.

    Every transported consumer or resource type must have explicit identifiers,
    in the same order in every patch. Patch parameters may otherwise differ.
    State blocks are ordered by patch, with consumers followed by resources
    inside each block.
    """

    parameters, matrix, consumer_rates, resource_rates = _validated_inputs(
        patch_parameters,
        connectivity,
        consumer_diffusion,
        resource_diffusion,
    )
    block_size = parameters[0].n_consumers + parameters[0].n_resources
    state_array = _state_array(state, len(parameters), block_size)
    return _validated_spatial_rhs(
        time,
        state_array,
        parameters,
        matrix,
        consumer_rates,
        resource_rates,
    )


def solve_spatial_micrm(
    patch_parameters: Sequence[MiCRMParameters],
    initial_state: ArrayLike,
    t_span: tuple[float, float],
    *,
    connectivity: ArrayLike,
    consumer_diffusion: ArrayLike = 0.0,
    resource_diffusion: ArrayLike = 0.0,
    t_eval: ArrayLike | None = None,
    **solver_options: Any,
):
    """Integrate coupled patch dynamics with :func:`scipy.integrate.solve_ivp`."""

    if "args" in solver_options:
        raise ValueError("args is managed internally by solve_spatial_micrm")
    if solver_options.get("vectorized", False):
        raise ValueError("solve_spatial_micrm does not support vectorized=True")

    parameters, matrix, consumer_rates, resource_rates = _validated_inputs(
        patch_parameters,
        connectivity,
        consumer_diffusion,
        resource_diffusion,
    )
    block_size = parameters[0].n_consumers + parameters[0].n_resources
    state_array = _state_array(
        initial_state,
        len(parameters),
        block_size,
        require_nonnegative=True,
    )
    evaluation_times = None if t_eval is None else np.asarray(t_eval, dtype=float)

    def model_rhs(time: float, state: FloatArray) -> FloatArray:
        return _validated_spatial_rhs(
            time,
            state,
            parameters,
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
