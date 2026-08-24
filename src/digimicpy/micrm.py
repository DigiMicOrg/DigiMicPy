"""Core microbial consumer-resource model dynamics."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from .model import MiCRMParameters


FloatArray = NDArray[np.float64]


def _state_array(
    state: ArrayLike,
    parameters: MiCRMParameters,
    *,
    require_nonnegative: bool = False,
) -> FloatArray:
    try:
        raw = np.asanyarray(state)
    except (TypeError, ValueError) as error:
        raise ValueError("state must be a one-dimensional numeric array") from error
    if np.iscomplexobj(raw):
        raise ValueError("state must contain only real values")
    try:
        state_array = np.array(state, dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError("state must be a one-dimensional numeric array") from error

    expected_shape = (parameters.n_consumers + parameters.n_resources,)
    if state_array.shape != expected_shape:
        raise ValueError(f"state must have shape {expected_shape}, got {state_array.shape}")
    if not np.all(np.isfinite(state_array)):
        raise ValueError("state must contain only finite values")
    if require_nonnegative and np.any(state_array < 0.0):
        raise ValueError("initial state must contain only nonnegative values")
    return state_array


def micrm_rhs(
    time: float,
    state: ArrayLike,
    parameters: MiCRMParameters,
) -> FloatArray:
    """Evaluate consumer and resource derivatives for a MiCRM state."""

    del time
    state_array = _state_array(state, parameters)
    consumers = state_array[: parameters.n_consumers]
    resources = state_array[parameters.n_consumers :]

    consumption = (
        consumers[:, np.newaxis]
        * resources[np.newaxis, :]
        * parameters.uptake
    )
    consumer_growth = np.sum(
        consumption * (1.0 - parameters.leakage_fraction),
        axis=1,
    )
    consumer_derivative = consumer_growth - consumers * parameters.mortality

    leaked_resources = np.einsum(
        "ij,ijk->k",
        consumption,
        parameters.leakage,
        optimize=True,
    )
    resource_derivative = (
        parameters.resource_supply
        - resources * parameters.resource_decay
        - consumption.sum(axis=0)
        + leaked_resources
    )
    return np.concatenate((consumer_derivative, resource_derivative))


def solve_micrm(
    parameters: MiCRMParameters,
    initial_state: ArrayLike,
    t_span: tuple[float, float],
    *,
    t_eval: ArrayLike | None = None,
    **solver_options: Any,
):
    """Integrate MiCRM dynamics with :func:`scipy.integrate.solve_ivp`."""

    if "args" in solver_options:
        raise ValueError("args is managed internally by solve_micrm")
    if solver_options.get("vectorized", False):
        raise ValueError("solve_micrm does not support vectorized=True")

    state_array = _state_array(
        initial_state,
        parameters,
        require_nonnegative=True,
    )
    evaluation_times = None if t_eval is None else np.asarray(t_eval, dtype=float)

    def model_rhs(time: float, state: FloatArray) -> FloatArray:
        return micrm_rhs(time, state, parameters)

    return solve_ivp(
        model_rhs,
        t_span,
        state_array,
        t_eval=evaluation_times,
        **solver_options,
    )
