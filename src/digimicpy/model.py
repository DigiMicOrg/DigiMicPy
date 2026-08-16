"""Validated data structures shared by DigiMicPy models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _float_array(name: str, values: ArrayLike) -> FloatArray:
    try:
        array = np.array(values, dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a rectangular numeric array") from error

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must contain only nonnegative values")
    return array


def _require_shape(name: str, array: FloatArray, shape: tuple[int, ...]) -> None:
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")


def _freeze(array: FloatArray) -> FloatArray:
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True, init=False)
class MiCRMParameters:
    """Validated parameters for a microbial consumer-resource model."""

    uptake: FloatArray
    mortality: FloatArray
    resource_supply: FloatArray
    resource_decay: FloatArray
    leakage: FloatArray
    leakage_fraction: FloatArray

    def __init__(
        self,
        uptake: ArrayLike,
        mortality: ArrayLike,
        resource_supply: ArrayLike,
        resource_decay: ArrayLike,
        leakage: ArrayLike,
        leakage_fraction: ArrayLike,
    ) -> None:
        uptake_array = _float_array("uptake", uptake)
        if uptake_array.ndim != 2:
            raise ValueError(f"uptake must be two-dimensional, got {uptake_array.ndim}")

        n_consumers, n_resources = uptake_array.shape
        if n_consumers == 0 or n_resources == 0:
            raise ValueError("uptake must describe at least one consumer and one resource")

        mortality_array = _float_array("mortality", mortality)
        supply_array = _float_array("resource_supply", resource_supply)
        decay_array = _float_array("resource_decay", resource_decay)
        leakage_array = _float_array("leakage", leakage)
        fraction_array = _float_array("leakage_fraction", leakage_fraction)

        _require_shape("mortality", mortality_array, (n_consumers,))
        _require_shape("resource_supply", supply_array, (n_resources,))
        _require_shape("resource_decay", decay_array, (n_resources,))
        _require_shape(
            "leakage",
            leakage_array,
            (n_consumers, n_resources, n_resources),
        )

        if fraction_array.ndim == 1:
            _require_shape("leakage_fraction", fraction_array, (n_resources,))
            fraction_array = np.broadcast_to(
                fraction_array,
                (n_consumers, n_resources),
            ).copy()
        elif fraction_array.ndim == 2:
            _require_shape(
                "leakage_fraction",
                fraction_array,
                (n_consumers, n_resources),
            )
        else:
            raise ValueError(
                "leakage_fraction must be a resource vector or consumer-resource matrix"
            )

        if np.any(fraction_array > 1.0):
            raise ValueError("leakage_fraction values must not exceed 1")
        if not np.allclose(
            leakage_array.sum(axis=2),
            fraction_array,
            rtol=1e-7,
            atol=1e-12,
        ):
            raise ValueError("leakage row sums must match leakage_fraction")

        object.__setattr__(self, "uptake", _freeze(uptake_array))
        object.__setattr__(self, "mortality", _freeze(mortality_array))
        object.__setattr__(self, "resource_supply", _freeze(supply_array))
        object.__setattr__(self, "resource_decay", _freeze(decay_array))
        object.__setattr__(self, "leakage", _freeze(leakage_array))
        object.__setattr__(self, "leakage_fraction", _freeze(fraction_array))

    @property
    def n_consumers(self) -> int:
        """Number of consumer populations represented by the model."""

        return self.uptake.shape[0]

    @property
    def n_resources(self) -> int:
        """Number of resources represented by the model."""

        return self.uptake.shape[1]