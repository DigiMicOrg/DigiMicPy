"""Random parameter generation for modular consumer-resource models."""

from __future__ import annotations

import operator

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def _positive_integer(name: str, value: int) -> int:
    try:
        integer = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error
    if isinstance(value, bool) or integer <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _positive_finite(name: str, value: float) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return number


def _leakage_fraction(value: float) -> float:
    fraction = float(value)
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("total_leakage must be finite and between 0 and 1")
    return fraction


def _generator(rng: np.random.Generator | None) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    return rng


def _module_indices(
    size: int,
    n_modules: int,
    rng: np.random.Generator,
) -> list[NDArray[np.int_]]:
    base_size, remainder = divmod(size, n_modules)
    module_sizes = np.full(n_modules, base_size, dtype=int)
    if remainder:
        selected = rng.choice(n_modules, remainder, replace=False)
        module_sizes[selected] += 1

    starts = np.cumsum(module_sizes) - module_sizes
    return [
        np.arange(start, start + module_size)
        for start, module_size in zip(starts, module_sizes)
    ]


def modular_uptake(
    n_consumers: int,
    n_resources: int,
    n_modules: int,
    specialization_ratio: float,
    *,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """Generate row-normalized uptake preferences with modular specialization."""

    consumers = _positive_integer("n_consumers", n_consumers)
    resources = _positive_integer("n_resources", n_resources)
    modules = _positive_integer("n_modules", n_modules)
    ratio = _positive_finite("specialization_ratio", specialization_ratio)
    if modules > min(consumers, resources):
        raise ValueError("n_modules must not exceed n_consumers or n_resources")

    generator = _generator(rng)
    resource_modules = _module_indices(resources, modules, generator)
    consumer_modules = _module_indices(consumers, modules, generator)
    uptake = generator.random((consumers, resources))

    for consumer_indices, resource_indices in zip(
        consumer_modules,
        resource_modules,
    ):
        uptake[np.ix_(consumer_indices, resource_indices)] *= ratio

    uptake /= uptake.sum(axis=1, keepdims=True)
    return uptake


def modular_leakage(
    n_resources: int,
    n_modules: int,
    specialization_ratio: float,
    total_leakage: float,
    *,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """Generate a leakage matrix whose rows sum to ``total_leakage``."""

    resources = _positive_integer("n_resources", n_resources)
    modules = _positive_integer("n_modules", n_modules)
    ratio = _positive_finite("specialization_ratio", specialization_ratio)
    fraction = _leakage_fraction(total_leakage)
    if modules > resources:
        raise ValueError("n_modules must not exceed n_resources")

    generator = _generator(rng)
    resource_modules = _module_indices(resources, modules, generator)
    leakage = generator.random((resources, resources))

    for source_index, source_resources in enumerate(resource_modules):
        for target_index, target_resources in enumerate(resource_modules):
            if source_index == target_index or source_index + 1 == target_index:
                leakage[np.ix_(source_resources, target_resources)] *= ratio

    leakage /= leakage.sum(axis=1, keepdims=True)
    leakage *= fraction
    return leakage


def generate_l_tensor(
    n_consumers: int,
    n_resources: int,
    n_modules: int,
    specialization_ratio: float,
    total_leakage: float,
    *,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """Generate one modular leakage matrix per consumer."""

    consumers = _positive_integer("n_consumers", n_consumers)
    generator = _generator(rng)
    matrices = [
        modular_leakage(
            n_resources,
            n_modules,
            specialization_ratio,
            total_leakage,
            rng=generator,
        )
        for _ in range(consumers)
    ]
    return np.stack(matrices)