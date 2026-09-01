"""Temperature responses for microbial consumer-resource parameters."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .model import MiCRMParameters


FloatArray = NDArray[np.float64]

# Boltzmann's constant in electronvolts per kelvin.
BOLTZMANN_CONSTANT = 8.617333262145e-5


def _real_array(name: str, values: ArrayLike) -> FloatArray:
    try:
        raw = np.asanyarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain only real values")
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _positive_array(name: str, values: ArrayLike) -> FloatArray:
    array = _real_array(name, values)
    if np.any(array <= 0.0):
        raise ValueError(f"{name} must contain only values greater than zero")
    return array


def _nonnegative_array(name: str, values: ArrayLike) -> FloatArray:
    array = _real_array(name, values)
    if np.any(array < 0.0):
        raise ValueError(f"{name} must contain only nonnegative values")
    return array


def _positive_scalar(name: str, value: float) -> float:
    array = _positive_array(name, value)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar")
    return float(array)


def _thermal_log_performance(
    temperature: ArrayLike,
    normalization: ArrayLike,
    activation_energy: ArrayLike,
    optimum_temperature: ArrayLike,
    deactivation_energy: ArrayLike,
    reference_temperature: ArrayLike,
    boltzmann_constant: float,
) -> FloatArray:
    temperatures = _positive_array("temperature", temperature)
    normalizations = _nonnegative_array("normalization", normalization)
    activation = _positive_array("activation_energy", activation_energy)
    optimum = _positive_array("optimum_temperature", optimum_temperature)
    deactivation = _positive_array("deactivation_energy", deactivation_energy)
    reference = _positive_array("reference_temperature", reference_temperature)
    constant = _positive_scalar("boltzmann_constant", boltzmann_constant)

    try:
        (
            temperatures,
            normalizations,
            activation,
            optimum,
            deactivation,
            reference,
        ) = np.broadcast_arrays(
            temperatures,
            normalizations,
            activation,
            optimum,
            deactivation,
            reference,
        )
    except ValueError as error:
        raise ValueError("thermal performance inputs could not be broadcast together") from error
    if np.any(deactivation <= activation):
        raise ValueError("deactivation_energy must exceed activation_energy")

    with np.errstate(divide="ignore", invalid="ignore"):
        log_increase = (
            -(activation / constant) * (1.0 / temperatures - 1.0 / reference)
        )
        log_loss_term = np.log(activation / (deactivation - activation)) + (
            deactivation / constant
        ) * (1.0 / optimum - 1.0 / temperatures)
        log_result = (
            np.log(normalizations)
            + log_increase
            - np.logaddexp(0.0, log_loss_term)
        )

    if np.any(np.isnan(log_result)) or np.any(np.isposinf(log_result)):
        raise ValueError("thermal performance is undefined for the supplied inputs")
    return np.asarray(log_result, dtype=float)


def thermal_performance(
    temperature: ArrayLike,
    normalization: ArrayLike,
    activation_energy: ArrayLike,
    optimum_temperature: ArrayLike,
    deactivation_energy: ArrayLike,
    reference_temperature: ArrayLike,
    *,
    boltzmann_constant: float = BOLTZMANN_CONSTANT,
) -> FloatArray:
    """Evaluate a high-temperature-deactivation thermal performance curve.

    Temperatures are absolute and therefore normally supplied in kelvin. Energies
    must use units consistent with ``boltzmann_constant``; the default expects
    electronvolts. Inputs follow NumPy broadcasting rules.
    """

    log_result = _thermal_log_performance(
        temperature,
        normalization,
        activation_energy,
        optimum_temperature,
        deactivation_energy,
        reference_temperature,
        boltzmann_constant,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        result_array = np.asarray(np.exp(log_result), dtype=float)
    if not np.all(np.isfinite(result_array)):
        raise ValueError("thermal performance is non-finite for the supplied inputs")
    return result_array


def thermal_scaling_factor(
    temperature: ArrayLike,
    activation_energy: ArrayLike,
    optimum_temperature: ArrayLike,
    deactivation_energy: ArrayLike,
    reference_temperature: ArrayLike,
    *,
    boltzmann_constant: float = BOLTZMANN_CONSTANT,
) -> FloatArray:
    """Return thermal performance relative to the reference temperature."""

    numerator_log = _thermal_log_performance(
        temperature,
        1.0,
        activation_energy,
        optimum_temperature,
        deactivation_energy,
        reference_temperature,
        boltzmann_constant,
    )
    denominator_log = _thermal_log_performance(
        reference_temperature,
        1.0,
        activation_energy,
        optimum_temperature,
        deactivation_energy,
        reference_temperature,
        boltzmann_constant,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(np.exp(numerator_log - denominator_log), dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError("thermal scaling factor is non-finite for the supplied inputs")
    return result


def _consumer_scale(name: str, values: ArrayLike, n_consumers: int) -> FloatArray:
    array = _positive_array(name, values)
    try:
        return np.broadcast_to(array, (n_consumers,)).copy()
    except ValueError as error:
        raise ValueError(
            f"{name} must be scalar or broadcast to ({n_consumers},)"
        ) from error


def temperature_adjusted_parameters(
    parameters: MiCRMParameters,
    temperature: float,
    reference_temperature: float,
    *,
    uptake_activation_energy: ArrayLike,
    uptake_optimum_temperature: ArrayLike,
    uptake_deactivation_energy: ArrayLike,
    mortality_activation_energy: ArrayLike,
    mortality_optimum_temperature: ArrayLike,
    mortality_deactivation_energy: ArrayLike,
    boltzmann_constant: float = BOLTZMANN_CONSTANT,
) -> MiCRMParameters:
    """Scale consumer uptake and mortality away from a reference temperature.

    The supplied ``parameters`` are interpreted as the rates at
    ``reference_temperature``. Consumer-specific thermal traits may be scalars
    or vectors with one value per consumer. Structural uptake preferences,
    leakage, resource supply, and resource decay remain unchanged.
    """

    if not isinstance(parameters, MiCRMParameters):
        raise TypeError("parameters must be a MiCRMParameters instance")
    fixed_temperature = _positive_scalar("temperature", temperature)
    fixed_reference = _positive_scalar(
        "reference_temperature",
        reference_temperature,
    )

    uptake_factor = thermal_scaling_factor(
        fixed_temperature,
        uptake_activation_energy,
        uptake_optimum_temperature,
        uptake_deactivation_energy,
        fixed_reference,
        boltzmann_constant=boltzmann_constant,
    )
    mortality_factor = thermal_scaling_factor(
        fixed_temperature,
        mortality_activation_energy,
        mortality_optimum_temperature,
        mortality_deactivation_energy,
        fixed_reference,
        boltzmann_constant=boltzmann_constant,
    )
    uptake_scale = _consumer_scale(
        "uptake thermal scaling factor",
        uptake_factor,
        parameters.n_consumers,
    )
    mortality_scale = _consumer_scale(
        "mortality thermal scaling factor",
        mortality_factor,
        parameters.n_consumers,
    )

    return MiCRMParameters(
        uptake=parameters.uptake * uptake_scale[:, np.newaxis],
        mortality=parameters.mortality * mortality_scale,
        resource_supply=parameters.resource_supply,
        resource_decay=parameters.resource_decay,
        leakage=parameters.leakage,
        leakage_fraction=parameters.leakage_fraction,
        consumer_ids=parameters.consumer_ids,
        resource_ids=parameters.resource_ids,
    )
