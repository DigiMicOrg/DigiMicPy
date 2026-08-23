"""Run one consumer-resource community across two temperature patches."""

import numpy as np

from digimicpy import (
    MiCRMParameters,
    distance_connectivity,
    solve_spatial_micrm,
    temperature_adjusted_parameters,
)


def build_reference_parameters() -> MiCRMParameters:
    """Construct parameters interpreted at the reference temperature."""

    return MiCRMParameters(
        uptake=[[0.8, 0.2], [0.3, 0.7]],
        mortality=[0.15, 0.2],
        resource_supply=[0.6, 0.4],
        resource_decay=[0.4, 0.4],
        leakage=[
            [[0.08, 0.02], [0.03, 0.07]],
            [[0.05, 0.05], [0.02, 0.08]],
        ],
        leakage_fraction=[0.1, 0.1],
        consumer_ids=["consumer-1", "consumer-2"],
        resource_ids=["resource-1", "resource-2"],
    )


def parameters_at_temperature(
    reference: MiCRMParameters,
    temperature: float,
) -> MiCRMParameters:
    """Evaluate shared consumer traits at one patch temperature."""

    return temperature_adjusted_parameters(
        reference,
        temperature,
        283.15,
        uptake_activation_energy=[0.7, 0.8],
        uptake_optimum_temperature=[303.15, 308.15],
        uptake_deactivation_energy=3.5,
        mortality_activation_energy=[0.5, 0.6],
        mortality_optimum_temperature=[306.15, 311.15],
        mortality_deactivation_energy=3.5,
    )


def run_simulation():
    """Integrate two coupled patches and return SciPy's solver result."""

    reference = build_reference_parameters()
    patch_parameters = [
        parameters_at_temperature(reference, 288.15),
        parameters_at_temperature(reference, 298.15),
    ]
    connectivity = distance_connectivity([[0.0], [1.0]], decay_rate=0.5)
    initial_state = np.array(
        [
            [0.05, 0.05, 1.0, 1.0],
            [0.05, 0.05, 1.0, 1.0],
        ]
    )
    result = solve_spatial_micrm(
        patch_parameters,
        initial_state,
        (0.0, 20.0),
        connectivity=connectivity,
        consumer_diffusion=0.01,
        resource_diffusion=0.05,
        t_eval=np.linspace(0.0, 20.0, 101),
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result


def main() -> None:
    result = run_simulation()
    final_state = result.y[:, -1].reshape((2, 4))
    print("Final patch states:")
    print(np.round(final_state, 4))


if __name__ == "__main__":
    main()
