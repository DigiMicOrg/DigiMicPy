"""Run and plot a small microbial consumer-resource simulation."""

import matplotlib.pyplot as plt
import numpy as np

from digimicpy import (
    MiCRMParameters,
    generate_l_tensor,
    modular_uptake,
    solve_micrm,
)


def build_parameters(seed: int = 42) -> MiCRMParameters:
    """Construct a reproducible parameter set for the example."""

    rng = np.random.default_rng(seed)
    n_consumers = 6
    n_resources = 4
    total_leakage = 0.1
    uptake = modular_uptake(
        n_consumers,
        n_resources,
        2,
        8.0,
        rng=rng,
    )
    leakage = generate_l_tensor(
        n_consumers,
        n_resources,
        2,
        8.0,
        total_leakage,
        rng=rng,
    )
    return MiCRMParameters(
        uptake=uptake,
        mortality=np.full(n_consumers, 0.2),
        resource_supply=np.full(n_resources, 0.5),
        resource_decay=np.full(n_resources, 0.5),
        leakage=leakage,
        leakage_fraction=np.full(n_resources, total_leakage),
    )


def run_simulation(seed: int = 42):
    """Run the example and return SciPy's integration result."""

    parameters = build_parameters(seed)
    initial_state = np.concatenate(
        (
            np.full(parameters.n_consumers, 0.01),
            np.ones(parameters.n_resources),
        )
    )
    result = solve_micrm(
        parameters,
        initial_state,
        (0.0, 20.0),
        t_eval=np.linspace(0.0, 20.0, 101),
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result


def plot_result(result, n_consumers: int) -> None:
    """Plot consumer and resource trajectories from a simulation result."""

    for index in range(n_consumers):
        plt.plot(result.t, result.y[index], label=f"Consumer {index + 1}")
    for index in range(n_consumers, result.y.shape[0]):
        plt.plot(
            result.t,
            result.y[index],
            linestyle="--",
            label=f"Resource {index - n_consumers + 1}",
        )
    plt.xlabel("Time")
    plt.ylabel("Abundance")
    plt.title("Microbial consumer-resource dynamics")
    plt.legend(ncol=2)
    plt.tight_layout()


def main() -> None:
    parameters = build_parameters()
    result = run_simulation()
    plot_result(result, parameters.n_consumers)
    plt.show()


if __name__ == "__main__":
    main()