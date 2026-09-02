"""Simulate two connected patches with different local communities."""

import numpy as np

from digimicpy import (
    MiCRMParameters,
    SpatialLayout,
    SpatialPatch,
    distance_connectivity,
    solve_spatial_micrm,
)


def build_layout() -> SpatialLayout:
    """Construct named patches with different consumer and resource pools."""

    nearshore = MiCRMParameters(
        uptake=[[0.9, 0.2], [0.4, 0.6]],
        mortality=[0.20, 0.15],
        resource_supply=[0.8, 0.5],
        resource_decay=[0.4, 0.3],
        leakage=np.zeros((2, 2, 2)),
        leakage_fraction=np.zeros(2),
        consumer_ids=["nearshore-specialist", "generalist"],
        resource_ids=["carbon", "nitrogen"],
    )
    offshore = MiCRMParameters(
        uptake=[
            [0.6, 0.4, 0.2],
            [0.5, 0.2, 0.4],
            [0.1, 0.8, 0.1],
        ],
        mortality=[0.12, 0.15, 0.18],
        resource_supply=[0.4, 0.7, 0.5],
        resource_decay=[0.3, 0.4, 0.35],
        leakage=np.zeros((3, 3, 3)),
        leakage_fraction=np.zeros(3),
        consumer_ids=["oligotroph", "generalist", "offshore-specialist"],
        # Shared resources are deliberately reordered relative to nearshore.
        resource_ids=["nitrogen", "phosphorus", "carbon"],
    )
    return SpatialLayout(
        [
            SpatialPatch(nearshore, volume=1.5, name="nearshore"),
            SpatialPatch(offshore, volume=3.0, name="offshore"),
        ]
    )


def run_simulation():
    """Integrate the heterogeneous landscape and return its layout and result."""

    layout = build_layout()
    initial_state = [
        # nearshore-specialist, generalist, carbon, nitrogen
        np.array([0.08, 0.05, 1.0, 0.8]),
        # oligotroph, generalist (potential colonist), offshore-specialist,
        # nitrogen, phosphorus, carbon
        np.array([0.04, 0.0, 0.06, 0.8, 1.2, 1.0]),
    ]
    result = solve_spatial_micrm(
        layout,
        initial_state,
        (0.0, 20.0),
        connectivity=distance_connectivity([[0.0], [1.0]], decay_rate=0.5),
        consumer_diffusion={"generalist": 0.02},
        resource_diffusion={"carbon": 0.05, "nitrogen": 0.05},
        t_eval=np.linspace(0.0, 20.0, 101),
    )
    if not result.success:
        raise RuntimeError(result.message)
    return layout, result


def main() -> None:
    layout, result = run_simulation()
    final_states = layout.unpack_state(result.y[:, -1])
    for patch, state in zip(layout.patches, final_states, strict=True):
        print(f"{patch.name} final state:")
        print(np.round(state, 4))


if __name__ == "__main__":
    main()
