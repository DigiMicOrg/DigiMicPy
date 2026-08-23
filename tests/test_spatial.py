import unittest

import numpy as np

from digimicpy import MiCRMParameters, micrm_rhs
from digimicpy.spatial import (
    distance_connectivity,
    solve_spatial_micrm,
    spatial_micrm_rhs,
)


def inert_parameters(
    n_consumers=1,
    n_resources=1,
    *,
    labeled=True,
    label_prefix="",
) -> MiCRMParameters:
    identifiers = {}
    if labeled:
        identifiers = {
            "consumer_ids": tuple(
                f"{label_prefix}consumer-{index}" for index in range(n_consumers)
            ),
            "resource_ids": tuple(
                f"{label_prefix}resource-{index}" for index in range(n_resources)
            ),
        }
    return MiCRMParameters(
        uptake=np.zeros((n_consumers, n_resources)),
        mortality=np.zeros(n_consumers),
        resource_supply=np.zeros(n_resources),
        resource_decay=np.zeros(n_resources),
        leakage=np.zeros((n_consumers, n_resources, n_resources)),
        leakage_fraction=np.zeros(n_resources),
        **identifiers,
    )


class SpatialMiCRMTests(unittest.TestCase):
    def test_distance_connectivity_is_symmetric_with_zero_diagonal(self):
        connectivity = distance_connectivity([[0.0, 0.0], [3.0, 4.0]])

        np.testing.assert_allclose(connectivity, [[0.0, np.exp(-5.0)], [np.exp(-5.0), 0.0]])

    def test_disconnected_rhs_matches_independent_patch_dynamics(self):
        first = MiCRMParameters(
            uptake=[[1.0]],
            mortality=[0.2],
            resource_supply=[0.5],
            resource_decay=[0.1],
            leakage=[[[0.0]]],
            leakage_fraction=[0.0],
        )
        second = MiCRMParameters(
            uptake=[[0.5]],
            mortality=[0.1],
            resource_supply=[0.8],
            resource_decay=[0.2],
            leakage=[[[0.0]]],
            leakage_fraction=[0.0],
        )
        state = np.array([[0.25, 0.75], [0.4, 0.5]])

        derivative = spatial_micrm_rhs(
            0.0,
            state,
            [first, second],
            np.zeros((2, 2)),
        )
        expected = np.concatenate(
            [micrm_rhs(0.0, state[0], first), micrm_rhs(0.0, state[1], second)]
        )

        np.testing.assert_allclose(derivative, expected)

    def test_transport_is_mass_conserving_for_every_state_variable(self):
        derivative = spatial_micrm_rhs(
            0.0,
            [[1.0, 2.0], [3.0, 6.0]],
            [inert_parameters(), inert_parameters()],
            [[0.0, 1.0], [1.0, 0.0]],
            consumer_diffusion=0.5,
            resource_diffusion=0.25,
        ).reshape((2, 2))

        np.testing.assert_allclose(derivative[0], [1.0, 1.0])
        np.testing.assert_allclose(derivative[1], [-1.0, -1.0])
        np.testing.assert_allclose(derivative.sum(axis=0), np.zeros(2))

    def test_equal_patch_states_have_zero_transport(self):
        derivative = spatial_micrm_rhs(
            0.0,
            [[2.0, 4.0], [2.0, 4.0]],
            [inert_parameters(), inert_parameters()],
            [[0.0, 0.7], [0.7, 0.0]],
            consumer_diffusion=0.5,
            resource_diffusion=0.25,
        )

        np.testing.assert_allclose(derivative, np.zeros(4))

    def test_transport_requires_matching_explicit_identities(self):
        calls = (
            lambda: spatial_micrm_rhs(
                0.0,
                [[1.0, 2.0], [3.0, 6.0]],
                [inert_parameters(labeled=False), inert_parameters(labeled=False)],
                [[0.0, 1.0], [1.0, 0.0]],
                consumer_diffusion=0.5,
            ),
            lambda: spatial_micrm_rhs(
                0.0,
                [[1.0, 2.0], [3.0, 6.0]],
                [inert_parameters(), inert_parameters(label_prefix="other-")],
                [[0.0, 1.0], [1.0, 0.0]],
                resource_diffusion=0.25,
            ),
        )

        for call in calls:
            with self.subTest(call=call):
                with self.assertRaisesRegex(ValueError, "_ids"):
                    call()

    def test_tolerated_connectivity_roundoff_is_symmetrized(self):
        derivative = spatial_micrm_rhs(
            0.0,
            [[1.0, 2.0], [3.0, 6.0]],
            [inert_parameters(), inert_parameters()],
            [[0.0, 1.0], [1.0 + 5e-13, 0.0]],
            consumer_diffusion=0.5,
            resource_diffusion=0.25,
        ).reshape((2, 2))

        np.testing.assert_allclose(derivative.sum(axis=0), np.zeros(2), atol=1e-15)

    def test_solver_matches_two_patch_diffusion_solution(self):
        result = solve_spatial_micrm(
            [inert_parameters(), inert_parameters()],
            [[1.0, 0.0], [3.0, 0.0]],
            (0.0, 1.0),
            connectivity=[[0.0, 1.0], [1.0, 0.0]],
            consumer_diffusion=0.5,
            t_eval=[0.0, 1.0],
            rtol=1e-10,
            atol=1e-12,
        )

        self.assertTrue(result.success, result.message)
        final_consumers = result.y[[0, 2], -1]
        expected_difference = 2.0 * np.exp(-1.0)
        np.testing.assert_allclose(final_consumers.sum(), 4.0, atol=1e-10)
        np.testing.assert_allclose(
            final_consumers[1] - final_consumers[0],
            expected_difference,
            rtol=1e-9,
        )

    def test_solver_preserves_standard_event_signature(self):
        def patches_nearly_equal(time, state):
            del time
            return state[2] - state[0] - 0.5

        patches_nearly_equal.terminal = True
        result = solve_spatial_micrm(
            [inert_parameters(), inert_parameters()],
            [[1.0, 0.0], [3.0, 0.0]],
            (0.0, 5.0),
            connectivity=[[0.0, 1.0], [1.0, 0.0]],
            consumer_diffusion=0.5,
            events=patches_nearly_equal,
            rtol=1e-9,
            atol=1e-11,
        )

        self.assertTrue(result.success, result.message)
        self.assertEqual(len(result.t_events[0]), 1)
        np.testing.assert_allclose(result.t_events[0][0], np.log(4.0), rtol=1e-4)

    def test_solver_preserves_standard_jacobian_signature(self):
        def jacobian(time, state):
            del time, state
            return np.array(
                [
                    [-0.5, 0.0, 0.5, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.5, 0.0, -0.5, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )

        result = solve_spatial_micrm(
            [inert_parameters(), inert_parameters()],
            [[1.0, 0.0], [3.0, 0.0]],
            (0.0, 0.1),
            connectivity=[[0.0, 1.0], [1.0, 0.0]],
            consumer_diffusion=0.5,
            method="BDF",
            jac=jacobian,
        )

        self.assertTrue(result.success, result.message)

    def test_invalid_spatial_inputs_are_rejected(self):
        invalid_calls = (
            lambda: spatial_micrm_rhs(
                0.0,
                [1.0, 1.0, 1.0, 1.0],
                [inert_parameters(), inert_parameters()],
                [[0.0, 1.0], [0.0, 0.0]],
            ),
            lambda: spatial_micrm_rhs(
                0.0,
                [1.0, 1.0, 1.0, 1.0],
                [inert_parameters(), inert_parameters()],
                [[1.0, 0.0], [0.0, 0.0]],
            ),
            lambda: spatial_micrm_rhs(
                0.0,
                [1.0, 1.0, 1.0, 1.0],
                [inert_parameters(), inert_parameters()],
                [[0.0, 1.0], [1.0, 0.0]],
                consumer_diffusion=-0.1,
            ),
            lambda: spatial_micrm_rhs(
                0.0,
                np.ones(5),
                [inert_parameters(), inert_parameters()],
                [[0.0, 1.0], [1.0, 0.0]],
            ),
            lambda: spatial_micrm_rhs(
                0.0,
                np.ones(5),
                [inert_parameters(), inert_parameters(n_consumers=2)],
                [[0.0, 1.0], [1.0, 0.0]],
            ),
        )

        for invalid_call in invalid_calls:
            with self.subTest(invalid_call=invalid_call):
                with self.assertRaises(ValueError):
                    invalid_call()


if __name__ == "__main__":
    unittest.main()
