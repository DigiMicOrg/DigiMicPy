import unittest
from decimal import Decimal

import numpy as np

from digimicpy import MiCRMParameters, SpatialLayout, SpatialPatch, micrm_rhs
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


def inert_parameters_with_ids(consumer_ids, resource_ids) -> MiCRMParameters:
    """Construct inert local dynamics with explicit biological identities."""

    return MiCRMParameters(
        uptake=np.zeros((len(consumer_ids), len(resource_ids))),
        mortality=np.zeros(len(consumer_ids)),
        resource_supply=np.zeros(len(resource_ids)),
        resource_decay=np.zeros(len(resource_ids)),
        leakage=np.zeros(
            (len(consumer_ids), len(resource_ids), len(resource_ids))
        ),
        leakage_fraction=np.zeros(len(resource_ids)),
        consumer_ids=consumer_ids,
        resource_ids=resource_ids,
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

    def test_numeric_object_states_remain_supported(self):
        parameters = [inert_parameters(), inert_parameters()]
        connectivity = np.zeros((2, 2))
        states = (
            np.array([1, 2, 3, 4], dtype=object),
            np.array([[1, 2], [3, 4]], dtype=object),
            [Decimal("1"), Decimal("2"), Decimal("3"), Decimal("4")],
        )

        for state in states:
            with self.subTest(state=state):
                derivative = spatial_micrm_rhs(
                    0.0,
                    state,
                    parameters,
                    connectivity,
                )
                np.testing.assert_allclose(derivative, np.zeros(4))

    def test_transport_requires_explicit_identities(self):
        with self.assertRaisesRegex(ValueError, "consumer_ids"):
            spatial_micrm_rhs(
                0.0,
                [[1.0, 2.0], [3.0, 6.0]],
                [inert_parameters(labeled=False), inert_parameters(labeled=False)],
                [[0.0, 1.0], [1.0, 0.0]],
                consumer_diffusion=0.5,
            )

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
        )

        for invalid_call in invalid_calls:
            with self.subTest(invalid_call=invalid_call):
                with self.assertRaises(ValueError):
                    invalid_call()


class HeterogeneousSpatialTests(unittest.TestCase):
    def test_layout_compiles_ragged_slices_id_maps_and_shared_ids(self):
        first = SpatialPatch(
            inert_parameters_with_ids(["consumer-a", "consumer-b"], ["resource-a"]),
            volume=2.0,
            name="estuary",
        )
        second = SpatialPatch(
            inert_parameters_with_ids(
                ["consumer-b"],
                ["resource-c", "resource-a", "resource-b"],
            ),
            volume=3.0,
            name="offshore",
        )

        layout = SpatialLayout([first, second])

        self.assertEqual(layout.state_size, 7)
        self.assertEqual(layout.patch_slices, (slice(0, 3), slice(3, 7)))
        self.assertEqual(layout.consumer_slices, (slice(0, 2), slice(3, 4)))
        self.assertEqual(layout.resource_slices, (slice(2, 3), slice(4, 7)))
        self.assertEqual(layout.consumer_indices[(0, "consumer-b")], 1)
        self.assertEqual(layout.consumer_indices[(1, "consumer-b")], 3)
        self.assertEqual(layout.resource_indices[(1, "resource-a")], 5)
        self.assertEqual(layout.shared_consumer_ids[(0, 1)], ("consumer-b",))
        self.assertEqual(layout.shared_resource_ids[(0, 1)], ("resource-a",))
        self.assertEqual(layout.volumes, (2.0, 3.0))

    def test_ragged_state_pack_and_unpack_round_trip(self):
        layout = SpatialLayout(
            [
                SpatialPatch(inert_parameters_with_ids(["a"], ["r"])),
                SpatialPatch(inert_parameters_with_ids(["b", "a"], ["r", "s"])),
            ]
        )
        patch_states = [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0, 6.0])]

        packed = layout.pack_state(patch_states)
        unpacked = layout.unpack_state(packed)

        np.testing.assert_array_equal(packed, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        np.testing.assert_array_equal(unpacked[0], patch_states[0])
        np.testing.assert_array_equal(unpacked[1], patch_states[1])
        patch_states[0][0] = 99.0
        unpacked[1][0] = 99.0
        self.assertEqual(packed[0], 1.0)
        self.assertEqual(packed[2], 3.0)

    def test_rectangular_and_flat_states_remain_supported(self):
        layout = SpatialLayout(
            [
                SpatialPatch(inert_parameters_with_ids(["a"], ["r"])),
                SpatialPatch(inert_parameters_with_ids(["a"], ["r"])),
            ]
        )
        rectangular = np.array([[1.0, 2.0], [3.0, 4.0]])

        np.testing.assert_array_equal(
            layout.pack_state(rectangular),
            layout.pack_state(rectangular.reshape(-1)),
        )

    def test_disconnected_heterogeneous_rhs_matches_local_models(self):
        first = MiCRMParameters(
            uptake=[[0.5]],
            mortality=[0.1],
            resource_supply=[0.7],
            resource_decay=[0.2],
            leakage=[[[0.0]]],
            leakage_fraction=[0.0],
            consumer_ids=["a"],
            resource_ids=["r"],
        )
        second = MiCRMParameters(
            uptake=[[0.2], [0.8]],
            mortality=[0.05, 0.3],
            resource_supply=[0.4],
            resource_decay=[0.1],
            leakage=[[[0.0]], [[0.0]]],
            leakage_fraction=[0.0],
            consumer_ids=["b", "a"],
            resource_ids=["r"],
        )
        layout = SpatialLayout([SpatialPatch(first), SpatialPatch(second)])
        patch_states = [np.array([0.2, 1.0]), np.array([0.3, 0.4, 0.8])]

        derivative = spatial_micrm_rhs(
            2.0,
            patch_states,
            layout,
            np.zeros((2, 2)),
        )
        expected = np.concatenate(
            [
                micrm_rhs(2.0, patch_states[0], first),
                micrm_rhs(2.0, patch_states[1], second),
            ]
        )

        np.testing.assert_allclose(derivative, expected)

    def test_transport_matches_shared_ids_not_local_positions(self):
        layout = SpatialLayout(
            [
                SpatialPatch(
                    inert_parameters_with_ids(
                        ["consumer-a", "consumer-b"],
                        ["resource-1", "resource-2"],
                    )
                ),
                SpatialPatch(
                    inert_parameters_with_ids(
                        ["consumer-c", "consumer-b", "consumer-a"],
                        ["resource-2", "resource-3"],
                    )
                ),
            ]
        )
        state = [
            [1.0, 2.0, 3.0, 4.0],
            [30.0, 10.0, 20.0, 8.0, 9.0],
        ]

        derivative = layout.unpack_state(
            spatial_micrm_rhs(
                0.0,
                state,
                layout,
                [[0.0, 1.0], [1.0, 0.0]],
                consumer_diffusion={
                    "consumer-a": 0.1,
                    "consumer-b": 0.2,
                    "consumer-c": 0.5,
                },
                resource_diffusion={"resource-2": 0.25},
            )
        )

        np.testing.assert_allclose(derivative[0], [1.9, 1.6, 0.0, 1.0])
        np.testing.assert_allclose(derivative[1], [0.0, -1.6, -1.9, -1.0, 0.0])

    def test_reordering_ids_is_only_a_coordinate_change(self):
        first = SpatialPatch(
            inert_parameters_with_ids(["a", "b"], ["r"])
        )
        layout_one = SpatialLayout(
            [first, SpatialPatch(inert_parameters_with_ids(["b", "a", "c"], ["r"]))]
        )
        layout_two = SpatialLayout(
            [first, SpatialPatch(inert_parameters_with_ids(["a", "c", "b"], ["r"]))]
        )
        first_state = [1.0, 2.0, 0.0]
        second_state_one = [10.0, 20.0, 30.0, 0.0]
        second_state_two = [20.0, 30.0, 10.0, 0.0]
        rates = {"a": 0.1, "b": 0.2, "c": 0.3}

        derivative_one = spatial_micrm_rhs(
            0.0,
            [first_state, second_state_one],
            layout_one,
            [[0.0, 1.0], [1.0, 0.0]],
            consumer_diffusion=rates,
        )
        derivative_two = spatial_micrm_rhs(
            0.0,
            [first_state, second_state_two],
            layout_two,
            [[0.0, 1.0], [1.0, 0.0]],
            consumer_diffusion=rates,
        )

        for patch_index in range(2):
            for identifier in ("a", "b"):
                self.assertAlmostEqual(
                    derivative_one[
                        layout_one.consumer_indices[(patch_index, identifier)]
                    ],
                    derivative_two[
                        layout_two.consumer_indices[(patch_index, identifier)]
                    ],
                )

    def test_structurally_absent_identity_does_not_colonise(self):
        layout = SpatialLayout(
            [
                SpatialPatch(inert_parameters_with_ids(["present"], ["r"])),
                SpatialPatch(inert_parameters_with_ids(["absent"], ["r"])),
            ]
        )

        derivative = spatial_micrm_rhs(
            0.0,
            [[5.0, 0.0], [0.0, 0.0]],
            layout,
            [[0.0, 1.0], [1.0, 0.0]],
            consumer_diffusion=1.0,
        )

        np.testing.assert_allclose(derivative, np.zeros(4))

    def test_zero_abundance_shared_identity_can_colonise_and_conserves_amount(self):
        layout = SpatialLayout(
            [
                SpatialPatch(
                    inert_parameters_with_ids(["colonist"], ["resource"]),
                    volume=2.0,
                ),
                SpatialPatch(
                    inert_parameters_with_ids(["colonist"], ["resource"]),
                    volume=1.0,
                ),
            ]
        )

        derivative = layout.unpack_state(
            spatial_micrm_rhs(
                0.0,
                [[4.0, 8.0], [0.0, 2.0]],
                layout,
                [[0.0, 3.0], [3.0, 0.0]],
                consumer_diffusion={"colonist": 0.5},
                resource_diffusion={"resource": 0.25},
            )
        )

        np.testing.assert_allclose(derivative[0], [-3.0, -2.25])
        np.testing.assert_allclose(derivative[1], [6.0, 4.5])
        np.testing.assert_allclose(
            2.0 * derivative[0] + derivative[1],
            np.zeros(2),
            atol=1e-15,
        )
        self.assertGreater(derivative[1][0], 0.0)

    def test_partial_id_keyed_rates_default_to_zero(self):
        layout = SpatialLayout(
            [
                SpatialPatch(inert_parameters_with_ids(["a", "b"], ["r"])),
                SpatialPatch(inert_parameters_with_ids(["a", "b"], ["r"])),
            ]
        )

        derivative = layout.unpack_state(
            spatial_micrm_rhs(
                0.0,
                [[1.0, 2.0, 0.0], [3.0, 8.0, 0.0]],
                layout,
                [[0.0, 1.0], [1.0, 0.0]],
                consumer_diffusion={"a": 0.5},
            )
        )

        np.testing.assert_allclose(derivative[0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(derivative[1], [-1.0, 0.0, 0.0])

    def test_positional_rates_require_identical_ordered_ids(self):
        reordered = SpatialLayout(
            [
                SpatialPatch(inert_parameters_with_ids(["a", "b"], ["r"])),
                SpatialPatch(inert_parameters_with_ids(["b", "a"], ["r"])),
            ]
        )

        for rates in ([0.1, 0.2], [0.0, 0.0]):
            with self.subTest(rates=rates):
                with self.assertRaisesRegex(ValueError, "ambiguous"):
                    spatial_micrm_rhs(
                        0.0,
                        [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]],
                        reordered,
                        [[0.0, 1.0], [1.0, 0.0]],
                        consumer_diffusion=rates,
                    )

        spatial_micrm_rhs(
            0.0,
            [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]],
            reordered,
            [[0.0, 1.0], [1.0, 0.0]],
            consumer_diffusion={"a": 0.1, "b": 0.2},
        )

    def test_unit_volume_layout_matches_homogeneous_wrapper(self):
        parameters = [
            inert_parameters(n_consumers=2, n_resources=2),
            inert_parameters(n_consumers=2, n_resources=2),
        ]
        layout = SpatialLayout([SpatialPatch(item) for item in parameters])
        state = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 8.0, 9.0, 10.0]])
        options = {
            "consumer_diffusion": [0.1, 0.2],
            "resource_diffusion": [0.3, 0.4],
        }

        wrapped = spatial_micrm_rhs(
            0.0,
            state,
            parameters,
            [[0.0, 0.7], [0.7, 0.0]],
            **options,
        )
        compiled = spatial_micrm_rhs(
            0.0,
            state,
            layout,
            [[0.0, 0.7], [0.7, 0.0]],
            **options,
        )

        np.testing.assert_allclose(compiled, wrapped, rtol=0.0, atol=0.0)

    def test_invalid_patch_metadata_and_identities_are_rejected(self):
        unlabeled = inert_parameters(labeled=False)
        with self.assertRaisesRegex(ValueError, "consumer_ids"):
            SpatialPatch(unlabeled)

        valid = inert_parameters()
        invalid_volumes = (0.0, -1.0, np.nan, np.inf, [1.0])
        for volume in invalid_volumes:
            with self.subTest(volume=volume):
                with self.assertRaises(ValueError):
                    SpatialPatch(valid, volume=volume)

        with self.assertRaises(ValueError):
            SpatialPatch(valid, name=" ")
        with self.assertRaises(TypeError):
            SpatialPatch(valid, name=3)
        with self.assertRaises(ValueError):
            inert_parameters_with_ids(["duplicate", "duplicate"], ["r"])
        with self.assertRaises(ValueError):
            inert_parameters_with_ids([["unhashable"]], ["r"])
        with self.assertRaisesRegex(ValueError, "must not contain None"):
            SpatialPatch(inert_parameters_with_ids([None], ["r"]))
        with self.assertRaisesRegex(ValueError, "stable identifiers"):
            SpatialPatch(inert_parameters_with_ids([np.nan], ["r"]))

    def test_invalid_ragged_states_are_rejected(self):
        layout = SpatialLayout(
            [
                SpatialPatch(inert_parameters_with_ids(["a"], ["r"])),
                SpatialPatch(inert_parameters_with_ids(["a", "b"], ["r"])),
            ]
        )
        invalid_states = (
            [[1.0, 2.0]],
            [[1.0, 2.0], [3.0, 4.0]],
            np.ones(4),
            [[1.0 + 2.0j, 2.0], [3.0, 4.0, 5.0]],
            [[1.0, np.nan], [3.0, 4.0, 5.0]],
        )

        for state in invalid_states:
            with self.subTest(state=state):
                with self.assertRaises(ValueError):
                    layout.pack_state(state)

    def test_invalid_id_keyed_rates_are_rejected(self):
        layout = SpatialLayout(
            [
                SpatialPatch(inert_parameters_with_ids(["a"], ["r"])),
                SpatialPatch(inert_parameters_with_ids(["a"], ["r"])),
            ]
        )
        invalid_rates = (
            {"unknown": 0.1},
            {"a": -0.1},
            {"a": np.nan},
            {"a": np.inf},
        )

        for rates in invalid_rates:
            with self.subTest(rates=rates):
                with self.assertRaises(ValueError):
                    spatial_micrm_rhs(
                        0.0,
                        [[1.0, 0.0], [2.0, 0.0]],
                        layout,
                        [[0.0, 1.0], [1.0, 0.0]],
                        consumer_diffusion=rates,
                    )

        homogeneous = SpatialLayout(
            [
                SpatialPatch(inert_parameters_with_ids(["a", "b"], ["r"])),
                SpatialPatch(inert_parameters_with_ids(["a", "b"], ["r"])),
            ]
        )
        for rates in ([], [0.0, 0.0, 0.0]):
            with self.subTest(rates=rates):
                with self.assertRaisesRegex(ValueError, "broadcast"):
                    spatial_micrm_rhs(
                        0.0,
                        [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]],
                        homogeneous,
                        [[0.0, 1.0], [1.0, 0.0]],
                        consumer_diffusion=rates,
                    )

    def test_heterogeneous_solver_preserves_event_and_jacobian_signatures(self):
        layout = SpatialLayout(
            [
                SpatialPatch(inert_parameters_with_ids(["a"], ["r"])),
                SpatialPatch(inert_parameters_with_ids(["b", "a"], ["r"])),
            ]
        )
        first_a = layout.consumer_indices[(0, "a")]
        second_a = layout.consumer_indices[(1, "a")]

        def patches_nearly_equal(time, state):
            del time
            return state[second_a] - state[first_a] - 0.5

        patches_nearly_equal.terminal = True

        def jacobian(time, state):
            del time, state
            matrix = np.zeros((layout.state_size, layout.state_size))
            matrix[first_a, first_a] = -0.5
            matrix[first_a, second_a] = 0.5
            matrix[second_a, first_a] = 0.5
            matrix[second_a, second_a] = -0.5
            return matrix

        result = solve_spatial_micrm(
            layout,
            [[1.0, 0.0], [0.0, 3.0, 0.0]],
            (0.0, 5.0),
            connectivity=[[0.0, 1.0], [1.0, 0.0]],
            consumer_diffusion={"a": 0.5},
            events=patches_nearly_equal,
            method="BDF",
            jac=jacobian,
            rtol=1e-9,
            atol=1e-11,
        )

        self.assertTrue(result.success, result.message)
        self.assertEqual(len(result.t_events[0]), 1)
        np.testing.assert_allclose(result.t_events[0][0], np.log(4.0), rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
