import unittest

import numpy as np

from digimicpy.micrm import micrm_rhs, solve_micrm
from digimicpy.model import MiCRMParameters


class MiCRMDynamicsTests(unittest.TestCase):
    def test_rhs_matches_hand_calculated_fluxes(self):
        parameters = MiCRMParameters(
            uptake=[[2.0, 1.0]],
            mortality=[0.5],
            resource_supply=[1.0, 2.0],
            resource_decay=[0.1, 0.2],
            leakage=[[[0.1, 0.1], [0.05, 0.05]]],
            leakage_fraction=[0.2, 0.1],
        )

        derivative = micrm_rhs(0.0, [3.0, 4.0, 5.0], parameters)

        np.testing.assert_allclose(derivative, [31.2, -20.25, -10.85])

    def test_rhs_is_independent_of_time(self):
        parameters = MiCRMParameters(
            uptake=[[1.0]],
            mortality=[0.2],
            resource_supply=[0.5],
            resource_decay=[0.1],
            leakage=[[[0.0]]],
            leakage_fraction=[0.0],
        )
        state = [0.25, 0.75]

        np.testing.assert_allclose(
            micrm_rhs(0.0, state, parameters),
            micrm_rhs(10.0, state, parameters),
        )

    def test_rhs_rejects_invalid_state(self):
        parameters = MiCRMParameters(
            uptake=[[1.0]],
            mortality=[0.2],
            resource_supply=[0.5],
            resource_decay=[0.1],
            leakage=[[[0.0]]],
            leakage_fraction=[0.0],
        )

        for state in ([1.0], [[1.0, 1.0]], [1.0, np.nan]):
            with self.subTest(state=state):
                with self.assertRaises(ValueError):
                    micrm_rhs(0.0, state, parameters)

    def test_solver_returns_expected_supply_decay_trajectory(self):
        parameters = MiCRMParameters(
            uptake=[[1.0]],
            mortality=[0.0],
            resource_supply=[1.0],
            resource_decay=[1.0],
            leakage=[[[0.0]]],
            leakage_fraction=[0.0],
        )
        initial_state = np.array([0.0, 0.0])

        result = solve_micrm(
            parameters,
            initial_state,
            (0.0, 1.0),
            t_eval=[0.0, 0.5, 1.0],
            rtol=1e-9,
            atol=1e-11,
        )

        self.assertTrue(result.success, result.message)
        self.assertEqual(result.y.shape, (2, 3))
        np.testing.assert_allclose(result.y[:, 0], initial_state)
        np.testing.assert_allclose(result.y[0], np.zeros(3), atol=1e-10)
        np.testing.assert_allclose(result.y[1, -1], 1.0 - np.exp(-1.0), rtol=1e-8)

    def test_solver_rejects_negative_initial_state(self):
        parameters = MiCRMParameters(
            uptake=[[1.0]],
            mortality=[0.0],
            resource_supply=[1.0],
            resource_decay=[1.0],
            leakage=[[[0.0]]],
            leakage_fraction=[0.0],
        )

        with self.assertRaisesRegex(ValueError, "nonnegative"):
            solve_micrm(parameters, [-0.1, 1.0], (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()