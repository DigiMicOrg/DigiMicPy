import unittest

import numpy as np

from digimicpy.model import MiCRMParameters


def valid_parameters(**overrides):
    values = {
        "uptake": np.array([[0.75, 0.25], [0.4, 0.6]]),
        "mortality": np.array([0.1, 0.2]),
        "resource_supply": np.array([1.0, 0.5]),
        "resource_decay": np.array([0.3, 0.4]),
        "leakage": np.array(
            [
                [[0.15, 0.05], [0.08, 0.12]],
                [[0.1, 0.1], [0.04, 0.16]],
            ]
        ),
        "leakage_fraction": np.array([0.2, 0.2]),
    }
    values.update(overrides)
    return MiCRMParameters(**values)


class MiCRMParametersTests(unittest.TestCase):
    def test_resource_leakage_vector_is_broadcast_per_consumer(self):
        parameters = valid_parameters()

        self.assertEqual(parameters.n_consumers, 2)
        self.assertEqual(parameters.n_resources, 2)
        self.assertEqual(parameters.leakage_fraction.shape, (2, 2))
        np.testing.assert_allclose(parameters.leakage_fraction, np.full((2, 2), 0.2))

    def test_consumer_specific_leakage_fraction_is_retained(self):
        leakage_fraction = np.array([[0.2, 0.2], [0.2, 0.2]])

        parameters = valid_parameters(leakage_fraction=leakage_fraction)

        np.testing.assert_allclose(parameters.leakage_fraction, leakage_fraction)

    def test_inputs_are_converted_to_float_arrays(self):
        parameters = valid_parameters(
            mortality=[1, 2],
            resource_supply=[1, 1],
            resource_decay=[1, 1],
        )

        self.assertTrue(np.issubdtype(parameters.mortality.dtype, np.floating))
        self.assertTrue(np.issubdtype(parameters.resource_supply.dtype, np.floating))
        self.assertTrue(np.issubdtype(parameters.resource_decay.dtype, np.floating))

    def test_invalid_shapes_raise_value_error(self):
        invalid_overrides = (
            {"uptake": np.ones(2)},
            {"mortality": np.ones(3)},
            {"resource_supply": np.ones(3)},
            {"resource_decay": np.ones(3)},
            {"leakage": np.ones((2, 2, 3))},
            {"leakage_fraction": np.ones((2, 2, 1))},
        )

        for override in invalid_overrides:
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    valid_parameters(**override)

    def test_invalid_values_raise_value_error(self):
        invalid_overrides = (
            {"uptake": np.array([[np.nan, 0.0], [0.0, 1.0]])},
            {"mortality": np.array([-0.1, 0.2])},
            {"resource_supply": np.array([np.inf, 0.5])},
            {"leakage_fraction": np.array([1.1, 0.2])},
            {"leakage": np.full((2, 2, 2), -0.1)},
            {"uptake": np.array([[1.0 + 2.0j, 0.0], [0.0, 1.0]])},
        )

        for override in invalid_overrides:
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    valid_parameters(**override)

    def test_leakage_rows_must_match_declared_fraction(self):
        with self.assertRaisesRegex(ValueError, "row sums"):
            valid_parameters(leakage_fraction=np.array([0.1, 0.1]))

    def test_equality_compares_array_values_without_ambiguity(self):
        first = valid_parameters()
        second = valid_parameters()

        self.assertIs(first == second, True)
        self.assertIs(first == valid_parameters(mortality=[0.1, 0.3]), False)

    def test_parameters_are_explicitly_unhashable(self):
        with self.assertRaises(TypeError):
            hash(valid_parameters())


if __name__ == "__main__":
    unittest.main()
