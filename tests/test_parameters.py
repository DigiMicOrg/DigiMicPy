import unittest

import numpy as np

from digimicpy.parameters import generate_l_tensor, modular_leakage, modular_uptake


class ParameterGenerationTests(unittest.TestCase):
    def test_modular_uptake_is_seeded_and_row_normalized(self):
        first = modular_uptake(5, 4, 2, 8.0, rng=np.random.default_rng(7))
        second = modular_uptake(5, 4, 2, 8.0, rng=np.random.default_rng(7))

        self.assertEqual(first.shape, (5, 4))
        np.testing.assert_allclose(first, second)
        np.testing.assert_allclose(first.sum(axis=1), np.ones(5))
        self.assertTrue(np.all(first >= 0.0))

    def test_modular_leakage_has_declared_row_sum(self):
        leakage = modular_leakage(
            5,
            2,
            4.0,
            0.2,
            rng=np.random.default_rng(11),
        )

        self.assertEqual(leakage.shape, (5, 5))
        np.testing.assert_allclose(leakage.sum(axis=1), np.full(5, 0.2))
        self.assertTrue(np.all(leakage >= 0.0))

    def test_generate_l_tensor_reuses_the_supplied_generator(self):
        first = generate_l_tensor(
            3,
            4,
            2,
            6.0,
            0.15,
            rng=np.random.default_rng(21),
        )
        second = generate_l_tensor(
            3,
            4,
            2,
            6.0,
            0.15,
            rng=np.random.default_rng(21),
        )

        self.assertEqual(first.shape, (3, 4, 4))
        np.testing.assert_allclose(first, second)
        np.testing.assert_allclose(first.sum(axis=2), np.full((3, 4), 0.15))

    def test_generation_does_not_advance_numpy_global_random_state(self):
        np.random.seed(31)
        expected = np.random.random()
        np.random.seed(31)

        modular_uptake(4, 4, 2, 3.0, rng=np.random.default_rng(31))

        self.assertEqual(np.random.random(), expected)

    def test_invalid_generation_inputs_raise_value_error(self):
        invalid_calls = (
            lambda: modular_uptake(0, 4, 2, 3.0),
            lambda: modular_uptake(4, 4, 0, 3.0),
            lambda: modular_uptake(4, 4, 5, 3.0),
            lambda: modular_uptake(4, 4, 2, 0.0),
            lambda: modular_leakage(4, 2, 3.0, -0.1),
            lambda: modular_leakage(4, 2, 3.0, 1.1),
        )

        for invalid_call in invalid_calls:
            with self.subTest(invalid_call=invalid_call):
                with self.assertRaises(ValueError):
                    invalid_call()


if __name__ == "__main__":
    unittest.main()