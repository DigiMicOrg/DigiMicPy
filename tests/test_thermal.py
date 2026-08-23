import unittest

import numpy as np

from digimicpy import MiCRMParameters
from digimicpy.thermal import (
    temperature_adjusted_parameters,
    thermal_performance,
    thermal_scaling_factor,
)


def base_parameters() -> MiCRMParameters:
    return MiCRMParameters(
        uptake=[[0.75, 0.25], [0.4, 0.6]],
        mortality=[0.1, 0.2],
        resource_supply=[1.0, 0.5],
        resource_decay=[0.3, 0.4],
        leakage=[
            [[0.15, 0.05], [0.08, 0.12]],
            [[0.1, 0.1], [0.04, 0.16]],
        ],
        leakage_fraction=[0.2, 0.2],
    )


class ThermalPerformanceTests(unittest.TestCase):
    def test_performance_matches_direct_formula(self):
        temperature = 293.15
        normalization = 0.25
        activation = 0.7
        optimum = 303.15
        deactivation = 3.5
        reference = 283.15
        boltzmann = 8.617333262145e-5
        expected = normalization * np.exp(
            -(activation / boltzmann) * (1.0 / temperature - 1.0 / reference)
        ) / (
            1.0
            + activation
            / (deactivation - activation)
            * np.exp(
                (deactivation / boltzmann)
                * (1.0 / optimum - 1.0 / temperature)
            )
        )

        np.testing.assert_allclose(
            thermal_performance(
                temperature,
                normalization,
                activation,
                optimum,
                deactivation,
                reference,
            ),
            expected,
        )

    def test_optimum_temperature_is_curve_peak(self):
        optimum = 303.15
        temperatures = np.linspace(optimum - 2.0, optimum + 2.0, 401)
        rates = thermal_performance(
            temperatures,
            1.0,
            0.7,
            optimum,
            3.5,
            283.15,
        )

        self.assertEqual(temperatures[np.argmax(rates)], optimum)

    def test_scaling_factor_is_one_at_reference_temperature(self):
        factors = thermal_scaling_factor(
            283.15,
            activation_energy=[0.7, 0.8],
            optimum_temperature=[303.15, 308.15],
            deactivation_energy=3.5,
            reference_temperature=283.15,
        )

        np.testing.assert_allclose(factors, np.ones(2))

    def test_scaling_factor_avoids_intermediate_overflow(self):
        factor = thermal_scaling_factor(
            1000.0,
            activation_energy=100.0,
            optimum_temperature=303.15,
            deactivation_energy=101.0,
            reference_temperature=283.15,
        )

        self.assertTrue(np.isfinite(factor))
        self.assertGreater(float(factor), 0.0)

    def test_performance_broadcasts_temperature_and_consumer_traits(self):
        rates = thermal_performance(
            np.array([[283.15], [293.15]]),
            normalization=[0.2, 0.3],
            activation_energy=[0.7, 0.8],
            optimum_temperature=[303.15, 308.15],
            deactivation_energy=3.5,
            reference_temperature=283.15,
        )

        self.assertEqual(rates.shape, (2, 2))
        self.assertTrue(np.all(rates > 0.0))

    def test_zero_normalization_produces_zero_performance(self):
        rate = thermal_performance(293.15, 0.0, 0.7, 303.15, 3.5, 283.15)

        np.testing.assert_allclose(rate, 0.0)

    def test_temperature_adjustment_preserves_structure_and_scales_rows(self):
        parameters = base_parameters()
        adjusted = temperature_adjusted_parameters(
            parameters,
            293.15,
            283.15,
            uptake_activation_energy=[0.7, 0.8],
            uptake_optimum_temperature=[303.15, 308.15],
            uptake_deactivation_energy=3.5,
            mortality_activation_energy=[0.5, 0.6],
            mortality_optimum_temperature=[306.15, 311.15],
            mortality_deactivation_energy=3.5,
        )

        uptake_ratio = adjusted.uptake / parameters.uptake
        mortality_ratio = adjusted.mortality / parameters.mortality
        np.testing.assert_allclose(uptake_ratio[:, 0], uptake_ratio[:, 1])
        self.assertTrue(np.all(uptake_ratio > 0.0))
        self.assertTrue(np.all(mortality_ratio > 0.0))
        np.testing.assert_allclose(adjusted.resource_supply, parameters.resource_supply)
        np.testing.assert_allclose(adjusted.resource_decay, parameters.resource_decay)
        np.testing.assert_allclose(adjusted.leakage, parameters.leakage)

    def test_invalid_thermal_inputs_are_rejected(self):
        invalid_calls = (
            lambda: thermal_scaling_factor(0.0, 0.7, 303.15, 3.5, 283.15),
            lambda: thermal_scaling_factor(293.15, 0.7, 303.15, 0.7, 283.15),
            lambda: thermal_scaling_factor(293.15, -0.7, 303.15, 3.5, 283.15),
            lambda: thermal_performance(
                293.15,
                1.0 + 2.0j,
                0.7,
                303.15,
                3.5,
                283.15,
            ),
            lambda: thermal_performance(
                293.15,
                [1.0, 1.0],
                [0.6, 0.7, 0.8],
                303.15,
                3.5,
                283.15,
            ),
            lambda: temperature_adjusted_parameters(
                base_parameters(),
                [293.15, 294.15],
                283.15,
                uptake_activation_energy=0.7,
                uptake_optimum_temperature=303.15,
                uptake_deactivation_energy=3.5,
                mortality_activation_energy=0.5,
                mortality_optimum_temperature=306.15,
                mortality_deactivation_energy=3.5,
            ),
        )

        for invalid_call in invalid_calls:
            with self.subTest(invalid_call=invalid_call):
                with self.assertRaises(ValueError):
                    invalid_call()

    def test_consumer_trait_shape_must_match_parameters(self):
        with self.assertRaisesRegex(ValueError, "broadcast"):
            temperature_adjusted_parameters(
                base_parameters(),
                293.15,
                283.15,
                uptake_activation_energy=[0.6, 0.7, 0.8],
                uptake_optimum_temperature=303.15,
                uptake_deactivation_energy=3.5,
                mortality_activation_energy=0.5,
                mortality_optimum_temperature=306.15,
                mortality_deactivation_energy=3.5,
            )


if __name__ == "__main__":
    unittest.main()
