---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Temperature control

DigiMicPy implements fixed-temperature scaling of consumer uptake and mortality.
An existing `MiCRMParameters` object is interpreted as the parameter set at a
reference temperature; thermal response curves then produce a new validated
parameter object for another fixed temperature.

```{note}
Time-varying temperature $T(t)$, thermal scaling of leakage or resource supply,
and random generation of thermal traits are not yet package APIs. They require a
custom non-autonomous derivative function and explicit trait assumptions.
```

## Thermal performance curve

The implemented curve is the modified Sharpe-Schoolfield parameterisation with
peak temperature explicit {cite}`smith2022thermal`:

$$
B(T)=B_0
\frac{
\exp\left[-\frac{E}{k_B}\left(\frac{1}{T}-\frac{1}{T_{\mathrm{ref}}}\right)\right]
}{
1 + \frac{E}{E_D-E}
\exp\left[\frac{E_D}{k_B}\left(\frac{1}{T_{\mathrm{pk}}}-\frac{1}{T}\right)\right]
}.
$$

| Symbol | Meaning | Required units with the default $k_B$ |
|---|---|---|
| $T$ | Evaluation temperature | kelvin |
| $T_{\mathrm{ref}}$ | Reference temperature | kelvin |
| $T_{\mathrm{pk}}$ | Temperature of peak performance | kelvin |
| $B_0$ | Nonnegative normalisation | trait units |
| $E$ | Activation energy | electronvolts |
| $E_D$ | Deactivation energy, greater than $E$ | electronvolts |

`thermal_performance` evaluates this equation directly. It uses log-space
arithmetic to avoid avoidable intermediate overflow and follows NumPy
broadcasting rules.

```{code-cell} ipython3
import numpy as np
from digimicpy import thermal_performance

temperatures = np.linspace(278.15, 318.15, 161)
rates = thermal_performance(
    temperatures,
    normalization=1.0,
    activation_energy=0.7,
    optimum_temperature=303.15,
    deactivation_energy=3.5,
    reference_temperature=283.15,
)
temperatures[np.argmax(rates)] - 273.15
```

## Scaling reference parameters

`temperature_adjusted_parameters` scales each uptake row and mortality entry by
$B(T)/B(T_{\mathrm{ref}})$. This makes the supplied parameter object exactly the
reference-temperature model, independent of the interpretation of $B_0$.

```{code-cell} ipython3
import numpy as np
from digimicpy import MiCRMParameters, temperature_adjusted_parameters

reference = MiCRMParameters(
    uptake=[[0.8, 0.2], [0.3, 0.7]],
    mortality=[0.15, 0.2],
    resource_supply=[0.6, 0.4],
    resource_decay=[0.4, 0.4],
    leakage=[
        [[0.08, 0.02], [0.03, 0.07]],
        [[0.05, 0.05], [0.02, 0.08]],
    ],
    leakage_fraction=[0.1, 0.1],
)

warm = temperature_adjusted_parameters(
    reference,
    temperature=298.15,
    reference_temperature=283.15,
    uptake_activation_energy=[0.7, 0.8],
    uptake_optimum_temperature=[303.15, 308.15],
    uptake_deactivation_energy=3.5,
    mortality_activation_energy=[0.5, 0.6],
    mortality_optimum_temperature=[306.15, 311.15],
    mortality_deactivation_energy=3.5,
)

warm.uptake, warm.mortality
```

The structural uptake distribution within each consumer row, leakage tensor,
resource supply, and resource decay are copied unchanged.

## Temperature comparisons

To compare fixed temperatures, generate the structural community once, create a
new parameter object for each temperature, and solve each from the same initial
state. This isolates acute physiological scaling from stochastic differences in
community structure.

For patches at different fixed temperatures, create one adjusted parameter set
per patch and pass them to `solve_spatial_micrm`; see {doc}`spatial`.

## Practical checks

- Convert Celsius to kelvin before calling the thermal API.
- Keep energies consistent with the selected Boltzmann constant.
- Require $0<E<E_D$ and use thermal optima supported by data.
- Avoid extrapolation beyond the fitted temperature range.
- Separate acute physiological scaling from adaptation or species sorting.
- Report which parameter families respond to temperature.

## References

```{bibliography}
:filter: docname in docnames
```
