# Temperature control

Temperature can be introduced into DigiMic by making physiological traits functions of temperature. The core idea is to replace fixed parameters such as uptake, maintenance, leakage, resource supply, or resource renewal with temperature-dependent functions evaluated during simulation.

This matters because temperature changes both single-strain metabolic traits and the effective interactions that emerge through resources.

## Temperature-dependent traits

A common trait response combines a Boltzmann-Arrhenius increase at low to moderate temperature with high-temperature deactivation:

$$
B(T)
=
B_0
\frac{
\exp\left[-\frac{E}{k_B}\left(\frac{1}{T}-\frac{1}{T_{\mathrm{ref}}}\right)\right]
}{
1 + \frac{E}{E_D-E}
\exp\left[\frac{E_D}{k_B}\left(\frac{1}{T_p}-\frac{1}{T}\right)\right]
}.
$$

where:

| Symbol | Meaning |
|---|---|
| $T$ | Temperature in Kelvin |
| $T_{\mathrm{ref}}$ | Reference temperature |
| $k_B$ | Boltzmann constant |
| $B_0$ | Trait value at the reference scale |
| $E$ | Activation energy |
| $E_D$ | High-temperature deactivation energy |
| $T_p$ | Peak temperature |

Different traits can have different $(B_0, E, T_p)$ values. For example, uptake and maintenance can be assigned correlated thermal parameters so that species differ in both growth potential and thermal sensitivity.

## Where temperature enters MiCRM

Temperature can control:

| Parameter | Possible temperature-dependent form |
|---|---|
| Uptake | $u_{i\alpha}(T)$ or a scalar multiplier on each row of $u$ |
| Maintenance | $m_i(T)$ |
| Leakage or CUE | $l_{i\alpha\beta}(T)$ or $\eta_{i\alpha}(T)$ |
| Resource supply | $\rho_\alpha(T)$ |
| Resource loss | $\omega_\alpha(T)$ |
| Self-renewing resources | $r_\alpha(T)$ and $K_\alpha(T)$ |

The simplest implementation keeps the structure of $u$ and $l$ fixed but multiplies uptake and maintenance by thermal performance curves. More detailed implementations can allow leakage and resource supply to change with temperature too.

## Temperature regimes

The temperature input can be:

| Regime | Example |
|---|---|
| Fixed | $T(t)=T_0$ |
| Step shift | $T(t)=T_1$ before a pulse and $T_2$ after |
| Sinusoidal | $T(t)=\bar T + A\sin(2\pi t/P)$ |
| Heatwave | Short interval with elevated $T$ |
| Empirical series | Interpolated field or incubator measurements |

For time-varying temperature, the ODE becomes non-autonomous because parameters are recalculated at each time point.

## Simulation workflow

1. Define a temperature function `T_of_t(t)`.
2. Define thermal trait functions for uptake, maintenance, and any resource parameters.
3. Inside the ODE function, evaluate `T = T_of_t(t)`.
4. Recalculate the temperature-dependent parameter values.
5. Integrate with a solver suitable for potentially stiff dynamics, such as `BDF`.
6. Record both the biological state and the temperature series.
7. Compare community composition, CUE, stability, and effective GLV interactions across temperature regimes.

```python
def thermal_trait(T, B0, E, T_ref, E_D, T_peak, kB=8.62e-5):
    arrhenius = np.exp(-(E / kB) * ((1.0 / T) - (1.0 / T_ref)))
    deactivation = 1.0 + (E / (E_D - E)) * np.exp((E_D / kB) * ((1.0 / T_peak) - (1.0 / T)))
    return B0 * arrhenius / deactivation

def dCdt_Rdt_temperature(t, y, params):
    T = params["T_of_t"](t)
    uptake_scale = thermal_trait(T, **params["uptake_temp"])
    maintenance = thermal_trait(T, **params["maintenance_temp"])
    # Use uptake_scale and maintenance inside the MiCRM equations.
```

## Practical checks

- Use Kelvin internally, even if inputs are displayed in Celsius.
- Check that trait curves remain biologically meaningful over the simulated temperature range.
- Avoid extrapolating far beyond the fitted thermal range.
- Compare fixed-temperature equilibria before running fluctuating regimes.
- Recompute effective GLV parameters separately at each equilibrium or time window.
- Report whether temperature changes only physiology or also resource supply and loss.


