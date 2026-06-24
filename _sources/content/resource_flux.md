# Resource processing flux

Resource processing fluxes describe how carbon or resource material moves through the MiCRM system. They are useful for connecting mechanistic simulations to measured functions such as respiration, carbon retention, resource depletion, and community-level CUE.

## Flux definitions

For species $i$ and resource $\alpha$, the gross uptake flux is:

$$
F^{\mathrm{uptake}}_{i\alpha}
= C_i u_{i\alpha}R_\alpha.
$$

The retained fraction is:

$$
\eta_{i\alpha}
= 1-\sum_\beta l_{i\alpha\beta}.
$$

The retained biomass-producing flux is:

$$
F^{\mathrm{retained}}_{i\alpha}
= C_i u_{i\alpha}R_\alpha\eta_{i\alpha}.
$$

The leakage flux from consumed resource $\alpha$ into resource $\beta$ is:

$$
F^{\mathrm{leak}}_{i\alpha\beta}
= C_i u_{i\alpha}R_\alpha l_{i\alpha\beta}.
$$

Maintenance loss for species $i$ is:

$$
F^{\mathrm{maintenance}}_i = m_i C_i.
$$

These terms separate the carbon that is consumed, retained, recycled, and lost from population growth.

## Resource balance

For leaching-style resources, the total flux balance for resource $\alpha$ is:

$$
\frac{dR_\alpha}{dt}
=
\rho_\alpha
- \omega_\alpha R_\alpha
- \sum_i F^{\mathrm{uptake}}_{i\alpha}
+ \sum_i\sum_\beta F^{\mathrm{leak}}_{i\beta\alpha}.
$$

At equilibrium, inflows and outflows balance:

$$
0
=
\rho_\alpha
- \omega_\alpha R_\alpha^*
- \sum_i C_i^*u_{i\alpha}R_\alpha^*
+ \sum_i\sum_\beta C_i^*u_{i\beta}R_\beta^*l_{i\beta\alpha}.
$$

This balance is useful for identifying which resources are depleted by direct consumption and which are replenished by cross-feeding.

## Species processing rate

The total gross processing rate of species $i$ in a given resource state $R$ is:

$$
U_i(R) = \sum_\alpha u_{i\alpha}R_\alpha.
$$

The biomass-scaled processing flux is:

$$
C_i U_i(R).
$$

For standardized comparisons, use a reference resource vector $R^0$:

$$
U_i^0 = U_i(R^0).
$$

This is the weighting term used in flux-weighted community CUE.

## Community processing flux

Total community gross uptake is:

$$
F^{\mathrm{uptake}}_{\mathrm{comm}}
= \sum_i\sum_\alpha C_i u_{i\alpha}R_\alpha.
$$

Total retained flux is:

$$
F^{\mathrm{retained}}_{\mathrm{comm}}
= \sum_i\sum_\alpha C_i u_{i\alpha}R_\alpha\eta_{i\alpha}.
$$

A direct flux-based CUE at a given resource state is:

$$
E_{\mathrm{direct}}
=
\frac{F^{\mathrm{retained}}_{\mathrm{comm}}}
{F^{\mathrm{uptake}}_{\mathrm{comm}}}.
$$

For prediction and comparison across communities, prefer the standardized reference-resource version:

$$
E_{\mathrm{flux}}
=
\frac{
\sum_i C_i^*\sum_\alpha u_{i\alpha}R_\alpha^0\eta_{i\alpha}
}{
\sum_i C_i^*\sum_\alpha u_{i\alpha}R_\alpha^0
}.
$$

## What to report

For each simulation or coalescence experiment, useful resource-flux summaries include:

| Quantity | Interpretation |
|---|---|
| Total uptake | How much resource material the community processes |
| Total retained flux | Carbon routed into biomass before maintenance losses |
| Total leakage flux | Strength of resource recycling and cross-feeding |
| Net growth flux | Retained flux minus maintenance loss |
| Resource drawdown | Residual resources after community assembly |
| Flux-weighted CUE | Efficiency of dominant carbon-processing pathways |

## Calculation workflow

```python
uptake_flux = C[:, None] * u * R[None, :]
eta = 1.0 - l.sum(axis=2)
retained_flux = uptake_flux * eta
leakage_flux = C[:, None, None] * u[:, :, None] * R[None, :, None] * l

community_uptake = uptake_flux.sum()
community_retained = retained_flux.sum()
community_leakage = leakage_flux.sum()
direct_cue = community_retained / (community_uptake + 1e-12)
```

When using equilibrium output, replace `C` and `R` with `C_eq` and `R_eq`. When comparing communities as predictors of coalescence outcome, use `R0` for standardized potential flux and use `C_eq` only as the assembled biomass weight.

This page is based on the attached flux-weighted CUE notes and the resource-processing calculations used in [EcoEngLab/CUE_predict_coalescence](https://github.com/EcoEngLab/CUE_predict_coalescence).
