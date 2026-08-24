# Implemented model formulation

This page records the equations implemented by DigiMicPy and maps their symbols
to the public parameter object. For platform-level motivation and shared
scientific workflows, see the
[DigiMic project overview](https://digimicorg.github.io/about/) and
[workflow documentation](https://digimicorg.github.io/workflows/).

```{figure} figures/MiCRM.png
:name: micrm-framework
:alt: MiCRM framework
:width: 70%

Microbial Consumer-Resource Model framework.
```

## Microbial Consumer-Resource Model

For $N$ consumers and $M$ resources, DigiMicPy implements:

$$
\frac{dC_i}{dt}
= C_i\left[\sum_{\alpha=1}^{M} u_{i\alpha}R_\alpha
\left(1-\sum_{\beta=1}^{M}l_{i\alpha\beta}\right)-m_i\right],
$$

$$
\frac{dR_\alpha}{dt}
= \rho_\alpha - \omega_\alpha R_\alpha
- \sum_{i=1}^{N}C_i u_{i\alpha}R_\alpha
+ \sum_{i=1}^{N}\sum_{\beta=1}^{M}
C_i u_{i\beta}R_\beta l_{i\beta\alpha}.
$$

Consumer biomass increases through retained resource uptake and decreases
through maintenance or mortality. Resource dynamics combine external supply,
abiotic loss, direct consumption, and replenishment from metabolic by-products.

| Symbol | Meaning | `MiCRMParameters` field |
|---|---|---|
| $C_i$ | Consumer biomass | first `N` state entries |
| $R_\alpha$ | Resource abundance | final `M` state entries |
| $u_{i\alpha}$ | Uptake rate or preference | `uptake` |
| $m_i$ | Maintenance or mortality | `mortality` |
| $\rho_\alpha$ | External resource input | `resource_supply` |
| $\omega_\alpha$ | Resource loss or washout | `resource_decay` |
| $l_{i\alpha\beta}$ | Fraction leaked from consumed $\alpha$ into $\beta$ | `leakage` |
| $\lambda_{i\alpha}$ | Total leaked fraction for an uptake channel | `leakage_fraction` |

The leakage tensor has shape `(N, M, M)`. Its row sums must equal the declared
`leakage_fraction`; `MiCRMParameters` checks this invariant during construction.

## Modular resource structure

The package includes reproducible generators for synthetic modular communities.
`modular_uptake` strengthens matched consumer-resource blocks and normalises
each consumer row. `generate_l_tensor` creates one by-product matrix per
consumer and normalises each consumed-resource row to `total_leakage`.

| Argument | Package interpretation |
|---|---|
| `n_modules` | Number of matched consumer-resource modules |
| `specialization_ratio` | Strength of favoured entries relative to background entries |
| `total_leakage` | Fraction of consumed material allocated to leaked resources |
| `rng` | Explicit `numpy.random.Generator` controlling reproducibility |

Effective GLV reduction is not part of the current DigiMicPy API. Its
scientific definition and interpretation are maintained in the
[platform workflow](https://digimicorg.github.io/workflows/effective-glv/).
