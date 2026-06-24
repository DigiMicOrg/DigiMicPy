# Carbon use efficiency

Carbon use efficiency (CUE) measures how much consumed carbon is retained in biomass rather than leaked, respired, or otherwise lost from growth. In DigiMic, CUE can be calculated at the species level and then summarised at the community level.

This page distinguishes biomass-weighted and flux-weighted community CUE. The distinction matters because a species can be abundant but process little resource, or rare but account for a large share of carbon flux.

## Species-level CUE

For species $i$ consuming resource $\alpha$, define the retained fraction:

$$
\eta_{i\alpha}
= 1 - \sum_\beta l_{i\alpha\beta}.
$$

Under a standardized reference resource environment $R^0$, the potential uptake flux of species $i$ is:

$$
U_i^0 = \sum_\alpha u_{i\alpha}R_\alpha^0.
$$

The potential retained carbon flux is:

$$
G_i^0 = \sum_\alpha u_{i\alpha}\eta_{i\alpha}R_\alpha^0.
$$

A gross species CUE is:

$$
\epsilon_i^{\mathrm{gross}}
= \frac{G_i^0}{U_i^0}.
$$

If maintenance cost is treated as a carbon loss, use a net CUE:

$$
\epsilon_i^{\mathrm{net}}
= \frac{G_i^0 - m_i}{U_i^0}.
$$

The CUE coalescence reference code uses this net form because maintenance reduces the carbon available for population growth.

## Biomass-weighted community CUE

The simplest community summary is the biomass-weighted mean CUE of the surviving assemblage:

$$
E_{\mathrm{biomass}}
= \frac{\sum_i C_i^*\epsilon_i}{\sum_i C_i^*}.
$$

This is easy to interpret: it describes the CUE of the taxa that are abundant after assembly. It is a useful community trait summary, but it is not always the best measure of carbon-processing efficiency.

## Flux-weighted community CUE

A more mechanistic community-level CUE weights each species by its biomass and potential uptake flux:

$$
E_{\mathrm{flux}}
= \frac{\sum_i C_i^*U_i^0\epsilon_i}{\sum_i C_i^*U_i^0}.
$$

Equivalently, using resource-specific retained fractions:

$$
E_{\mathrm{flux}}
=
\frac{
\sum_i C_i^*\sum_\alpha u_{i\alpha}R_\alpha^0\eta_{i\alpha}
}{
\sum_i C_i^*\sum_\alpha u_{i\alpha}R_\alpha^0
}.
$$

This quantity is closer to "retained carbon per unit consumed carbon" for the assembled community under a standardized resource environment.

## Why use a standardized resource environment?

Using $R^0$ avoids circularity. If CUE were calculated using the post-assembly resource state $R^*$, then the metric would partly depend on the outcome it is trying to explain. A standard reference environment makes species CUE an intrinsic trait-like quantity, while biomass and flux weights capture how that trait is expressed in the assembled community.

Common choices for $R^0$ include:

- the initial resource vector used for assembly;
- the chemostat carrying-capacity vector;
- the mean resource environment across scenarios;
- a user-defined experimental medium.

The same $R^0$ should be used when comparing communities.

## CUE and coalescence

CUE is useful in coalescence analysis because it links community composition to resource depletion and invasion thresholds:

$$
E_{\mathrm{flux}}
\rightarrow R^*
\rightarrow \mathrm{invasion\ growth}
\rightarrow \mathrm{coalescence\ outcome}.
$$

Higher CUE communities can retain more consumed carbon in biomass and may draw down limiting resources more deeply. This can make invasion harder for low-efficiency competitors, but the outcome also depends on leakage structure, facilitation, and resource overlap.

## Calculation workflow

1. Choose a reference resource vector $R^0$.
2. Compute $\eta_{i\alpha}$ from the leakage tensor.
3. Compute $U_i^0$, $G_i^0$, and species CUE $\epsilon_i$.
4. Simulate MiCRM to get equilibrium biomasses $C_i^*$.
5. Filter extinct species with a threshold such as `1e-5`.
6. Compute both $E_{\mathrm{biomass}}$ and $E_{\mathrm{flux}}$.
7. Compare CUE with dominance, resource depletion, facilitation, and post-coalescence stability.

```python
eta = 1.0 - l.sum(axis=2)
Ui0 = np.sum(u * R0[None, :], axis=1)
Gi0 = np.sum(u * eta * R0[None, :], axis=1)
species_cue = (Gi0 - m) / (Ui0 + 1e-12)

community_cue_biomass = np.average(species_cue[survivors], weights=C_eq[survivors])
community_cue_flux = np.average(
    species_cue[survivors],
    weights=C_eq[survivors] * Ui0[survivors],
)
```

The species and community CUE definitions here are based on [EcoEngLab/CUE_predict_coalescence](https://github.com/EcoEngLab/CUE_predict_coalescence) and the attached notes on flux-weighted community CUE.
