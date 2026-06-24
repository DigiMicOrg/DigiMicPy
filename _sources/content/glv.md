# From MiCRM to GLV

The Microbial Consumer-Resource Model (MiCRM) tracks species and resources explicitly. For some analyses, it is helpful to reduce the model to species-only dynamics so that apparent interactions between species can be inspected directly. This reduced model is an effective Generalized Lotka-Volterra (effective GLV or eGLV) model.

The eGLV reduction is not a replacement for MiCRM. It is a local summary of how species affect one another through the resource environment near a specified equilibrium.

## GLV form

The species-only model is:

$$
\frac{dC_i}{dt}
= C_i \left(r_i + \sum_j \alpha_{ij} C_j \right),
$$

where:

| Symbol | Meaning |
|---|---|
| $C_i$ | Biomass of consumer $i$ |
| $r_i$ | Effective intrinsic growth rate |
| $\alpha_{ij}$ | Effective effect of species $j$ on species $i$ |

Negative $\alpha_{ij}$ means species $j$ suppresses species $i$ near the reference environment. Positive $\alpha_{ij}$ means species $j$ facilitates species $i$, usually by modifying the resource pool in a beneficial way.

## MiCRM growth term

For MiCRM with leakage, define the retained fraction:

$$
\eta_{i\alpha} = 1 - \sum_\beta l_{i\alpha\beta}.
$$

The per-capita growth contribution from resources is:

$$
g_i(R) = \sum_\alpha u_{i\alpha}\eta_{i\alpha}R_\alpha.
$$

The consumer equation can be written compactly as:

$$
\frac{dC_i}{dt}
= C_i \left(g_i(R) - m_i\right).
$$

Resources mediate all apparent species interactions. If species $j$ changes resources, and species $i$ depends on those resources, then species $j$ has an effective interaction with species $i$.

## Eliminating resources locally

Let $(\hat C, \hat R)$ be a MiCRM equilibrium. The resource equation can be written as:

$$
\frac{dR}{dt} = F(R, C).
$$

At equilibrium, $F(\hat R, \hat C)=0$. Locally, the implicit function theorem gives the resource response to a small change in consumer biomass:

$$
\frac{\partial \hat R}{\partial C}
= -\left(\frac{\partial F}{\partial R}\right)^{-1}
\frac{\partial F}{\partial C}.
$$

The effective interaction matrix is then:

$$
\alpha_{ij}
= \sum_\alpha
u_{i\alpha}\eta_{i\alpha}
\frac{\partial \hat R_\alpha}{\partial C_j}.
$$

After $\alpha$ is known, choose $r$ so that the eGLV has the same equilibrium as the MiCRM:

$$
r_i
= g_i(\hat R) - m_i - \sum_j \alpha_{ij}\hat C_j.
$$

This is the core calculation used in DigiMic-style eGLV analysis.

## Simple no-leakage example

For a classical consumer-resource model with logistic resources:

$$
\frac{dR_\alpha}{dt}
= r_\alpha R_\alpha\left(1-\frac{R_\alpha}{K_\alpha}\right)
- \sum_j a_{j\alpha}R_\alpha C_j,
$$

$$
\frac{dC_i}{dt}
= C_i\left(\sum_\alpha w_{i\alpha}R_\alpha - m_i\right),
$$

the effective parameters near the resource carrying capacities have the simple form:

$$
r_i^{\mathrm{eff}}
= \sum_\alpha w_{i\alpha}K_\alpha - m_i,
$$

$$
\alpha_{ij}
= -\sum_\alpha
\frac{w_{i\alpha}a_{j\alpha}K_\alpha}{r_\alpha}.
$$

This simple expression is useful for intuition. In MiCRM, leakage and resource recycling make the response matrix more complex, so the equilibrium-based calculation above is preferred.

## Computational workflow

1. Simulate MiCRM until the derivative norm is small.
2. Extract equilibrium biomasses $\hat C$ and resources $\hat R$.
3. Build the resource Jacobian $\partial F/\partial R$.
4. Build the consumer-to-resource response matrix $\partial F/\partial C$.
5. Solve the linear system for $\partial \hat R/\partial C$.
6. Calculate $\alpha$ and $r$.
7. Keep the surviving subsystem if the analysis should focus only on extant species.
8. Compare MiCRM and eGLV trajectories after a small perturbation.

```python
alpha, r = calculate_elv_params(C_eq, R_eq, N, M, u, l, m, rho, omega, lambda_vec)

def dCdt_elv(t, C):
    return C * (r + alpha @ C)
```

## How to read the interaction matrix

The rows of $\alpha$ are affected species. The columns are the species causing the effect.

| Pattern | Interpretation |
|---|---|
| $\alpha_{ij}<0$ and $\alpha_{ji}<0$ | Net competition |
| $\alpha_{ij}>0$ and $\alpha_{ji}>0$ | Net mutual facilitation |
| Opposite signs | Exploitation-like or asymmetric facilitation |
| Weak off-diagonal values | Weak apparent coupling through resources |

The reference calculations for this page follow the effective GLV derivations in the MQB interactions notebook and the equilibrium-based implementations in [EcoEngLab/coalescence_robustness](https://github.com/EcoEngLab/coalescence_robustness) and [DaniDuan/temp_interactions](https://github.com/DaniDuan/temp_interactions).
