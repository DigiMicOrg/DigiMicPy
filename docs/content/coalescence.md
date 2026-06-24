# Coalescence

Community coalescence describes what happens when two previously assembled microbiomes are brought into the same environment. In DigiMic this means integrating two parental MiCRM communities to equilibrium, combining their species into one shared resource environment, and measuring which species persist after the merged system settles.

This workflow is useful for questions such as:

- Does one parental community dominate after mixing?
- Do structurally complementary communities coexist more easily?
- Does resource overlap increase competition?
- Do leakage and cross-feeding increase facilitation?
- Is the coalesced community more or less stable than either parent?

The current page defines the intended theory and calculation workflow. The implementation can later be wrapped into helper functions, but all quantities can already be computed from the MiCRM state variables and parameter matrices.

## Parent communities

For two parental communities, write the MiCRM parameters as:

| Quantity | Community 1 | Community 2 |
|---|---:|---:|
| Consumers | $N_1$ | $N_2$ |
| Resources | $M$ | $M$ |
| Uptake matrix | $u^{(1)}$ | $u^{(2)}$ |
| Leakage tensor | $l^{(1)}$ | $l^{(2)}$ |
| Mortality vector | $m^{(1)}$ | $m^{(2)}$ |
| Equilibrium biomass | $\hat C^{(1)}$ | $\hat C^{(2)}$ |
| Equilibrium resources | $\hat R^{(1)}$ | $\hat R^{(2)}$ |

If both communities use the same resource pool, the coalesced system is formed by stacking species-level parameters:

$$
u^{(3)} =
\begin{bmatrix}
u^{(1)} \\
u^{(2)}
\end{bmatrix},
\qquad
l^{(3)} =
\begin{bmatrix}
l^{(1)} \\
l^{(2)}
\end{bmatrix},
\qquad
m^{(3)} =
\begin{bmatrix}
m^{(1)} \\
m^{(2)}
\end{bmatrix}.
$$

The initial biomass of the merged community is usually the concatenated parental equilibrium:

$$
C^{(3)}(0) =
\begin{bmatrix}
\hat C^{(1)} \\
\hat C^{(2)}
\end{bmatrix}.
$$

The initial resource vector can be a standard reference environment $R^0$, the resource state of one parent, an average of both parents, or a user-specified post-mixing environment. The choice should be reported because it changes the invasion conditions experienced immediately after mixing.

## Partially overlapping resources

When parental communities are assembled on different but partially overlapping resource sets, first build a union resource list. Each parental uptake matrix and leakage tensor is then embedded into that common resource space.

For species from community 1, set uptake to zero for resources not available in community 1 before coalescence. Do the same for community 2. This keeps parental physiology intact while allowing the merged community to experience a broader resource environment.

The resource-overlap ratio is:

$$
\Omega_R = \frac{|R^{(1)} \cap R^{(2)}|}{|R^{(1)} \cup R^{(2)}|}.
$$

Higher $\Omega_R$ usually means stronger direct competition, while lower overlap can create more room for complementarity if leaked resources from one community are usable by the other.

## Competition and facilitation metrics

A simple community-level competition metric is the mean cosine similarity between uptake rows:

$$
\mathrm{competition}
= \frac{2}{N(N-1)}
\sum_{i<j}
\frac{u_i \cdot u_j}{||u_i||\,||u_j||}.
$$

This is high when species consume similar resources.

An effective leakage profile for species $i$ is:

$$
L^{\mathrm{eff}}_{i\beta}
= \sum_\alpha u_{i\alpha} l_{i\alpha\beta}.
$$

This records which resources are produced as by-products after accounting for what the species consumes. A practical facilitation score compares the resources leaked by species $i$ with the uptake preferences of the other species:

$$
\mathrm{facilitation}_i
= \sum_\beta L^{\mathrm{eff}}_{i\beta}
\left(\frac{1}{N-1}\sum_{j \ne i} u_{j\beta}\right).
$$

High values indicate that a species tends to leak resources that other species can use.

## Coalescence outcomes

After integrating the merged MiCRM to equilibrium, calculate:

| Metric | Meaning |
|---|---|
| Survivors | Species with $C_i^* > \theta$, where $\theta$ is a small threshold such as `1e-5` |
| Origin biomass | Total biomass of species originating from each parent |
| Dominant parent | Parent with larger biomass contribution in the coalesced equilibrium |
| Similarity to parents | Bray-Curtis or other abundance similarity between the merged state and each parent |
| Resource drawdown | Total or resource-wise residual resource abundance |
| Stability | Leading real eigenvalue of the MiCRM or effective GLV Jacobian |
| Feasibility | Whether all surviving species have positive equilibrium biomass |

The basic simulation workflow is:

```python
# 1. Assemble two parental communities
sol1 = solve_micrm(params1)
sol2 = solve_micrm(params2)

# 2. Build the merged parameter set
u3 = np.vstack([u1, u2])
l3 = np.vstack([l1, l2])
m3 = np.concatenate([m1, m2])
C0_3 = np.concatenate([C1_eq, C2_eq])

# 3. Integrate the coalesced community
sol3 = solve_micrm(u3, l3, m3, C0_3, R0_mix)

# 4. Measure dominance, survivors, resources, stability, and CUE
```

## Interpretation

Coalescence is not only a species-mixing experiment. It is also a test of whether two resource-processing systems can coexist. Communities with very similar uptake profiles tend to compete strongly. Communities with complementary uptake and leakage can create stabilising cross-feeding if the by-products of one group match the demands of another.

The reference implementation used for this page is the coalescence workflow in [EcoEngLab/coalescence_robustness](https://github.com/EcoEngLab/coalescence_robustness), especially its community stacking, competition-facilitation metrics, effective GLV reduction, and post-merge stability calculations.
