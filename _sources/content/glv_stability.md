# GLV stability analysis

GLV stability analysis uses the effective species-only model derived from MiCRM:

$$
\frac{dC_i}{dt}
= C_i\left(r_i+\sum_j\alpha_{ij}C_j\right).
$$

This analysis is useful when you want to interpret stability in terms of species interactions rather than explicit resource dynamics.

## Species-only Jacobian

For the effective GLV, the full species Jacobian is:

$$
J^{\mathrm{GLV}}_{ij}
= \delta_{ij}\left(r_i+\sum_k\alpha_{ik}C_k^*\right)
+ C_i^*\alpha_{ij}.
$$

For surviving species at equilibrium, the per-capita term is zero, so:

$$
J^{\mathrm{GLV}} = \mathrm{diag}(C^*)\alpha.
$$

Use only species with $C_i^*>\theta$ when analysing the surviving subsystem. A common threshold is `1e-5`.

## Local stability

The surviving effective GLV equilibrium is locally stable when:

$$
\max_k \mathrm{Re}(\lambda_k(J^{\mathrm{GLV}})) < 0.
$$

The sign pattern and magnitude of $\alpha$ help explain why stability changes. Strong negative self-effects usually stabilise the system, while strong positive feedbacks can reduce stability.

## Feasibility

For a GLV system, a feasible equilibrium has positive biomass for all included species:

$$
C^* = -\alpha^{-1}r,
\qquad
C_i^*>0 \ \mathrm{for\ all}\ i.
$$

When $r$ is uncertain, feasibility can be treated probabilistically by asking how often random growth-rate vectors imply a positive equilibrium. One practical approach uses:

$$
\Sigma = \alpha^{-1}(\alpha^{-1})^T
$$

and estimates the positive orthant probability of a multivariate normal distribution.

## Workflow

1. Simulate MiCRM to equilibrium.
2. Calculate effective GLV parameters $r$ and $\alpha$.
3. Keep the surviving species subsystem.
4. Build $J^{\mathrm{GLV}}=\mathrm{diag}(C^*)\alpha$.
5. Compute the leading real eigenvalue.
6. Compare the result with the full MiCRM stability analysis.
