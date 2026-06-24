# MiCRM stability analysis

MiCRM stability analysis uses the full consumer-resource Jacobian. It asks whether a small perturbation in either species biomass or resource abundance will decay back to the equilibrium.

## Fixed point

Let the equilibrium be:

$$
x^* =
\begin{bmatrix}
C^* \\
R^*
\end{bmatrix}.
$$

Before computing stability, check that the derivative is small:

$$
\max_k \left|\frac{dx_k}{dt}\right| < \epsilon.
$$

A typical tolerance is `1e-5`.

## Full Jacobian

With retained fraction:

$$
\eta_{i\alpha}=1-\sum_\beta l_{i\alpha\beta},
$$

the MiCRM Jacobian has four blocks:

$$
J =
\begin{bmatrix}
J_{CC} & J_{CR} \\
J_{RC} & J_{RR}
\end{bmatrix}.
$$

For consumer rows:

$$
(J_{CC})_{ij}
= \delta_{ij}\left(\sum_\alpha u_{i\alpha}\eta_{i\alpha}R_\alpha^* - m_i\right),
$$

$$
(J_{CR})_{i\alpha}
= C_i^* u_{i\alpha}\eta_{i\alpha}.
$$

For resource rows with leaching-style abiotic loss $-\omega_\alpha R_\alpha$:

$$
(J_{RC})_{\alpha i}
= -u_{i\alpha}R_\alpha^*
+ \sum_\beta u_{i\beta}R_\beta^*l_{i\beta\alpha},
$$

$$
(J_{RR})_{\alpha\gamma}
= -\omega_\alpha\delta_{\alpha\gamma}
- \sum_i C_i^*u_{i\alpha}\delta_{\alpha\gamma}
+ \sum_i C_i^*u_{i\gamma}l_{i\gamma\alpha}.
$$

If the resource supply model is constant, chemostat, or self-renewing, replace the first term in $J_{RR}$ with the corresponding derivative of the resource supply function.

## Local stability and reactivity

Compute the eigenvalues of $J$. The equilibrium is locally stable when:

$$
\max_k \mathrm{Re}(\lambda_k(J)) < 0.
$$

The leading real eigenvalue gives a continuous stability metric. More negative values indicate faster local recovery, while values close to zero indicate slow return.

A stable system can still amplify perturbations transiently. Reactivity is measured from:

$$
H = \frac{J+J^T}{2}.
$$

If the largest eigenvalue of $H$ is positive, some perturbation directions grow initially even if the equilibrium is asymptotically stable.

## Workflow

1. Integrate MiCRM to equilibrium.
2. Build the full consumer-resource Jacobian.
3. Compute the leading real eigenvalue.
4. Compute reactivity if transient amplification matters.
5. Repeat across parameter scenarios, coalescence pairs, or temperature regimes.
