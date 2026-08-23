# Analysis

This section focuses on model interpretation after a MiCRM simulation has been run. The main tasks are to reduce MiCRM to an effective species-interaction model and to analyse stability at both the consumer-resource and species-only levels.

```{important}
The current package exposes trajectories and the pure MiCRM derivative needed
for these calculations, but it does not yet provide eGLV conversion, Jacobian,
stability, reactivity, or feasibility helper functions. The pages in this
section are mathematical and manual-computation guidance.
```

The recommended order is:

1. Simulate MiCRM and confirm the system has reached a numerical equilibrium.
2. Convert the equilibrium to an effective GLV model if species-level interactions are needed.
3. Analyse full MiCRM stability using the consumer-resource Jacobian.
4. Analyse effective GLV stability using the surviving species interaction matrix.

## Perturbation experiments

A direct resilience experiment can reuse the final state as a new initial
condition, modify selected entries, and integrate again:

```python
perturbed = result.y[:, -1].copy()
perturbed[0] *= 0.5  # reduce the first consumer
perturbed[parameters.n_consumers] += 0.5  # pulse the first resource

post = solve_micrm(
    parameters,
    perturbed,
    (0.0, 25.0),
    t_eval=np.linspace(0.0, 25.0, 150),
)
```

Compare recovery time or distance from the pre-perturbation state only after
checking both solver results and confirming that the reference endpoint was
close to equilibrium.
