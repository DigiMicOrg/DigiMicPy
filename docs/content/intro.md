# DigiMicPy

DigiMicPy is the Python implementation of the DigiMic microbial
consumer-resource modelling framework. It provides a validated model core,
reproducible parameter generators, fixed-temperature trait scaling, and
conservative coupling between spatial patches.

For the project vision, shared scientific workflows, implementation comparison,
training resources, team, and support, visit the
[DigiMic platform site](https://digimicorg.github.io/).

## Implemented and tested

| Area | Package support |
|---|---|
| Core MiCRM parameters, right-hand side, and solver | Public API |
| Modular uptake and leakage generators | Public API |
| Fixed-temperature uptake and mortality scaling | Public API |
| Conservative undirected spatial patches | Public API |
| Coalescence, CUE, and resource-flux calculations | Explicit NumPy recipes |
| eGLV conversion and stability helpers | Not implemented |

Start with {doc}`useinfo` for an executable simulation, {doc}`theo` for the
equations implemented by the package, and {doc}`api` for the supported public
interface.

## Scope of this book

This documentation records how DigiMicPy behaves: installation, array shapes,
state ordering, validation, solver use, implemented extensions, and recipes
that operate on package outputs. Shared scientific definitions and
interpretation live in the
[platform workflows](https://digimicorg.github.io/workflows/) and are linked from the
relevant recipe.

```{important}
Package helpers are part of the supported interface. Recipes are transparent
calculations using public outputs but are not stable helper APIs. Proposed
capabilities are documented only on the platform site until implemented and
tested here.
```
