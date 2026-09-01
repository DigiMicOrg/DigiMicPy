# DigiMicPy

DigiMicPy is the Python package for constructing and simulating microbial
consumer-resource models. The current public API provides validated MiCRM
parameters, reproducible modular uptake and leakage generators, a pure model
right-hand side, fixed-temperature trait scaling, conservative spatial
transport, and SciPy integration wrappers.

For the project vision, package directory, shared scientific workflows,
training resources, team, funding, and general support, visit the
[DigiMic platform site](https://digimicorg.github.io/).

## Package documentation

| Page | Package-level content |
|---|---|
| {doc}`useinfo` | Install and run a reproducible simulation |
| {doc}`theo` | Equations and symbols implemented by DigiMicPy |
| {doc}`details` | State layout, array shapes, validation, and solver behaviour |
| {doc}`api` | Supported top-level Python interface |
| {doc}`advanced_usage` | Explicit recipes using DigiMicPy parameters and outputs |
| {doc}`thermal` | Fixed-temperature scaling implemented by DigiMicPy |
| {doc}`spatial` | Conservative coupling between labeled MiCRM patches |
| {doc}`analysis` | Endpoint and numerical stability diagnostics |
| {doc}`support` | Package issue-reporting guidance |

## Documentation boundary

This book describes the behaviour of DigiMicPy. Shared scientific definitions,
interpretation, reporting guidance, project information, and material that is
not implemented by this package are maintained on the DigiMic platform site.

```{important}
Functions exported from `digimicpy` form the supported package interface.
Recipes in this book are transparent calculations using that interface; they
are not additional package APIs unless stated otherwise.
```
