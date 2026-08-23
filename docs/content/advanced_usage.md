# Advanced usage

These pages describe DigiMic extensions that go beyond a single baseline MiCRM run. They are written as theory and usage notes first, so the web documentation can be useful before every helper function is wrapped into the Python package.

```{important}
Temperature scaling and conservative spatial patches have tested package APIs.
Coalescence, carbon-use-efficiency, and resource-flux pages currently describe
manual NumPy workflows rather than dedicated `digimicpy` helper functions.
```

Use this section when you want to:

- merge independently assembled microbial communities;
- calculate species-level or community-level carbon use efficiency;
- introduce temperature-dependent traits and environmental regimes;
- couple matching consumers and resources across spatial patches;
- quantify resource-processing fluxes through uptake, retention, leakage, and maintenance.
