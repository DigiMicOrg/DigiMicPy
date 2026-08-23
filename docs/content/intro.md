# About

## Digital Microbiome

**DigiMic** (Digital Microbiome) is an open modelling framework for predicting how microbial communities assemble, respond to environmental change, and process carbon. The current Python package, **DigiMicPy**, starts from the Microbial Consumer-Resource Model (MiCRM): species consume resources, leak metabolic by-products, compete through shared demand, and facilitate one another through cross-feeding.

The longer-term Digital Microbiome goal is to connect three layers in a single transparent workflow:

1. **Metabolic modelling and parameterisation** from strain-level data, traits, taxa, and omics.
2. **Microbiome modelling and prediction** with MiCRM, effective GLV reductions, stability analysis, coalescence experiments, carbon use efficiency, and temperature-dependent traits.
3. **Microbiome data and validation** against lab and real-world freshwater microbiome observations, including community composition, abundance, resource chemistry, carbon fluxes, and responses to fluctuating temperature, nutrient, and chemical regimes.

```{figure} figures/DigiMic.jpg
:name: digimic-workflow
:alt: Conceptual Digital Microbiome workflow linking metabolic modelling, microbiome modelling, and microbiome data.
:width: 100%

Digital Microbiome workflow: strain-level traits and metabolic modelling parameterise predictive microbiome dynamics, which are then compared with lab and field data.
```

## What DigiMicPy currently implements

DigiMicPy is intended for exploratory and mechanistic microbiome modelling,
especially when a question depends on how species transform shared resources.
The tested package API currently supports:

- generating synthetic microbial communities with modular resource preferences;
- simulating consumer and resource trajectories through time;
- comparing communities under different leakage, supply, mortality, or resource-loss regimes;
- scaling uptake and mortality across fixed temperatures;
- coupling matching consumers and resources across undirected spatial patches.

## Implementation status

The documentation separates implemented code from mathematical workflows that
are useful for planning analyses:

| Area | Status |
|---|---|
| Core MiCRM parameters, RHS, and solver | Implemented and tested |
| Modular uptake and leakage generators | Implemented and tested |
| Fixed-temperature uptake/mortality scaling | Implemented and tested |
| Conservative undirected spatial patches | Implemented and tested |
| Coalescence, CUE, resource-flux summaries | Documented manual workflows |
| Effective GLV conversion and stability helpers | Theory only; package API planned |

Start with {doc}`useinfo` for an executable package example and {doc}`api` for
the supported public interface. Advanced and analysis pages state explicitly
when their calculations are pseudocode or manual NumPy workflows.


## Development and community contribution

DigiMic is actively under development. Our aim is to establish a transparent core workflow that can support a growing set of modular extensions as new modelling, data-integration, and analysis needs arise.

Rather than treating every capability as a fixed part of the package, DigiMic is designed to accommodate optional components that can be integrated into the core workflow when they are useful for a particular research question. These may include new parameterisation methods, metabolic-model interfaces, host-response modules, inference tools, experimental-design workflows, or domain-specific analysis functions.

We welcome researchers from across microbiology, ecology, metabolic modelling, bioinformatics, environmental science, and related fields to use DigiMic and help shape its development. Contributions may take the form of independently developed extensions, code contributions, examples and datasets, or clearly defined feature requests motivated by real research needs.
