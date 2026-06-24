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

## What DigiMic helps you study

DigiMic is intended for exploratory and mechanistic microbiome modelling, especially when the question depends on how species transform shared resources rather than on species interactions alone. Typical uses include:

- generating synthetic microbial communities with modular resource preferences;
- simulating consumer and resource trajectories through time;
- comparing communities under different leakage, supply, mortality, or resource-loss regimes;
- studying cross-feeding and metabolic by-product structure;
- reducing MiCRM dynamics to effective species interactions for interpretation;
- analysing local stability, reactivity, feasibility, and return rates around equilibria;
- simulating community coalescence by merging pre-assembled microbiomes;
- calculating species-level and community-level carbon use efficiency (CUE);
- adding temperature-dependent uptake, maintenance, leakage, or resource-supply traits.

## Current implementation

The current Python version contains:

- `src/param.py`: utilities for modular uptake matrices and leakage tensors;
- `src/micrm.py`: a complete minimal MiCRM simulation script;
- `docs/content/*.ipynb`: executable documentation pages used to generate this website;
- `docs/content/*.md`: theory and usage notes for extensions that are being added to the package;
- `docs/content/figures/`: conceptual figures for the modelling framework and workflow.

The documentation is organised into basic theory, technical details, a runnable basic usage example, advanced usage notes, analysis pages, support information, and contact details.


## Development and community contribution

DigiMic is actively under development. Our aim is to establish a transparent core workflow that can support a growing set of modular extensions as new modelling, data-integration, and analysis needs arise.

Rather than treating every capability as a fixed part of the package, DigiMic is designed to accommodate optional components that can be integrated into the core workflow when they are useful for a particular research question. These may include new parameterisation methods, metabolic-model interfaces, host-response modules, inference tools, experimental-design workflows, or domain-specific analysis functions.

We welcome researchers from across microbiology, ecology, metabolic modelling, bioinformatics, environmental science, and related fields to use DigiMic and help shape its development. Contributions may take the form of independently developed extensions, code contributions, examples and datasets, or clearly defined feature requests motivated by real research needs.
