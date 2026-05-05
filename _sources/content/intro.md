# DigiMic project

## Digital Microbiome simulation for consumer-resource ecology

**DigiMic** (Digital Microbiome) is a lightweight modelling package for simulating microbial community dynamics with the Microbial Consumer-Resource Model (MiCRM). It focuses on the ecological mechanisms that shape microbiomes: resource competition, metabolic leakage, cross-feeding, maintenance costs, environmental resource supply, and community-level responses to perturbations.

This repository contains the Python implementation, **DigiMicPy**. It is designed as a transparent research codebase: the equations are explicit, the parameter generators are small enough to inspect, and simulations can be run directly from Python notebooks or scripts. A graphical interface is planned as the project develops.

## What DigiMic helps you study

DigiMic is intended for exploratory and mechanistic microbiome modelling, especially when the question depends on how species transform shared resources rather than on species interactions alone. Typical uses include:

- generating synthetic microbial communities with modular resource preferences;
- simulating consumer and resource trajectories through time;
- comparing communities under different leakage, supply, mortality, or resource-loss regimes;
- studying cross-feeding and metabolic by-product structure;
- reducing MiCRM dynamics to effective species interactions for interpretation;
- analysing local stability, reactivity, and return rates around equilibria.

## Model overview

The core state variables are consumer biomasses $C_i$ and resource abundances $R_\alpha$. Each consumer takes up resources according to an uptake matrix $u_{i\alpha}$, converts part of the uptake into growth, and leaks the rest into other resources through a leakage tensor $l_{i\alpha\beta}$. Resources also enter and leave the environment through supply and decay terms.

DigiMic therefore keeps both sides of microbiome dynamics visible:

| Layer | What it represents | Examples in the code |
|---|---|---|
| Consumers | Microbial populations or strains | $N$, $C_i$, $m_i$ |
| Resources | Metabolites, nutrients, or abstract resource classes | $M$, $R_\alpha$, $\rho_\alpha$, $\omega_\alpha$ |
| Uptake | Consumer-resource preferences | `modular_uptake` |
| Leakage | Metabolic by-products and cross-feeding | `modular_leakage`, `generate_l_tensor` |
| Dynamics | Coupled ODEs for consumers and resources | `solve_ivp`, `dCdt_Rdt` |

## Current implementation

The current Python version contains:

- `src/param.py`: utilities for modular uptake matrices and leakage tensors;
- `src/micrm.py`: a complete minimal MiCRM simulation script;
- `docs/content/*.ipynb`: executable documentation pages used to generate this website;
- `docs/content/figures/`: conceptual figures for the modelling framework and workflow.

The documentation is organised as follows:

- **The basic theory** introduces the MiCRM equations and their interpretation.
- **Technical details** explains the simulation workflow and data structures.
- **Basic usage** gives a runnable Python example.
- **Analysis** describes how DigiMic outputs can be used for stability and perturbation analysis.

## Project team

DigiMic is developed by PawarLab / EcoEngLab at Imperial College London. The Python package is authored by Yan Zhu and Samraat Pawar.
