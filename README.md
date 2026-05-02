# Koopman Epidemics

The project studies early outbreak detection and intervention selection in a multi-agent epidemic simulation. It combines agent-based epidemic dynamics, Koopman-inspired representation learning, supervised outbreak classification, and counterfactual mobility interventions to analyze trajectories near tipping boundaries.

## Overview

The simulation represents a population of agents that move through a shared environment, interact through co-location, and follow heterogeneous disease progression after infection. The learning component uses short early trajectory windows to estimate whether the final epidemic outcome will remain contained or become a major outbreak.

A Koopman-inspired latent representation is used to encode aggregate epidemic observables into a compact dynamical space. The learned representation supports short-horizon forecasting and provides features for outbreak-risk estimation. Counterfactual experiments then test whether small localized mobility restrictions can reduce the final attack rate or move a trajectory below the outbreak threshold.

## Main Capabilities

- Multi-agent epidemic simulation with structured mobility, heterogeneous susceptibility, viral-load progression, and local transmission through co-location.
- Generation of aggregate epidemic observables for contained, near-threshold, and major-outbreak regimes.
- Koopman-inspired latent representation learning from short epidemic trajectory windows.
- Early-warning outbreak classification from limited initial observations.
- Counterfactual intervention analysis based on paired baseline and intervention simulations.
- Evaluation of final attack rate, peak infection burden, forecast structure, classifier performance, and intervention effects.
- Reproducible experimentation through configurable parameters and fixed random seeds.

## Typical Workflow

A typical experimental workflow consists of:

1. Configuring the simulation and learning parameters.
2. Generating baseline epidemic trajectories across multiple random seeds or parameter regimes.
3. Building early trajectory windows from aggregate epidemic observables.
4. Training Koopman-inspired representations for short-horizon epidemic forecasting.
5. Training and evaluating an outbreak classifier from early-window features.
6. Running counterfactual intervention experiments on selected trajectories.
7. Comparing baseline and intervention outcomes through attack rate, outbreak status, and peak infection burden.
 
## Citation

A detailed description of the architecture and examples can be found in this paper:

> Florin Leon, *Koopman Representations for Early Outbreak Warning and Minimal Counterfactual Intervention in Multi-Agent Epidemic Simulations*, 2026

## Note

The implementations are intended as reference programs rather than optimized systems. The programs are distributed in the hope that they will be useful, but without any warranty; without even the implied warranty of merchantability or fitness for a particular purpose.




