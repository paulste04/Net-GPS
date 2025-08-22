Net-GPS: Generalized Propensity Scores for Network-Interdependent Continuous Treatments
--------------------------------------------------------------------------------------

This repository contains simulation code for the MSc Statistics thesis:

"Generalized Propensity Scores for Network-Interdependent Continuous Treatment Variables"
ETH Zürich – Paul Stephan

The project develops the Net-GPS estimator, combining:
- Generalized Propensity Scores (GPS) for continuous treatments
- Spatial Autoregressive (SAR) outcome models for network spillovers
- Simulation-based estimation of global and local dose-response functions (DRFs)

--------------------------------------------------------------------------------------
Files
- Assumption Violation Simulations.py   → Bias under assumption violations
- Net-GPS Global_MC_DRF.py              → Global dose-response simulations
- Net-GPS Local Estimand_Final.py       → Local dose-response estimation
- Network Example.py                    → Minimal network example
- PO Continuous Binary.py               → Mixed continuous/binary treatment setup
- PSM Balance.py                        → Balance checks
- Robustness Network Draws.py           → Robustness across random networks
- Robustness network size.py            → Robustness across network sizes
- Robustness_Specification.py           → Robustness to model specification

--------------------------------------------------------------------------------------
Requirements
Python 3.11+ with numpy, pandas, matplotlib, seaborn, statsmodels, networkx, libpysal, spreg, patsy

--------------------------------------------------------------------------------------
Usage
Clone repo and run, e.g.:
    python Net-GPS_Global_MC_DRF.py

--------------------------------------------------------------------------------------
Citation
Stephan, P. (2025). Generalized Propensity Scores for Network-Interdependent Continuous Treatment Variables. MSc Thesis, ETH Zürich.

--------------------------------------------------------------------------------------
AI Disclaimer
The written text of the thesis was partly polished with ChatGPT.
Parts of the code (comments, improvements, structure) were also improved with ChatGPT assistance.
