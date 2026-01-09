# Reproducible Code for "Supervised clustering using SOM for severity-based pattern detection in urban traffic crashes"


This repository contains the source code and a reduced sample of the data used to
reproduce the main results reported in the paper.

------------------------------------------------------------
1. Repository contents
------------------------------------------------------------

  - Sample.RData
    Reduced dataset used to demonstrate the full analysis workflow. 
    This sample preserves the structure and variable definitions of the original data but,
    due to its reduced size, does not yield the exact numerical results reported in the paper.

  - R_codes_paper.R
    Including required R packages, Data Partitioning and Model Configuration, Comparison of models,
    Global interpretation with iml, Local interpretation: Shapley values, Self-Organizing Map (SOM) training, 
    Generation of figures reported in the paper.
    

------------------------------------------------------------
2. Software requirements
------------------------------------------------------------

The analysis was developed and tested using:

- R version >= 4.5.0
