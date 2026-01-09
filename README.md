# Reproducible Code for "Supervised clustering using SOM for severity-based pattern detection in urban traffic crashes"


This repository contains the source code and a reduced sample of the data used to
reproduce the main results reported in the paper.

------------------------------------------------------------
1. Repository contents
------------------------------------------------------------

  - [Sample.RData](https://github.com/AnnaSalazar/Supervised-Clustering/blob/main/Sample.RData)
    Reduced dataset used to demonstrate the full analysis workflow. 
    This sample preserves the structure and variable definitions of the original data but,
    due to its reduced size, does not yield the exact numerical results reported in the paper.

  - [R_codes_paper.R](https://github.com/AnnaSalazar/Supervised-Clustering/blob/main/R_codes_paper.R)
    Including required R packages, Data Partitioning and Model Configuration, Comparison of models,
    Global interpretation with iml, Local interpretation: Shapley values, Self-Organizing Map (SOM) training, 
    Generation of figures reported in the paper.
    

------------------------------------------------------------
2. Software requirements
------------------------------------------------------------

The analysis was developed and tested using:

- R version >= 4.5.0

------------------------------------------------------------
3. How to reproduce the results
------------------------------------------------------------

1. Clone or download this repository.
2. Open R or RStudio and set the working directory to the repository root.
3. Run the script.


------------------------------------------------------------
4. Data availability
------------------------------------------------------------

The file `Sample.RData` contains a reduced random sample of the original data,
included solely for reproducibility and demonstration purposes. The code structure
and results obtained with this sample are consistent with those reported in the
paper, although numerical values may differ.

------------------------------------------------------------
5. Reproducibility notes
------------------------------------------------------------

- Random seeds are fixed where applicable to ensure reproducibility.
- Minor numerical differences may occur across platforms or R versions.
- The repository reflects the version of the code corresponding to the
  accepted manuscript.

------------------------------------------------------------
6. Contact
------------------------------------------------------------

For questions regarding the code or reproducibility, please contact:

[Anna Salazar]
[Universitat de Barcelona]
[asalazar@ub.edu]

