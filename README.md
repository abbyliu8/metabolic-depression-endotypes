# Metabolic-Depressive Endotypes: Cross-Cycle NHANES Analysis

This repository contains the code for reproducing the analysis presented in the paper "Β-VAE Decodes Time-Consistent Clinical Phenotypes of Depression and Metabolic Comorbidities in American Adults."


## Abstract

This study identifies metabolic-depressive endotypes using unsupervised machine learning applied to NHANES 2017-2018 data, with temporal stability validation using NHANES 2021-2023 data.

## Quick Start

### Requirements
- R version ≥ 4.0
- RStudio (recommended)
- Internet connection for data download

### Installation and Execution
```r
# 1. Clone or download this repository
# 2. Open the project in RStudio
# 3. Run setup script to install dependencies
source("setup.R")

# 4. Execute main analysis
source("main.R")
Repository Structure
├── README.md           # Project overview
├── setup.R             # Environment setup and package installation
├── main.R              # Main analysis script
├── data/               # Data folder (NHANES data will be downloaded here)
├── results/            # Output folder
│   ├── figures/        # Generated plots
│   └── tables/         # Result tables
└── LICENSE             # MIT License
Data Sources

NHANES 2017-2018 (Cycle J): Discovery cohort
NHANES 2021-2023 (Cycle L): Validation cohort
Data automatically downloaded via the NHANES R package

Methodology

Dimensionality Reduction: β-Variational Autoencoder (β-VAE)
Clustering: Dirichlet Process Gaussian Mixture Model
Validation: Cross-cycle temporal stability assessment
Metrics: Adjusted Rand Index (ARI), Normalised Mutual Information (NMI)

Key Variables

Depression: PHQ-9 item-level responses (DPQ010-DPQ090)
Metabolic: Fasting glucose, insulin, lipids, BMI, blood pressure, hs-CRP

Results
Running the analysis will generate:

Endotype heatmaps showing metabolic-depressive patterns
UMAP projections of identified endotypes
Cross-cycle validation metrics
Clinical characterisation tables

Contact

Email: uctqhhl@ucl.ac.uk
Institution: University College London

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

National Health and Nutrition Examination Survey (NHANES) data
CDC National Centre for Health Statistics
R Core Team and package developers
