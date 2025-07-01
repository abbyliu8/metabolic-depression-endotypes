# setup.R - Environment Setup Script
# Metabolic-Depressive Endotypes Analysis Project

cat("=== Project Environment Setup Started ===\n")

# Check R version
if (getRversion() < "4.0.0") {
  stop("R version 4.0 or higher is required")
}

# Create folder structure
folders <- c("data", "results", "results/figures", "results/tables")
for (folder in folders) {
  if (!dir.exists(folder)) {
    dir.create(folder, recursive = TRUE)
    cat("Created folder:", folder, "\n")
  }
}

# Required R packages
required_packages <- c(
  "tidyverse",    # Data manipulation and visualization
  "nhanesA",      # NHANES data access
  "cluster",      # Clustering analysis
  "mclust",       # Gaussian mixture models
  "pheatmap",     # Heatmap visualization
  "ggplot2",      # Advanced plotting
  "corrplot",     # Correlation plots
  "here",         # File path management
  "mice",         # Missing data imputation
  "sva",          # Batch effect correction (ComBat)
  "factoextra",   # Clustering visualization
  "ggpubr"        # Publication-ready plots
)

# Check and install missing packages
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) {
  cat("Installing missing packages:", paste(new_packages, collapse = ", "), "\n")
  install.packages(new_packages, dependencies = TRUE)
}

# Load essential packages
library(tidyverse)
library(here)

cat("=== Environment Setup Completed ===\n")
cat("Installed", length(required_packages), "required packages\n")
cat("Project ready for analysis\n")
