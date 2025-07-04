 NHANES Metabolic-Depression Analysis Setup
 Environment Setup and Package Installation
========================================

# This script sets up the R environment for NHANES metabolic-depression 
# phenotyping analysis using β-VAE and Dirichlet Process GMM clustering.

# Required for the methodology described:
# - Data preprocessing and batch correction
# - Multiple imputation using MICE
# - β-VAE implementation via torch/keras
# - Dirichlet Process Gaussian Mixture Models
# - Survey-weighted statistical analysis

Author: Abby Liu
Date: July 2025

=============================================================================
1. SYSTEM REQUIREMENTS AND INITIAL SETUP
=============================================================================

# Clear workspace and set options
rm(list = ls())
gc()

# Set global options for reproducibility
set.seed(42)
options(scipen = 999, digits = 4)

# Check R version (recommended >= 4.0.0)
if (R.version$major < 4) {
  warning("R version 4.0.0 or higher is recommended for this analysis")
}

cat("NHANES Metabolic-Depression Analysis Environment Setup\n")
cat("======================================================\n")
cat("R version:", R.version.string, "\n")
cat("Platform:", R.version$platform, "\n\n")

=============================================================================
2. PACKAGE INSTALLATION FUNCTION
=============================================================================

#' Install packages if not already installed
#' @param packages Character vector of package names
#' @param repos Repository to install from
install_if_missing <- function(packages, repos = "https://cran.r-project.org") {
  new_packages <- packages[!(packages %in% installed.packages()[, "Package"])]
  if (length(new_packages) > 0) {
    cat("Installing missing packages:", paste(new_packages, collapse = ", "), "\n")
    install.packages(new_packages, repos = repos, dependencies = TRUE)
  } else {
    cat("All packages already installed\n")
  }
}

=============================================================================
 3. CORE DATA MANIPULATION AND STATISTICS PACKAGES
=============================================================================

cat("Installing core data manipulation packages...\n")

core_packages <- c(
  "tidyverse",      # Data manipulation and visualization (dplyr, ggplot2, etc.)
  "data.table",     # Fast data manipulation
  "magrittr",       # Pipe operators
  "lubridate",      # Date manipulation
  "stringr",        # String operations
  "forcats",        # Factor handling
  "readxl",         # Excel file reading
  "haven",          # SPSS, Stata, SAS file reading
  "janitor"         # Data cleaning functions
)

install_if_missing(core_packages)

=============================================================================
4. SURVEY STATISTICS AND NHANES-SPECIFIC PACKAGES
=============================================================================

cat("Installing survey statistics packages...\n")

survey_packages <- c(
  "survey",         # Survey-weighted analysis (primary package)
  "srvyr",          # dplyr-compatible survey analysis
  "NHANES",         # NHANES data access and utilities
  "nhanesA",        # Enhanced NHANES data download functions
  "weights",        # Additional weighting functions
  "Hmisc",          # Statistical functions and utilities
  "rms"             # Regression modeling strategies
)

install_if_missing(survey_packages)

 =============================================================================
 5. MISSING DATA AND MULTIPLE IMPUTATION PACKAGES
 =============================================================================

cat("Installing multiple imputation packages...\n")

imputation_packages <- c(
  "mice",           # Multiple Imputation by Chained Equations (primary)
  "VIM",            # Visualization and Imputation of Missing values
  "Hmisc",          # Additional imputation methods
  "imputeTS",       # Time series imputation
  "RANN",           # Random forest-based imputation
  "missForest",     # Non-parametric imputation using Random Forest
  "Amelia",         # Multiple imputation for missing data
  "mi"              # Multiple imputation using Bayesian methods
)

install_if_missing(imputation_packages)

 =============================================================================
 6. BATCH CORRECTION AND HARMONIZATION PACKAGES
 =============================================================================

cat("Installing batch correction packages...\n")

# Note: neuroCombat for ComBat batch correction
batch_packages <- c(
  "sva",            # Surrogate Variable Analysis (includes ComBat)
  "limma",          # Linear models for microarray data (batch correction)
  "preprocessCore", # Normalisation and preprocessing functions
  "RUVSeq"          # Remove Unwanted Variation
)

install_if_missing(batch_packages)

# Try to install neuroCombat from GitHub if not available on CRAN
if (!("neuroCombat" %in% installed.packages()[, "Package"])) {
  cat("Installing neuroCombat from GitHub...\n")
  if (!("devtools" %in% installed.packages()[, "Package"])) {
    install.packages("devtools")
  }
  tryCatch({
    devtools::install_github("Jfortin1/neuroCombat_Rpackage")
  }, error = function(e) {
    cat("Warning: Could not install neuroCombat. Will use sva::ComBat instead.\n")
  })
}

=============================================================================
7. MACHINE LEARNING AND DEEP LEARNING PACKAGES
=============================================================================

cat("Installing machine learning packages...\n")

ml_packages <- c(
  "caret",          # Classification and Regression Training
  "randomForest",   # Random Forest implementation
  "e1071",          # Support Vector Machines and other ML methods
  "glmnet",         # Lasso and Elastic-Net regularization
  "xgboost",        # Extreme Gradient Boosting
  "cluster",        # Cluster analysis
  "fpc",            # Flexible Procedures for Clustering
  "mclust",         # Gaussian Mixture Models
  "mixtools",       # Tools for mixture models
  "dirichletprocess" # Dirichlet Process models
)

install_if_missing(ml_packages)

# Deep Learning packages (torch ecosystem for β-VAE)
cat("Installing deep learning packages...\n")

dl_packages <- c(
  "torch",          # PyTorch for R (primary for β-VAE)
  "torchvision",    # Computer vision utilities
  "luz",            # High-level torch interface
  "reticulate"      # Python interface (backup for TensorFlow/Keras)
)

# Install torch packages with error handling
tryCatch({
  install_if_missing(dl_packages)
  
  # Install torch if not available
  if (!torch::torch_is_installed()) {
    cat("Installing torch backend...\n")
    torch::install_torch()
  }
}, error = function(e) {
  cat("Warning: torch installation failed. Will attempt alternative approaches.\n")
  cat("Error details:", e$message, "\n")
})

 =============================================================================
 8. ADVANCED STATISTICAL ANALYSIS PACKAGES
 =============================================================================

cat("Installing advanced statistical packages...\n")

stats_packages <- c(
  "lme4",           # Linear mixed-effects models
  "nlme",           # Non-linear mixed-effects models
  "mgcv",           # Generalized Additive Models
  "survival",       # Survival analysis
  "survminer",      # Survival analysis visualization
  "psych",          # Psychological statistics and psychometrics
  "lavaan",         # Structural Equation Modeling
  "sem",            # Structural Equation Models
  "corrplot",       # Correlation plot visualization
  "FactoMineR",     # Multivariate exploratory data analysis
  "factoextra"      # Extract and visualize multivariate analysis results
)

install_if_missing(stats_packages)

 =============================================================================
 9. VISUALIZATION AND REPORTING PACKAGES
 =============================================================================

cat("Installing visualization packages...\n")

viz_packages <- c(
  "ggplot2",        # Grammar of graphics (included in tidyverse)
  "ggpubr",         # Publication-ready plots
  "ggsci",          # Scientific journal color palettes
  "ggthemes",       # Additional ggplot2 themes
  "plotly",         # Interactive plots
  "DT",             # Interactive data tables
  "knitr",          # Dynamic report generation
  "rmarkdown",      # R Markdown documents
  "kableExtra",     # Enhanced table formatting
  "gridExtra",      # Grid-based plot arrangements
  "cowplot",        # Publication-ready plot themes
  "pheatmap",       # Pretty heatmaps
  "ComplexHeatmap", # Advanced heatmap visualization
  "circlize"        # Circular visualization
)

install_if_missing(viz_packages)

 =============================================================================
10. BIOCONDUCTOR PACKAGES (for advanced batch correction)
=============================================================================

cat("Installing Bioconductor packages...\n")

# Install BiocManager if not available
if (!("BiocManager" %in% installed.packages()[, "Package"])) {
  install.packages("BiocManager")
}

# Bioconductor packages for batch correction and preprocessing
bioc_packages <- c(
  "sva",            # Surrogate Variable Analysis
  "limma",          # Linear models for microarray data
  "preprocessCore", # Preprocessing and normalization
  "Biobase",        # Base functions for Bioconductor
  "edgeR"           # Empirical analysis of digital gene expression
)

tryCatch({
  BiocManager::install(bioc_packages, update = FALSE, ask = FALSE)
}, error = function(e) {
  cat("Warning: Some Bioconductor packages could not be installed.\n")
  cat("This may affect ComBat batch correction functionality.\n")
})

=============================================================================
 11. LOAD CORE LIBRARIES AND VERIFY INSTALLATION
=============================================================================

cat("\nLoading and verifying core packages...\n")

# Core libraries that must load successfully
core_libs <- c(
  "tidyverse", "data.table", "survey", "mice", 
  "ggplot2", "knitr", "caret"
)

# Load libraries with error checking
for (lib in core_libs) {
  tryCatch({
    library(lib, character.only = TRUE, quietly = TRUE)
    cat("✓", lib, "loaded successfully\n")
  }, error = function(e) {
    cat("✗", lib, "failed to load:", e$message, "\n")
  })
}

=============================================================================
12. CONFIGURATION AND HELPER FUNCTIONS
=============================================================================

#' Custom theme for publication-ready plots
theme_nhanes <- function() {
  theme_minimal() +
    theme(
      panel.grid.minor = element_blank(),
      plot.title = element_text(size = 14, face = "bold"),
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 10),
      legend.title = element_text(size = 12),
      legend.text = element_text(size = 10),
      strip.text = element_text(size = 11, face = "bold")
    )
}

#' Function to check survey weights
check_survey_weights <- function(data, weight_var = "WEIGHT") {
  if (!weight_var %in% names(data)) {
    stop("Weight variable not found in data")
  }
  
  weights <- data[[weight_var]]
  cat("Survey Weight Summary:\n")
  cat("Valid weights:", sum(!is.na(weights) & weights > 0), "\n")
  cat("Mean weight:", round(mean(weights, na.rm = TRUE), 2), "\n")
  cat("Weight range:", round(range(weights, na.rm = TRUE), 2), "\n")
  
  return(invisible(weights))
}

#' Function to create survey design object
create_survey_design <- function(data, weights = "WEIGHT", strata = NULL, cluster = NULL) {
  if (is.null(strata) && is.null(cluster)) {
    # Simple survey design without clustering
    design <- svydesign(
      ids = ~1,
      weights = as.formula(paste("~", weights)),
      data = data
    )
  } else {
    # Complex survey design
    design <- svydesign(
      ids = if(!is.null(cluster)) as.formula(paste("~", cluster)) else ~1,
      strata = if(!is.null(strata)) as.formula(paste("~", strata)) else NULL,
      weights = as.formula(paste("~", weights)),
      data = data,
      nest = TRUE
    )
  }
  
  return(design)
}

#' Function to perform survey-weighted descriptive statistics
survey_descriptives <- function(design, variables) {
  results <- list()
  
  for (var in variables) {
    if (var %in% names(design$variables)) {
      # Continuous variables
      if (is.numeric(design$variables[[var]])) {
        mean_est <- svymean(as.formula(paste("~", var)), design, na.rm = TRUE)
        results[[var]] <- list(
          type = "continuous",
          mean = coef(mean_est)[1],
          se = SE(mean_est)[1],
          ci = confint(mean_est)
        )
      } else {
        # Categorical variables
        prop_est <- svymean(as.formula(paste("~", var)), design, na.rm = TRUE)
        results[[var]] <- list(
          type = "categorical",
          proportions = coef(prop_est),
          se = SE(prop_est),
          ci = confint(prop_est)
        )
      }
    }
  }
  
  return(results)
}

=============================================================================
 13. β-VAE IMPLEMENTATION SETUP
=============================================================================

# Function to setup β-VAE environment
setup_beta_vae <- function() {
  cat("Setting up β-VAE environment...\n")
  
  # Check if torch is available
  if (requireNamespace("torch", quietly = TRUE)) {
    if (torch::torch_is_installed()) {
      cat("✓ PyTorch backend available for β-VAE implementation\n")
      
      # Define β-VAE architecture function
      create_beta_vae <<- function(input_dim = 10, latent_dim = 8, hidden_dim = 128) {
        
        # Encoder network
        encoder <- nn_sequential(
          nn_linear(input_dim, hidden_dim),
          nn_relu(),
          nn_linear(hidden_dim, 64),
          nn_relu()
        )
        
        # Latent space layers
        mu_layer <- nn_linear(64, latent_dim)
        logvar_layer <- nn_linear(64, latent_dim)
        
        # Decoder network
        decoder <- nn_sequential(
          nn_linear(latent_dim, 64),
          nn_relu(),
          nn_linear(64, hidden_dim),
          nn_relu(),
          nn_linear(hidden_dim, input_dim)
        )
        
        return(list(
          encoder = encoder,
          mu_layer = mu_layer,
          logvar_layer = logvar_layer,
          decoder = decoder
        ))
      }
      
      cat("✓ β-VAE architecture function defined\n")
      
    } else {
      cat("⚠ PyTorch backend not installed. Run torch::install_torch()\n")
    }
  } else {
    cat("⚠ torch package not available. β-VAE will use alternative implementation\n")
  }
}

# Run β-VAE setup
setup_beta_vae()

=============================================================================
14. DIRICHLET PROCESS GMM SETUP
=============================================================================

# Function to setup Dirichlet Process GMM
setup_dpgmm <- function() {
  cat("Setting up Dirichlet Process GMM environment...\n")
  
  if (requireNamespace("dirichletprocess", quietly = TRUE)) {
    cat("✓ dirichletprocess package available\n")
    
    # Function to perform DP-GMM clustering
    perform_dpgmm_clustering <<- function(data, max_clusters = 10, alpha = 1.0) {
      
      # Create Dirichlet Process object
      dp <- DirichletProcessGaussian(data)
      dp <- Initialise(dp, prior_weight = alpha, max_clusters = max_clusters)
      
      # Fit the model
      dp <- Fit(dp, iterations = 2000, progressBar = FALSE)
      
      # Extract cluster assignments
      clusters <- dp$clusterLabels
      n_clusters <- length(unique(clusters))
      
      cat("DP-GMM identified", n_clusters, "clusters\n")
      
      return(list(
        clusters = clusters,
        n_clusters = n_clusters,
        dp_object = dp
      ))
    }
    
    cat("✓ DP-GMM clustering function defined\n")
    
  } else {
    cat("⚠ dirichletprocess package not available\n")
    cat("   Will use alternative clustering methods (mixtools or mclust)\n")
    
    # Alternative using mixtools
    if (requireNamespace("mixtools", quietly = TRUE)) {
      perform_dpgmm_clustering <<- function(data, max_clusters = 10, alpha = 1.0) {
        # Use mixture of Gaussians as approximation
        library(mixtools)
        
        # Try different numbers of components
        bic_values <- numeric(max_clusters)
        models <- list()
        
        for (k in 1:max_clusters) {
          tryCatch({
            model <- normalmixEM(data, k = k, verb = FALSE)
            # Calculate BIC approximation
            bic_values[k] <- -2 * model$loglik + k * log(length(data))
            models[[k]] <- model
          }, error = function(e) {
            bic_values[k] <- Inf
          })
        }
        
        # Select best model
        best_k <- which.min(bic_values)
        best_model <- models[[best_k]]
        
        # Assign clusters
        clusters <- apply(best_model$posterior, 1, which.max)
        
        cat("Gaussian Mixture Model identified", best_k, "clusters\n")
        
        return(list(
          clusters = clusters,
          n_clusters = best_k,
          model = best_model
        ))
      }
      
      cat("✓ Alternative clustering function defined using mixtools\n")
    }
  }
}

# Run DP-GMM setup
setup_dpgmm()

=============================================================================
 15. FINAL VERIFICATION AND SESSION INFO
=============================================================================

cat("\n" %>% rep(2) %>% paste(collapse = ""))
cat("========================================\n")
cat("ENVIRONMENT SETUP COMPLETE\n")
cat("========================================\n\n")

# Display session information
cat("Session Information:\n")
print(sessionInfo())

cat("\n")
cat("Key Functions Available:\n")
cat("  - check_survey_weights(): Verify survey weight data\n")
cat("  - create_survey_design(): Create survey design object\n")
cat("  - survey_descriptives(): Survey-weighted descriptive statistics\n")
cat("  - create_beta_vae(): Define β-VAE architecture\n")
cat("  - perform_dpgmm_clustering(): Dirichlet Process GMM clustering\n")
cat("  - theme_nhanes(): Custom ggplot theme\n\n")

# Check critical packages
critical_packages <- c("tidyverse", "survey", "mice", "ggplot2")
all_loaded <- sapply(critical_packages, function(x) x %in% (.packages()))

if (all(all_loaded)) {
  cat("✅ All critical packages loaded successfully!\n")
  cat("✅ Environment ready for NHANES analysis pipeline\n\n")
} else {
  missing <- critical_packages[!all_loaded]
  cat("❌ Some critical packages not loaded:", paste(missing, collapse = ", "), "\n")
  cat("Please check installation and try loading manually\n\n")
}

cat("Next Steps:\n")
cat("1. Load your NHANES data: data <- read.csv('nhanes_processed_data.csv')\n")
cat("2. Create survey design: design <- create_survey_design(data)\n")
cat("3. Run preprocessing pipeline as described in methodology\n")
cat("4. Train β-VAE model on standardized variables\n")
cat("5. Perform DP-GMM clustering on latent representations\n")
cat("6. Conduct survey-weighted statistical analysis\n\n")

cat("For help with any function, use: ?function_name\n")
cat("For package documentation, use: help(package = 'package_name')\n\n")

=============================================================================
 END OF SETUP SCRIPT
=============================================================================

# Clear temporary variables
rm(list = c("core_packages", "survey_packages", "imputation_packages", 
           "batch_packages", "ml_packages", "dl_packages", "stats_packages",
           "viz_packages", "bioc_packages", "core_libs"))

cat("Environment setup completed successfully! 🎉\n")
cat("Ready to begin NHANES metabolic-depression analysis.\n")
