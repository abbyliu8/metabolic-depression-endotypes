# Data Directory

## Structure
- `raw/`: Original NHANES XPT files downloaded from CDC
- `processed/`: Cleaned and merged datasets ready for analysis

## Data Sources

### NHANES 2017-2018 (Discovery Cohort)
- **Demographics**: DEMO_J.xpt
- **Insulin**: INS_J.xpt  
- **Blood Pressure**: BPX_J.xpt
- **Body Measurements**: BMX_J.xpt
- **Depression**: DPQ_J.xpt
- **Laboratory**: GLU_J.xpt, TRIGLY_J.xpt, HDL_J.xpt, HSCRP_J.xpt
- **Fasting**: FASTQX_J.xpt

### NHANES 2021-2023 (Validation Cohort)
- **Demographics**: P_DEMO_L.xpt
- **Insulin**: P_INS_L.xpt
- **Blood Pressure**: P_BPX_L.xpt
- **Body Measurements**: P_BMX_L.xpt
- **Depression**: P_DPQ_L.xpt
- **Laboratory**: P_GLU_L.xpt, P_TRIGLY_L.xpt, P_HDL_L.xpt, P_HSCRP_L.xpt
- **Fasting**: P_FASTQX_L.xpt

## Usage
Data files will be automatically downloaded when running `main.R`.
Manual download instructions available in project documentation.
