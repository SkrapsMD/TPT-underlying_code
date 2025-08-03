# Data Construction Pipeline

This directory contains the complete pipeline for mapping HS commodity codes to BEA economic categories and constructing trade weights for tariff analysis.

## Overview

The pipeline transforms raw HS commodity trade data into structured formats suitable for BEA economic analysis and tariff impact assessment. It consists of 8 main scripts that must be run in sequence.

## Quick Start

```bash
python run_pipeline.py
```

This will run all pipeline steps in sequence and create validation reports.

## Pipeline Steps

### 01_naics_crosswalk_formation.py
**Purpose**: Creates NAICS 2017-2022 crosswalk mapping for consistent code translation

### 02_schott_data_compiler.py  
**Purpose**: Creates corrected HS-to-NAICS mappings for 2023 trade data
- Maps 2023 HS codes to 2017 NAICS codes (required for BEA consistency)
- Uses hierarchical matching when direct mappings fail

### 03_hs_naics_bea_mapping.py
**Purpose**: Bridges HS codes to BEA economic categories via NAICS
- Creates hierarchical matching from NAICS to BEA codes
- Builds complete HS → NAICS → BEA bridge

### 04_country_trade_data_processing.py  
**Purpose**: Processes raw 2024 trade data and applies HS-to-BEA mappings
- Loads continent-specific trade data
- Creates clean country-level datasets

### 05_bea_hs_aggregation.py
**Purpose**: Aggregates trade data to BEA levels and creates HS hierarchies
- Creates 4 BEA aggregation levels (Detail, U.Summary, Summary, Sector)  
- Calculates compositional weights

### 06_trade_weights_calculation.py
**Purpose**: Creates country-specific trade weights for tariff analysis
- Calculates direct weights (global denominators)
- Calculates indirect weights (regional denominators)

### 07_data_validation.py
**Purpose**: Validates constructed data against benchmark import values

### 08_tiva_benchmark_comparison.py  
**Purpose**: Compares trade weights with TiVA benchmark data and creates visualizations

## Final Outputs

After successful completion, check these locations:

### Main Results
- `data/final/bea_import_weights.json` - Final trade weights for economic analysis
- `data/final/bea_section_weights.json` - Sectoral composition weights

### Validation  
- `validations/08_TiVA_Import_Values_Comparison/02_TiVA_vs_HS_Import_Charts.html` - Interactive validation dashboard
- `validations/` - Detailed validation files for each step

### Working Data
- `data/working/` - Intermediate processing files for debugging

## Configuration

Edit `data_paths.json` to match your local file structure:

```json
{
  "base_paths": {
    "project_root": "/your/path/to/project",
    "raw_data": "/your/path/to/data/raw",
    "working_data": "/your/path/to/data/working", 
    "final_data": "/your/path/to/data/final"
  }
}
```

## Data Requirements

Ensure you have the following raw data files:
- `data/raw/hs10/` - 2024 trade data by continent
- `data/raw/BEA_codes/` - BEA economic classification codes  
- `data/raw/naics_crosswalks/` - NAICS code mappings
- Schott concordance files for HS-NAICS mappings

## Key Dependencies

- pandas
- country_converter
- json
- os

Scripts must be run in sequence due to dependencies between outputs and inputs.