# Economic Calculations

This directory contains the economic impact calculations that use the trade weights from the data construction pipeline to compute tariff price pulse effects.

## Overview

The calculations component takes the trade weights and HS-BEA mappings from the data construction pipeline and computes direct and indirect economic impacts of tariff changes.

## Quick Start

```bash
python run_calculations.py
```

## Component Structure

### Core Calculations (`code/core/`)

These are the fundamental calculation scripts that must be run in sequence:

#### 01_read_pce_data.py
**Purpose**: Loads and processes Personal Consumption Expenditure (PCE) data
- Reads BEA PCE detailed data
- Creates consumption category mappings

#### 02_read_io_data.py  
**Purpose**: Loads and processes Input-Output (IO) table data
- Reads BEA IO tables for inter-industry relationships
- Prepares matrices for impact calculations

#### 03_validate_inputs_outputs.py
**Purpose**: Validates data consistency between inputs and outputs
- Checks data alignment and completeness
- Generates validation reports

#### 04_main_tariff_calculations.py
**Purpose**: Computes main tariff impact calculations
- Calculates direct import effects
- Computes indirect effects through IO relationships
- Generates impact matrices

### Analysis Scripts (`code/analysis/`)

These scripts perform various analyses and validations of the calculation results:

#### 01_bea_calculations.py & 01_nipa_calculations.py
**Purpose**: Alternative calculation methods using BEA and NIPA frameworks

#### 02_bea_country_ranking_analysis.py & 02_nipa_country_ranking_analysis.py  
**Purpose**: Ranks countries by economic impact magnitude

#### 03_bea_product_ranking_analysis.py & 03_nipa_product_ranking_analysis.py
**Purpose**: Ranks products/sectors by economic impact

#### 04_import_effect_comparison.py
**Purpose**: Compares import rankings with economic effect rankings

#### 05_cosine_similarity_analysis.py
**Purpose**: Analyzes similarity between import patterns and economic effects

#### 06_within_country_analysis.py
**Purpose**: Analyzes effects within individual countries

#### 07_pce_flattener.py  
**Purpose**: Creates flattened PCE data for analysis

#### 08_weighted_correlation_analysis.py
**Purpose**: Computes weighted correlations and Spearman rank statistics

## Final Outputs

After successful completion, key outputs include:

### Main Results
- `data/final/indirect_BEA_matrix_2023.json` - Indirect economic impact matrix (BEA framework)
- `data/final/indirect_matrix_2023.json` - Indirect economic impact matrix (alternative framework)

### Analysis Results  
- Various CSV files with country and product rankings
- Correlation and similarity analysis results
- Validation and comparison reports

## Configuration

Edit `data_paths.json` to match your local file structure and point to the data construction outputs:

```json
{
  "base_paths": {
    "project_root": "/your/path/to/project",
    "data_construction_outputs": "/path/to/1_data_construction/data/final",
    "raw_data": "/your/path/to/calculations/raw/data"
  }
}
```

## Data Requirements

### From Data Construction Pipeline
- Trade weights JSON files from `1_data_construction/data/final/`
- HS-BEA mapping files

### Additional Raw Data
- BEA PCE detailed data tables
- BEA Input-Output tables  
- Tariff rate data from Trade War Tracker

## Dependencies

- pandas
- numpy
- json
- os
- scipy (for correlation calculations)

## Notes

- Core calculations should be run before analysis scripts
- Some analysis scripts can be run independently for specific analyses
- Check validation outputs to ensure data quality before using results