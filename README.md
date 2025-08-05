# Trade Data Processing Pipeline

This repository contains a comprehensive trade data processing and economic analysis pipeline for tariff price pulse analysis. The system transforms raw HS commodity trade data into structured formats suitable for BEA economic analysis and tariff impact assessment.

## Repository Structure

```
TPT-underlying_code/
├── run_complete_pipeline.py          # 🚀 MAIN ENTRY POINT - Run complete pipeline
├── HS to BEA Data/                    # Core trade data processing pipeline
│   ├── run_pipeline.py               # Run HS-to-BEA pipeline only
│   ├── code/                         # Processing scripts (renamed for clarity)
│   │   ├── naics_crosswalk_builder.py
│   │   ├── hs_naics_mapping_compiler.py
│   │   ├── hs_to_bea_bridge_builder.py
│   │   ├── trade_data_mapper.py
│   │   ├── bea_hs_aggregator.py
│   │   ├── trade_weights_calculator.py
│   │   ├── initial_data_validator.py
│   │   └── tiva_import_values_validator.py
│   ├── data/
│   │   ├── raw/                      # Raw input data
│   │   ├── working/                  # 📊 Important intermediate data
│   │   └── final/                    # 🎯 Main pipeline outputs
│   └── validations/                  # 📋 Validation reports (.html, .tex files)
├── Alternative_Census_Mapping/        # Alternative mapping methodology (extracted)
│   ├── code/                         # Alternative census mapping scripts
│   ├── data/                         # Alternative mapping data
│   └── validations/                  # Alternative mapping validation reports
├── Calculations/                      # Economic calculations and analysis
│   ├── code/
│   │   ├── core/                     # Core economic calculations
│   │   │   ├── pce_data_loader.py
│   │   │   ├── input_output_data_loader.py
│   │   │   ├── data_validation_checks.py
│   │   │   └── economic_effect_calculations.py
│   │   └── testing/                  # 🧪 Economic analysis & validation (updated)
│   │       ├── bea_tariff_effects_calculation.py
│   │       ├── nipa_tariff_effects_calculation.py
│   │       ├── bea_country_rank_analysis.py
│   │       ├── nipa_country_rank_analysis.py
│   │       ├── bea_product_rank_analysis.py
│   │       ├── nipa_product_rank_analysis.py
│   │       ├── import_vs_effect_rank_comparison.py
│   │       ├── import_vs_effect_cosine_similarity.py
│   │       ├── within_country_cosine_similarity_analysis.py
│   │       ├── pce_data_flattener.py
│   │       ├── weighted_cosine_spearman_rank_analysis.py
│   │       └── data_loader.py
│   ├── data/                         # Calculation input/output data
│   └── validations/                  # Economic analysis validation outputs
├── Map BEA Regions/                   # Regional mapping utilities
├── Images for Macroblog/              # Visualization outputs
└── Resources/                         # Additional resources
```

## Quick Start

### 🚀 Run Complete Pipeline
```bash
python run_complete_pipeline.py
```

### 🎯 Run Specific Components
```bash
# Run only HS to BEA data processing
python run_complete_pipeline.py --component hs_to_bea

# Run only alternative census mapping comparison
python run_complete_pipeline.py --component alternative_census

# Run only economic calculations
python run_complete_pipeline.py --component calculations

# Run only economic analysis & testing
python run_complete_pipeline.py --component analysis

# Run validation scripts only
python run_complete_pipeline.py --validate-only
```

### 📂 Run Individual Pipelines
```bash
# HS to BEA Data pipeline only
cd "HS to BEA Data"
python run_pipeline.py

# Alternative Census Mapping pipeline only  
cd "Alternative_Census_Mapping/code"
python run_alternative_census_mapping_pipeline.py
```

## Key Improvements Made

### 📝 Descriptive File Names
- **Before**: `01_Schott_Data_Compiler.py` → **After**: `hs_naics_mapping_compiler.py`
- **Before**: `02_HS_to_Naics_to_BEA.py` → **After**: `hs_to_bea_bridge_builder.py`
- **Before**: `01_BEA_calculation.py` → **After**: `bea_tariff_effects_calculation.py`

### 🏗️ Cleaner Structure
- **Extracted Alternative Census Mapping** from nested location to top-level component
- **Standardized naming** across all components
- **Clear separation** of core processing vs. analysis/testing

### 🎯 Standardized Output Naming
- **Validation outputs** clearly labeled with source script names
- **HTML and TEX files** appropriately organized by generating script
- **CSV outputs** organized for easy identification

### 🚀 Single Entry Point
- **`run_complete_pipeline.py`** - Master orchestrator for entire system
- **Component-specific runners** - Individual pipeline control
- **Flexible execution** - Run complete pipeline or specific components

## Pipeline Components

### 1. HS to BEA Data Pipeline
**Purpose**: Core trade data processing that maps HS commodity codes to BEA economic categories

**Key Outputs**:
- `data/final/bea_hs_section_weights.json` - Final trade weights for analysis
- `data/final/trade_weights.json` - Regional trade weight mappings
- `data/working/` - Important intermediate processing data
- `validations/` - HTML/TEX validation reports

### 2. Alternative Census Mapping Pipeline  
**Purpose**: Alternative methodology comparison using Census Bureau mappings

**Key Outputs**:
- `validations/` - Comparison reports between Census and Schott methodologies
- `data/` - Alternative mapping results

### 3. Economic Calculations
**Purpose**: Core economic impact calculations using input-output analysis

**Scripts**:
- `pce_data_loader.py` - Load Personal Consumption Expenditure data
- `input_output_data_loader.py` - Load BEA input-output tables
- `data_validation_checks.py` - Validate calculation inputs/outputs
- `economic_effect_calculations.py` - Core economic effect calculations

### 4. Economic Analysis & Testing
**Purpose**: Validation, comparison, and testing of economic calculations

**Analysis Types**:
- **Tariff Effects**: BEA vs NIPA calculation comparisons
- **Ranking Analysis**: Country and product rank correlations
- **Similarity Analysis**: Cosine similarity and Spearman rank comparisons
- **Data Processing**: PCE flattening and weighted analysis

## Configuration

Each component has its own `data_paths.json` configuration file:
- `HS to BEA Data/data_paths.json` - Main pipeline paths
- `Alternative_Census_Mapping/data_paths.json` - Alternative mapping paths  
- `Calculations/data_paths.json` - Economic calculation paths

Update these files to match your local file structure before running.

## Dependencies

Install required Python packages:
```bash
pip install pandas numpy matplotlib plotly country_converter scipy openpyxl
```

## Validation and Quality Assurance

The pipeline includes comprehensive validation:
- **Data Quality Checks**: Automated validation of mappings and transformations
- **Benchmark Comparisons**: Validation against TiVA and other benchmark data
- **Visual Validation**: HTML dashboards and charts for manual review
- **Statistical Validation**: Correlation and similarity analyses

## Replication Package Standards

This repository follows replication package best practices:
- ✅ **Single entry point** for complete reproduction
- ✅ **Clear file naming** indicating function and purpose  
- ✅ **Standardized output naming** for easy identification
- ✅ **Comprehensive documentation** of each component
- ✅ **Automated validation** and quality checks
- ✅ **Modular design** allowing component-by-component execution

## Output Locations

### 🎯 Main Outputs (Most Important)
- `HS to BEA Data/data/final/` - Final trade weights and HS section weights
- `HS to BEA Data/validations/` - Key validation reports (HTML/TEX files)

### 📊 Important Intermediate Data  
- `HS to BEA Data/data/working/` - Intermediate processing files (crucial for understanding pipeline)

### 📋 Validation Reports
- `*.html` files - Interactive validation dashboards
- `*.tex` files - LaTeX-formatted validation reports  
- `*.csv` files - Supporting data (helpful but not always primary focus)

## Getting Help

For pipeline execution issues:
1. Check component-specific log outputs
2. Review validation HTML files for data quality issues
3. Ensure `data_paths.json` files are configured correctly
4. Verify all required input data files are present

## Authors

Trade Data Processing Pipeline - Federal Reserve Bank of Atlanta