# Trade Price Pulse (TPT) - Underlying Code Replication Package

This replication package contains the complete pipeline for constructing tariff price pulse analysis data, from raw trade data processing through final economic calculations.

## Quick Start - Final Outputs

**The main results of this project can be found in:**
- `3_final_outputs/trade_weights/` - Final trade weights and import mappings
- `3_final_outputs/import_calculations/` - Economic impact calculations  
- `3_final_outputs/validation_reports/` - Data validation and quality reports

## Repository Structure

```
TPT-underlying_code/
├── 1_data_construction/          # HS-to-BEA mapping and trade weights construction
├── 2_economic_calculations/      # Tariff impact calculations and analysis
├── 3_final_outputs/             # Final results and deliverables
├── 4_supplementary/             # Additional materials and figures
└── shared_utilities/            # Common functions and styles
```

## Running the Complete Pipeline

### Option 1: Run Everything (Recommended)
```bash
python run_complete_pipeline.py
```

This runs both data construction and economic calculations in sequence.

### Option 2: Run Individual Components
```bash
# Step 1: Data construction (creates trade weights)
cd 1_data_construction
python run_pipeline.py

# Step 2: Economic calculations (computes tariff impacts)
cd ../2_economic_calculations  
python run_calculations.py
```

### Option 3: Run Individual Scripts
See the README files in each numbered directory for detailed instructions.

## Main Components

### 1. Data Construction Pipeline (`1_data_construction/`)
Transforms raw HS commodity trade data into structured datasets for economic analysis.

**Key Outputs:**
- Trade weights by region and BEA category
- HS-to-BEA economic category mappings
- Import value benchmarks and validations

**Run:** `cd 1_data_construction && python run_pipeline.py`

### 2. Economic Calculations (`2_economic_calculations/`)
Computes tariff price impacts using the constructed trade data.

**Key Outputs:**
- Direct and indirect economic impact matrices
- Country and product ranking analyses
- Correlation and similarity analyses

**Run:** `cd 2_economic_calculations && python run_calculations.py`

### 3. Final Outputs (`3_final_outputs/`)
Consolidated results ready for downstream analysis and publication.

**Contents:**
- `trade_weights/` - Final JSON files with trade weights and mappings
- `import_calculations/` - Economic impact calculation results
- `validation_reports/` - Data quality and benchmark comparisons
- `figures/` - Key charts and visualizations

### 4. Supplementary Materials (`4_supplementary/`)
Additional supporting materials and alternative analyses.

## Key Configuration

Both main components use `data_paths.json` files to configure input/output locations. Update these files to match your local directory structure before running.

## Data Requirements

### Raw Data Needed:
- 2024 HS-10 trade data by continent (`data/raw/hs10/`)
- BEA economic classification codes (`data/raw/BEA_codes/`)
- NAICS code crosswalks (`data/raw/naics_crosswalks/`)
- Schott HS-NAICS concordance files

### Intermediate Data:
The pipeline creates extensive intermediate files in `data/working/` folders for debugging and validation.

## Dependencies

- Python 3.7+
- pandas
- country_converter  
- json
- os

## Output Validation

Each component generates validation files to verify data quality:
- Mapping coverage and accuracy reports
- Statistical consistency checks  
- Benchmark comparisons with external data sources
- Interactive HTML dashboards for validation review

## Citation

If you use this code or data, please cite:
[Add appropriate citation information]

## Support

For questions about the methodology, see the detailed documentation in each component's README file.
For technical issues, check the validation reports for data quality indicators.

---

**Last Updated:** [Current Date]
**Version:** 2.0 (Replication Package)