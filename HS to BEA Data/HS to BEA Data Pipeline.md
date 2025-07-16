# Trade Data Processing Pipeline

This repository contains the underlying data construction pipeline for tariff price pulse analysis. The pipeline processes raw trade data through multiple stages to create analytical datasets for economic impact assessment.

## Overview

The pipeline transforms raw HS commodity trade data into structured formats suitable for BEA economic analysis and tariff impact assessment. It consists of 5 main scripts that must be run in sequence.

## Setup

### 1. Configure File Paths

Edit `data_paths.json` to match your local file structure:

```json
{
  "base_paths": {
    "project_root": "/your/path/to/project",
    "underlying_data_root": "/your/path/to/Underlying_Data_Construction",
    "raw_data": "/your/path/to/data/raw",
    "working_data": "/your/path/to/data/working",
    "final_data": "/your/path/to/data/final",
    "code": "/your/path/to/code",
    "validations": "/your/path/to/validations"
  }
}
```

### 2. Data Requirements

Ensure you have the following raw data files:
- `data/raw/hs10/`: 2024 trade data by continent
- `data/raw/BEA_codes/`: BEA economic classification codes
- `data/raw/naics crosswalks/`: NAICS code mappings
- Schott concordance files for HS-NAICS mappings

## Pipeline Scripts

### 01_Schott_Data_Compiler.py
**Purpose**: Creates corrected HS-to-NAICS mappings for 2023 trade data

**Key Functions**:
- Maps 2023 HS codes to 2017 NAICS codes (required for BEA consistency)
- Uses hierarchical matching when direct mappings fail
- Handles NAICS classification changes between 2017 and 2022

**Key Assumptions**:
- Hierarchical matching levels: HS-8 → HS-6 → HS-4 (lines 579-616)
- Tolerance for many-to-many mappings in complex cases
- Priority: DIRECT_2017 > SIMPLE_CROSSWALK > HIERARCHICAL_HS

**Outputs**:
- `03_hs_naics_mapping_2023_corrected_naicsMDS.csv`: Final corrected mapping
- Validation files in `validations/01_Schott_Data_Compiler/`

### 02_HS_to_Naics_to_BEA.py
**Purpose**: Bridges HS codes to BEA economic categories via NAICS

**Key Functions**:
- Processes BEA data to extract valid NAICS codes
- Creates hierarchical matching from NAICS to BEA codes
- Builds complete HS → NAICS → BEA bridge

**Key Assumptions**:
- Hierarchical trimming: 6-digit → 5-digit → 4-digit → 3-digit → 2-digit NAICS
- BEA range expansion (e.g., "3331-9" becomes individual codes)
- Priority matching: exact match first, then progressive trimming

**Outputs**:
- `03_complete_hs_to_bea_mapping.csv`: Complete HS-to-BEA bridge
- `02_BEA_hierarchy.csv`: BEA level mappings

### 03_Map_country_trade_data.py
**Purpose**: Processes raw 2024 trade data and applies HS-to-BEA mappings

**Key Functions**:
- Loads continent-specific trade data
- Applies HS-to-BEA mapping from script 02
- Creates clean country-level datasets

**Key Assumptions**:
- Continent processing: Asia, Europe, North America, South America, Oceana
- Skips `combined_data.csv` files in favor of individual files
- Missing BEA mappings are handled gracefully

**Outputs**:
- `{continent}_combined.csv`: Raw combined data
- `{continent}_processed.csv`: Data with BEA mappings applied

### 04_Aggregate_BEA_and_HS.py
**Purpose**: Aggregates trade data to BEA levels and creates HS hierarchies

**Key Functions**:
- Creates 4 BEA aggregation levels (Detail, U.Summary, Summary, Sector)
- Builds HS code hierarchy (HS10 → HS8 → HS6 → HS4 → HS2 → HS_Section)
- Calculates compositional weights

**Key Assumptions**:
- HS hierarchy based on code length: 9-digit vs 10-digit commodity codes
- U.Summary code replacements: 'S004 ' → 'Used', 'S003 '/'S009 ' → 'Other'
- Weight calculation: (HS value within BEA) / (total BEA value)

**Outputs**:
- `aggregated_data/country_{level}/`: BEA aggregated data
- `hs_weights/{level}/`: HS compositional weights
- `bea_hs_section_weights.json`: Final JSON for analysis

### 05_Trade_weights.py
**Purpose**: Creates country-specific trade weights for tariff analysis

**Key Functions**:
- Calculates direct weights (global denominators)
- Calculates indirect weights (regional denominators)
- Validates weights sum to 1.0

**Key Assumptions**:
- Regional definitions:
  - Single countries: CAN, MEX, CHN, JPN (indirect weight = 1.0)
  - Europe: All European countries
  - RoAsia: Asia + Oceania excluding China/Japan
  - RoWorld: All other countries
- Weight tolerance: 0.0001 for validation

**Outputs**:
- `{level}_trade_weights.csv`: Detailed weight data
- `trade_weights.json`: Final weights in direct/indirect structure

## Key Configuration Points

### Modifying Regional Definitions (05_Trade_weights.py)
```python
# Lines 35-46: assign_region function
def assign_region(row):
    iso3 = row['iso3']
    continent = row['continent']
    
    if iso3 in ['CAN', 'MEX', 'CHN', 'JPN']:  # Modify single-country regions
        return iso3
    elif continent == 'Europe':
        return 'Europe'
    # ... modify other regional assignments
```

### Changing Hierarchical Matching Levels (01_Schott_Data_Compiler.py)
```python
# Lines 579-616: HIERARCHICAL_HS matching
# Modify the HS trimming levels in hierarchical_hs_match function
```

### Adjusting BEA Code Replacements (04_Aggregate_BEA_and_HS.py)
```python
# Line 24: U.Summary code replacements
usummary_df['usummary_code'] = usummary_df['usummary_code'].replace({
    'S004 ': 'Used', 
    'S003 ': 'Other', 
    'S009 ': 'Other'
})
```

## Running the Pipeline

1. Update `data_paths.json` with your file paths
2. Run scripts in sequence: 01 → 02 → 03 → 04 → 05
3. Check validation files in `validations/` folders
4. Final outputs will be in `data/final/`

## Validation

Each script creates validation files to verify data quality:
- **01**: NAICS mapping validation and method breakdowns
- **02**: BEA matching success rates and hierarchy validation
- **03**: Data quality checks and mapping coverage
- **04**: Country-level total preservation across aggregations
- **05**: Weight sum validation (must equal 1.0)

## Output Structure

```
data/
├── working/
│   ├── 01_Schott_Data_Compiler/
│   ├── 02_HS_to_Naics_to_BEA/
│   ├── 03_Map_country_trade_data/
│   ├── 04_Aggregate_BEA_and_HS/
│   └── 05_Trade_weights/
├── final/
│   ├── bea_hs_section_weights.json
│   └── trade_weights.json
└── validations/
    ├── 01_Schott_Data_Compiler/
    ├── 02_HS_to_Naics_to_BEA/
    ├── 03_Map_country_trade_data/
    ├── 04_Aggregate_BEA_and_HS/
    └── 05_Trade_weights/
```

## Key Dependencies

- `pandas`: Data manipulation
- `country_converter`: ISO3 country codes and continent mapping
- `json`: Configuration file handling
- `os`: File path operations

## Notes

- Scripts must be run in sequence due to dependencies
- Validation files should be checked after each step
- Final JSON files are optimized for downstream tariff analysis
- Regional definitions can be modified in `05_Trade_weights.py`
- HS hierarchy logic handles both 9-digit (i.e. missing leading 0's) and 10-digit commodity codes