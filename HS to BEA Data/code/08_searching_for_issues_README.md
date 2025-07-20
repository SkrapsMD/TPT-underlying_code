# 08_searching_for_issues.py - BEA Mapping Issue Explorer

## Purpose
This tool investigates potential mapping inconsistencies that may contribute to discrepancies between our HS-to-BEA trade mappings and TiVA benchmark data.

## Key Problem Addressed
We perform hierarchical mapping in **two stages**:
1. **Stage 1** (01_Schott_Data_Compiler.py): HS codes → NAICS codes
2. **Stage 2** (03_Map_country_trade_data.py): HS codes → BEA detail codes (with strength scores)

This dual-stage process can create inconsistencies where an HS code gets different mappings through different pathways.

## Main Functions

### 1. `find_problematic_bea_codes(min_discrepancy_pct=30, top_n=10)`
Identifies BEA codes with the largest discrepancies from TiVA data using the results from 07_TiVA_Import_Values_Comparison.py.

**Example:**
```python
explorer = BEAMappingExplorer()
problematic = explorer.find_problematic_bea_codes(min_discrepancy_pct=50, top_n=5)
```

### 2. `explore_bea_code(usummary_code, show_weak_only=False, strength_threshold=0.8)`
Analyzes all HS codes currently mapped to a specific BEA U.Summary code.

**Shows:**
- HS codes from both mapping stages
- Mapping consistency between stages
- Trade values by region
- Weak mappings (strength < threshold)
- Alternative mapping possibilities

**Example:**
```python
result = explorer.explore_bea_code('334X')  # Computer and electronics
weak_only = explorer.explore_bea_code('334X', show_weak_only=True, strength_threshold=0.8)
```

### 3. `comprehensive_analysis(bea_code)`
Runs a complete analysis combining mapping exploration with TiVA discrepancy context.

**Example:**
```python
analysis = explorer.comprehensive_analysis('334X')
```

### 4. `simulate_mapping_change(usummary_code, hs_code, new_detail_code)`
Simulates the impact of changing an HS code's mapping on regional trade totals.

**Example:**
```python
explorer.simulate_mapping_change('334X', '1234567890', '335')
```

## Data Sources Used
- **HS-to-BEA mappings**: 03_complete_hs_to_bea_mapping.csv
- **Hierarchical matches**: 3_Hierarchical_Matches.csv (from stage 2)
- **Trade data**: Combined regional files from 03_Map_country_trade_data
- **BEA hierarchy**: 02_BEA_hierarchy.csv
- **TiVA discrepancies**: 03_large_discrepancies_table.csv

## Usage Workflow

1. **Find problematic codes:**
   ```python
   explorer = BEAMappingExplorer()
   problems = explorer.find_problematic_bea_codes()
   ```

2. **Analyze a specific code:**
   ```python
   analysis = explorer.comprehensive_analysis('334X')
   ```

3. **Look for weak mappings:**
   ```python
   weak_mappings = explorer.explore_bea_code('334X', show_weak_only=True)
   ```

4. **Test alternative mappings:**
   ```python
   # For each weak mapping, simulate changing to alternative
   explorer.simulate_mapping_change('334X', 'weak_hs_code', 'alternative_bea_detail')
   ```

## Key Insights from Initial Testing

1. **Services codes** (2211, 213, 512) show 100% discrepancies as expected (HS=0, TiVA>0)
2. **Electronics codes** (334X) show mapping but zero trade values, suggesting:
   - Potential HS code format mismatches
   - Missing trade data for these specific codes
   - Successful identification of mapping issues

### 5. `test_alternative_mappings(usummary_code, strength_threshold=0.8)` **[NEW!]**
Automatically tests alternative mappings for all weak codes in a BEA category and shows the complete impact analysis.

**Shows:**
- Which HS codes have weak mappings
- What their alternative BEA mappings would be  
- Regional trade value changes
- Potential impact on TiVA discrepancies

**Example:**
```python
# Test alternatives for BEA code with weak mappings
impact = explorer.test_alternative_mappings('3251')
# Shows: 3 weak HS codes, $237M trade value, moves to BEA 3254 (pharmaceuticals)
```

## Real Example Output

For BEA code **3251** (Basic chemical manufacturing):
- **3 weak HS codes** with mapping strength 0.875
- **$237.7 million** in trade values affected
- All alternatives point to **3254** (Pharmaceutical manufacturing)
- **TiVA Impact**: Small changes in discrepancy percentages (0.1-1.1%)

For BEA code **336111** (Automobile manufacturing):
- **0 weak mappings** - all automotive codes have strong mappings
- **$211.8 billion** total trade value with high confidence

## Updated Investigation Workflow

1. **Find problematic codes:**
   ```python
   problems = explorer.find_problematic_bea_codes(min_discrepancy_pct=30)
   ```

2. **Test alternative mappings automatically:**
   ```python
   impact = explorer.test_alternative_mappings('3251')
   ```

3. **Run comprehensive analysis:**
   ```python
   analysis = explorer.comprehensive_analysis('3251')  # Includes alternative testing
   ```

4. **Evaluate TiVA impact:**
   - Review discrepancy changes shown in the output
   - Determine if alternative mappings reduce overall discrepancies
   - Consider implementing changes for codes with significant improvements

## Key Insights

- **Weak mappings are relatively rare** (32 out of thousands of HS codes)
- **Alternative mappings can move substantial trade values** (up to hundreds of millions)
- **TiVA discrepancy improvements are typically small** (0.1-1% changes)
- **Some BEA categories have no weak mappings** (like automobiles - very clean data)

This tool provides complete end-to-end analysis for investigating and quantifying the impact of hierarchical mapping uncertainties on TiVA comparison results.