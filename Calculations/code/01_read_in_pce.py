"""
01_read_in_pce.py

This script processes PCE (Personal Consumption Expenditure) bridge data to create normalized import shares 
for TiVA calculations. It bridges 2017 Detail-level PCE data to 2023 U.Summary-level data by applying 
growth rates and validates the consistency of the transformation.

INPUTS:
- pcebridge_2017_detail.xlsx: 2017 Detail-level PCE data with producer/purchaser prices
- pcebridge_97_23.xlsx: Summary-level PCE data for growth rate calculations (1997-2023)
- BEA hierarchy crosswalk from HS to BEA Data project (Detail -> U.Summary -> Summary mapping)

OUTPUTS:
- C.csv: Normalized U.Summary shares for TiVA calculations (saved to calculations_dir/TiVA/138/2023/)
- Multiple validation files in validations/01_read_in_pce/: -- TOO MANY VALIDATIONS... more like intermediates to just check that things make sense
  - 00_detail_vs_usum_validation_2017.csv: Validates Detail vs U.Summary totals
  - 01-04_*_pivot_table_*.csv: Pivot tables for both price types and aggregation levels
  - 05_summary_PCE_growth_rates_2017_2023.csv: Growth rates used for projection
  - 06_NOT_NORMALIZED_usummary_pivot_table_2017.csv: U.Summary data before normalization
  - 07_summary_level_comparison_*.csv: Comparison of constructed vs PCE bridge Summary data
  - 08_nipa_distribution_changes_2017.csv: NIPA line share changes validation

PROCESS:
1. Load 2017 Detail PCE data and create pivot tables by Detail and U.Summary codes
2. Calculate Summary-level growth rates from 2017 to 2023 using purchaser prices
3. Apply growth rates to U.Summary data to project to 2023 values
4. Handle special codes (S003+S009->Other, S004->Used) and reorder by TiVA ordering
5. Normalize by total purchaser price to create expenditure shares
6. Validate consistency at each step and identify sources of any data loss

VALIDATION:
- Compares totals across Detail/U.Summary/Summary levels to identify mapping losses
- Validates NIPA line share consistency between 2017 and 2023 projections
- Tracks the ~$300B (~2%) difference between projected and target PCE values -- Still haven't figured out this one yet... But because the distributios are unchanged I'm more prone to think its not a big deal.
"""

import os
import json
import numpy as np
import pandas as pd
# Load data paths and set up standard directory variables
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)  # Go up one level to Calculations/
data_paths_file = os.path.join(project_dir, "data_paths.json")
with open(data_paths_file, 'r') as f:
    data_paths = json.load(f)
# Set up clean directory variables using project_root from config
project_root = data_paths['base_paths']['project_root']
raw_data_dir = os.path.join(project_root, data_paths['base_paths']['raw_data'])
working_data_dir = os.path.join(project_root, data_paths['base_paths']['working_data'])
calculations_dir = os.path.join(working_data_dir, "Components for Calculations")
validations_dir = os.path.join(project_root, data_paths['base_paths']['validations'])
hs_to_bea_data_dir = os.path.join(project_root, data_paths['base_paths']['hs_to_bea_data'])



# Part 1: Create the 2017 Detail and Underlying Summary pivot tables
year = 2017
year_to = 2023
data = pd.read_excel(os.path.join(raw_data_dir, 'pcebridge_2017_detail.xlsx'), sheet_name = f'{year}', header = 4)
data = data.rename(columns = {
    'Commodity Code': 'Detail', 
    'Commodity Description':'Detail Description', 
    'Unnamed: 4': 'prodPrice',
    'Unnamed: 5': 'transpCost', 
    'Wholesale': 'wholesaleCost',
    'Retail': 'retailCost',
    'Unnamed: 8': 'purchPrice'
})

# Create pivot of the Purchaser Price value and the Producer Price value

pivot_tables_detail = {}
for price_type in ['purchPrice', 'prodPrice']:
    data[price_type] = pd.to_numeric(data[price_type], errors='coerce')
    pivot_tables_detail[price_type] = data.pivot_table(
        index='Detail',
        columns='NIPA Line',
        values=price_type,
        aggfunc='sum'
    ).fillna(0)
    if price_type == 'purchPrice':
        output_file = os.path.join(validations_dir, f"01_read_in_pce/02_{price_type}_pivot_table_DETAIL_{year}.csv")
    elif price_type == 'prodPrice':
        output_file = os.path.join(validations_dir, f"01_read_in_pce/01_{price_type}_pivot_table_DETAIL_{year}.csv")
    pivot_tables_detail[price_type].to_csv(output_file)
# These outputs are used as a comparison with the direct BEA data -- which comes at the detail level in 2017. So checking these, it looks like we have the pivot working correctly.



#NOTE: We prefer the producer price pivot table (we believe this is the additive markup assumption). If we want to change the assumption, we can change this here. -- This is the Detail Level Table
main_table_detail = pivot_tables_detail['prodPrice']

# Create a crosswalk from Detail to U.Summary level.
# crosswalk = pd.read_csv(os.path.join(hs_to_bea_data_dir, 'data', 'working', '02_HS_to_Naics_to_BEA', '02_BEA_hierarchy.csv')) # This is an older version of the crosswalk, we will use the one in the hs_to_bea_data_dir
crosswalk = pd.read_csv(os.path.join(hs_to_bea_data_dir, 'data', 'working', '02_HS_to_Naics_to_BEA', '02_BEA_hierarchy.csv'))
crosswalk_usum = crosswalk[['Detail','U.Summary']].copy()
crosswalk_usum['Detail'] = crosswalk_usum['Detail'].astype(str)
data['Detail'] = data['Detail'].astype(str)

crosswalk_sum = crosswalk[['U.Summary','Summary']].drop_duplicates()
crosswalk_sum['U.Summary'] = crosswalk_sum['U.Summary'].astype(str)
crosswalk_sum['Summary'] = crosswalk_sum['Summary'].astype(str)

data_with_usum = pd.merge(data, crosswalk_usum, how = 'left', on = 'Detail')

pivot_tables_usum = {}
for price_type in ['purchPrice', 'prodPrice']:
    data_with_usum[price_type] = pd.to_numeric(data_with_usum[price_type], errors='coerce')
    pivot_tables_usum[price_type] = data_with_usum.pivot_table(
        index='U.Summary',
        columns='NIPA Line',
        values=price_type,
        aggfunc='sum'
    ).fillna(0)
    if price_type == 'purchPrice':
        output_file = os.path.join(validations_dir, f"01_read_in_pce/04_{price_type}_pivot_table_USUM_{year}.csv")
    elif price_type == 'prodPrice':
        output_file = os.path.join(validations_dir, f"01_read_in_pce/03_{price_type}_pivot_table_USUM_{year}.csv")
    pivot_tables_usum[price_type].to_csv(output_file)

#NOTE: We prefer the producer price pivot table (we believe this is the additive markup assumption). If we want to change the assumption, we can change this here. -- This is the Underlying Summary Level Table
main_table_usum = pivot_tables_usum['prodPrice']

# VALIDATION: Check that Detail level sums match U.Summary level sums for both price types
print("=== VALIDATION: Detail vs U.Summary Level Totals ===")
for price_type in ['purchPrice', 'prodPrice']:
    detail_total = pivot_tables_detail[price_type].values.sum()
    usum_total = pivot_tables_usum[price_type].values.sum()
    difference = detail_total - usum_total
    pct_difference = (difference / detail_total) * 100 if detail_total != 0 else 0
    
    print(f"{price_type.upper()} - Detail Total: ${detail_total*1000000:,.2f}")
    print(f"{price_type.upper()} - U.Summary Total: ${usum_total*1000000:,.2f}")
    print(f"{price_type.upper()} - Difference: ${difference*1000000:,.2f} ({pct_difference:.4f}%)")
    print("---")

# Save validation results
validation_results = []
for price_type in ['purchPrice', 'prodPrice']:
    detail_total = pivot_tables_detail[price_type].values.sum()
    usum_total = pivot_tables_usum[price_type].values.sum()
    difference = detail_total - usum_total
    pct_difference = (difference / detail_total) * 100 if detail_total != 0 else 0
    
    validation_results.append({
        'price_type': price_type,
        'detail_total': detail_total,
        'usum_total': usum_total,
        'difference': difference,
        'pct_difference': pct_difference
    })

validation_df = pd.DataFrame(validation_results)
validation_df.to_csv(os.path.join(validations_dir, f"01_read_in_pce/00_detail_vs_usum_validation_{year}.csv"), index=False)


# Part 2: Calculate the Summary Level Growth rates from 2017 to 2023 --- We use the Purchaser Price growth rate (note, purchaser prices sum to PCE)
def calculate_summary_growthRates(year_from=year, year_to=year_to, type ='purchPrice', use_detail_to_summary=True):
    # Load data for the specified years
    pce_data = pd.ExcelFile(os.path.join(raw_data_dir, 'pcebridge_97_23.xlsx'))
    pce_from = pce_data.parse(sheet_name=str(year_from), header=4)
    pce_to = pce_data.parse(sheet_name=str(year_to), header=4)
    
    # Rename columns for both DataFrames
    pce_from = pce_from.rename(columns={
        'Commodity Code': 'Summary',
        'Unnamed: 4': 'prodPrice',
        'Unnamed: 8': 'purchPrice'
    })
    pce_to = pce_to.rename(columns={
        'Commodity Code': 'Summary',
        'Unnamed: 4': 'prodPrice',
        'Unnamed: 8': 'purchPrice'
    })
    
    # Calculate total purchaser prices for both years
    total_purchPrice_from = pce_from['purchPrice'].sum()
    total_purchPrice_to = pce_to['purchPrice'].sum()
    total_prodPrice_to = pce_to['prodPrice'].sum()
    total_prodPrice_from = pce_from['prodPrice'].sum()
    
    if use_detail_to_summary:
        # Use Detail->Summary mapping for the FROM year (maintains consistency with our constructed data)
        # Load the FROM year detail data
        detail_from = pd.read_excel(os.path.join(raw_data_dir, 'pcebridge_2017_detail.xlsx'), sheet_name=str(year_from), header=4)
        detail_from = detail_from.rename(columns={
            'Commodity Code': 'Detail',
            'Unnamed: 4': 'prodPrice',
            'Unnamed: 8': 'purchPrice'
        })
        detail_from['Detail'] = detail_from['Detail'].astype(str)
        
        # Create Detail->Summary crosswalk within function
        crosswalk_detail_to_summary_func = crosswalk[['Detail','Summary']].drop_duplicates().copy()
        crosswalk_detail_to_summary_func['Detail'] = crosswalk_detail_to_summary_func['Detail'].astype(str)
        crosswalk_detail_to_summary_func['Summary'] = crosswalk_detail_to_summary_func['Summary'].astype(str)
        
        # Create Detail->Summary mapping for FROM year
        detail_from_with_summary = pd.merge(detail_from, crosswalk_detail_to_summary_func, how='left', on='Detail')
        summary_from = detail_from_with_summary.groupby('Summary')[type].sum().reset_index()
        summary_from['Summary'] = summary_from['Summary'].astype(str)
        
        # Use direct PCE data for TO year (this is the target we want to grow to)
        pce_to['Summary'] = pce_to['Summary'].astype(str)
        summary_to = pce_to.groupby('Summary')[type].sum().reset_index()
    else:
        # Use direct PCE data for both years (original approach)
        pce_from['Summary'] = pce_from['Summary'].astype(str)
        pce_to['Summary'] = pce_to['Summary'].astype(str)
        summary_from = pce_from.groupby('Summary')[type].sum().reset_index()
        summary_to = pce_to.groupby('Summary')[type].sum().reset_index()
    
    summary_ratios = pd.merge(summary_from, summary_to, on='Summary', suffixes=('_from', '_to'))
    summary_ratios['Summary'] = summary_ratios['Summary'].astype(str)
    summary_ratios['growth_ratio'] = summary_ratios[f'{type}_to'] / summary_ratios[f'{type}_from']
    
    # Output the ratios
    suffix = "_detail_to_summary" if use_detail_to_summary else ""
    summary_ratios.to_csv(os.path.join(validations_dir, f"01_read_in_pce/05_summary_PCE_growth_rates_{year_from}_{year_to}{suffix}.csv"), index=False)
    return summary_ratios, total_purchPrice_from, total_purchPrice_to, total_prodPrice_to, total_prodPrice_from

summary_ratios, total_purchPrice_from, total_purchPrice_to, total_prodPrice_to, total_prodPrice_from = calculate_summary_growthRates(year_from=year, year_to=year_to, type = 'purchPrice', use_detail_to_summary=True)

# VALIDATION: Compare U.Summary totals with Summary totals from calculate_summary_growthRates
print("=== VALIDATION: U.Summary vs Summary Level Totals (2017) ===")
usum_purchPrice_total = pivot_tables_usum['purchPrice'].values.sum()
usum_prodPrice_total = pivot_tables_usum['prodPrice'].values.sum()

print(f"PURCHASER PRICE (2017) - U.Summary Total: ${usum_purchPrice_total*1000000:,.2f}")
print(f"PURCHASER PRICE (2017) - Summary Total: ${total_purchPrice_from*1000000:,.2f}")
purchPrice_difference = usum_purchPrice_total - total_purchPrice_from
purchPrice_pct_diff = (purchPrice_difference / usum_purchPrice_total) * 100 if usum_purchPrice_total != 0 else 0
print(f"PURCHASER PRICE (2017) - Difference: ${purchPrice_difference*1000000:,.2f} ({purchPrice_pct_diff:.4f}%)")
print("---")

print(f"PRODUCER PRICE (2017) - U.Summary Total: ${usum_prodPrice_total*1000000:,.2f}")
print(f"PRODUCER PRICE (2017) - Summary Total: ${total_prodPrice_from*1000000:,.2f}")
prodPrice_difference = usum_prodPrice_total - total_prodPrice_from
prodPrice_pct_diff = (prodPrice_difference / usum_prodPrice_total) * 100 if usum_prodPrice_total != 0 else 0
print(f"PRODUCER PRICE (2017) - Difference: ${prodPrice_difference*1000000:,.2f} ({prodPrice_pct_diff:.4f}%)")
print("---")

# VALIDATION: Create direct Detail -> Summary mapping and compare totals
print("=== VALIDATION: Detail -> Summary vs Summary Level Totals (2017) ===")
crosswalk_detail_to_summary = crosswalk[['Detail','Summary']].drop_duplicates().copy()
crosswalk_detail_to_summary['Detail'] = crosswalk_detail_to_summary['Detail'].astype(str)
crosswalk_detail_to_summary['Summary'] = crosswalk_detail_to_summary['Summary'].astype(str)

# Merge detail data directly with summary codes
data_with_summary = pd.merge(data, crosswalk_detail_to_summary, how='left', on='Detail')

# Create summary level pivot tables from detail data
pivot_tables_summary_from_detail = {}
for price_type in ['purchPrice', 'prodPrice']:
    data_with_summary[price_type] = pd.to_numeric(data_with_summary[price_type], errors='coerce')
    pivot_tables_summary_from_detail[price_type] = data_with_summary.pivot_table(
        index='Summary',
        columns='NIPA Line',
        values=price_type,
        aggfunc='sum'
    ).fillna(0)

# Compare totals
for price_type in ['purchPrice', 'prodPrice']:
    detail_to_summary_total = pivot_tables_summary_from_detail[price_type].values.sum()
    pce_bridge_total = total_purchPrice_from if price_type == 'purchPrice' else total_prodPrice_from
    
    print(f"{price_type.upper()} (2017) - Detail->Summary Total: ${detail_to_summary_total*1000000:,.2f}")
    print(f"{price_type.upper()} (2017) - PCE Bridge Total: ${pce_bridge_total*1000000:,.2f}")
    difference = detail_to_summary_total - pce_bridge_total
    pct_difference = (difference / detail_to_summary_total) * 100 if detail_to_summary_total != 0 else 0
    print(f"{price_type.upper()} (2017) - Difference: ${difference*1000000:,.2f} ({pct_difference:.4f}%)")
    print("---")

# Save detailed comparison by Summary code
for price_type in ['purchPrice', 'prodPrice']:
    # Get totals by Summary code from detail data
    detail_summary_totals = data_with_summary.groupby('Summary')[price_type].sum().reset_index()
    detail_summary_totals.columns = ['Summary', f'detail_to_summary_{price_type}']
    detail_summary_totals['Summary'] = detail_summary_totals['Summary'].astype(str)
    
    # Get totals by Summary code from PCE bridge data
    # We need to recreate the PCE bridge data here since it's not available in outer scope
    pce_data = pd.ExcelFile(os.path.join(raw_data_dir, 'pcebridge_97_23.xlsx'))
    pce_2017 = pce_data.parse(sheet_name=str(year), header=4)
    pce_2017 = pce_2017.rename(columns={
        'Commodity Code': 'Summary',
        'Unnamed: 4': 'prodPrice',
        'Unnamed: 8': 'purchPrice'
    })
    pce_2017['Summary'] = pce_2017['Summary'].astype(str)
    
    pce_bridge_summary = pce_2017.groupby('Summary')[price_type].sum().reset_index()
    pce_bridge_summary.columns = ['Summary', f'pce_bridge_{price_type}']
    pce_bridge_summary['Summary'] = pce_bridge_summary['Summary'].astype(str)
    
    # Merge and compare
    summary_comparison = pd.merge(detail_summary_totals, pce_bridge_summary, on='Summary', how='outer').fillna(0)
    summary_comparison['difference'] = summary_comparison[f'detail_to_summary_{price_type}'] - summary_comparison[f'pce_bridge_{price_type}']
    summary_comparison['pct_difference'] = (summary_comparison['difference'] / summary_comparison[f'detail_to_summary_{price_type}']) * 100
    
    # Save comparison
    summary_comparison.to_csv(os.path.join(validations_dir, f"01_read_in_pce/07_summary_level_comparison_{price_type}_{year}.csv"), index=False)
    
    # Print largest discrepancies
    print(f"=== LARGEST DISCREPANCIES - {price_type.upper()} (2017) ===")
    largest_discrepancies = summary_comparison.reindex(summary_comparison['difference'].abs().sort_values(ascending=False).index).head(10)
    for _, row in largest_discrepancies.iterrows():
        print(f"Summary {row['Summary']}: ${row['difference']*1000000:,.2f} ({row['pct_difference']:.2f}%)")
    print("---")
# Part 3: Apply the growth rates to create 2023 (or year_to) Underlying Summary pivot table by using the ratio from the summary level growth rates. 
usummary_pivot = main_table_usum.reset_index()
usummary_with_summary = pd.merge(usummary_pivot, crosswalk_sum, how='left', on='U.Summary')
usummary_with_ratios = pd.merge(usummary_with_summary, summary_ratios[['Summary', 'growth_ratio']], how='left', on='Summary')

# Apply the growth ratios to the NIPA columns. NOTE: The assumption here is that the growth rate applies consistently across all NIPA lines for a given Summary/Usummary -- No cell blocks, just the Summary ratio.
nipa_cols = [col for col in usummary_pivot.columns if col != 'U.Summary']
for col in nipa_cols:
    usummary_with_ratios[col] = usummary_with_ratios[col] * usummary_with_ratios['growth_ratio']

usummary_final = usummary_with_ratios[['U.Summary'] + nipa_cols].copy()
usummary_final.set_index('U.Summary', inplace=True)

# Part 4: Handle special rows and reorder according to TiVA ordering.... We want to bring in the TiVA ordering from a representative file we use in HS to BEA Data (Note, this file is really one we create later in this pipeline, but we bring it here for consistency.)
q_tiva = pd.read_csv(os.path.join(hs_to_bea_data_dir, 'data', 'raw', 'BEA_codes', 'q.csv'))
sum_order_tiva = q_tiva['U.Summary Code'].tolist()

# Handle Special Rows
usummary_final.index = usummary_final.index.str.strip()

# 1 Rename S004 to "Used" 
if 'S004' in usummary_final.index:
    usummary_final = usummary_final.rename(index={'S004': 'Used'})

# 2 Combine S003 and S009 into "Other" 
s003_row = usummary_final.loc['S003'] if 'S003' in usummary_final.index else 0 
s009_row = usummary_final.loc['S009'] if 'S009' in usummary_final.index else 0
usummary_final.loc['Other'] = s003_row + s009_row 

usummary_final = usummary_final.drop(index=['S003','S009'], errors='ignore')

# Apply the custom TiVA ordering 
sum_order_tiva = ['Other' if x in ['S003', 'S009'] else x for x in sum_order_tiva]
sum_order_tiva = list(dict.fromkeys(sum_order_tiva))  # Remove duplicates while preserving order

usummary_final = usummary_final.reindex(sum_order_tiva, fill_value=0)

# Save the non-normalized 2023 (or year_to) Underlying Summary Pivot Table 
usummary_final.to_csv(os.path.join(validations_dir, f"01_read_in_pce/06_NOT_NORMALIZED_usummary_pivot_table_{year}.csv"))

# See if we lose any PCE values from the growth rates being applied (i.e. see if the pre-application sum of PCE = the sum of all now...)
total_sum = usummary_final.values.sum()
print(f"Total sum of PCE values after applying growth rates: ${total_sum*1000000:,.2f}")
print(f"Total sum of PCE values before applying growth rates: ${total_prodPrice_to*1000000:,.2f}") 
print(f"Difference in sums: ${(total_sum - total_prodPrice_to)*1000000:,.2f} ({(total_sum - total_prodPrice_to) / total_prodPrice_to * 100:,.2f}%)")
""" -- There are about 300 Billion less in PCE ... Where is this coming from? This is not a small amount....
Total sum of PCE values after applying growth rates: $15,293,631,176,900.40
Total sum of PCE values before applying growth rates: $15,596,759,000,000.00
Difference in sums: $-303,127,823,099.60 (-1.94%)

So we validate below whether this is coming from distribution changes etc, and it doesn't look like it. This is a funky 2%. The BEA does say that rounding errors in their Summary level statistics can lead
to differences, but this is a bit big (granted, their statistics are published at the millions of dollars level so I can imagine how in the aggregation from 402 to 71 levels they may lost a few million, but this is quite 
large... I checked the distributions and it seems fine. Will return here later.)
"""
# Note: we want to compare this with the total PRODUCER price (despite the fact that we are using the purchaser prices to calculate the growth rates), because the total_sum is denominated in producer prices. 

# VALIDATION: Check NIPA line distributions between 2017 U.Summary and 2023 U.Summary data
print("=== VALIDATION: NIPA Line Share Changes (2017 vs 2023 U.Summary) ===")
# Get 2017 U.Summary data (before growth rates applied) and apply same special handling
usum_2017 = main_table_usum.copy()
# Apply same special handling to 2017 data for fair comparison
usum_2017.index = usum_2017.index.str.strip()
# 1 Rename S004 to "Used" 
if 'S004' in usum_2017.index:
    usum_2017 = usum_2017.rename(index={'S004': 'Used'})
# 2 Combine S003 and S009 into "Other" 
s003_row = usum_2017.loc['S003'] if 'S003' in usum_2017.index else 0 
s009_row = usum_2017.loc['S009'] if 'S009' in usum_2017.index else 0
usum_2017.loc['Other'] = s003_row + s009_row 
usum_2017 = usum_2017.drop(index=['S003','S009'], errors='ignore')
# Get 2023 U.Summary data (after growth rates applied and special handling)
usum_2023 = usummary_final.copy()  # This already has S003/S009 combined, S004 renamed, etc.
# Find common U.Summary codes and NIPA columns
common_codes = usum_2017.index.intersection(usum_2023.index)
common_nipa_cols = [col for col in usum_2017.columns if col in usum_2023.columns]
print(f"Comparing {len(common_codes)} U.Summary codes across {len(common_nipa_cols)} NIPA lines")
print("---")
# Calculate share distributions for each U.Summary code
distribution_changes = []
for code in common_codes:
    # 2017 shares
    usum_2017_row = usum_2017.loc[code, common_nipa_cols]
    usum_2017_total = usum_2017_row.sum()
    usum_2017_shares = usum_2017_row / usum_2017_total if usum_2017_total > 0 else usum_2017_row * 0
    # 2023 shares
    usum_2023_row = usum_2023.loc[code, common_nipa_cols]
    usum_2023_total = usum_2023_row.sum()
    usum_2023_shares = usum_2023_row / usum_2023_total if usum_2023_total > 0 else usum_2023_row * 0
    # Calculate share change (sum of absolute differences)
    share_change = abs(usum_2017_shares - usum_2023_shares).sum()
    # Create row with shares for each NIPA column
    row_data = {
        'code': code,
        'usum_2017_total': usum_2017_total,
        'usum_2023_total': usum_2023_total,
        'share_change': share_change
    }
    # Add individual NIPA shares
    for nipa_col in common_nipa_cols:
        row_data[f'share_2017_{nipa_col}'] = usum_2017_shares[nipa_col]
        row_data[f'share_2023_{nipa_col}'] = usum_2023_shares[nipa_col]
        row_data[f'share_diff_{nipa_col}'] = abs(usum_2017_shares[nipa_col] - usum_2023_shares[nipa_col])
    distribution_changes.append(row_data)
# Convert to DataFrame and sort by share change
dist_df = pd.DataFrame(distribution_changes)
dist_df = dist_df.sort_values('share_change', ascending=False)
# Print top 10 codes with largest share changes
print("TOP 10 U.SUMMARY CODES WITH LARGEST NIPA SHARE CHANGES:")
for i, row in dist_df.head(10).iterrows():
    print(f"{row['code']}: Share change = {row['share_change']:.4f}")
    print(f"  2017 Total: ${row['usum_2017_total']*1000000:,.2f}, 2023 Total: ${row['usum_2023_total']*1000000:,.2f}")
print("---")
# Calculate overall share similarity
overall_share_change = dist_df['share_change'].mean()
print(f"Average share change across all U.Summary codes: {overall_share_change:.4f}")
print(f"Codes with >10% share change: {len(dist_df[dist_df['share_change'] > 0.1])}")
print(f"Codes with >20% share change: {len(dist_df[dist_df['share_change'] > 0.2])}")
# If shares are identical, overall_share_change should be 0
if overall_share_change < 0.001:
    print("✓ NIPA shares are preserved - the $300B loss is likely elsewhere")
else:
    print("✗ NIPA shares are changing - this could explain the $300B loss")
# Save detailed results
dist_df.to_csv(os.path.join(validations_dir, f"01_read_in_pce/08_nipa_distribution_changes_{year}.csv"), index=False)
print("---")


# Final Step, normalize the producer prices denominated usummary with purchaser prices to 
# create the share of total expenditures on each BEA code contributes to each NIPA line

normalized_final = usummary_final/total_purchPrice_to
normalized_final = normalized_final.replace([np.inf, -np.inf, np.nan], 0)
# Check the rows: 
pivot_rows = normalized_final.index.tolist()
expected_rows = sum_order_tiva
missing_rows = [row for row in expected_rows if row not in pivot_rows]
if missing_rows:
    print(f"Warning: The following expected rows are missing: {missing_rows}")
else:
    print("All expected rows are present")

# Check for row order match
if pivot_rows == expected_rows:
    print("Row order matches expected TiVA ordering")
else:
    print("Row order doesn't match expected TiVA ordering - reordering...")
    normalized_final = normalized_final.reindex(expected_rows).fillna(0)


# We use this path in the calculations... NAMED for what it is.
normalized_final.to_csv(os.path.join(calculations_dir, f'TiVA','138',f'{year_to}','C.csv'))
