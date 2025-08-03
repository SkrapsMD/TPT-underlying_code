# Aggregate trade data by country and NAICS codes, then map to BEA codes
# This script processes the merged trade data files, aggregates by country and NAICS,
# and applies hierarchical matching to map both naics_2024 and naics_2022_mapped to BEA codes

import json
import os
import pandas as pd
import glob

# Manual BEA mappings for specific HS codes that don't map through hierarchical matching
# These are based on the unmapped analysis and provide direct BEA code assignments
MANUAL_BEA_MAPPINGS = {
    # Automotive HS codes (currently mapping to NAICS 336110) -> BEA 336111
    '8703220110': '336111',  # Passenger cars
    '8703800060': '336111',  # Passenger cars  
    '8703800020': '336111',  # Passenger cars
    
    # Additional automotive HS codes from comparison analysis that should map to automotive not noncomparable
    '8703800080': '336111',  # Passenger cars - $715M (consolidated_naics in original)
    '8703220190': '336111',  # Passenger cars - $40.8M (consolidated_naics in original)
    '8703800045': '336111',  # Passenger cars - $374K (consolidated_naics in original)
    
    # Battery HS codes (currently mapping to NAICS 335910) -> BEA 335911
    '8506100090': '335911',  # Primary batteries
    '8506500090': '335911',  # Primary batteries
    '8506500010': '335911',  # Primary batteries
    '8506401010': '335911',  # Primary batteries
    '8506800090': '335911',  # Primary batteries
    '8506600010': '335911',  # Primary batteries
    '8506100010': '335911',  # Primary batteries
    '8506600090': '335911',  # Primary batteries
    '8506401090': '335911',  # Primary batteries
    '8506800010': '335911'   # Primary batteries
}

# Load data paths configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
data_paths_file = os.path.join(script_dir, '..', '..', 'data_paths.json')

with open(data_paths_file, 'r') as f:
    data_paths = json.load(f)

# Set up paths
input_path = os.path.join(data_paths['base_paths']['underlying_data_root'], 
                         'data', 'working', 'Alternative_Census_Mapping', 'combined_data')

bea_mapping_path = os.path.join(data_paths['base_paths']['underlying_data_root'], 
                               'data', 'working', '02_HS_to_Naics_to_BEA', '01_BEA_naics_mapping.csv')

output_path = os.path.join(data_paths['base_paths']['underlying_data_root'], 
                          'data', 'working', 'Alternative_Census_Mapping')

def hierarchical_match(naics_code, bea_naics_set):
    """
    Match NAICS code to BEA NAICS by progressively trimming digits
    Returns (matched_bea_code, match_level) or (None, None) if no match
    """
    # Handle special cases for noncomparable imports
    if naics_code in ['910000', '930000', '980000', '990000']:
        return 'S00300', 'noncomparable_imports'
    
    # Clean the NAICS code (remove X wildcards for matching)
    clean_code = str(naics_code).replace('X', '').strip()
    
    if not clean_code or clean_code == 'nan':
        return None, None
    
    # Try exact match first
    if clean_code in bea_naics_set:
        return clean_code, 'exact'
    
    # Progressive trimming from 6 digits down to 2
    for level in range(len(clean_code) - 1, 1, -1):
        trimmed = clean_code[:level]
        if trimmed in bea_naics_set:
            return trimmed, f'level_{level}'
    
    return None, None

def load_bea_mapping_data(bea_mapping_path):
    """Load BEA NAICS mapping data"""
    print(f"Loading BEA mapping data from: {bea_mapping_path}")
    
    try:
        bea_df = pd.read_csv(bea_mapping_path, dtype=str)
        
        # Clean up any whitespace
        for col in bea_df.columns:
            if bea_df[col].dtype == 'object':
                bea_df[col] = bea_df[col].str.strip()
        
        # Get unique NAICS codes for hierarchical matching
        bea_naics_codes = set(bea_df['naics'].dropna().unique())
        
        # Create NAICS-to-BEA code lookup dictionary
        naics_to_bea = {}
        for _, row in bea_df.iterrows():
            if pd.notna(row['naics']) and pd.notna(row['Code']):
                naics_to_bea[row['naics']] = row['Code']
        
        print(f"Loaded {len(bea_df):,} BEA mapping records")
        print(f"Found {len(bea_naics_codes):,} unique NAICS codes")
        print(f"Created {len(naics_to_bea):,} NAICS-to-BEA mappings")
        print(f"Columns: {list(bea_df.columns)}")
        
        return bea_df, bea_naics_codes, naics_to_bea
    except Exception as e:
        print(f"Error loading BEA mapping data: {e}")
        return None, None, None

def aggregate_trade_data(input_path):
    """Aggregate all trade data files by country and NAICS codes"""
    print(f"\nAggregating trade data from: {input_path}")
    
    # Find all merged trade data files
    trade_files = glob.glob(os.path.join(input_path, "*_with_naics_mapping.csv"))
    print(f"Found {len(trade_files)} trade data files:")
    for file in trade_files:
        print(f"  - {os.path.basename(file)}")
    
    if not trade_files:
        print("No trade data files found. Exiting.")
        return None
    
    # Load and concatenate all trade data
    all_data = []
    total_records = 0
    
    for trade_file in trade_files:
        continent_name = os.path.basename(trade_file).replace('_with_naics_mapping.csv', '')
        print(f"\n  Loading {continent_name} data...")
        
        try:
            df = pd.read_csv(trade_file, dtype={'hs_code': str, 'naics_2024': str, 'naics_2022_mapped': str})
            print(f"    Records: {len(df):,}")
            
            # Add continent identifier
            df['continent'] = continent_name
            all_data.append(df)
            total_records += len(df)
            
        except Exception as e:
            print(f"    Error loading {continent_name}: {e}")
    
    if not all_data:
        print("No data loaded successfully. Exiting.")
        return None
    
    # Concatenate all data
    print(f"\nCombining all data...")
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Total combined records: {len(combined_df):,}")
    
    # Apply manual BEA mappings before aggregation
    print(f"Applying manual BEA mappings...")
    combined_df['manual_bea_2024'] = combined_df['hs_code'].map(MANUAL_BEA_MAPPINGS)
    combined_df['manual_bea_2022'] = combined_df['hs_code'].map(MANUAL_BEA_MAPPINGS)
    
    manual_mapped_count = combined_df['manual_bea_2024'].notna().sum()
    manual_mapped_value = combined_df[combined_df['manual_bea_2024'].notna()]['impVal'].sum()
    
    print(f"  Manual mappings applied to {manual_mapped_count:,} records")
    print(f"  Import value with manual mappings: ${manual_mapped_value:,.0f}")
    
    # Aggregate by Country, naics_2024, and naics_2022_mapped
    print(f"Aggregating by Country and NAICS codes...")
    
    # Group by and sum import values, keeping manual BEA mappings
    aggregated_df = combined_df.groupby(['Country', 'naics_2024', 'naics_2022_mapped'], as_index=False).agg({
        'impVal': 'sum',
        'continent': 'first',  # Keep the continent info
        'hs_code': 'count',    # Count number of HS codes contributing to this aggregation
        'manual_bea_2024': 'first',  # Keep manual BEA mappings (should be consistent within groups)
        'manual_bea_2022': 'first'   # Keep manual BEA mappings (should be consistent within groups)
    })
    
    # Rename the hs_code count column
    aggregated_df = aggregated_df.rename(columns={'hs_code': 'hs_code_count'})
    
    print(f"Aggregated to {len(aggregated_df):,} country-NAICS combinations")
    print(f"Unique countries: {aggregated_df['Country'].nunique():,}")
    print(f"Unique naics_2024 codes: {aggregated_df['naics_2024'].nunique():,}")
    print(f"Unique naics_2022_mapped codes: {aggregated_df['naics_2022_mapped'].nunique():,}")
    print(f"Total import value: ${aggregated_df['impVal'].sum():,.0f}")
    
    return aggregated_df

def apply_bea_mapping(aggregated_df, bea_naics_codes, naics_to_bea):
    """Apply hierarchical BEA mapping to both NAICS columns, with manual override"""
    print(f"\nApplying BEA mapping to NAICS codes...")
    
    # Initialize result columns
    aggregated_df['bea_2024'] = None
    aggregated_df['bea_2024_match_level'] = None
    aggregated_df['bea_2022'] = None
    aggregated_df['bea_2022_match_level'] = None
    
    # Apply hierarchical matching to naics_2024 (with manual override)
    print("  Mapping naics_2024 to BEA codes...")
    manual_2024_count = 0
    for idx, row in aggregated_df.iterrows():
        # Check for manual mapping first
        if pd.notna(row['manual_bea_2024']):
            aggregated_df.at[idx, 'bea_2024'] = row['manual_bea_2024']
            aggregated_df.at[idx, 'bea_2024_match_level'] = 'manual_mapping'
            manual_2024_count += 1
        else:
            # Use hierarchical matching: NAICS -> NAICS -> BEA
            naics_2024 = row['naics_2024']
            if pd.notna(naics_2024) and str(naics_2024).strip():
                # Step 1: Find matching NAICS code using hierarchical matching
                matched_naics, match_level = hierarchical_match(naics_2024, bea_naics_codes)
                if matched_naics:
                    # Step 2: Convert matched NAICS to BEA code
                    if matched_naics == 'S00300':
                        # Special case: S00300 is already a BEA code, not a NAICS code
                        bea_code = 'S00300'
                    else:
                        bea_code = naics_to_bea.get(matched_naics)
                    aggregated_df.at[idx, 'bea_2024'] = bea_code
                    aggregated_df.at[idx, 'bea_2024_match_level'] = match_level
    
    # Apply hierarchical matching to naics_2022_mapped (with manual override)
    print("  Mapping naics_2022_mapped to BEA codes...")
    manual_2022_count = 0
    for idx, row in aggregated_df.iterrows():
        # Check for manual mapping first
        if pd.notna(row['manual_bea_2022']):
            aggregated_df.at[idx, 'bea_2022'] = row['manual_bea_2022']
            aggregated_df.at[idx, 'bea_2022_match_level'] = 'manual_mapping'
            manual_2022_count += 1
        else:
            # Use hierarchical matching: NAICS -> NAICS -> BEA
            naics_2022 = row['naics_2022_mapped']
            if pd.notna(naics_2022) and str(naics_2022).strip():
                # Step 1: Find matching NAICS code using hierarchical matching
                matched_naics, match_level = hierarchical_match(naics_2022, bea_naics_codes)
                if matched_naics:
                    # Step 2: Convert matched NAICS to BEA code
                    if matched_naics == 'S00300':
                        # Special case: S00300 is already a BEA code, not a NAICS code
                        bea_code = 'S00300'
                    else:
                        bea_code = naics_to_bea.get(matched_naics)
                    aggregated_df.at[idx, 'bea_2022'] = bea_code
                    aggregated_df.at[idx, 'bea_2022_match_level'] = match_level
    
    # Analyze mapping results
    bea_2024_mapped = aggregated_df['bea_2024'].notna().sum()
    bea_2022_mapped = aggregated_df['bea_2022'].notna().sum()
    total_records = len(aggregated_df)
    
    print(f"  BEA mapping results:")
    print(f"    naics_2024 -> BEA: {bea_2024_mapped:,}/{total_records:,} ({bea_2024_mapped/total_records*100:.1f}%)")
    print(f"    naics_2022_mapped -> BEA: {bea_2022_mapped:,}/{total_records:,} ({bea_2022_mapped/total_records*100:.1f}%)")
    print(f"    Manual mappings applied: {manual_2024_count:,} (2024), {manual_2022_count:,} (2022)")
    
    return aggregated_df

def analyze_discrepancies(aggregated_df):
    """Analyze and flag discrepancies between the two BEA mappings"""
    print(f"\nAnalyzing discrepancies between BEA mappings...")
    
    # Create discrepancy flags
    aggregated_df['mapping_discrepancy'] = 'none'
    
    # Check different scenarios
    both_mapped = (aggregated_df['bea_2024'].notna()) & (aggregated_df['bea_2022'].notna())
    only_2024_mapped = (aggregated_df['bea_2024'].notna()) & (aggregated_df['bea_2022'].isna())
    only_2022_mapped = (aggregated_df['bea_2024'].isna()) & (aggregated_df['bea_2022'].notna())
    neither_mapped = (aggregated_df['bea_2024'].isna()) & (aggregated_df['bea_2022'].isna())
    
    # For records where both are mapped, check if they map to different BEA codes
    both_mapped_different = both_mapped & (aggregated_df['bea_2024'] != aggregated_df['bea_2022'])
    both_mapped_same = both_mapped & (aggregated_df['bea_2024'] == aggregated_df['bea_2022'])
    
    # Assign discrepancy types
    aggregated_df.loc[both_mapped_same, 'mapping_discrepancy'] = 'both_mapped_same'
    aggregated_df.loc[both_mapped_different, 'mapping_discrepancy'] = 'both_mapped_different'
    aggregated_df.loc[only_2024_mapped, 'mapping_discrepancy'] = 'only_2024_mapped'
    aggregated_df.loc[only_2022_mapped, 'mapping_discrepancy'] = 'only_2022_mapped'
    aggregated_df.loc[neither_mapped, 'mapping_discrepancy'] = 'neither_mapped'
    
    # Print discrepancy analysis
    discrepancy_counts = aggregated_df['mapping_discrepancy'].value_counts()
    total_records = len(aggregated_df)
    
    print(f"  Discrepancy analysis:")
    for discrepancy_type, count in discrepancy_counts.items():
        percentage = (count / total_records) * 100
        print(f"    {discrepancy_type}: {count:,} ({percentage:.1f}%)")
    
    # Calculate import value impacts
    print(f"\n  Import value by discrepancy type:")
    discrepancy_value = aggregated_df.groupby('mapping_discrepancy')['impVal'].sum()
    total_value = aggregated_df['impVal'].sum()
    
    for discrepancy_type, value in discrepancy_value.items():
        percentage = (value / total_value) * 100
        print(f"    {discrepancy_type}: ${value:,.0f} ({percentage:.1f}%)")
    
    # Show examples of different BEA mappings
    different_mappings = aggregated_df[aggregated_df['mapping_discrepancy'] == 'both_mapped_different']
    if len(different_mappings) > 0:
        print(f"\n  Examples of different BEA mappings:")
        sample_different = different_mappings[['Country', 'naics_2024', 'naics_2022_mapped', 
                                            'bea_2024', 'bea_2022', 'impVal']].head(5)
        for _, row in sample_different.iterrows():
            print(f"    {row['Country']}: {row['naics_2024']}→{row['bea_2024']} vs {row['naics_2022_mapped']}→{row['bea_2022']} (${row['impVal']:,.0f})")
    
    return aggregated_df

def main():
    """Main execution function"""
    print("=" * 70)
    print("AGGREGATING TRADE DATA AND MAPPING TO BEA CODES")
    print("=" * 70)
    
    # Load BEA mapping data
    bea_df, bea_naics_codes, naics_to_bea = load_bea_mapping_data(bea_mapping_path)
    if bea_df is None or bea_naics_codes is None or naics_to_bea is None:
        print("Failed to load BEA mapping data. Exiting.")
        return
    
    # Aggregate trade data
    aggregated_df = aggregate_trade_data(input_path)
    if aggregated_df is None:
        print("Failed to aggregate trade data. Exiting.")
        return
    
    # Apply BEA mapping
    aggregated_df = apply_bea_mapping(aggregated_df, bea_naics_codes, naics_to_bea)
    
    # Analyze discrepancies
    aggregated_df = analyze_discrepancies(aggregated_df)
    
    # Validation: Check import value totals
    print(f"\n" + "=" * 70)
    print("VALIDATION: IMPORT VALUE TOTALS")
    print("=" * 70)
    
    total_original_value = aggregated_df['impVal'].sum()
    total_bea_2022_mapped_value = aggregated_df[aggregated_df['bea_2022'].notna()]['impVal'].sum()
    total_bea_2024_mapped_value = aggregated_df[aggregated_df['bea_2024'].notna()]['impVal'].sum()
    
    print(f"Original total import value: ${total_original_value:,.0f}")
    print(f"Import value with BEA mapping via 2022 NAICS: ${total_bea_2022_mapped_value:,.0f}")
    print(f"Import value with BEA mapping via 2024 NAICS: ${total_bea_2024_mapped_value:,.0f}")
    
    # Calculate percentages
    pct_2022_mapped = (total_bea_2022_mapped_value / total_original_value) * 100
    pct_2024_mapped = (total_bea_2024_mapped_value / total_original_value) * 100
    
    print(f"\nCoverage by import value:")
    print(f"  2022 NAICS → BEA mapping: {pct_2022_mapped:.1f}%")
    print(f"  2024 NAICS → BEA mapping: {pct_2024_mapped:.1f}%")
    
    # Check for any discrepancies
    value_lost_2022 = total_original_value - total_bea_2022_mapped_value
    value_lost_2024 = total_original_value - total_bea_2024_mapped_value
    
    if value_lost_2022 > 0:
        print(f"  Import value not mapped via 2022 NAICS: ${value_lost_2022:,.0f} ({(value_lost_2022/total_original_value)*100:.1f}%)")
    if value_lost_2024 > 0:
        print(f"  Import value not mapped via 2024 NAICS: ${value_lost_2024:,.0f} ({(value_lost_2024/total_original_value)*100:.1f}%)")
    
    # Save results
    print(f"\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    # Create output directory if needed
    os.makedirs(output_path, exist_ok=True)
    
    # Save main aggregated data with BEA mappings
    main_output_path = os.path.join(output_path, '02_aggregated_with_bea_mapping.csv')
    aggregated_df.to_csv(main_output_path, index=False)
    print(f"Main results saved: {main_output_path}")
    print(f"  Records: {len(aggregated_df):,}")
    
    # Save discrepancy analysis
    discrepancy_output_path = os.path.join(output_path, '03_bea_mapping_discrepancies.csv')
    discrepancy_df = aggregated_df[aggregated_df['mapping_discrepancy'] != 'both_mapped_same'].copy()
    discrepancy_df = discrepancy_df.sort_values(['mapping_discrepancy', 'impVal'], ascending=[True, False])
    discrepancy_df.to_csv(discrepancy_output_path, index=False)
    print(f"Discrepancy analysis saved: {discrepancy_output_path}")
    print(f"  Records with discrepancies: {len(discrepancy_df):,}")
    
    # Save summary statistics
    summary_stats = {
        'total_country_naics_combinations': len(aggregated_df),
        'total_import_value': aggregated_df['impVal'].sum(),
        'unique_countries': aggregated_df['Country'].nunique(),
        'unique_naics_2024': aggregated_df['naics_2024'].nunique(),
        'unique_naics_2022': aggregated_df['naics_2022_mapped'].nunique(),
        'unique_bea_2024': aggregated_df['bea_2024'].nunique(),
        'unique_bea_2022': aggregated_df['bea_2022'].nunique(),
        'bea_2024_mapping_rate': (aggregated_df['bea_2024'].notna().sum() / len(aggregated_df)) * 100,
        'bea_2022_mapping_rate': (aggregated_df['bea_2022'].notna().sum() / len(aggregated_df)) * 100,
        'discrepancy_rate': (len(discrepancy_df) / len(aggregated_df)) * 100
    }
    
    summary_df = pd.DataFrame([summary_stats])
    summary_output_path = os.path.join(output_path, '04_aggregation_summary.csv')
    summary_df.to_csv(summary_output_path, index=False)
    print(f"Summary statistics saved: {summary_output_path}")
    
    print(f"\nProcessing complete!")
    print(f"Final dataset: {len(aggregated_df):,} country-NAICS combinations")
    print(f"Total import value: ${aggregated_df['impVal'].sum():,.0f}")
    print(f"BEA mapping success: {summary_stats['bea_2024_mapping_rate']:.1f}% (2024), {summary_stats['bea_2022_mapping_rate']:.1f}% (2022)")
    print(f"Discrepancy rate: {summary_stats['discrepancy_rate']:.1f}%")

if __name__ == "__main__":
    main()