# Compare Census mapping data vs Schott mapping data

import json
import os
import pandas as pd
import glob

# Import the census mapping reader function from the main script
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Load data paths configuration
data_paths_file = os.path.join(script_dir, '..', '..', 'data_paths.json')

with open(data_paths_file, 'r') as f:
    data_paths = json.load(f)

# Set up paths
raw_data_base = data_paths['base_paths']['raw_data']
alt_census_path = os.path.join(raw_data_base, 'Alt_Census_Crosswalks')
alt_census_structure_path = os.path.join(alt_census_path, 'structure')

def read_census_mapping(year):
    """Read census mapping data for a given year by parsing structure file and reading fixed-width data"""
    # Parse structure file to get column specifications
    structure_file = os.path.join(alt_census_structure_path, f'imp-stru_{year}.txt')
    column_specs = []
    column_names = []
    with open(structure_file, 'r') as f:
        lines = f.readlines()
    # Parse column positions from structure file
    for line in lines:
        line = line.strip()
        if line and not line.startswith('-') and not line.startswith('CHARACTER'):
            parts = line.split()
            if len(parts) >= 2:
                char_position = parts[0]
                try:
                    if '-' in char_position:
                        # Parse character range (e.g., "1-10")
                        start, end = char_position.split('-')
                        start_pos = int(start) - 1  # Convert to 0-indexed
                        end_pos = int(end)
                    else:
                        # Parse single position (e.g., "261")
                        pos = int(char_position)
                        start_pos = pos - 1  # Convert to 0-indexed
                        end_pos = pos
                    column_specs.append((start_pos, end_pos))
                    column_names.append(parts[1])
                except ValueError:
                    continue
    # Read the data file
    imp_code_file = os.path.join(alt_census_path, f'imp-code_{year}.txt')
    df = pd.read_fwf(imp_code_file, colspecs=column_specs, names=column_names, 
                     dtype=str, encoding='latin-1')
    # Clean up whitespace and add year
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()
    df['YEAR'] = year
    return df

def read_schott_mapping(file_path):
    """Read Schott mapping data"""
    df = pd.read_csv(file_path, dtype=str)
    # Clean up any whitespace
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()
    return df

def compare_mappings(census_df, schott_df, year):
    """Compare Census and Schott mappings for a given year"""
    
    print(f"\n{'='*60}")
    print(f"COMPARING CENSUS vs SCHOTT MAPPINGS FOR {year}")
    print(f"{'='*60}")
    
    # Prepare Census data
    census_clean = census_df[['COMMODITY', 'NAICS']].copy()
    census_clean = census_clean.rename(columns={'COMMODITY': 'hs_code', 'NAICS': 'naics'})
    census_clean['hs_code'] = census_clean['hs_code'].astype(str).str.zfill(10)  # Ensure 10-digit format
    
    # Prepare Schott data (use the naicsX column which seems to be the updated version)
    schott_clean = schott_df[['commodity', 'naicsX']].copy()
    schott_clean = schott_clean.rename(columns={'commodity': 'hs_code', 'naicsX': 'naics'})
    schott_clean['hs_code'] = schott_clean['hs_code'].astype(str).str.zfill(10)  # Ensure 10-digit format
    
    print(f"Census data: {len(census_clean):,} HS codes")
    print(f"Schott data: {len(schott_clean):,} HS codes")
    
    # Find common HS codes
    census_hs_codes = set(census_clean['hs_code'])
    schott_hs_codes = set(schott_clean['hs_code'])
    
    common_hs_codes = census_hs_codes.intersection(schott_hs_codes)
    census_only = census_hs_codes - schott_hs_codes
    schott_only = schott_hs_codes - census_hs_codes
    
    print(f"\nHS Code Coverage:")
    print(f"  Common HS codes: {len(common_hs_codes):,}")
    print(f"  Census only: {len(census_only):,}")
    print(f"  Schott only: {len(schott_only):,}")
    print(f"  Coverage overlap: {len(common_hs_codes)/max(len(census_hs_codes), len(schott_hs_codes)):.1%}")
    
    if len(census_only) > 0:
        print(f"  Census-only examples: {list(census_only)[:5]}")
    if len(schott_only) > 0:
        print(f"  Schott-only examples: {list(schott_only)[:5]}")
    
    # Compare NAICS mappings for common HS codes
    census_mapping = dict(zip(census_clean['hs_code'], census_clean['naics']))
    schott_mapping = dict(zip(schott_clean['hs_code'], schott_clean['naics']))
    
    matching_mappings = 0
    different_mappings = 0
    mapping_differences = []
    
    for hs_code in common_hs_codes:
        census_naics = census_mapping[hs_code]
        schott_naics = schott_mapping[hs_code]
        
        if census_naics == schott_naics:
            matching_mappings += 1
        else:
            different_mappings += 1
            mapping_differences.append({
                'hs_code': hs_code,
                'census_naics': census_naics,
                'schott_naics': schott_naics
            })
    
    print(f"\nNAICS Mapping Comparison (for common HS codes):")
    print(f"  Matching mappings: {matching_mappings:,}")
    print(f"  Different mappings: {different_mappings:,}")
    print(f"  Mapping agreement: {matching_mappings/len(common_hs_codes):.1%}")
    
    # Show examples of different mappings
    if len(mapping_differences) > 0:
        print(f"\nExample mapping differences:")
        for i, diff in enumerate(mapping_differences[:10]):
            print(f"  HS {diff['hs_code']}: Census={diff['census_naics']}, Schott={diff['schott_naics']}")
        if len(mapping_differences) > 10:
            print(f"  ... and {len(mapping_differences)-10} more differences")
    
    # Analyze NAICS code distributions
    census_naics_codes = set(census_clean['naics'])
    schott_naics_codes = set(schott_clean['naics'])
    
    common_naics = census_naics_codes.intersection(schott_naics_codes)
    census_only_naics = census_naics_codes - schott_naics_codes
    schott_only_naics = schott_naics_codes - census_naics_codes
    
    print(f"\nNAICS Code Distribution:")
    print(f"  Census unique NAICS: {len(census_naics_codes):,}")
    print(f"  Schott unique NAICS: {len(schott_naics_codes):,}")
    print(f"  Common NAICS codes: {len(common_naics):,}")
    print(f"  Census-only NAICS: {len(census_only_naics):,}")
    print(f"  Schott-only NAICS: {len(schott_only_naics):,}")
    
    if len(census_only_naics) > 0:
        print(f"  Census-only NAICS examples: {sorted(list(census_only_naics))[:5]}")
    if len(schott_only_naics) > 0:
        print(f"  Schott-only NAICS examples: {sorted(list(schott_only_naics))[:5]}")
    
    return {
        'year': year,
        'census_hs_count': len(census_hs_codes),
        'schott_hs_count': len(schott_hs_codes),
        'common_hs_count': len(common_hs_codes),
        'matching_mappings': matching_mappings,
        'different_mappings': different_mappings,
        'mapping_agreement_rate': matching_mappings/len(common_hs_codes) if len(common_hs_codes) > 0 else 0,
        'census_naics_count': len(census_naics_codes),
        'schott_naics_count': len(schott_naics_codes),
        'mapping_differences': mapping_differences
    }

# Main execution
if __name__ == "__main__":
    print("CENSUS vs SCHOTT MAPPING COMPARISON")
    print("="*60)
    
    # Load Census data for 2023 and 2024
    years_to_compare = ['2023', '2024']
    census_data = {}
    
    for year in years_to_compare:
        try:
            census_df = read_census_mapping(year)
            census_data[year] = census_df
            print(f"Loaded Census {year} data: {len(census_df):,} records")
        except Exception as e:
            print(f"Error loading Census {year} data: {e}")
    
    # Load Schott data
    schott_data_path = os.path.join(data_paths['base_paths']['underlying_data_root'], 
                                   'data', 'working', '01_Schott_Data_Compiler', 
                                   '02_hs_naics_mapping_2023_imports.csv')
    
    try:
        schott_df = read_schott_mapping(schott_data_path)
        print(f"Loaded Schott 2023 data: {len(schott_df):,} records")
    except Exception as e:
        print(f"Error loading Schott data: {e}")
        schott_df = None
    
    # Perform comparisons
    comparison_results = {}
    
    if schott_df is not None:
        # Compare Census 2023 vs Schott 2023
        if '2023' in census_data:
            comparison_results['2023'] = compare_mappings(census_data['2023'], schott_df, '2023')
        
        # Compare Census 2024 vs Schott 2023 (if Schott 2024 is not available)
        if '2024' in census_data:
            print(f"\n{'='*60}")
            print(f"COMPARING CENSUS 2024 vs SCHOTT 2023 (Cross-year comparison)")
            print(f"{'='*60}")
            print("Note: This compares different years since Schott 2024 may not be available")
            comparison_results['2024_vs_2023'] = compare_mappings(census_data['2024'], schott_df, '2024 vs 2023')
    
    # Generate comprehensive comparison CSV
    print(f"\n{'='*60}")
    print("GENERATING COMPARISON CSV")
    print(f"{'='*60}")
    
    if schott_df is not None and '2023' in census_data:
        # Create comprehensive comparison for 2023
        census_2023 = census_data['2023'][['COMMODITY', 'NAICS']].copy()
        census_2023 = census_2023.rename(columns={'COMMODITY': 'hs_code', 'NAICS': 'naics_census'})
        census_2023['hs_code'] = census_2023['hs_code'].astype(str).str.zfill(10)
        
        schott_2023 = schott_df[['commodity', 'naicsX']].copy()
        schott_2023 = schott_2023.rename(columns={'commodity': 'hs_code', 'naicsX': 'naics_schott'})
        schott_2023['hs_code'] = schott_2023['hs_code'].astype(str).str.zfill(10)
        
        # Full outer join to keep all HS codes from both datasets
        comparison_df = pd.merge(schott_2023, census_2023, on='hs_code', how='outer', suffixes=('_schott', '_census'))
        
        # Add comparison columns
        comparison_df['hs_in_schott'] = comparison_df['naics_schott'].notna()
        comparison_df['hs_in_census'] = comparison_df['naics_census'].notna()
        comparison_df['naics_match'] = comparison_df['naics_schott'] == comparison_df['naics_census']
        
        # Add source information
        def get_source(row):
            if row['hs_in_schott'] and row['hs_in_census']:
                return 'both'
            elif row['hs_in_schott']:
                return 'schott_only'
            else:
                return 'census_only'
        
        comparison_df['source'] = comparison_df.apply(get_source, axis=1)
        
        # Reorder columns for clarity
        column_order = [
            'hs_code',
            'naics_schott', 
            'naics_census',
            'naics_match',
            'hs_in_schott',
            'hs_in_census', 
            'source'
        ]
        comparison_df = comparison_df[column_order]
        
        # Sort by HS code
        comparison_df = comparison_df.sort_values('hs_code')
        
        # Save to CSV
        output_path = os.path.join(data_paths['base_paths']['underlying_data_root'], 
                                  'data', 'working', 'Alternative_Census_Mapping')
        
        # Create directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        csv_output_path = os.path.join(output_path, '01_Schott_vs_Census_hs_mappings.csv')
        comparison_df.to_csv(csv_output_path, index=False)
        
        print(f"Comparison CSV saved: {csv_output_path}")
        print(f"Total records: {len(comparison_df):,}")
        print(f"  Both datasets: {len(comparison_df[comparison_df['source'] == 'both']):,}")
        print(f"  Schott only: {len(comparison_df[comparison_df['source'] == 'schott_only']):,}")
        print(f"  Census only: {len(comparison_df[comparison_df['source'] == 'census_only']):,}")
        print(f"  NAICS matches: {len(comparison_df[comparison_df['naics_match'] == True]):,}")
        print(f"  NAICS differs: {len(comparison_df[(comparison_df['source'] == 'both') & (comparison_df['naics_match'] == False)]):,}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for comparison_name, results in comparison_results.items():
        print(f"\n{comparison_name.upper()}:")
        print(f"  HS Code Coverage: {results['common_hs_count']:,}/{max(results['census_hs_count'], results['schott_hs_count']):,} " +
              f"({results['common_hs_count']/max(results['census_hs_count'], results['schott_hs_count']):.1%})")
        print(f"  NAICS Mapping Agreement: {results['matching_mappings']:,}/{results['common_hs_count']:,} " +
              f"({results['mapping_agreement_rate']:.1%})")
        print(f"  Different mappings: {results['different_mappings']:,}")