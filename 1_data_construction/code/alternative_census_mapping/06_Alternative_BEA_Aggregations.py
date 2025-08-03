# Create BEA aggregations using Alternative Census mappings
# This replicates the 04_Aggregate_BEA_and_HS.py approach but using our new Census-based mappings
# Then validates against the original aggregations

import os
import pandas as pd
import json
import country_converter as coco

# Load data paths configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
data_paths_file = os.path.join(script_dir, '..', '..', 'data_paths.json')

with open(data_paths_file, 'r') as f:
    data_paths = json.load(f)

def load_alternative_census_data():
    """Load our Alternative Census mapping aggregated data"""
    print("Loading Alternative Census aggregated data...")
    
    aggregated_data_path = os.path.join(data_paths['base_paths']['underlying_data_root'], 
                                       'data', 'working', 'Alternative_Census_Mapping', 
                                       '02_aggregated_with_bea_mapping.csv')
    
    df = pd.read_csv(aggregated_data_path, dtype=str)
    df['impVal'] = pd.to_numeric(df['impVal'])
    
    # Use bea_2022 as our detail_code (since it has 100% coverage)
    df['detail_code'] = df['bea_2022']
    
    # Filter out any records without BEA mapping
    df_with_bea = df[df['detail_code'].notna()].copy()
    df_with_bea['detail_code'] = df_with_bea['detail_code'].astype(str).str.strip()
    
    # Add ISO3 country codes
    df_with_bea['iso3'] = coco.convert(df_with_bea['Country'], to='iso3')
    
    # Handle cases where coco.convert returns lists or None
    def clean_iso3(iso3_value):
        if isinstance(iso3_value, list):
            return iso3_value[0] if iso3_value else 'UNK'
        elif iso3_value is None or pd.isna(iso3_value):
            return 'UNK'
        else:
            return iso3_value
    
    df_with_bea['iso3'] = df_with_bea['iso3'].apply(clean_iso3)
    
    # Print any countries that couldn't be converted
    unknown_countries = df_with_bea[df_with_bea['iso3'] == 'UNK']['Country'].unique()
    if len(unknown_countries) > 0:
        print(f"  Warning: Countries with unknown ISO3 codes: {unknown_countries}")
    
    print(f"Loaded Alternative Census data: {len(df_with_bea)} rows")
    print(f"Total import value: ${df_with_bea['impVal'].sum():,.0f}")
    print(f"Unique countries: {df_with_bea['Country'].nunique()}")
    print(f"Unique BEA detail codes: {df_with_bea['detail_code'].nunique()}")
    
    return df_with_bea

def load_bea_hierarchy():
    """Load BEA hierarchy mapping"""
    print("Loading BEA hierarchy mapping...")
    
    bea_hierarchy_path = os.path.join(data_paths['base_paths']['underlying_data_root'], 
                                     'data', 'working', '02_HS_to_Naics_to_BEA', '02_BEA_hierarchy.csv')
    bea_hierarchy = pd.read_csv(bea_hierarchy_path)
    
    # Create mapping dictionaries with trimmed values
    detail_to_usummary = dict(zip(bea_hierarchy['Detail'].str.strip(), bea_hierarchy['U.Summary'].str.strip()))
    detail_to_summary = dict(zip(bea_hierarchy['Detail'].str.strip(), bea_hierarchy['Summary'].str.strip()))
    detail_to_sector = dict(zip(bea_hierarchy['Detail'].str.strip(), bea_hierarchy['Sector'].str.strip()))
    
    print(f"BEA hierarchy loaded: {len(bea_hierarchy)} mappings")
    
    return detail_to_usummary, detail_to_summary, detail_to_sector

def create_bea_aggregations(df, detail_to_usummary, detail_to_summary, detail_to_sector):
    """Create the four BEA aggregation levels"""
    print("Creating BEA aggregations...")
    
    # Create output directories
    base_output_dir = os.path.join(data_paths['base_paths']['underlying_data_root'], 
                                  'data', 'working', 'Alternative_Census_Mapping', 'bea_aggregations')
    detail_dir = os.path.join(base_output_dir, 'country_detail')
    usummary_dir = os.path.join(base_output_dir, 'country_usummary')
    summary_dir = os.path.join(base_output_dir, 'country_summary')
    sector_dir = os.path.join(base_output_dir, 'country_sector')
    
    for dir_path in [detail_dir, usummary_dir, summary_dir, sector_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Store original country totals for validation
    print("  Calculating original country totals...")
    original_country_totals = df.groupby('Country')['impVal'].sum().reset_index()
    
    # Version 1: Country X detail_code level (already aggregated in our data)
    print("  Creating Country X detail_code aggregation...")
    detail_aggregated = df.groupby(['Country', 'detail_code', 'iso3'])['impVal'].sum().reset_index()
    
    # Save detail level data
    detail_output_path = os.path.join(detail_dir, 'all_continents_detail.csv')
    detail_aggregated.to_csv(detail_output_path, index=False)
    print(f"    Saved detail aggregated data: {len(detail_aggregated)} rows")
    
    # Version 2: Country X U.Summary level
    print("  Creating Country X U.Summary aggregation...")
    detail_aggregated['usummary_code'] = detail_aggregated['detail_code'].map(detail_to_usummary)
    detail_with_usummary = detail_aggregated[detail_aggregated['usummary_code'].notna()].copy()
    detail_with_usummary['usummary_code'] = detail_with_usummary['usummary_code'].astype(str).str.strip()
    
    usummary_aggregated = detail_with_usummary.groupby(['Country', 'usummary_code', 'iso3'])['impVal'].sum().reset_index()
    usummary_output_path = os.path.join(usummary_dir, 'all_continents_usummary.csv')
    usummary_aggregated.to_csv(usummary_output_path, index=False)
    print(f"    Saved U.Summary aggregated data: {len(usummary_aggregated)} rows")
    
    # Version 3: Country X Summary level
    print("  Creating Country X Summary aggregation...")
    detail_aggregated['summary_code'] = detail_aggregated['detail_code'].map(detail_to_summary)
    detail_with_summary = detail_aggregated[detail_aggregated['summary_code'].notna()].copy()
    detail_with_summary['summary_code'] = detail_with_summary['summary_code'].astype(str).str.strip()
    
    summary_aggregated = detail_with_summary.groupby(['Country', 'summary_code', 'iso3'])['impVal'].sum().reset_index()
    summary_output_path = os.path.join(summary_dir, 'all_continents_summary.csv')
    summary_aggregated.to_csv(summary_output_path, index=False)
    print(f"    Saved Summary aggregated data: {len(summary_aggregated)} rows")
    
    # Version 4: Country X Sector level
    print("  Creating Country X Sector aggregation...")
    detail_aggregated['sector_code'] = detail_aggregated['detail_code'].map(detail_to_sector)
    detail_with_sector = detail_aggregated[detail_aggregated['sector_code'].notna()].copy()
    detail_with_sector['sector_code'] = detail_with_sector['sector_code'].astype(str).str.strip()
    
    sector_aggregated = detail_with_sector.groupby(['Country', 'sector_code', 'iso3'])['impVal'].sum().reset_index()
    sector_output_path = os.path.join(sector_dir, 'all_continents_sector.csv')
    sector_aggregated.to_csv(sector_output_path, index=False)
    print(f"    Saved Sector aggregated data: {len(sector_aggregated)} rows")
    
    return {
        'original': original_country_totals,
        'detail': detail_aggregated,
        'usummary': usummary_aggregated,
        'summary': summary_aggregated,
        'sector': sector_aggregated
    }

def validate_aggregations(aggregations):
    """Validate that country totals are preserved across all aggregation levels"""
    print("Validating country totals across all aggregation levels...")
    
    # Get country totals for each level
    original_totals = aggregations['original'].rename(columns={'impVal': 'original_impVal'})
    detail_totals = aggregations['detail'].groupby('Country')['impVal'].sum().reset_index().rename(columns={'impVal': 'detail_impVal'})
    usummary_totals = aggregations['usummary'].groupby('Country')['impVal'].sum().reset_index().rename(columns={'impVal': 'usummary_impVal'})
    summary_totals = aggregations['summary'].groupby('Country')['impVal'].sum().reset_index().rename(columns={'impVal': 'summary_impVal'})
    sector_totals = aggregations['sector'].groupby('Country')['impVal'].sum().reset_index().rename(columns={'impVal': 'sector_impVal'})
    
    # Merge all validation data
    validation_df = original_totals.merge(detail_totals, on='Country', how='left') \
                                  .merge(usummary_totals, on='Country', how='left') \
                                  .merge(summary_totals, on='Country', how='left') \
                                  .merge(sector_totals, on='Country', how='left')
    
    # Calculate differences
    for level in ['detail', 'usummary', 'summary', 'sector']:
        validation_df[f'{level}_difference'] = validation_df['original_impVal'] - validation_df[f'{level}_impVal']
        validation_df[f'{level}_pct_difference'] = (validation_df[f'{level}_difference'] / validation_df['original_impVal']) * 100
    
    # Save validation results
    validation_output_dir = os.path.join(data_paths['base_paths']['underlying_data_root'], 
                                       'data', 'working', 'Alternative_Census_Mapping', 'validations')
    os.makedirs(validation_output_dir, exist_ok=True)
    
    validation_path = os.path.join(validation_output_dir, 'Alternative_Country_Aggregation_Validation.csv')
    validation_df.to_csv(validation_path, index=False)
    
    # Print validation summary
    print("  Validation Summary:")
    for level in ['detail', 'usummary', 'summary', 'sector']:
        total_original = validation_df['original_impVal'].sum()
        total_level = validation_df[f'{level}_impVal'].sum()
        pct_diff = ((total_original - total_level) / total_original) * 100
        print(f"    {level.capitalize()}: ${total_level:,.0f} ({pct_diff:+.3f}% difference)")
    
    return validation_df

def compare_with_original_aggregations(our_aggregations):
    """Compare our aggregations with the original ones from 04_Aggregate_BEA_and_HS"""
    print("Comparing with original BEA aggregations...")
    
    # Load original aggregations
    original_base_dir = os.path.join(data_paths['base_paths']['underlying_data_root'], 
                                   'data', 'working', '04_Aggregate_BEA_and_HS', 'aggregated_data')
    
    validation_base = data_paths['validation_outputs']['base_path']
    validation_subdir = data_paths['validation_outputs']['subdirectories']['Alternative_Census_Mappings']
    validation_output_dir = os.path.join(data_paths['base_paths']['underlying_data_root'], validation_base, validation_subdir, 'comparisons with original mapping')
    os.makedirs(validation_output_dir, exist_ok=True)
    
    comparison_results = {}
    
    for level in ['detail', 'usummary', 'summary', 'sector']:
        print(f"  Comparing {level} level...")
        
        # Load original data
        original_path = os.path.join(original_base_dir, f'country_{level}', f'all_continents_{level}.csv')
        
        if not os.path.exists(original_path):
            print(f"    Warning: Original {level} file not found at {original_path}")
            continue
        
        original_df = pd.read_csv(original_path)
        
        # Get our data
        our_df = our_aggregations[level].copy()
        
        # Standardize column names for comparison
        if level == 'detail':
            original_df = original_df.rename(columns={'detail_code': 'bea_code'})
            our_df = our_df.rename(columns={'detail_code': 'bea_code'})
        elif level == 'usummary':
            original_df = original_df.rename(columns={'usummary_code': 'bea_code'})
            our_df = our_df.rename(columns={'usummary_code': 'bea_code'})
        elif level == 'summary':
            original_df = original_df.rename(columns={'summary_code': 'bea_code'})
            our_df = our_df.rename(columns={'summary_code': 'bea_code'})
        elif level == 'sector':
            original_df = original_df.rename(columns={'sector_code': 'bea_code'})
            our_df = our_df.rename(columns={'sector_code': 'bea_code'})
        
        # Merge for comparison
        comparison_df = original_df[['Country', 'bea_code', 'impVal']].merge(
            our_df[['Country', 'bea_code', 'impVal']], 
            on=['Country', 'bea_code'], 
            how='outer', 
            suffixes=('_original', '_alternative')
        )
        
        # Fill NaN values with 0 for calculations
        comparison_df['impVal_original'] = comparison_df['impVal_original'].fillna(0)
        comparison_df['impVal_alternative'] = comparison_df['impVal_alternative'].fillna(0)
        
        # Calculate differences
        comparison_df['impVal_difference'] = comparison_df['impVal_alternative'] - comparison_df['impVal_original']
        
        # Calculate percentage difference (avoiding division by zero)
        comparison_df['impVal_pct_difference'] = 0.0
        mask = comparison_df['impVal_original'] != 0
        comparison_df.loc[mask, 'impVal_pct_difference'] = (
            comparison_df.loc[mask, 'impVal_difference'] / comparison_df.loc[mask, 'impVal_original']
        ) * 100
        
        # Flag records that exist in only one dataset
        comparison_df['data_source'] = 'both'
        comparison_df.loc[comparison_df['impVal_original'] == 0, 'data_source'] = 'alternative_only'
        comparison_df.loc[comparison_df['impVal_alternative'] == 0, 'data_source'] = 'original_only'
        
        # Sort by absolute difference (largest differences first)
        comparison_df['abs_difference'] = comparison_df['impVal_difference'].abs()
        comparison_df = comparison_df.sort_values('abs_difference', ascending=False)
        
        # Save detailed comparison
        comparison_output_path = os.path.join(validation_output_dir, f'08_{level}_level_comparison.csv')
        comparison_df.to_csv(comparison_output_path, index=False)
        
        # Calculate summary statistics
        total_original = comparison_df['impVal_original'].sum()
        total_alternative = comparison_df['impVal_alternative'].sum()
        total_difference = total_alternative - total_original
        total_pct_difference = (total_difference / total_original) * 100 if total_original != 0 else 0
        
        # Count records by source
        both_count = len(comparison_df[comparison_df['data_source'] == 'both'])
        alt_only_count = len(comparison_df[comparison_df['data_source'] == 'alternative_only'])
        orig_only_count = len(comparison_df[comparison_df['data_source'] == 'original_only'])
        
        comparison_results[level] = {
            'total_original': total_original,
            'total_alternative': total_alternative,
            'total_difference': total_difference,
            'total_pct_difference': total_pct_difference,
            'records_both': both_count,
            'records_alternative_only': alt_only_count,
            'records_original_only': orig_only_count,
            'total_records': len(comparison_df)
        }
        
        print(f"    {level.capitalize()} comparison:")
        print(f"      Original total: ${total_original:,.0f}")
        print(f"      Alternative total: ${total_alternative:,.0f}")
        print(f"      Difference: ${total_difference:,.0f} ({total_pct_difference:+.2f}%)")
        print(f"      Records: {both_count} both, {alt_only_count} alt-only, {orig_only_count} orig-only")
        print(f"      Saved comparison: {comparison_output_path}")
    
    # Save summary of all comparisons
    summary_df = pd.DataFrame(comparison_results).T
    summary_output_path = os.path.join(validation_output_dir, '09_comparison_summary.csv')
    summary_df.to_csv(summary_output_path, index=True)
    print(f"  Saved comparison summary: {summary_output_path}")
    
    return comparison_results

def main():
    """Main execution function"""
    print("=" * 80)
    print("CREATING BEA AGGREGATIONS WITH ALTERNATIVE CENSUS MAPPINGS")
    print("=" * 80)
    
    # Load Alternative Census data
    df = load_alternative_census_data()
    
    # Load BEA hierarchy
    detail_to_usummary, detail_to_summary, detail_to_sector = load_bea_hierarchy()
    
    # Create BEA aggregations
    aggregations = create_bea_aggregations(df, detail_to_usummary, detail_to_summary, detail_to_sector)
    
    # Validate aggregations
    validation_df = validate_aggregations(aggregations)
    
    # Compare with original aggregations
    comparison_results = compare_with_original_aggregations(aggregations)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Alternative Census aggregations created successfully!")
    print(f"Total import value processed: ${df['impVal'].sum():,.0f}")
    
    print(f"\nComparison with original aggregations:")
    for level, results in comparison_results.items():
        print(f"  {level.capitalize()}: {results['total_pct_difference']:+.2f}% difference")
    
    print(f"\nOutput files saved to:")
    print(f"  BEA aggregations: data/working/Alternative_Census_Mapping/bea_aggregations/")
    print(f"  Validations: validations/Alternative_Census_Mappings/")

if __name__ == "__main__":
    main()