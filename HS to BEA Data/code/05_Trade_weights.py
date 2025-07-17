import os
import json 
import pandas as pd 
import country_converter as coco
from main_pipeline_run import get_data_path

"""
Description: Creates country-specific trade weights for tariff analysis using two different weighting schemes.
This script takes the BEA-aggregated trade data from 04_Aggregate_BEA_and_HS and calculates import weights
that can be used to assess the economic impact of trade policies on different countries and regions.

The core challenge we're solving:
- Trade policies affect countries differently based on their import patterns
- We need weights that reflect both global trade relationships and regional dependencies
- Different BEA aggregation levels (detail, U.Summary, summary, sector) require consistent weighting

This script creates two types of weights:
1. DIRECT weights: Use global import totals as denominators (country's share of world imports)
2. INDIRECT weights: Use regional import totals as denominators (country's share of regional imports)

Regional definitions:
- Single-country regions: CAN, MEX, CHN, JPN (weights = 1.0 for indirect)
- Europe: All European countries combined
- RoAsia: Asia + Oceania excluding China and Japan
- RoWorld: All other countries not in above regions

The approach:
1. Load BEA aggregated data from all 4 levels (detail, usummary, summary, sector)
2. Apply region assignments using country_converter and custom logic
3. Calculate denominators for both world and regional weighting schemes
4. Compute weights and validate they sum to 1.0 within each grouping
5. Create comprehensive output files and final JSON for downstream analysis

Main outputs:
- Working files: {level}_trade_weights.csv for each BEA level (verification data)
- Final output: trade_weights.json with direct/indirect structure matching bea_import_weights.json
- Validation: 1_weights_sum_to_one.csv showing weight validation across all levels

The weights are structured as:
- Direct: weight = country_import_value / world_total_for_bea_code
- Indirect: weight = country_import_value / regional_total_for_bea_code

These weights enable analysis of how trade policies affect different countries based on their
global vs regional import dependencies across different BEA economic sectors.
"""

base_path = get_data_path('working', '04_Aggregate_BEA_and_HS')
aggregated_path = os.path.join(base_path, 'aggregated_data')

detail_df = pd.read_csv(os.path.join(aggregated_path, 'country_detail', 'all_continents_detail.csv'))
usummary_df = pd.read_csv(os.path.join(aggregated_path, 'country_usummary', 'all_continents_usummary.csv'))
summary_df = pd.read_csv(os.path.join(aggregated_path, 'country_summary', 'all_continents_summary.csv'))
sector_df = pd.read_csv(os.path.join(aggregated_path, 'country_sector', 'all_continents_sector.csv'))

usummary_df['usummary_code'] = usummary_df['usummary_code'].replace({'S004 ': 'Used', 'S003 ': 'Other', 'S009 ': 'Other'})

print(f"Loaded BEA aggregated data: {len(detail_df)} detail, {len(usummary_df)} usummary, {len(summary_df)} summary, {len(sector_df)} sector rows")

print("\nValidation: Total impVal across BEA aggregation levels")
detail_total = detail_df['impVal'].sum()
usummary_total = usummary_df['impVal'].sum()
summary_total = summary_df['impVal'].sum()
sector_total = sector_df['impVal'].sum()
print(f"Detail total: ${detail_total:,.0f}")
print(f"U.Summary total: ${usummary_total:,.0f}")
print(f"Summary total: ${summary_total:,.0f}")
print(f"Sector total: ${sector_total:,.0f}")

totals = [detail_total, usummary_total, summary_total, sector_total]
if all(abs(t - totals[0]) < 1 for t in totals):
    print("All aggregation levels have consistent total impVal")
else:
    print("Warning: Aggregation levels have different total impVal")


"""
We assign regions in two ways. The first is based on just a pure continental mapping using country_converter - when the BEA says Asia and Pacific we 
interpret that as Asia and Oceania. The BEA descriptions are not super easy to find, but they are burried somewher, so I went and scrounged them up.
With these descriptions we create a second version of the region mapping using the BEAs specific countries and regions. One thing to note about this
is that many of them do not have ISO codes nor appropraite mappings within the regex, so I will want to, depending on the quality of this mapping, 
go back and fix some of these by hands and comparing them to the US Census Bureau data... Ideally there will be nice concordances. 

To get the BEA maps, which come in the form of an obnoxious .pdf, I wrote some pdf reading code in the new folder "Map BEA Regions", that reads them in
and extracts them to be a .csv file. We run that here (just by calling the .py file) and then use the extracted csvs to create the RoAsia and Europe regions. 
-- CAN, MEX, CHN, and JPN are single countries so these are easy, and rest of world is just the set of all countries not yet assigned a region.

The two csvs from Map BEA REgions are:
- 'Map BEA Regions/data/final/BEA_TiVA_Europe.csv'
- 'Map BEA Regions/data/final/BEA_TiVA_Asia_and_Pacific.csv

Each of these has a column 'iso3' which determines whether a country is in that mapping. WE construct a few very simple validation rules to see the difference between this
mapping and what we achieved with the country converter including, but not limited to: a.) how may countries are in the mapping, b.) How many countries in the country_converter 
mapping are not in the bea mapping, c.) how many countries in the bea mapping are not in the country_converter mapping, and d.) what are the relative sums of impVal between 
the two mappings (i.e. total imports from EUROPE under the country_converter mapping vs. total imports from EUROPE under the bea mapping, same for the new version.).
"""

# Load BEA region mapping files
bea_europe_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Map BEA Regions', 'data', 'final', 'BEA_TiVA_Europe.csv')
bea_asia_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Map BEA Regions', 'data', 'final', 'BEA_TiVA_Asia_and_Pacific.csv')

bea_europe_df = pd.read_csv(bea_europe_path)
bea_asia_df = pd.read_csv(bea_asia_path)

# Create sets of ISO3 codes for BEA regions
bea_europe_iso3 = set(bea_europe_df['iso3'].dropna().unique())
bea_asia_iso3 = set(bea_asia_df['iso3'].dropna().unique())

print(f"Loaded BEA region mappings:")
print(f"  Europe: {len(bea_europe_iso3)} countries")
print(f"  Asia and Pacific: {len(bea_asia_iso3)} countries")

def assign_region_country_converter(row):
    """Original country converter mapping"""
    iso3 = row['iso3']
    continent = row['continent']
    
    if iso3 in ['CAN', 'MEX', 'CHN', 'JPN']:
        return iso3
    elif continent == 'Europe':
        return 'Europe'
    elif continent in ['Asia', 'Oceania'] and iso3 not in ['CHN', 'JPN']:
        return 'RoAsia'
    else:
        return 'RoWorld'

def assign_region_bea(row):
    """BEA-specific region mapping"""
    iso3 = row['iso3']
    
    if iso3 in ['CAN', 'MEX', 'CHN', 'JPN']:
        return iso3
    elif iso3 in bea_europe_iso3:
        return 'Europe'
    elif iso3 in bea_asia_iso3:
        return 'RoAsia'
    else:
        return 'RoWorld'

# Apply both mappings to all dataframes
for df in [detail_df, usummary_df, summary_df, sector_df]:
    df['continent'] = coco.convert(df['iso3'], to='Continent_7', not_found=None)
    df['alt_region'] = df.apply(assign_region_country_converter, axis=1)  # Country converter mapping
    df['region'] = df.apply(assign_region_bea, axis=1)  # BEA mapping (new primary)

# Create validation comparison between the two mappings
print("\n" + "="*60)
print("VALIDATION: Comparing Country Converter vs BEA Region Mappings")
print("="*60)

# Use detail_df for comprehensive validation
validation_data = []

# Compare mapping coverage
cc_region_counts = detail_df['alt_region'].value_counts()
bea_region_counts = detail_df['region'].value_counts()

print(f"\nRegion distribution comparison:")
print(f"{'Region':<10} {'CountryConverter':<18} {'BEA Mapping':<12} {'Difference':<10}")
print("-" * 55)

for region in ['CAN', 'MEX', 'CHN', 'JPN', 'Europe', 'RoAsia', 'RoWorld']:
    cc_count = cc_region_counts.get(region, 0)
    bea_count = bea_region_counts.get(region, 0)
    diff = bea_count - cc_count
    print(f"{region:<10} {cc_count:<18} {bea_count:<12} {diff:<10}")

# Compare import values by region
print(f"\nImport value comparison by region:")
print(f"{'Region':<10} {'CountryConverter ($)':<20} {'BEA Mapping ($)':<18} {'Difference ($)':<15} {'% Change':<10}")
print("-" * 80)

cc_region_values = detail_df.groupby('alt_region')['impVal'].sum()
bea_region_values = detail_df.groupby('region')['impVal'].sum()

for region in ['CAN', 'MEX', 'CHN', 'JPN', 'Europe', 'RoAsia', 'RoWorld']:
    cc_val = cc_region_values.get(region, 0)
    bea_val = bea_region_values.get(region, 0)
    diff = bea_val - cc_val
    pct_change = (diff / cc_val * 100) if cc_val > 0 else 0
    print(f"{region:<10} {cc_val:<20,.0f} {bea_val:<18,.0f} {diff:<15,.0f} {pct_change:<10.1f}%")

# Find countries that switched regions
region_switches = detail_df[detail_df['alt_region'] != detail_df['region']][['iso3', 'Country', 'alt_region', 'region', 'impVal']].drop_duplicates()
region_switches = region_switches.groupby(['iso3', 'Country', 'alt_region', 'region'])['impVal'].sum().reset_index()
region_switches = region_switches.sort_values('impVal', ascending=False)

print(f"\nCountries that switched regions (top 10 by import value):")
print(f"{'ISO3':<5} {'Country':<25} {'CC Region':<10} {'BEA Region':<10} {'Import Value ($)':<15}")
print("-" * 75)

for i, (_, row) in enumerate(region_switches.head(10).iterrows()):
    print(f"{row['iso3']:<5} {row['Country'][:24]:<25} {row['alt_region']:<10} {row['region']:<10} {row['impVal']:<15,.0f}")

# Check for unmapped countries in BEA system
unmapped_countries = detail_df[detail_df['region'] == 'RoWorld'][['iso3', 'Country']].drop_duplicates()
cc_unmapped = detail_df[detail_df['alt_region'] == 'RoWorld'][['iso3', 'Country']].drop_duplicates()

print(f"\nUnmapped countries comparison:")
print(f"  Countries in RoWorld (Country Converter): {len(cc_unmapped)}")
print(f"  Countries in RoWorld (BEA mapping): {len(unmapped_countries)}")

# Countries only in BEA Europe but not in Country Converter Europe
bea_only_europe = detail_df[(detail_df['region'] == 'Europe') & (detail_df['alt_region'] != 'Europe')][['iso3', 'Country']].drop_duplicates()
cc_only_europe = detail_df[(detail_df['alt_region'] == 'Europe') & (detail_df['region'] != 'Europe')][['iso3', 'Country']].drop_duplicates()

print(f"\nEurope mapping differences:")
print(f"  Countries in BEA Europe but not CC Europe: {len(bea_only_europe)}")
if len(bea_only_europe) > 0:
    print(f"    {list(bea_only_europe['iso3'])}")
print(f"  Countries in CC Europe but not BEA Europe: {len(cc_only_europe)}")
if len(cc_only_europe) > 0:
    print(f"    {list(cc_only_europe['iso3'])}")

# Countries only in BEA RoAsia but not in Country Converter RoAsia
bea_only_asia = detail_df[(detail_df['region'] == 'RoAsia') & (detail_df['alt_region'] != 'RoAsia')][['iso3', 'Country']].drop_duplicates()
cc_only_asia = detail_df[(detail_df['alt_region'] == 'RoAsia') & (detail_df['region'] != 'RoAsia')][['iso3', 'Country']].drop_duplicates()

print(f"\nRoAsia mapping differences:")
print(f"  Countries in BEA RoAsia but not CC RoAsia: {len(bea_only_asia)}")
if len(bea_only_asia) > 0:
    print(f"    {list(bea_only_asia['iso3'])}")
print(f"  Countries in CC RoAsia but not BEA RoAsia: {len(cc_only_asia)}")
if len(cc_only_asia) > 0:
    print(f"    {list(cc_only_asia['iso3'])}")

# Save detailed validation report
validation_report = {
    'mapping_comparison': {
        'region_counts': {
            'country_converter': cc_region_counts.to_dict(),
            'bea_mapping': bea_region_counts.to_dict()
        },
        'import_values': {
            'country_converter': cc_region_values.to_dict(),
            'bea_mapping': bea_region_values.to_dict()
        },
        'countries_switched': len(region_switches),
        'total_switch_value': region_switches['impVal'].sum()
    },
    'regional_differences': {
        'europe': {
            'bea_only': list(bea_only_europe['iso3']) if len(bea_only_europe) > 0 else [],
            'cc_only': list(cc_only_europe['iso3']) if len(cc_only_europe) > 0 else []
        },
        'roasia': {
            'bea_only': list(bea_only_asia['iso3']) if len(bea_only_asia) > 0 else [],
            'cc_only': list(cc_only_asia['iso3']) if len(cc_only_asia) > 0 else []
        }
    }
}

# Save validation files
validation_dir = os.path.join(get_data_path('validation'), '05_Trade_weights')
os.makedirs(validation_dir, exist_ok=True)

# Save detailed region switches
region_switches_path = os.path.join(validation_dir, '2_region_mapping_comparison.csv')
region_switches.to_csv(region_switches_path, index=False)

# Save summary validation as text
validation_text_path = os.path.join(validation_dir, '2_region_mapping_validation.txt')
with open(validation_text_path, 'w') as f:
    f.write("VALIDATION: Country Converter vs BEA Region Mappings\n")
    f.write("="*60 + "\n\n")
    
    f.write("Region distribution comparison:\n")
    f.write(f"{'Region':<10} {'CountryConverter':<18} {'BEA Mapping':<12} {'Difference':<10}\n")
    f.write("-" * 55 + "\n")
    for region in ['CAN', 'MEX', 'CHN', 'JPN', 'Europe', 'RoAsia', 'RoWorld']:
        cc_count = cc_region_counts.get(region, 0)
        bea_count = bea_region_counts.get(region, 0)
        diff = bea_count - cc_count
        f.write(f"{region:<10} {cc_count:<18} {bea_count:<12} {diff:<10}\n")
    
    f.write(f"\nImport value comparison by region:\n")
    f.write(f"{'Region':<10} {'CountryConverter ($)':<20} {'BEA Mapping ($)':<18} {'Difference ($)':<15} {'% Change':<10}\n")
    f.write("-" * 80 + "\n")
    for region in ['CAN', 'MEX', 'CHN', 'JPN', 'Europe', 'RoAsia', 'RoWorld']:
        cc_val = cc_region_values.get(region, 0)
        bea_val = bea_region_values.get(region, 0)
        diff = bea_val - cc_val
        pct_change = (diff / cc_val * 100) if cc_val > 0 else 0
        f.write(f"{region:<10} {cc_val:<20,.0f} {bea_val:<18,.0f} {diff:<15,.0f} {pct_change:<10.1f}%\n")
    
    f.write(f"\nCountries that switched regions: {len(region_switches)}\n")
    f.write(f"Total import value affected: ${region_switches['impVal'].sum():,.0f}\n")
    
    f.write(f"\nEurope mapping differences:\n")
    f.write(f"  Countries in BEA Europe but not CC Europe: {len(bea_only_europe)}\n")
    if len(bea_only_europe) > 0:
        f.write(f"    {list(bea_only_europe['iso3'])}\n")
    f.write(f"  Countries in CC Europe but not BEA Europe: {len(cc_only_europe)}\n")
    if len(cc_only_europe) > 0:
        f.write(f"    {list(cc_only_europe['iso3'])}\n")
    
    f.write(f"\nRoAsia mapping differences:\n")
    f.write(f"  Countries in BEA RoAsia but not CC RoAsia: {len(bea_only_asia)}\n")
    if len(bea_only_asia) > 0:
        f.write(f"    {list(bea_only_asia['iso3'])}\n")
    f.write(f"  Countries in CC RoAsia but not BEA RoAsia: {len(cc_only_asia)}\n")
    if len(cc_only_asia) > 0:
        f.write(f"    {list(cc_only_asia['iso3'])}\n")

print(f"\nValidation files saved:")
print(f"  {region_switches_path}")
print(f"  {validation_text_path}")
print("="*60)

print("\nCalculating denominators for trade weights...")

def calculate_denominators(df, level_name):
    bea_col = {'detail': 'detail_code', 'usummary': 'usummary_code', 'summary': 'summary_code', 'sector': 'sector_code'}[level_name]
    
    world_denominators = df.groupby(bea_col)['impVal'].sum().to_dict()
    regional_denominators = {}
    
    for country in ['CAN', 'MEX', 'CHN', 'JPN']:
        country_data = df[df['iso3'] == country]
        if len(country_data) > 0:
            regional_denominators[country] = country_data.groupby(bea_col)['impVal'].sum().to_dict()
    
    for region in ['Europe', 'RoAsia', 'RoWorld']:
        region_data = df[df['region'] == region]
        if len(region_data) > 0:
            regional_denominators[region] = region_data.groupby(bea_col)['impVal'].sum().to_dict()
    
    return world_denominators, regional_denominators

denominators = {}
for level, df in [('detail', detail_df), ('usummary', usummary_df), ('summary', summary_df), ('sector', sector_df)]:
    world_denom, regional_denom = calculate_denominators(df, level)
    denominators[level] = {'world': world_denom, 'regional': regional_denom}
    
    print(f"\n{level.capitalize()} level denominators:")
    print(f"  World: {len(world_denom)} BEA codes")
    print(f"  Regional: {len(regional_denom)} regions")
    for region, codes in regional_denom.items():
        print(f"    {region}: {len(codes)} BEA codes")

print("\n" + "="*50)
print("VALIDATION CHECKS")
print("="*50)

for level, df in [('detail', detail_df), ('usummary', usummary_df), ('summary', summary_df), ('sector', sector_df)]:
    missing_regions = df[df['region'].isna()]
    if len(missing_regions) > 0:
        print(f" {level}: {len(missing_regions)} rows missing region assignment")
        print(f"  Countries: {missing_regions['iso3'].unique()}")
    else:
        print(f" {level}: All countries have region assignments")

print(f"\nSingle-country regions validation:")
for country in ['CAN', 'MEX', 'CHN', 'JPN']:
    country_in_data = any(country in df['iso3'].values for df in [detail_df, usummary_df, summary_df, sector_df])
    if country_in_data:
        print(f" {country}: Found in data (will have weight = 1)")
    else:
        print(f" {country}: Not found in data")

print(f"\nRegion distribution (using detail level):")
region_counts = detail_df['region'].value_counts()
for region, count in region_counts.items():
    print(f"  {region}: {count} rows")

print("="*50)

print("\nCalculating weights and validating they sum to 1...")

validation_dir = os.path.join(get_data_path('validation'), '05_Trade_weights')
os.makedirs(validation_dir, exist_ok=True)

def calculate_and_validate_weights(df, level_name, denominators):
    bea_col = {'detail': 'detail_code', 'usummary': 'usummary_code', 'summary': 'summary_code', 'sector': 'sector_code'}[level_name]
    
    df['world_weight'] = df.apply(lambda row: row['impVal'] / denominators['world'][row[bea_col]], axis=1)
    
    def calc_regional_weight(row):
        region = row['region']
        bea_code = row[bea_col]
        if region in denominators['regional'] and bea_code in denominators['regional'][region]:
            return row['impVal'] / denominators['regional'][region][bea_code]
        return 0
    
    df['regional_weight'] = df.apply(calc_regional_weight, axis=1)
    
    world_weight_sums = df.groupby(bea_col)['world_weight'].sum().reset_index()
    world_weight_sums['weight_sum_diff'] = world_weight_sums['world_weight'] - 1.0
    world_weight_sums['level'] = level_name
    world_weight_sums['weight_type'] = 'world'
    world_weight_sums['passes_validation'] = abs(world_weight_sums['weight_sum_diff']) <= 0.0001
    
    regional_weight_sums = df.groupby([bea_col, 'region'])['regional_weight'].sum().reset_index()
    regional_weight_sums['weight_sum_diff'] = regional_weight_sums['regional_weight'] - 1.0
    regional_weight_sums['level'] = level_name
    regional_weight_sums['weight_type'] = 'regional'
    regional_weight_sums['passes_validation'] = abs(regional_weight_sums['weight_sum_diff']) <= 0.0001
    
    world_issues = world_weight_sums[~world_weight_sums['passes_validation']]
    regional_issues = regional_weight_sums[~regional_weight_sums['passes_validation']]
    
    print(f"\n{level_name.capitalize()} level weight validation:")
    if len(world_issues) == 0:
        print(f"  World weights: All {len(world_weight_sums)} BEA codes sum to 1.0")
    else:
        print(f"  World weights: {len(world_issues)} BEA codes don't sum to 1.0")
    
    if len(regional_issues) == 0:
        print(f"  regional weights: All {len(regional_weight_sums)} BEA code-region combinations sum to 1.0")
    else:
        print(f"  Regional weights: {len(regional_issues)} BEA code-region combinations don't sum to 1.0")
    return world_weight_sums, regional_weight_sums

all_world_validations = []
all_regional_validations = []

for level, df in [('detail', detail_df), ('usummary', usummary_df), ('summary', summary_df), ('sector', sector_df)]:
    world_val, regional_val = calculate_and_validate_weights(df, level, denominators[level])
    all_world_validations.append(world_val)
    all_regional_validations.append(regional_val)

all_validations = pd.concat([pd.concat(all_world_validations, ignore_index=True), pd.concat(all_regional_validations, ignore_index=True)], ignore_index=True)

validation_path = os.path.join(validation_dir, '1_weights_sum_to_one.csv')
all_validations.to_csv(validation_path, index=False)

print(f"\nValidation results saved to: {validation_path}")
print(f"Total validation checks: {len(all_validations)}")
print(f"Failed validations: {len(all_validations[~all_validations['passes_validation']])}")

print("\n" + "="*50)
print("WEIGHT VALIDATION COMPLETE")
print("="*50)

# Step 3: Save output files with trade values and weights
print("\nSaving output files...")

# Create output directory
output_dir = get_data_path('working', '05_Trade_weights')
os.makedirs(output_dir, exist_ok=True)

# Save each BEA level with all necessary columns
for level, df in [('detail', detail_df), ('usummary', usummary_df), ('summary', summary_df), ('sector', sector_df)]:
    # Select relevant columns for output
    if level == 'detail':
        bea_col = 'detail_code'
    elif level == 'usummary':
        bea_col = 'usummary_code'
    elif level == 'summary':
        bea_col = 'summary_code'
    elif level == 'sector':
        bea_col = 'sector_code'
    # Create output dataframe with all necessary information including both region mappings
    output_df = df[['Country', 'iso3', bea_col, 'impVal', 'continent', 'region', 'alt_region', 'world_weight', 'regional_weight']].copy()
    # Add denominators for verification
    output_df['world_denominator'] = output_df[bea_col].map(denominators[level]['world'])
    def get_regional_denominator(row):
        region = row['region']
        bea_code = row[bea_col]
        if region in denominators[level]['regional'] and bea_code in denominators[level]['regional'][region]:
            return denominators[level]['regional'][region][bea_code]
        return None
    
    output_df['regional_denominator'] = output_df.apply(get_regional_denominator, axis=1)
    
    # Save to CSV
    output_path = os.path.join(output_dir, f'{level}_trade_weights.csv')
    output_df.to_csv(output_path, index=False)
    
    print(f"Saved {level} level: {output_path}")
    print(f"  Columns: {list(output_df.columns)}")
    print(f"  Rows: {len(output_df)}")

print(f"\nAll output files saved to: {output_dir}")
print("\nCreating final JSON output for U.Summary level...")

bea_codes_path = get_data_path('raw', 'BEA_codes', '402_use')
# Actually, let's use the q.csv file you mentioned
bea_codes_path = os.path.join(get_data_path('raw', 'BEA_codes'), 'q.csv')
bea_codes_df = pd.read_csv(bea_codes_path)
usummary_order = bea_codes_df['U.Summary Code'].tolist()

print(f"Loaded {len(usummary_order)} U.Summary codes in order")

# Create the JSON structure
def create_trade_weights_json(df, usummary_order):
    """Create the final JSON structure with direct and indirect weights"""
    
    result = {
        "direct": {},
        "indirect": {}
    }
    
    # Get all unique countries, sorted by ISO3 code
    countries = sorted(df['iso3'].unique())
    
    for country in countries:
        country_data = df[df['iso3'] == country]
        
        # Initialize country entries
        result["direct"][country] = {}
        result["indirect"][country] = {}
        
        # Fill in weights for each U.Summary code in the specified order
        for usummary_code in usummary_order:
            # Find the row for this country-usummary combination
            row = country_data[country_data['usummary_code'] == usummary_code]
            
            if len(row) > 0:
                # Use the actual weights
                result["direct"][country][usummary_code] = float(row.iloc[0]['world_weight'])
                result["indirect"][country][usummary_code] = float(row.iloc[0]['regional_weight'])
            else:
                # Use 0 if no data for this combination
                result["direct"][country][usummary_code] = 0.0
                result["indirect"][country][usummary_code] = 0.0
    
    return result

# Create the JSON structure using usummary data
final_json = create_trade_weights_json(usummary_df, usummary_order)

# Save to final data directory
final_output_dir = get_data_path('final')
os.makedirs(final_output_dir, exist_ok=True)

final_json_path = os.path.join(final_output_dir, 'trade_weights.json')

with open(final_json_path, 'w') as f:
    json.dump(final_json, f, indent=2)

print(f"Final JSON output saved to: {final_json_path}")
print(f"Countries included: {len(final_json['direct'])}")
print(f"U.Summary codes per country: {len(usummary_order)}")

print("="*50)