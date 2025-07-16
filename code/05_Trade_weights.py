import os
import json 
import pandas as pd 
import country_converter as coco
from main_pipeline_run import get_data_path

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

def assign_region(row):
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

for df in [detail_df, usummary_df, summary_df, sector_df]:
    df['continent'] = coco.convert(df['iso3'], to='Continent_7', not_found=None)
    df['region'] = df.apply(assign_region, axis=1)

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
    # Create output dataframe with all necessary information
    output_df = df[['Country', 'iso3', bea_col, 'impVal', 'continent', 'region', 'world_weight', 'regional_weight']].copy()
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