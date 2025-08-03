import os
import json
import pandas as pd
import numpy as np
# Load data paths and set up standard directory variables
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))  # Go up two levels to Calculations/
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
figures_dir = os.path.join(validations_dir,'04_main_calculations')
final_data_dir = os.path.join(project_root, data_paths['base_paths']['final_data'])

# Read BEA Results Validation Constant 10% data
country_effects = pd.read_csv(os.path.join(validations_dir, 'BEA Results Validations/Constant 10%/figures/02_country_effects.csv'))
product_effects = pd.read_csv(os.path.join(validations_dir, 'BEA Results Validations/Constant 10%/figures/03_product_effects.csv'))

# Read HS to BEA import data
import_data = pd.read_csv(os.path.join(hs_to_bea_data_dir, 'data', 'working', '04_Aggregate_BEA_and_HS', 'aggregated_data', 'country_usummary', 'all_continents_usummary.csv'))

# Create country-level aggregation
country_imports = import_data.groupby('iso3')['impVal'].sum().reset_index()
country_imports['import_rank'] = country_imports['impVal'].rank(method='dense', ascending=False).astype(int)

# Create product-level aggregation  
product_imports = import_data.groupby('usummary_code')['impVal'].sum().reset_index()
product_imports['import_rank'] = product_imports['impVal'].rank(method='dense', ascending=False).astype(int)

# Add ranking columns to effects data - separate ranks for total, direct, and indirect
country_effects['total_effect_rank'] = country_effects['Total'].rank(method='dense', ascending=False).astype(int)
country_effects['direct_effect_rank'] = country_effects['Direct'].rank(method='dense', ascending=False).astype(int)
country_effects['indirect_effect_rank'] = country_effects['Indirect'].rank(method='dense', ascending=False).astype(int)

product_effects['total_effect_rank'] = product_effects['Total'].rank(method='dense', ascending=False).astype(int)
product_effects['direct_effect_rank'] = product_effects['Direct'].rank(method='dense', ascending=False).astype(int)
product_effects['indirect_effect_rank'] = product_effects['Indirect'].rank(method='dense', ascending=False).astype(int)

# Merge country data
country_comparison = pd.merge(country_effects, country_imports, left_on='ISO', right_on='iso3', how='inner')
country_comparison = country_comparison[['ISO', 'country_name', 'total_effect_rank', 'direct_effect_rank', 'indirect_effect_rank', 'import_rank']]

# Merge product data
product_comparison = pd.merge(product_effects, product_imports, left_on='BEA_Code', right_on='usummary_code', how='inner')
product_comparison = product_comparison[['BEA_Code', 'BEA_Description', 'total_effect_rank', 'direct_effect_rank', 'indirect_effect_rank', 'import_rank']]

# Save comparison files to BEA Results Validations/Constant 10%/figures directory
output_dir = os.path.join(validations_dir, 'BEA Results Validations', 'Constant 10%', 'figures')
os.makedirs(output_dir, exist_ok=True)
country_comparison.to_csv(os.path.join(output_dir, '04_country_comparison.csv'), index=False)
product_comparison.to_csv(os.path.join(output_dir, '04_product_comparison.csv'), index=False)

print("="*80)
print("HTML TABLE ROWS FOR VALIDATION PAGE")
print("="*80)

print("\n" + "="*50)
print("COUNTRY COMPARISON TABLE ROWS")
print("="*50)
print("<!-- Copy these rows into the country table in 99_calculations_validation.html -->")

# Generate HTML table rows for countries
for _, row in country_comparison.iterrows():
    iso = row['ISO']
    country = str(row['country_name']).replace('"', '&quot;')  # Escape quotes for HTML
    total_rank = row['total_effect_rank']
    direct_rank = row['direct_effect_rank']
    indirect_rank = row['indirect_effect_rank'] 
    import_rank = row['import_rank']
    
    print(f'<tr><td>{iso}</td><td>{country}</td><td>{total_rank}</td><td>{direct_rank}</td><td>{indirect_rank}</td><td>{import_rank}</td></tr>')

print("\n" + "="*50) 
print("PRODUCT COMPARISON TABLE ROWS")
print("="*50)
print("<!-- Copy these rows into the product table in 99_calculations_validation.html -->")

# Generate HTML table rows for products
for _, row in product_comparison.iterrows():
    bea_code = row['BEA_Code']
    description = str(row['BEA_Description']).replace('"', '&quot;').replace(',', '&#44;')  # Escape quotes and commas for HTML
    total_rank = row['total_effect_rank']
    direct_rank = row['direct_effect_rank'] 
    indirect_rank = row['indirect_effect_rank']
    import_rank = row['import_rank']
    
    print(f'<tr><td>{bea_code}</td><td>{description}</td><td>{total_rank}</td><td>{direct_rank}</td><td>{indirect_rank}</td><td>{import_rank}</td></tr>')

print("\n" + "="*80)
print("COPY THE ABOVE HTML ROWS INTO 99_calculations_validation.html")
print("="*80)