"""
TiVA Import Values R² Analysis Tables

This script compares HS-to-BEA mapped import values with TiVA data to generate
R² coefficient tables for validation analysis.
"""

import os
import sys
import json
import copy
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from colorama import Fore, Style, init
from pathlib import Path

# Initialize colorama
init()

# Add parent directory to path for shared modules
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..'))
from main_pipeline_run import get_data_path
from shared_validation_styles import get_shared_css, get_shared_javascript

# Load data paths configuration
data_paths_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_paths.json')
with open(data_paths_file, 'r') as f:
    data_paths = json.load(f)

# Load trade weights data
trade_weights_path = os.path.join(data_paths['base_paths']['working_data'], '05_Trade_weights', 'usummary_trade_weights.csv')
trade_weights_df = pd.read_csv(trade_weights_path)

# Display summary statistics
nonzero_codes_count = trade_weights_df[trade_weights_df['regional_denominator'] > 0]['usummary_code'].nunique()
print(f"{Fore.GREEN}HS-to-BEA Data: {nonzero_codes_count} unique codes with non-zero import values{Style.RESET_ALL}")

# Extract unique regions and create regional dataframes
regions = [r for r in trade_weights_df['region'].unique() if r != 'region']
regional_dfs = {}
for region in regions:
    regional_dfs[region] = trade_weights_df[trade_weights_df['region'] == region][['usummary_code', 'regional_denominator']].drop_duplicates()

# Create world denominator mapping
world_df = trade_weights_df[['usummary_code', 'world_denominator']].drop_duplicates()

# Standardize S003 code to 'Other' across all dataframes
for region in regions:
    regional_dfs[region]['usummary_code'] = regional_dfs[region]['usummary_code'].replace({'S003': 'Other'})
world_df['usummary_code'] = world_df['usummary_code'].replace({'S003': 'Other'})

# Save validation results
validation_dir = os.path.join(data_paths['base_paths']['validations'], '08_TiVA_Import_Values_R2_Tables')
os.makedirs(validation_dir, exist_ok=True)

# Load BEA hierarchy for usummary code names
bea_hierarchy_path = os.path.join(data_paths['base_paths']['working_data'], '02_HS_to_Naics_to_BEA', '02_BEA_hierarchy.csv')
bea_hierarchy_df = pd.read_csv(bea_hierarchy_path)
usummary_names = bea_hierarchy_df[['U.Summary', 'undersum title']].drop_duplicates()
usummary_names = usummary_names.rename(columns={'U.Summary': 'usummary_code', 'undersum title': 'usummary_name'})
usummary_names['usummary_code'] = usummary_names['usummary_code'].astype(str)

# Define TiVA data directory and region mapping
tiva_dir = os.path.join(data_paths['base_paths']['underlying_data_root'], 'data', 'raw', 'TiVA Tables')
REGION_MAPPING = {
    'CAN.csv': 'CAN',
    'CHN.csv': 'CHN', 
    'EUR.csv': 'Europe',
    'JPN.csv': 'JPN',
    'MEX.csv': 'MEX',
    'RoAsia.csv': 'RoAsia',
    'RoW.csv': 'RoWorld',
    'WholeWorld.csv': 'world'
}

# Define exclusion lists for different analyses
AUTOMOTIVE_CODES = ['336111', '336112', '33612', '3362BP']
CHEMICAL_CODES = ['3251', '3252', '3254', '325X']
COMPUTER_CODES = ['3341', '3342', '3344', '3345', '334X']

# Store results for all regions
regional_r2_results = {}

for tiva_filename, region_name in REGION_MAPPING.items():
    print(f"{Fore.CYAN}Processing {tiva_filename} for region {region_name}{Style.RESET_ALL}")
    
    # Load TiVA data
    tiva_path = os.path.join(tiva_dir, tiva_filename)
    tiva_df = pd.read_csv(tiva_path)
    
    # Extract TiVA import values (first and last columns)
    tiva_imports = tiva_df.iloc[:, [0, -1]].copy()
    tiva_imports.columns = ['usummary_code', 'tiva_imports']
    tiva_imports['usummary_code'] = tiva_imports['usummary_code'].astype(str)
    tiva_imports['tiva_imports'] = tiva_imports['tiva_imports'] * 1_000_000  # Convert to dollars
    
    # Get corresponding HS-to-BEA data
    if region_name == 'world':
        hs_imports = world_df.copy()
        hs_imports.columns = ['usummary_code', 'hs_imports']
    else:
        hs_imports = regional_dfs[region_name].copy()
        hs_imports.columns = ['usummary_code', 'hs_imports']
    
    hs_imports['usummary_code'] = hs_imports['usummary_code'].astype(str)
    
    # Merge HS and TiVA data
    merged_data = hs_imports.merge(tiva_imports, on='usummary_code', how='outer').fillna(0)
    
    # Prepare logged data for analysis (remove zeros and infinities)
    logged_data = merged_data.copy()
    logged_data = logged_data[(logged_data['hs_imports'] > 0) & (logged_data['tiva_imports'] > 0)]
    logged_data['hs_imports_log'] = np.log(logged_data['hs_imports'])
    logged_data['tiva_imports_log'] = np.log(logged_data['tiva_imports'])
    # Create filtered datasets for different analyses
    analysis_datasets = {
        'All Goods': logged_data,
        'No Automobiles': logged_data[~logged_data['usummary_code'].isin(AUTOMOTIVE_CODES)],
        'No Chemicals': logged_data[~logged_data['usummary_code'].isin(CHEMICAL_CODES)],
        'No Computers': logged_data[~logged_data['usummary_code'].isin(COMPUTER_CODES)],
        'No Auto, Chem, Comp': logged_data[~logged_data['usummary_code'].isin(
            AUTOMOTIVE_CODES + CHEMICAL_CODES + COMPUTER_CODES
        )]
    }
    
    # Calculate R² scores for each analysis
    r2_scores = {}
    for analysis_name, dataset in analysis_datasets.items():
        if len(dataset) > 1:  # Ensure we have enough data points
            r2_scores[analysis_name] = r2_score(dataset['hs_imports_log'], dataset['tiva_imports_log'])
        else:
            r2_scores[analysis_name] = np.nan
    
    regional_r2_results[region_name] = {'r2_scores': r2_scores}

region_names = {
    'CAN': 'Canada',
    'CHN': 'China',
    'Europe': 'Europe',
    'JPN': 'Japan',
    'MEX': 'Mexico',
    'RoAsia': 'Rest of Asia',
    'RoWorld': 'Rest of World',
    'world': 'World Total'
}
# Create DataFrame with R² scores for all regions
r2_results_df = (
    pd.DataFrame({k: v["r2_scores"] for k, v in regional_r2_results.items()})
        .T # rows: region_name
        .rename_axis("Country/Region")
)
r2_results_df.index = r2_results_df.index.map(region_names)



# Create progressive LaTeX tables (1 column, then 2, then 3, etc.)
table_configs = [
    {"cols": 1, "suffix": "all_goods", "title": "All Goods"},
    {"cols": 2, "suffix": "excl_auto", "title": "Excluding Automobiles"},
    {"cols": 3, "suffix": "excl_auto_chem", "title": "Excluding Automobiles and Chemicals"},
    {"cols": 4, "suffix": "excl_auto_chem_comp", "title": "Excluding Automobiles, Chemicals, and Computers"},
    {"cols": 5, "suffix": "complete", "title": "Complete Analysis"}
]
footnote = r"""
\vspace{0.01cm}\\
\raggedright
{\scriptsize Note: Results from logged import values of the 43 goods BEA underlying summary codes. Total import values for each region is given as the sum of all
commodities as intermediate inputs and final uses from the Import Use table.
The BEA–TiVA tables provide trade in both goods and services.}
"""
for config in table_configs:
    # Select subset of columns
    subset_df = r2_results_df.iloc[:, :config["cols"]]
    
    # Generate LaTeX table
    latex_table = subset_df.round(3).to_latex(
        index=True,
        header=True,
        float_format="%.3f",
        caption=f"$R^2$ of BEA--TiVA Import Values vs. HS‐to‐BEA Import Values: {config['title']}",
        label=f"correlation_coefficent_regions",
        column_format="l" + "c" * len(subset_df.columns),
        escape=False,
        position = "htbp",
        na_rep="."
    )
    # Add footnote
    full_table = latex_table.replace(r"\end{tabular}", r"\end{tabular}" + footnote)
    full_table = full_table.replace(r"\begin{table}[htbp]", r"\begin{table}[htbp]"+"\n"+"\centering")
    # Save table
    table_path = os.path.join(validation_dir, f'R2_table_{config["suffix"]}.tex')
    Path(table_path).write_text(full_table)
    
    print(f"{Fore.GREEN}Created LaTeX table: {table_path}{Style.RESET_ALL}")
