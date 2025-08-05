import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy import linalg
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
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
#############################################################################################################################################
# Read in the specific BEA TiVA Regions 
europe_df = pd.read_csv(os.path.join(os.path.dirname(project_root), 'Map BEA Regions', 'data', 'final', 'BEA_TiVA_Europe.csv'))
asia_pacific_df = pd.read_csv(os.path.join(os.path.dirname(project_root), 'Map BEA Regions', 'data', 'final', 'BEA_TiVA_Asia_and_Pacific.csv'))
EUR_iso_codes = set(europe_df['iso3'].dropna().unique())
RoAsia_iso_codes = set(asia_pacific_df['iso3'].dropna().unique())

# Direct and Indirect BEA Matrices for Calculations. 
with open(os.path.join(final_data_dir, 'direct_BEA_matrix_2023.json'), 'r') as f:
    direct = json.load(f)
with open(os.path.join(final_data_dir, 'indirect_BEA_matrix_2023.json'), 'r') as f:
    indirect = json.load(f)

# Load BEA industry mappings for better output labeling
# Get the sum_order_tiva from the TiVA ordering (same as in 01_read_in_pce.py)
q_tiva = pd.read_csv(os.path.join(hs_to_bea_data_dir, 'data', 'raw', 'BEA_codes', 'q.csv'))
sum_order_tiva = q_tiva['U.Summary Code'].tolist()
# Apply same transformations as in 01_read_in_pce.py
sum_order_tiva = ['Other' if x in ['S003', 'S009'] else x for x in sum_order_tiva]
sum_order_tiva = list(dict.fromkeys(sum_order_tiva))  # Remove duplicates while preserving order

# Load BEA hierarchy for industry descriptions
bea_hierarchy = pd.read_csv(os.path.join(hs_to_bea_data_dir, 'data', 'working', '02_HS_to_Naics_to_BEA', '02_BEA_hierarchy.csv'))
bea_descriptions = bea_hierarchy[['U.Summary', 'undersum title']].drop_duplicates()
bea_descriptions.columns = ['BEA_Code', 'BEA_Description']

# Create BEA industry mapping DataFrame that matches the 140 industries
bea_industry_mapping = pd.DataFrame({
    'BEA_Industry': range(140),
    'BEA_Code': sum_order_tiva[:140]  # Ensure we only take 140 codes
})

# Merge with descriptions
bea_industry_mapping = bea_industry_mapping.merge(bea_descriptions, on='BEA_Code', how='left')

matrices = {
    'direct': {key.replace('data_', '').replace('data', 'data'): np.array(value) 
                for key, value in direct.items() if key.startswith('data')},
    'indirect': {key.replace('data_', '').replace('data', 'data'): np.array(value) 
                for key, value in indirect.items() if key.startswith('data')}
}

# Validate that the codes are in the correct size and shape for what the calculations
print("=" * 50)
print("DIRECT BEA MATRICES")
print("=" * 50)
for name, matrix in matrices['direct'].items():
    rows, cols = matrix.shape
    color = '\033[92m' if matrix.shape == (140, 140) else '\033[91m'
    print(f"{color}direct_BEA_{name}: {rows} rows, {cols} columns\033[0m")

print("\n" + "=" * 50)
print("INDIRECT BEA MATRICES")
print("=" * 50)
for name, matrix in matrices['indirect'].items():
    rows, cols = matrix.shape
    color = '\033[92m' if matrix.shape == (140, 140) else '\033[91m'
    print(f"{color}indirect_BEA_{name}: {rows} rows, {cols} columns\033[0m")
print("\n" + "=" * 50)
print("BEA CALCULATION WORK BEGINS")
print("=" * 50)

# Read in the trade data and the BEA HS section weights. 
with open(os.path.join(hs_to_bea_data_dir, 'data', 'final', 'bea_hs_section_weights.json'), 'r') as f:
    bea_hs_section_weights = json.load(f)

with open(os.path.join(hs_to_bea_data_dir, 'data', 'final', 'trade_weights.json'), 'r') as f:
    trade_weights = json.load(f)

with open(os.path.join(raw_data_dir, 'Trade War Tracker Data', 'jsons', 'hsSection_tariffs_CURRENT_VERSION.json'), 'r') as f:
    tariffs = json.load(f) 

"""
What is this code doing here? Simple, we are reading in the trade_weights.json data
which contains information on the share of each BEA Code's total import to the US 
that originate with that given ISO code. 

Then we also read in the bea_hs_section_weights.json data which tells us the importance
of each HS section to the total BEA underlying summary code. This let's us calculate the 
effective tariff on that BEA code

So the process is as follows: input HS-Section level tariffs, they get multiplied by the country 
specific bea_hs_section weights, which gives us the increase in the effective tariff rate on 
that country and BEA code (coarse aggregation at the HS section level). 
"""
def determine_data(iso_code): 
    """
    Description: This function determines the data type for each ISO code based 
    on the 7 bea regions in the data: CAN, CHN, MEX, JAP, EUR, RoW, RoAsia 
    
    Args:
        iso_code: the country iso code being run at the moment. 
    """
    if iso_code == "CAN":
        return "CAN"
    elif iso_code == "CHN":
        return "CHN" 
    elif iso_code == "JPN":
        return "JAP"
    elif iso_code == "MEX":
        return "MEX"
    elif iso_code in EUR_iso_codes:
        return "EUR"
    elif iso_code in RoAsia_iso_codes:
        return "RoAsia" 
    else:
        return "RoW"

def calculate_pct_chg_tariffs(iso_code):
    tariff_changes = {}
    country_data = tariffs["HSSection_Tariffs_07_09"][iso_code]
    for section in range(1, 22):
        section_str = str(section)
        original = country_data["original"][section_str]
        current = country_data["current"][section_str]
        tariff_changes[section_str] = (current - original)/(100+original)
    return tariff_changes

def calculate_tradeWeighted_tariffs(iso_code, constant_rate = None, method = 'indirect'):
    """
    Description: This calculate the particular trade-weighted tariff rate for a given 
    country ISO3 code. This is the Tau object that we use in the calculations. 
    Args: 
        iso_code (str): The ISO3 code of the country for which to calculate the tariff rate.
        constant_rate (float, optional): If provided, this will be used as a constant rate instead of calculating from the Trade War Tracker data.
        method (str): method for using the trade weights, either 'direct' or 'indirect'. direct means that its the global share, indirect means that its the regional share TiVA formatted.
    """
    if constant_rate is None:
        tariff_changes = calculate_pct_chg_tariffs(iso_code) # TWT Data for the HS Sections (21)
    else: 
        tariff_changes = {str(i): constant_rate for i in range(1, 22)} # Constant rate for all sections
    # Calculate the BEA Code level effective tariff (i.e. apply the bea_hs_section_weights) by code.
    bea_tariff = {} # This is an empty mapping where we will store the BEA level tariffs as we calculate them. 
    country_data = bea_hs_section_weights[iso_code]
    bea_codes = country_data.keys()
    for code in bea_codes:
        for section, weight in country_data[code].items():
            if section in tariff_changes:
                bea_tariff[code] = bea_tariff.get(code, 0) + weight * tariff_changes[section] # this is the BEA level effective tariff rate on that country and bea code. 
    
    country_trade_weights = trade_weights[method][iso_code]
    tau_vector_dict = {key: 0.0 for key in country_trade_weights} # This creates the final output vector which is a trade weighted vector of the bea level tariffs. 
    for code in country_trade_weights: 
        if code in bea_tariff:
            tau_vector_dict[code] = bea_tariff[code] * country_trade_weights[code]
        else:
            tau_vector_dict[code] = 0.0
    
    tau_vector = np.array(list(tau_vector_dict.values()))  # Convert the dictionary to a numpy array for calculations
    
    return tau_vector, tau_vector_dict

def create_country_results(iso_code, constant_rate):
    """
    Description: This function creates the output for a given country ISO code. It uses the other
    functions to calculate the trade-weighted tariffs, and then uses them to calculate the consumer price effect from that set 
    of tariffs. It then returns the results for both the direct and indirect matrices as 1 by 140 vectors where each 
    row is a BEA industry.
    
    args: 
        iso_code (str): The ISO3 code of the country for which to calculate the tariffs.
        constant_rate (float, optional): If provided, this will be used as a constant rate instead of calculating from the Trade War Tracker (TWT) data.
    """
    tau_vector, tau_vector_dict = calculate_tradeWeighted_tariffs(iso_code, constant_rate)
    
    data = determine_data(iso_code)
    
    # Calculate the direct and indirect effects using the BEA matrices
    # tau_vector is (140,), matrices are (140, 140), so we need matrices.T to get (140, 140)
    # Then tau_vector @ matrices.T gives us (140,) which is what we want
    direct_effect = np.dot(matrices['direct'][data].T, tau_vector)
    indirect_effect = np.dot(matrices['indirect'][data].T, tau_vector)
    
    # Create DataFrames with BEA industry indices, codes, and descriptions
    direct_effects_df = pd.DataFrame({
        'BEA_Industry': range(140),
        'direct_effect': direct_effect
    })
    direct_effects_df = direct_effects_df.merge(bea_industry_mapping, on='BEA_Industry', how='left')
    
    indirect_effects_df = pd.DataFrame({
        'BEA_Industry': range(140),
        'indirect_effect': indirect_effect
    })
    indirect_effects_df = indirect_effects_df.merge(bea_industry_mapping, on='BEA_Industry', how='left')
    
    total_effects_df = pd.DataFrame({
        'BEA_Industry': range(140),
        'total_effect': direct_effect + indirect_effect
    })
    total_effects_df = total_effects_df.merge(bea_industry_mapping, on='BEA_Industry', how='left')
    
    # Calculate the total effect by summing the values in the direct effects and the indirect effects individually, 
    direct_sum = np.sum(direct_effect)
    indirect_sum = np.sum(indirect_effect)
    total_effect = direct_sum + indirect_sum
    
    # Create the results dictionary
    results = {
        'iso_code': iso_code,
        'tau_vector': tau_vector_dict,
        'direct_effects_df': direct_effects_df,
        'indirect_effects_df': indirect_effects_df,
        'total_effects_df': total_effects_df,
        'direct_sum': direct_sum,
        'indirect_sum': indirect_sum,
        'total_effect': total_effect
    }
    return results

## Function to run all the countries and save results (no hierarchical aggregation)
def run_all_countries(constant_rate, aggregate = True):
    if aggregate: 
        results = {}
        direct_effect_sum = 0
        indirect_effect_sum = 0
        total_effect_sum = 0
        
        # Collect all country data first
        country_data = {}
        for iso_code in tariffs["HSSection_Tariffs_07_09"].keys():
            if iso_code == "EU" or iso_code == "PSE":
                continue
            country_results = create_country_results(iso_code, constant_rate)
            direct_effect_sum += country_results['direct_sum']
            indirect_effect_sum += country_results['indirect_sum']
            total_effect_sum += country_results['total_effect']
            
            country_data[iso_code] = {
                'direct': country_results['direct_effects_df']['direct_effect'],
                'indirect': country_results['indirect_effects_df']['indirect_effect'],
                'total': country_results['total_effects_df']['total_effect']
            }
        
        # Create DataFrames efficiently using concat
        direct_country_df = pd.DataFrame({iso: data['direct'] for iso, data in country_data.items()})
        indirect_country_df = pd.DataFrame({iso: data['indirect'] for iso, data in country_data.items()})
        total_country_df = pd.DataFrame({iso: data['total'] for iso, data in country_data.items()})
        
        # Create aggregated DataFrames with BEA industry mapping
        aggregated_direct_df = pd.concat([bea_industry_mapping, direct_country_df], axis=1)
        aggregated_indirect_df = pd.concat([bea_industry_mapping, indirect_country_df], axis=1)
        aggregated_total_df = pd.concat([bea_industry_mapping, total_country_df], axis=1)
        
        # Add "All Countries Effect" column
        aggregated_direct_df['All Countries Effect'] = direct_country_df.sum(axis=1)
        aggregated_indirect_df['All Countries Effect'] = indirect_country_df.sum(axis=1)
        aggregated_total_df['All Countries Effect'] = total_country_df.sum(axis=1)
        
        # Determine output directory (BEA Results instead of NIPA)
        if constant_rate is None:
            output_dir = os.path.join(validations_dir, 'BEA Results Validations', 'TWT Data', 'All Countries')
        else:
            output_dir = os.path.join(validations_dir, 'BEA Results Validations', 'Constant 10%', 'All Countries')
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save BEA industry level results
        aggregated_direct_df.to_csv(os.path.join(output_dir, 'BEA_direct_effects.csv'), index=False)
        aggregated_indirect_df.to_csv(os.path.join(output_dir, 'BEA_indirect_effects.csv'), index=False)
        aggregated_total_df.to_csv(os.path.join(output_dir, 'BEA_total_effects.csv'), index=False)
        
        print(f"\nSum of all direct effects across all countries: {direct_effect_sum*100 :.5f}%")
        print(f"Sum of all indirect effects across all countries: {indirect_effect_sum*100 :.5f}%")
        print(f"Sum of all total_effects across all countries: {total_effect_sum*100 :.5f}%")
        print(f"BEA Results saved to: {output_dir}")
    else: 
        pass 
    
run_all_countries(constant_rate = 0.1, aggregate= True )
run_all_countries(constant_rate = None, aggregate= True )

# Function to read BEA effects (simplified version without hierarchical levels)
def read_BEA_effects(constant_rate=None):
    """
    Description: Reads and returns the BEA industry effects
    
    Args:
        constant_rate (float, optional): Whether to read from TWT Data (None) or Constant 10% results
    
    Returns:
        tuple: (direct_effects_df, indirect_effects_df, total_effects_df)
    """
    # Determine which directory to read from
    if constant_rate is None:
        data_dir = os.path.join(validations_dir, 'BEA Results Validations', 'TWT Data', 'All Countries')
    else:
        data_dir = os.path.join(validations_dir, 'BEA Results Validations', 'Constant 10%', 'All Countries')
    
    # Read the BEA files
    direct_df = pd.read_csv(os.path.join(data_dir, 'BEA_direct_effects.csv'))
    indirect_df = pd.read_csv(os.path.join(data_dir, 'BEA_indirect_effects.csv'))
    total_df = pd.read_csv(os.path.join(data_dir, 'BEA_total_effects.csv'))
    
    print(f"Retrieved BEA effects: {len(direct_df)} industries")
    
    return direct_df, indirect_df, total_df