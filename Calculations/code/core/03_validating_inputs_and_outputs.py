import os
import json
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import warnings
from pandas.errors import PerformanceWarning
CORNFLOWERBLUE   = '\033[38;2;100;149;237m'
INDIANRED = '\033[38;2;205;92;92m'
RESET     = '\033[0m'
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
###############################################################################################
# SET UP USEFUL FUNCTIONS
def check_matrix(data, id="Summary Code"):
    # 1.) check if Square
    n_rows = data.shape[0]
    n_cols = data.shape[1]-1
    print(f"Matrix Shape (excluding ID): {n_rows} x {n_cols}")
    if n_rows != n_cols:
        print(f"{INDIANRED}Warning!!!: The Matrix is not square: {n_rows} x {n_cols} {RESET}")
    # 2.) Check Mapping existence    
    cols_data = [c for c in data.columns if c != id]
    row_ids = data[id].tolist()
    unique_ids = set(row_ids)
    
    missing = [h for h in unique_ids if h not in cols_data]
    if missing:
        print(f"{INDIANRED}Warning!!!: The following codes are missing from the data: {missing} {RESET}")
        mapping_exist = False
    else:
        mapping_exist = True
        print(f"{CORNFLOWERBLUE}Success!!!: The mapping exists for all codes {RESET}")
    
    # 3.) Ordering
    if mapping_exist:
        if row_ids == cols_data:
            print(f'{CORNFLOWERBLUE}Success!!!: The ordering is correct {RESET}')
        else:
            print(f'{INDIANRED} Warning: Row HEaders do not match column Headers. {RESET} Mismatches:')
            for i, (rid, col) in enumerate(zip(row_ids, cols_data)):
                if rid != col: 
                    print(f"     at position {i}: row-header = '{rid}' vs column-header = '{col}'")
    else: 
            print("Skipping order since some identifiers were missing. ")


################################################################################################

# Load BEA Mapping
map = pd.read_csv(os.path.join(hs_to_bea_data_dir, 'data', 'working', '02_HS_to_Naics_to_BEA', '02_BEA_hierarchy.csv'))


# years = [2017, 2019, 2022, 2023] # If we want to do this again we need to make sure we are very explicit about the way wae treat the TiVA import stuff. 
years = [2023]

# Key Row and Column NAmes: 
total_intermediate_row = "Total Intermediate"
total_intermediate_column = "Total Intermediate"
total_value_added_row = "Total Value Added"
total_output_row = "Total Industry Output"

total_PCE_column = "F010" # for TiVA Use
total_compensation_row = "V001" # TiVA Use
total_commodity_column = "Total Commodity Output" # for TiVA Use


tiva_use_data ={} # Full Data
tiva_use_inputs = {} # Intermediate Input Section / Matrix
tiva_use_intermediates = {} # Total Intermedaite Inputs Row (just for the industries)
tiva_use_valueAdded ={} # Total Value Added Row (just for the industries)
tiva_use_outputs ={} # Total Output Section (just for the industries)
tiva_use_com_output ={} # Total Output of Commodity (column)
tiva_use_compensation = {} # Total Compensation Row (industries)
for year in years: 
    tiva_use_data[year] = pd.read_csv(os.path.join(working_data_dir, 'TiVA Tables',f'{year}', f"useTiVA_{year}.csv"))
    intermediate_row = tiva_use_data[year][tiva_use_data[year]['U.Summary Code']== total_intermediate_row].index[0]
    intermediate_column = tiva_use_data[year].columns.get_loc(total_intermediate_column)
    value_added_row = tiva_use_data[year][tiva_use_data[year]['U.Summary Code']== total_value_added_row].index[0]
    output_row = tiva_use_data[year][tiva_use_data[year]['U.Summary Code']== total_output_row].index[0]
    compensation_row = tiva_use_data[year][tiva_use_data[year]['U.Summary Code']== total_compensation_row].index[0]
    
    tiva_use_inputs[year] = tiva_use_data[year].iloc[:intermediate_row, :intermediate_column]
    tiva_use_intermediates[year] = tiva_use_data[year].iloc[intermediate_row:intermediate_row+1, :intermediate_column]
    
    tiva_use_intermediates[year]['Used'] = 0
    tiva_use_intermediates[year]['Other'] = 0
    
    tiva_use_valueAdded[year] = tiva_use_data[year].iloc[value_added_row:value_added_row+1, :intermediate_column]
    tiva_use_outputs[year] = tiva_use_data[year].iloc[output_row:output_row+1, :intermediate_column]
    tiva_use_outputs[year]['Used']=0
    tiva_use_outputs[year]['Other'] = 0
    
    com_output_column = tiva_use_data[year].columns.get_loc(total_commodity_column)
    names_col = tiva_use_data[year].columns.get_loc('U.Summary Code')
    cols_to_extract = [names_col, com_output_column]
    tiva_use_com_output[year] = tiva_use_data[year].iloc[:intermediate_row, cols_to_extract]
    
    tiva_use_compensation[year] = tiva_use_data[year].iloc[compensation_row:compensation_row+1, :intermediate_column]
    tiva_use_compensation[year]['Used'] = 0
    tiva_use_compensation[year]['Other'] = 0

    tiva_use_inputs[year]['Used'] = 0
    tiva_use_inputs[year]['Other'] = 0
    print(f"{INDIANRED} Checking the  TiVA Use Input Matrix for at U.Summary Level {RESET}")
    check_matrix(tiva_use_inputs[year], id = 'U.Summary Code')
    

impGroups = ['CAN','CHN','EUR','JAP', 'MEX','RoAsia','RoW']

tiva_import_inputs = {} # Just the Intermediate input section (the data is already in this format...)
tiva_import_intermediates = {} # Total Intermediates for each Industry (column). 
tiva_import_groups = {}

for year in years: 
    tiva_import_inputs[year] = pd.read_csv(os.path.join(working_data_dir, 'TiVA Tables', f'{year}', f'importTiVA_{year}.csv'))
    
    # For 2023 data, truncate to only include columns up to GSLE to match older data structure
    if year == 2023 and 'GSLE' in tiva_import_inputs[year].columns:
        intermediate_column = tiva_import_inputs[year].columns.get_loc('GSLE')
        tiva_import_inputs[year] = tiva_import_inputs[year].iloc[:, :intermediate_column + 1]
    
    tiva_import_inputs[year]['Used'] = 0
    tiva_import_inputs[year]['Other'] = 0  
    print(f"{INDIANRED} Checking the  TiVA Import Input Matrix for at U.Summary Level {RESET}")
    check_matrix(tiva_import_inputs[year], id = 'U.Summary Code')
    
    tiva_import_groups[year] = {}
    for country in impGroups:
        country_file_path = os.path.join(working_data_dir, 'TiVA Tables', f'{year}', f'impGroups', f'imp_{country}_{year}.csv')
        try:
            tiva_import_groups[year][country]= pd.read_csv(country_file_path)
            tiva_import_groups[year][country]['Used'] = 0
            tiva_import_groups[year][country]['Other'] = 0  
            print(f"{INDIANRED} Checking the TiVA Import Input Matrix for {country} at U.Summary Level {RESET}")
            check_matrix(tiva_import_groups[year][country], id='U.Summary Code')
        except FileNotFoundError: 
            print(f"file for {country} in {year} not found. ")

        
total_commodity_output_row = "Total Commodity Output" # For TiVA Make table  -- so long as we do the same cleaning... 
total_industry_output_col= "Total Industry Output" # For TiVA Make table  -- so long as we do the same cleaning... 

tiva_make_data = {}
tiva_make_outputs = {} # Intermediate part of the make table
tiva_make_commodity_output = {} # Total Output for each commodity (useful for the market share table) 
tiva_make_industry_output = {} # Total Output for each industry

for year in years:
    tiva_make_data[year] = pd.read_csv(os.path.join(working_data_dir, 'TiVA Tables', f'{year}', f'makeTiVA_{year}.csv'))
    
    #1.) Get the Total Commodity Output row
    commodity_output_row = tiva_make_data[year][tiva_make_data[year]['U.Summary Code'] == total_commodity_output_row].index[0]
    tiva_make_commodity_output[year] = tiva_make_data[year].iloc[commodity_output_row:commodity_output_row+1, :-1]
    
    # Add the 0 vectors for the Used and Other industries
    tiva_make_data[year] = tiva_make_data[year][tiva_make_data[year]['U.Summary Code'] != "Total Commodity Output"]
    tiva_make_data[year].loc[len(tiva_make_data[year])] = 0
    tiva_make_data[year].iloc[-1, 0] = "Used"
    tiva_make_data[year].loc[len(tiva_make_data[year])] = 0
    tiva_make_data[year].iloc[-1, 0] = "Other"
    
    #2.) Get the Total Industry Output Column
    industry_output_col = tiva_make_data[year].columns.get_loc(total_industry_output_col)
    names_col = tiva_make_data[year].columns.get_loc('U.Summary Code')
    cols_to_extract = [names_col, industry_output_col]
    tiva_make_industry_output[year] = tiva_make_data[year].iloc[:, cols_to_extract]
    
    # 3.) Get the Commodity Outputs  by Industry
    tiva_make_outputs[year] = tiva_make_data[year].iloc[:, :-1]
    print(f"{INDIANRED} Checking the Make Output Matrix for TiVA at U.Summary Level {RESET}")

    check_matrix(tiva_make_outputs[year], id = 'U.Summary Code')

# Outputs data to the working directory for components for calculations
# Variable Names and Definitions:  NOTE TO SIMON AND SALOME -- A lot of this hierarchy has BEEN REMOVED TO MAKE THIS STRUCTURE A BIT SMALLER AND LESS CONFUSING. 
"""
m - Number of Industries
n - Number of Commodities
-------------------------------------------------

g - Total Industry Output (Use Table) [1 x m]  *

q - Total Commodity Output (Use Table) [n x 1]  *

c - Total Compensation of Employees (Use Table) [1 x m]

U - Intermediate Inputs from Use Table [n x m] -- includes Other and Used  (0)

V - Output from the Make Table [m x n] -- includes Other and Used  (0)

M - Input matrix from the Import Use Table [n x m] -- includes Other and Used (0)

--------------------------------------------------
* - can be switched out for the Make table. 

Data gets saved to the data/working/Components for Calculations directory under:

|-- BEA
|    | -- 71
|    |     |-- 2017
|    |     |-- 2019
|    |     |-- 2022
|-- TiVA
|    | -- 71
|    |     |-- 2017
|    |     |-- 2019
|    |     |-- 2022
|    | -- 138
|    |     |-- 2017
|    |     |-- 2019
|    |     |-- 2022

Their names will be reflected as their variable names: M.csv -> M matrix, etc. 

They will always come with the name column included (U.Summary Code of Summary Code).
"""

# g 
# summary_use_outputs
# tiva_use_outputs 

# q
# summary_use_com_output
# tiva_use_com_output

# c
# summary_use_compensation
# tiva_use_compensation

# u
# summary_use_intermediates
# tiva_use_intermediates

# U
# summary_use_inputs
# tiva_use_inputs

# V
# summary_make_outputs -- We don't have this table yet... need to create it in the reading in step.... Raw Data is there, need to read it in....
# tiva_make_outputs 

# M 
# summary_import_inputs
# tiva_import_inputs

for year in years: 
    g = tiva_use_outputs[year]
    g.to_csv(os.path.join(calculations_dir, 'TiVA', '138', f'{year}', 'g.csv'), index = False)
    del g
        
    q = tiva_use_com_output[year]
    q.to_csv(os.path.join(calculations_dir, 'TiVA', '138', f'{year}', 'q.csv'), index = False)
    del q 
    
    c = tiva_use_compensation[year]
    c.to_csv(os.path.join(calculations_dir, 'TiVA', '138', f'{year}', 'c.csv'), index = False)
    del c
    
    u = tiva_use_intermediates[year]
    u.to_csv(os.path.join(calculations_dir, 'TiVA', '138', f'{year}', 'u.csv'), index = False)
    del u
    
    U = tiva_use_inputs[year]
    U.to_csv(os.path.join(calculations_dir, 'TiVA', '138', f'{year}', 'U.csv'), index = False)
    del U
    
    V = tiva_make_outputs[year]
    V.to_csv(os.path.join(calculations_dir, 'TiVA', '138', f'{year}', 'V.csv'), index = False)
    del V
    
    M = tiva_import_inputs[year]
    M.to_csv(os.path.join(calculations_dir, 'TiVA', '138', f'{year}', 'M.csv'), index = False)
    del M
    for country in impGroups:
        if country in tiva_import_groups[year]:
            country_data = tiva_import_groups[year][country]
            country_data.to_csv(os.path.join(calculations_dir, 'TiVA', '138', f'{year}', 'impGroups', f'M_{country}.csv'), index = False)
    # Do the same for the Summary level data when it's

## NOTE: The w_m_star_* and w_d_star variables were calculated by hand in excel (it was just a simple element wise division of the PCE from the Use table by the PCE from the representative import tables. 
# It is super easy to validate. by hand, you will see its correct. I will write this explicitly later. Will be easy, but low VA right now imo. 