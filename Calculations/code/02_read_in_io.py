import os 
import time
import json
import pickle
import numpy as np
import pandas as pd 
from tqdm import tqdm
# Load data paths and set up standard directory variables
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)  # Go up one level to Calculations/
data_paths_file = os.path.join(project_dir, "data_paths.json")
with open(data_paths_file, 'r') as f:
    data_paths = json.load(f)
# Set up clean directory variables using project_root from config
project_root = data_paths['base_paths']['project_root']
raw_data_path = os.path.join(project_root, data_paths['base_paths']['raw_data'])
working_data_path = os.path.join(project_root, data_paths['base_paths']['working_data'])
calculations_dir = os.path.join(working_data_path, "Components for Calculations")
validations_dir = os.path.join(project_root, data_paths['base_paths']['validations'])
hs_to_bea_data_dir = os.path.join(project_root, data_paths['base_paths']['hs_to_bea_data'])


bea_map = pd.read_csv(os.path.join(hs_to_bea_data_dir, 'data', 'working', '02_HS_to_Naics_to_BEA', '02_BEA_hierarchy.csv'))
bea_map_cln = bea_map[['Sector','Summary','U.Summary','Detail']]

# Define the years we want to process
years = [2017, 2019, 2022, 2023]

# Loop over the years
for year in years:
    # Process in the TiVA Tables:
    tiva_code_maps = {}

    # Adjust file paths to include the year
    importTiVA = pd.read_csv(os.path.join(raw_data_path, 'TiVA Tables', str(year), f'import_{year}.csv'), header = 3)
    useTiVA = pd.read_csv(os.path.join(raw_data_path, 'TiVA Tables', str(year), f'use_{year}.csv'), header = 3)
    makeTiVA = pd.read_csv(os.path.join(raw_data_path, 'TiVA Tables', str(year), f'make_{year}.csv'), header = 3)

    tiva_files = {
        f'importTiVA_{year}': importTiVA,
        f'useTiVA_{year}': useTiVA,
        f'makeTiVA_{year}': makeTiVA
    }

    for name, df in tqdm(tiva_files.items(), desc=f"Processing Core TiVA Files for {year}"):
        df = df.rename(columns = {'IOCode': 'U.Summary Code'})
        if df.columns[0].startswith("Unnamed:"):
            df.rename(columns={df.columns[0]: "U.Summary Code"}, inplace=True)
        # Fill missing column headers with the first non-column header value from row_mapping
        for idx, col in enumerate(df.columns):
            if col.startswith("Unnamed: "):
                new_name = df.iloc[0, idx]
                print(f"\033[92mReplacing column '{col}' with '{new_name}' at index {idx}.\033[0m")
                df.rename(columns={col: new_name}, inplace=True)
        
        row_mapping = dict(zip(df.columns[2:].astype(str), df.iloc[0, 2:].astype(str)))  # Exclude first two columns
        column_mapping = dict(zip(df.iloc[2:, 0].astype(str), df.iloc[2:, 1].astype(str)))  # Exclude first two rows
        tiva_code_maps.update(row_mapping)
        tiva_code_maps.update(column_mapping)
        
        # Handle missing U.Summary Code
        if df['U.Summary Code'].isna().any():
            second_column = df.columns[1]
            for idx, value in df[df['U.Summary Code'].isna()][second_column].items():
                print(f"\033[92mMissing 'U.Summary Code' at index {idx} was replaced with '{value}' from column '{second_column}'.\033[0m")
                df.at[idx, 'U.Summary Code'] = value  # Use .at for setting values
            print(f"Filled missing 'U.Summary Code' values using '{second_column}' column.")
        
        df = df[df['U.Summary Code'] != "IOCode"]
        df = df[df['U.Summary Code'] != 'Name']
        df = df[df['U.Summary Code'] != "Legend/Footnotes"]
        for col in df.columns:
            if col == "Industries/Commodities":
                df = df.drop(columns = {'Industries/Commodities'}) 
            elif col == "Commodities/Industries":
                df = df.drop(columns = {'Commodities/Industries'})
            
        df.fillna(0, inplace=True)
        df.to_csv(os.path.join(working_data_path, 'TiVA Tables', str(year), name + '.csv'), index = False)
        # Create a mapping of to the Sector level so we can compare at several levels: 
        if name == f"useTiVA_{year}":
            df = df[['U.Summary Code', 'F010']]
            df = df.rename(columns = {'F010': 'TiVA USum PCE'})
            df['TiVA USum PCE'] = df['TiVA USum PCE'].astype(float)
            df = pd.merge(df, bea_map_cln, left_on = 'U.Summary Code', right_on = 'U.Summary', how='left')
            df = df.drop(columns = {'Detail'})
            df = df.drop_duplicates()
            df['TiVA USum PCE'] = df.groupby('U.Summary')['TiVA USum PCE'].transform('sum')
            df['TiVA Summary PCE'] = df.groupby('Summary')['TiVA USum PCE'].transform('sum')
            df['TiVA Sector PCE'] = df.groupby('Sector')['TiVA USum PCE'].transform('sum')
            df = df[['U.Summary', 'Summary','Sector', 'TiVA USum PCE', 'TiVA Summary PCE', 'TiVA Sector PCE']]
            df.to_csv(os.path.join(working_data_path, 'TiVA Tables', 'PCE', name + '.csv'), index = False)
            #df.to_csv(os.path.join(working_data_path, 'TiVA Tables',f'{year}' ,f'PCE_{year}' + '.csv'), index = False)


    for import_file in tqdm([f for f in os.listdir(os.path.join(raw_data_path, 'TiVA Tables', str(year), 'impGroups')) if f.endswith('.csv')], desc=f"Processing TiVA Import Groups for {year}"):
        import_file_name = import_file.split('.')[0]
        import_file = pd.read_csv(os.path.join(raw_data_path, 'TiVA Tables', str(year), 'impGroups', import_file), header = 3)
        if import_file.columns[0].startswith("Unnamed:"):
            import_file.rename(columns={import_file.columns[0]: "U.Summary CDode"}, inplace=True)
        # Fill missing column headers with the first non-column header value from row_mapping
        for idx, col in enumerate(import_file.columns):
            if col.startswith("Unnamed: "):
                new_name = import_file.iloc[0, idx]
                print(f"\033[92mReplacing column '{col}' with '{new_name}' at index {idx}.\033[0m")
                import_file.rename(columns={col: new_name}, inplace=True)
        
        row_mapping = dict(zip(import_file.columns[2:].astype(str), import_file.iloc[0, 2:].astype(str)))  # Exclude first two columns
        column_mapping = dict(zip(import_file.iloc[2:, 0].astype(str), import_file.iloc[2:, 1].astype(str)))  # Exclude first two rows
        tiva_code_maps.update(row_mapping)
        tiva_code_maps.update(column_mapping)
        import_file = import_file[import_file['IOCode']!="IOCode"]
        
        # For 2023 data, truncate to only include columns up to GSLE to match older data structure 
        if year == 2023 and 'GSLE' in import_file.columns:
            gsle_index = import_file.columns.get_loc('GSLE')
            import_file = import_file.iloc[:, :gsle_index + 1]
        
        import_file = import_file.drop(columns = {'Commodities/Industries'}) 
        import_file = import_file.rename(columns={'IOCode': 'U.Summary Code'})
        
        # For 2023 data, create PCE files from impGroups since they now have F010 column
        if year == 2023 and 'F010' in import_file.columns:
            pce_df = import_file[['U.Summary Code', 'F010']].copy()
            pce_df = pce_df.rename(columns = {'F010': 'TiVA USum PCE'})
            pce_df['TiVA USum PCE'] = pce_df['TiVA USum PCE'].astype(float)
            pce_df = pd.merge(pce_df, bea_map_cln, left_on = 'U.Summary Code', right_on = 'U.Summary', how='left')
            pce_df = pce_df.drop(columns = {'Detail'})
            pce_df = pce_df.drop_duplicates()
            pce_df['TiVA USum PCE'] = pce_df.groupby('U.Summary')['TiVA USum PCE'].transform('sum')
            pce_df['TiVA Summary PCE'] = pce_df.groupby('Summary')['TiVA USum PCE'].transform('sum')
            pce_df['TiVA Sector PCE'] = pce_df.groupby('Sector')['TiVA USum PCE'].transform('sum')
            pce_df = pce_df[['U.Summary', 'Summary','Sector', 'TiVA USum PCE', 'TiVA Summary PCE', 'TiVA Sector PCE']]
            pce_df.to_csv(os.path.join(working_data_path, 'TiVA Tables', 'PCE', import_file_name + '.csv'), index = False)
        
        import_file.fillna(0, inplace=True)
        
        zero_code_index = import_file[import_file['U.Summary Code'] == "Other"].index
        if not zero_code_index.empty:
            import_file = import_file.iloc[:zero_code_index[0] + 1]
        import_file = import_file[import_file['U.Summary Code'] != "Legend/Footnotes"]
        
        import_file.to_csv(os.path.join(working_data_path, 'TiVA Tables', str(year), 'impGroups', import_file_name + '.csv'), index = False)
    with open(os.path.join(working_data_path, 'TiVA Tables', str(year), f'tiva_codes_desc_map_{str(year)}.pkl'), 'wb') as f:
        pickle.dump(tiva_code_maps, f)