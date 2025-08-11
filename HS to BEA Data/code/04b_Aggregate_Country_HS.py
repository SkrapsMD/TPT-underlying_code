import os 
import pandas as pd 
import json 
from main_pipeline_run import get_data_path

"""
This code will take the country-HS level data and keep a measure of import Value and Share for each 

hs4, hs2 and hssection code. We use these to create some weighting.json files. 
"""
data_paths_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_paths.json')
with open(data_paths_file, 'r') as f:
    data_paths = json.load(f)

def print_validation(data):
    print(data.columns )
    # Sum for Canada, China, Mexico. 
    countries = ['CAN','CHN','MEX']
    for country in countries:
        print(f"Validation for {country}:")
        print(data[data['iso3'] == country]['impVal'].sum())
    print("\n Validation for Whole World")
    print(data['impVal'].sum())

def create_bea_json(df, output_dir, varname):
    """
    This creates the generalized json file for the 2024 import values so we can more effectively create 
    import weights. 
    
    structure will be data[isoCode][HS_Section/HS2/HS4] - impVal
    and a special isoCode for the World. 
    """
    result = {} 
    for country_iso3 in df['iso3'].unique():
        result[country_iso3] = {}
        country_df = df[df['iso3'] == country_iso3]
        for _, row in country_df.iterrows():
            code = str(row[varname])
            impVal = row['impVal']
            result[country_iso3][code] = impVal

    with open(output_dir, 'w') as f:
        json.dump(result, f)

base_input_dir = os.path.join(data_paths['base_paths']['working_data'], '04_Aggregate_BEA_and_HS', 'hs_weights')

hs_types = {'hs_section':'HS_Section','hs2':'HS2','hs4':'HS4'}
for type, varname in hs_types.items():
    data = pd.read_csv(os.path.join(base_input_dir, 'detail' ,f'{type}_weights.csv'), index_col = False)
    data = data[['iso3',varname, 'impVal']]
    data = data.groupby(['iso3',varname]).agg({'impVal':'sum'})
    data.reset_index(inplace = True)
    print_validation(data) # This looks good, I don't think we are missing anything obvious here. 

    global_data = data.groupby(varname).agg({'impVal':'sum'}).reset_index()
    global_data['iso3'] = 'GLOBAL'  
    data = pd.concat([data, global_data], axis = 0).set_index('iso3')
    data.to_csv(os.path.join(base_input_dir, 'country_HS' ,f'{varname}.csv'), index = True)
    
    # Output to .json file 
    output_dir = os.path.join(data_paths['base_paths']['final_data'], f'country_{varname}_impVals.json')
    create_bea_json(data.reset_index(), output_dir, varname)
