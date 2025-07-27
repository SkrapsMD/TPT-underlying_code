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

# TWT Scenario 
TWT_direct = pd.read_csv(os.path.join(validations_dir,'Results Validations/TWT Data/All Countries/disaggregated/disaggregated_direct_effects.csv'))
TWT_indirect = pd.read_csv(os.path.join(validations_dir,'Results Validations/TWT Data/All Countries/disaggregated/disaggregated_indirect_effects.csv'))
TWT_total = pd.read_csv(os.path.join(validations_dir,'Results Validations/TWT Data/All Countries/disaggregated/disaggregated_total_effects.csv'))

# 10% Scenario
TEN_direct = pd.read_csv(os.path.join(validations_dir,'Results Validations/Constant 10%/All Countries/disaggregated/disaggregated_direct_effects.csv'))
TEN_indirect = pd.read_csv(os.path.join(validations_dir,'Results Validations/Constant 10%/All Countries/disaggregated/disaggregated_indirect_effects.csv'))
TEN_total = pd.read_csv(os.path.join(validations_dir,'Results Validations/Constant 10%/All Countries/disaggregated/disaggregated_total_effects.csv'))

