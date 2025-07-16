import os
import subprocess
import sys
from datetime import datetime

"""
HS to BEA Data Pipeline Runner

This script runs the complete pipeline for constructing trade weights and import values
by mapping HS commodity codes to BEA economic categories through NAICS codes.

The pipeline processes raw trade data through multiple stages:
1. NAICS crosswalk formation
2. HS-NAICS mapping compilation  
3. HS-NAICS-BEA mapping creation
4. Country trade data processing
5. BEA aggregation and HS weighting
6. Trade weights calculation
7. Initial data validation
8. TiVA benchmark comparison

All outputs are saved to data/working/, data/final/, and validations/ folders.
"""

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'      # Success
    RED = '\033[91m'        # Error/Failed  
    BLUE = '\033[94m'       # General notices
    BOLD = '\033[1m'
    END = '\033[0m'

def colored_print(message, color=Colors.BLUE):
    """Print message with color formatting"""
    print(f"{color}{message}{Colors.END}")

def run_script(script_name, description):
    """Run a Python script and handle errors."""
    print(f"\n{'='*60}")
    colored_print(f"RUNNING: {script_name}", Colors.BLUE)
    colored_print(f"DESCRIPTION: {description}", Colors.BLUE)
    print(f"{'='*60}")
    
    script_path = os.path.join("code", script_name)
    
    try:
        # Run the script and capture output
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, 
                              text=True, 
                              cwd=os.getcwd())
        
        if result.returncode == 0:
            colored_print(f"SUCCESS: {script_name} completed successfully", Colors.GREEN)
            if result.stdout:
                colored_print("OUTPUT:", Colors.BLUE)
                print(result.stdout)
        else:
            colored_print(f"ERROR: {script_name} failed with return code {result.returncode}", Colors.RED)
            if result.stderr:
                colored_print("ERROR OUTPUT:", Colors.RED)
                print(result.stderr)
            if result.stdout:
                colored_print("STDOUT:", Colors.BLUE)
                print(result.stdout)
            return False
            
    except Exception as e:
        colored_print(f"EXCEPTION: Failed to run {script_name}: {str(e)}", Colors.RED)
        return False
    
    return True

def main():
    """Run the complete HS to BEA data pipeline."""
    
    # Ensure we're in the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    colored_print("STARTING HS TO BEA DATA PIPELINE", Colors.BLUE)
    colored_print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", Colors.BLUE)
    colored_print(f"Working directory: {os.getcwd()}", Colors.BLUE)
    
    # Pipeline steps with descriptions
    pipeline_steps = [
        ("00_Naics_crosswalk_formation.py", 
         "Creates NAICS 2017-2022 crosswalk mapping for consistent code translation"),
        
        ("01_Schott_Data_Compiler.py", 
         "Compiles and corrects HS-NAICS mappings from Schott data using hierarchical matching"),
        
        ("02_HS_to_Naics_to_BEA.py", 
         "Maps HS codes through NAICS to BEA economic categories for trade analysis"),
        
        ("03_Map_country_trade_data.py", 
         "Processes 2024 trade data by continent and applies HS-to-BEA mappings"),
        
        ("04_Aggregate_BEA_and_HS.py", 
         "Aggregates trade data by BEA categories and creates HS commodity weights"),
        
        ("05_Trade_weights.py", 
         "Calculates final trade weights by region and BEA category for economic analysis"),
        
        ("06_Validate_Initial_HS_Data.py", 
         "Validates constructed data against June 2025 benchmark import values"),
        
        ("07_TiVA_Import_Values_Comparison.py", 
         "Compares our constructed trade weights with TiVA benchmark data and creates visualizations")
    ]
    
    # Track success/failure
    successful_steps = []
    failed_steps = []
    
    # Run each step
    for script_name, description in pipeline_steps:
        success = run_script(script_name, description)
        
        if success:
            successful_steps.append(script_name)
        else:
            failed_steps.append(script_name)
            colored_print(f"Pipeline stopped at {script_name} due to error", Colors.RED)
            break
    
    # Final summary
    print(f"\n{'='*60}")
    colored_print("PIPELINE SUMMARY", Colors.BLUE)
    print(f"{'='*60}")
    colored_print(f"Successful steps: {len(successful_steps)}", Colors.GREEN)
    for step in successful_steps:
        colored_print(f"   - {step}", Colors.GREEN)
    
    if failed_steps:
        colored_print(f"Failed steps: {len(failed_steps)}", Colors.RED)
        for step in failed_steps:
            colored_print(f"   - {step}", Colors.RED)
    
    colored_print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", Colors.BLUE)
    
    if len(successful_steps) == len(pipeline_steps):
        colored_print("PIPELINE COMPLETED SUCCESSFULLY!", Colors.GREEN)
        colored_print("Check the following directories for outputs:", Colors.BLUE)
        colored_print("   - data/working/: Intermediate processing files", Colors.BLUE)
        colored_print("   - data/final/: Final trade weights and HS section weights", Colors.BLUE)
        colored_print("   - validations/: Validation files and comparison charts", Colors.BLUE)
        colored_print("   - validations/07_TiVA_Import_Values_Comparison/02_TiVA_vs_HS_Import_Charts.html: Interactive validation dashboard", Colors.BLUE)
        return True
    else:
        colored_print(f"PIPELINE FAILED at step {len(successful_steps) + 1} of {len(pipeline_steps)}", Colors.RED)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)