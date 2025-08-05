#!/usr/bin/env python3
"""
Trade Data Processing Master Pipeline
=====================================

This is the main entry point for the complete Trade Data Processing pipeline.
It orchestrates all components of the tariff price pulse analysis system.

Components:
1. HS to BEA Data Pipeline - Core trade data processing
2. Alternative Census Mapping Pipeline - Alternative mapping methodology comparison  
3. Economic Calculations - Core tariff effects calculations
4. Economic Analysis & Testing - Validation and comparative analysis

Usage:
    python run_complete_pipeline.py [--component COMPONENT] [--validate-only]
    
Options:
    --component: Run specific component only (hs_to_bea, alternative_census, calculations, analysis)
    --validate-only: Run validation scripts only
    --help: Show this help message

Author: Trade Data Processing Pipeline
Date: Generated for repository restructuring
"""

import os
import sys
import subprocess
import argparse
import json
from datetime import datetime
import time

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'      # Success
    RED = '\033[91m'        # Error/Failed  
    BLUE = '\033[94m'       # General notices
    YELLOW = '\033[93m'     # Warnings
    MAGENTA = '\033[95m'    # Headers
    BOLD = '\033[1m'
    END = '\033[0m'

def colored_print(message, color=Colors.BLUE):
    """Print message with color formatting"""
    print(f"{color}{message}{Colors.END}")

class MasterPipeline:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.start_time = datetime.now()
        self.results = {}
        
    def run_component_pipeline(self, component_path, pipeline_script, component_name):
        """Run a component's pipeline script"""
        colored_print(f"\n{'='*60}", Colors.MAGENTA)
        colored_print(f"RUNNING COMPONENT: {component_name}", Colors.MAGENTA)
        colored_print(f"{'='*60}", Colors.MAGENTA)
        
        script_path = os.path.join(self.script_dir, component_path, pipeline_script)
        
        if not os.path.exists(script_path):
            colored_print(f"ERROR: Pipeline script not found: {script_path}", Colors.RED)
            return False
            
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=os.path.dirname(script_path),
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                colored_print(f"SUCCESS: {component_name} completed successfully", Colors.GREEN)
                if result.stdout:
                    print(result.stdout)
                return True
            else:
                colored_print(f"ERROR: {component_name} failed with return code {result.returncode}", Colors.RED)
                if result.stderr:
                    colored_print("ERROR OUTPUT:", Colors.RED)
                    print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            colored_print(f"ERROR: {component_name} timed out after 1 hour", Colors.RED)
            return False
        except Exception as e:
            colored_print(f"ERROR: Failed to run {component_name}: {str(e)}", Colors.RED)
            return False

    def run_hs_to_bea_pipeline(self):
        """Run the main HS to BEA data processing pipeline"""
        return self.run_component_pipeline(
            "HS to BEA Data", 
            "run_pipeline.py",
            "HS to BEA Data Pipeline"
        )
    
    def run_alternative_census_pipeline(self):
        """Run the alternative census mapping pipeline"""
        return self.run_component_pipeline(
            "Alternative_Census_Mapping/code", 
            "run_alternative_census_mapping_pipeline.py",
            "Alternative Census Mapping Pipeline"
        )
    
    def run_calculations_pipeline(self):
        """Run the economic calculations pipeline"""
        colored_print(f"\n{'='*60}", Colors.MAGENTA)
        colored_print(f"RUNNING COMPONENT: Economic Calculations", Colors.MAGENTA)
        colored_print(f"{'='*60}", Colors.MAGENTA)
        
        calc_dir = os.path.join(self.script_dir, "Calculations", "code", "core")
        
        if not os.path.exists(calc_dir):
            colored_print(f"ERROR: Calculations directory not found: {calc_dir}", Colors.RED)
            return False
            
        # Run core calculation scripts in order
        core_scripts = [
            "pce_data_loader.py",
            "input_output_data_loader.py", 
            "data_validation_checks.py",
            "economic_effect_calculations.py"
        ]
        
        for script in core_scripts:
            colored_print(f"Running {script}...", Colors.BLUE)
            script_path = os.path.join(calc_dir, script)
            
            if not os.path.exists(script_path):
                colored_print(f"WARNING: Script not found: {script}", Colors.YELLOW)
                continue
                
            try:
                result = subprocess.run(
                    [sys.executable, script_path],
                    cwd=calc_dir,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout per script
                )
                
                if result.returncode == 0:
                    colored_print(f"SUCCESS: {script} completed", Colors.GREEN)
                else:
                    colored_print(f"ERROR: {script} failed", Colors.RED)
                    if result.stderr:
                        print(result.stderr)
                    return False
                    
            except Exception as e:
                colored_print(f"ERROR: Failed to run {script}: {str(e)}", Colors.RED)
                return False
        
        return True
    
    def run_analysis_pipeline(self):
        """Run the economic analysis and testing pipeline"""
        colored_print(f"\n{'='*60}", Colors.MAGENTA)
        colored_print(f"RUNNING COMPONENT: Economic Analysis & Testing", Colors.MAGENTA)
        colored_print(f"{'='*60}", Colors.MAGENTA)
        
        analysis_dir = os.path.join(self.script_dir, "Calculations", "code", "testing")
        
        if not os.path.exists(analysis_dir):
            colored_print(f"ERROR: Analysis directory not found: {analysis_dir}", Colors.RED)
            return False
            
        # Run analysis scripts
        analysis_scripts = [
            "bea_tariff_effects_calculation.py",
            "nipa_tariff_effects_calculation.py",
            "bea_country_rank_analysis.py",
            "nipa_country_rank_analysis.py",
            "bea_product_rank_analysis.py", 
            "nipa_product_rank_analysis.py",
            "import_vs_effect_rank_comparison.py",
            "import_vs_effect_cosine_similarity.py",
            "within_country_cosine_similarity_analysis.py",
            "pce_data_flattener.py",
            "weighted_cosine_spearman_rank_analysis.py"
        ]
        
        success_count = 0
        for script in analysis_scripts:
            colored_print(f"Running {script}...", Colors.BLUE)
            script_path = os.path.join(analysis_dir, script)
            
            if not os.path.exists(script_path):
                colored_print(f"WARNING: Script not found: {script}", Colors.YELLOW)
                continue
                
            try:
                result = subprocess.run(
                    [sys.executable, script_path],
                    cwd=analysis_dir,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout per script
                )
                
                if result.returncode == 0:
                    colored_print(f"SUCCESS: {script} completed", Colors.GREEN)
                    success_count += 1
                else:
                    colored_print(f"WARNING: {script} failed (continuing with other scripts)", Colors.YELLOW)
                    if result.stderr:
                        print(result.stderr)
                    
            except Exception as e:
                colored_print(f"WARNING: Failed to run {script}: {str(e)} (continuing)", Colors.YELLOW)
        
        colored_print(f"Analysis pipeline completed: {success_count}/{len(analysis_scripts)} scripts successful", Colors.BLUE)
        return success_count > 0  # Success if at least one script ran
    
    def run_complete_pipeline(self, component=None, validate_only=False):
        """Run the complete pipeline or specific component"""
        
        colored_print("TRADE DATA PROCESSING MASTER PIPELINE", Colors.MAGENTA + Colors.BOLD)
        colored_print(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}", Colors.BLUE)
        colored_print(f"Working directory: {self.script_dir}", Colors.BLUE)
        
        if validate_only:
            colored_print("Running in VALIDATION-ONLY mode", Colors.YELLOW)
        
        # Define pipeline steps
        if component:
            if component == "hs_to_bea":
                steps = [("HS to BEA Data", self.run_hs_to_bea_pipeline)]
            elif component == "alternative_census":
                steps = [("Alternative Census Mapping", self.run_alternative_census_pipeline)]
            elif component == "calculations":
                steps = [("Economic Calculations", self.run_calculations_pipeline)]
            elif component == "analysis":
                steps = [("Economic Analysis", self.run_analysis_pipeline)]
            else:
                colored_print(f"ERROR: Unknown component '{component}'", Colors.RED)
                return False
        else:
            steps = [
                ("HS to BEA Data", self.run_hs_to_bea_pipeline),
                ("Alternative Census Mapping", self.run_alternative_census_pipeline),
                ("Economic Calculations", self.run_calculations_pipeline),
                ("Economic Analysis", self.run_analysis_pipeline)
            ]
        
        # Run pipeline steps
        for step_name, step_func in steps:
            if validate_only and "Calculation" in step_name:
                colored_print(f"SKIPPING: {step_name} (validation-only mode)", Colors.YELLOW)
                continue
                
            self.results[step_name] = step_func()
            
            if not self.results[step_name]:
                colored_print(f"Pipeline stopped at {step_name} due to error", Colors.RED)
                break
        
        # Final summary
        self.print_summary()
        
        return all(self.results.values())
    
    def print_summary(self):
        """Print pipeline execution summary"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        colored_print(f"\n{'='*60}", Colors.MAGENTA)
        colored_print("PIPELINE EXECUTION SUMMARY", Colors.MAGENTA + Colors.BOLD)
        colored_print(f"{'='*60}", Colors.MAGENTA)
        
        colored_print(f"Total runtime: {duration}", Colors.BLUE)
        colored_print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}", Colors.BLUE)
        
        success_count = sum(self.results.values())
        total_count = len(self.results)
        
        colored_print(f"\nComponent Results ({success_count}/{total_count} successful):", Colors.BLUE)
        for component, success in self.results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            color = Colors.GREEN if success else Colors.RED
            colored_print(f"  {component}: {status}", color)
        
        if success_count == total_count:
            colored_print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!", Colors.GREEN + Colors.BOLD)
            colored_print("\nKey Output Locations:", Colors.BLUE)
            colored_print("  📁 HS to BEA Data/data/final/ - Final trade weights and mappings", Colors.BLUE)
            colored_print("  📁 HS to BEA Data/data/working/ - Intermediate processing data", Colors.BLUE)
            colored_print("  📁 HS to BEA Data/validations/ - Data validation reports (HTML/TEX)", Colors.BLUE)
            colored_print("  📁 Alternative_Census_Mapping/validations/ - Alternative mapping comparisons", Colors.BLUE)
            colored_print("  📁 Calculations/validations/ - Economic analysis results", Colors.BLUE)
        else:
            colored_print(f"\n❌ PIPELINE FAILED: {total_count - success_count} component(s) failed", Colors.RED)

def main():
    parser = argparse.ArgumentParser(
        description="Trade Data Processing Master Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_complete_pipeline.py                    # Run complete pipeline
  python run_complete_pipeline.py --component hs_to_bea      # Run only HS to BEA pipeline
  python run_complete_pipeline.py --validate-only            # Run validation scripts only
        """
    )
    
    parser.add_argument(
        '--component', 
        choices=['hs_to_bea', 'alternative_census', 'calculations', 'analysis'],
        help='Run specific component only'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Run validation scripts only'
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = MasterPipeline()
    success = pipeline.run_complete_pipeline(
        component=args.component,
        validate_only=args.validate_only
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()