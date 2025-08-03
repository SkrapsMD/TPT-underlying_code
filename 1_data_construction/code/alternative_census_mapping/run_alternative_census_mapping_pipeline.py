#!/usr/bin/env python3
"""
Pipeline script to run all Alternative Census Mapping scripts in the correct order.

This pipeline processes the Alternative Census Bureau HS-to-BEA mapping approach
and compares it against the original Schott mapping methodology.

Execution Order:
1. 01_Read_in_census_mappings.py - Read raw Census mapping data
2. 02_check_census_vs_schott.py - Compare Census vs Schott mappings  
3. 03_merge_mapping_with_trade_data.py - Merge mappings with trade data
4. 04_Aggregate_to_NAICS_and_BEA.py - Aggregate to NAICS and BEA levels
5. 05_validate_unmapped_codes.py - Validate unmapped HS codes
6. 06_Alternative_BEA_Aggregations.py - Create BEA aggregations
7. 07_Compare_Unmapped_with_Original_Mappings.py - Compare unmapped codes
8. 08_Create_Regional_Mapping_Comparison.py - Regional comparison visualization
9. 09_Compare_Original_Census_Mappings.py - Compare original census mappings

Usage:
    python run_alternative_census_mapping_pipeline.py
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_script(script_name, script_dir):
    """Run a Python script and handle errors."""
    script_path = os.path.join(script_dir, script_name)
    
    if not os.path.exists(script_path):
        print(f"❌ ERROR: Script not found: {script_path}")
        return False
    
    print(f"🔄 Running {script_name}...")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {script_name} completed successfully ({duration:.1f}s)")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {script_name} failed with return code {result.returncode}")
            if result.stderr:
                print(f"   Error: {result.stderr}")
            if result.stdout:
                print(f"   Output: {result.stdout}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ {script_name} timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False

def main():
    """Run the complete Alternative Census Mapping pipeline."""
    
    print("🚀 Starting Alternative Census Mapping Pipeline")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the scripts to run in order
    scripts = [
        "01_Read_in_census_mappings.py",
        "02_check_census_vs_schott.py", 
        "03_merge_mapping_with_trade_data.py",
        "04_Aggregate_to_NAICS_and_BEA.py",
        "05_validate_unmapped_codes.py",
        "06_Alternative_BEA_Aggregations.py",
        "07_Compare_Unmapped_with_Original_Mappings.py",
        "08_Create_Regional_Mapping_Comparison.py",
        "09_Compare_Original_Census_Mappings.py"
    ]
    
    # Track success/failure
    results = {}
    total_start_time = time.time()
    
    # Run each script in sequence
    for i, script in enumerate(scripts, 1):
        print(f"\n📋 Step {i}/{len(scripts)}: {script}")
        success = run_script(script, script_dir)
        results[script] = success
        
        if not success:
            print(f"\n💥 Pipeline failed at step {i}: {script}")
            print("❌ Stopping execution due to failure")
            break
    
    # Summary
    total_duration = time.time() - total_start_time
    successful = sum(results.values())
    total = len(results)
    
    print("\n" + "=" * 60)
    print("📊 PIPELINE SUMMARY")
    print("=" * 60)
    print(f"⏰ Total runtime: {total_duration:.1f} seconds")
    print(f"✅ Successful: {successful}/{total} scripts")
    
    if successful == total:
        print("🎉 All scripts completed successfully!")
        print("\n📁 Output files can be found in:")
        print("   - validations/Alternative_Census_Mappings/")
        print("   - data/working/Alternative_Census_Mapping/")
    else:
        print("❌ Pipeline completed with errors")
        print("\n🔍 Failed scripts:")
        for script, success in results.items():
            if not success:
                print(f"   - {script}")
    
    print(f"\n⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Exit with appropriate code
    sys.exit(0 if successful == total else 1)

if __name__ == "__main__":
    main()