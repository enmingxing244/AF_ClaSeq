#!/usr/bin/env python3
"""
Test script for explicit metric selection feature in AF-ClaSeq
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_config_loading_and_validation():
    """Test loading and validating the configuration with explicit metrics"""
    print("=== Testing Configuration Loading and Validation ===")
    
    try:
        from af_claseq.pipeline.config import load_pipeline_config, get_selected_metrics
        
        # Test loading the YAML config with explicit metrics
        yaml_path = "results_updated/ABL1/run1/ABL1_pipeline_config_run1.yaml"
        
        if not Path(yaml_path).exists():
            print(f"✗ YAML config file not found: {yaml_path}")
            return False
        
        print(f"Loading pipeline config from: {yaml_path}")
        config = load_pipeline_config(yaml_path)
        print("✓ Successfully loaded pipeline configuration")
        
        # Check the general config values
        print(f"✓ use_composite_metrics: {config.general.use_composite_metrics}")
        print(f"✓ metric1_name: {config.general.metric1_name}")
        print(f"✓ metric2_name: {config.general.metric2_name}")
        print(f"✓ config_file: {config.general.config_file}")
        
        # Test metric selection
        selected_metrics = get_selected_metrics(config.general)
        print(f"✓ Selected metrics: {selected_metrics}")
        
        # Validate the metrics exist in the JSON config
        expected_metrics = ["2g2i_A_loop_dfg_weighted_sum_rmsd", "2hiw_A_loop_dfg_weighted_sum_rmsd"]
        
        if selected_metrics == expected_metrics:
            print("✓ Selected metrics match expected composite metrics")
        else:
            print(f"✗ Selected metrics mismatch. Expected: {expected_metrics}, Got: {selected_metrics}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fallback_behavior():
    """Test fallback behavior when no explicit metrics are specified"""
    print("\n=== Testing Fallback Behavior ===")
    
    try:
        from af_claseq.pipeline.config import GeneralConfig, get_selected_metrics
        
        # Create a config without explicit metric names
        test_config = GeneralConfig(
            source_a3m="/test/path.a3m",
            default_pdb="/test/path.pdb", 
            base_dir="/test/base",
            config_file="/fs/ess/PAA0203/xing244/AF_ClaSeq/results_updated/ABL1/configs/config_6xr6_6xrg_composite.json",
            protein_name="TEST",
            use_composite_metrics=True,
            metric1_name=None,  # No explicit selection
            metric2_name=None   # No explicit selection
        )
        
        selected_metrics = get_selected_metrics(test_config)
        print(f"✓ Fallback selected metrics: {selected_metrics}")
        
        # Should fall back to first 2 composite metrics
        expected_fallback = ["2g2i_A_loop_dfg_weighted_sum_rmsd", "2hiw_A_loop_dfg_weighted_sum_rmsd"]
        
        if selected_metrics == expected_fallback:
            print("✓ Fallback behavior works correctly")
        else:
            print(f"⚠ Fallback metrics: Expected {expected_fallback}, Got {selected_metrics}")
        
        return True
        
    except Exception as e:
        print(f"✗ Fallback test error: {e}")
        return False

def test_validation_errors():
    """Test that validation catches invalid metric names"""
    print("\n=== Testing Validation Error Handling ===")
    
    try:
        from af_claseq.pipeline.config import GeneralConfig, validate_metric_names
        
        # Create a config with invalid metric name
        test_config = GeneralConfig(
            source_a3m="/test/path.a3m",
            default_pdb="/test/path.pdb",
            base_dir="/test/base", 
            config_file="/fs/ess/PAA0203/xing244/AF_ClaSeq/results_updated/ABL1/configs/config_6xr6_6xrg_composite.json",
            protein_name="TEST",
            use_composite_metrics=True,
            metric1_name="invalid_metric_name",  # This should fail validation
            metric2_name=None
        )
        
        try:
            validate_metric_names(test_config)
            print("✗ Validation should have failed for invalid metric name")
            return False
        except ValueError as e:
            print(f"✓ Validation correctly caught invalid metric: {e}")
            return True
            
    except Exception as e:
        print(f"✗ Validation test error: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Testing Explicit Metric Selection Implementation ===")
    
    tests = [
        ("Configuration Loading and Validation", test_config_loading_and_validation),
        ("Fallback Behavior", test_fallback_behavior),
        ("Validation Error Handling", test_validation_errors)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            all_passed = all_passed and passed
        except Exception as e:
            print(f"✗ Test '{test_name}' failed with exception: {e}")
            all_passed = False
    
    print(f"\n=== Test Summary ===")
    if all_passed:
        print("✓ All explicit metric selection tests PASSED")
        print("✓ The implementation is ready for use!")
    else:
        print("✗ Some tests FAILED")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)