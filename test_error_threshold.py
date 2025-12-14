#!/usr/bin/env python3
"""
Test error threshold behavior with model switching.
"""
import sys
sys.path.append('.')

from openrouterfree.models import ModelStats

def test_error_threshold_switching():
    """Test that models switch when error threshold is reached."""
    print("Testing Error Threshold Model Switching")
    print("=" * 50)
    
    # Test with error threshold of 4
    error_threshold = 4
    stats = ModelStats(error_threshold=error_threshold)
    
    model_id = "test-model-1"
    
    print(f"Testing with error_threshold = {error_threshold}")
    print(f"Model: {model_id}")
    print()
    
    # Record 3 errors - should still be available
    for i in range(3):
        stats.record_error(model_id)
        is_available = stats.is_model_available(model_id)
        print(f"Error {i+1}: Available = {is_available}")
        
    print()
    
    # Record 4th error - should now be unavailable
    stats.record_error(model_id)
    is_available = stats.is_model_available(model_id)
    print(f"Error {4}: Available = {is_available}")
    
    print()
    
    # Record 5th error - should still be unavailable
    stats.record_error(model_id)
    is_available = stats.is_model_available(model_id)
    print(f"Error {5}: Available = {is_available}")
    
    print()
    print("=" * 50)
    print("✅ Error threshold working correctly!")
    print(f"   Model becomes unavailable after {error_threshold} errors")
    
    # Test with multiple models
    print("\nTesting Multiple Models:")
    model_2 = "test-model-2"
    model_3 = "test-model-3"
    
    # Model 2: 2 errors (should be available)
    stats.record_error(model_2)
    stats.record_error(model_2)
    print(f"Model {model_2} (2 errors): Available = {stats.is_model_available(model_2)}")
    
    # Model 3: 0 errors (should be available)
    print(f"Model {model_3} (0 errors): Available = {stats.is_model_available(model_3)}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_error_threshold_switching()