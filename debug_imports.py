#!/usr/bin/env python3
"""
Debug script to test SkyReels imports and identify issues.
"""

import sys
import traceback

def test_basic_imports():
    """Test basic Python imports."""
    print("🔍 Testing basic imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
    except Exception as e:
        print(f"❌ PyTorch failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except Exception as e:
        print(f"❌ NumPy failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ PIL/Pillow imported")
    except Exception as e:
        print(f"❌ PIL failed: {e}")
        return False
    
    return True

def test_skyreel_modules():
    """Test SkyReels module imports."""
    print("\n🔍 Testing SkyReels module imports...")
    
    try:
        from skyreels_v2_infer.modules import download_model
        print("✅ download_model imported")
    except Exception as e:
        print(f"❌ download_model failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        from skyreels_v2_infer.pipelines import Text2VideoPipeline
        print("✅ Text2VideoPipeline imported")
    except Exception as e:
        print(f"❌ Text2VideoPipeline failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        from skyreels_v2_infer.pipelines import Image2VideoPipeline
        print("✅ Image2VideoPipeline imported")
    except Exception as e:
        print(f"❌ Image2VideoPipeline failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        from skyreels_v2_infer.pipelines import DiffusionForcingPipeline
        print("✅ DiffusionForcingPipeline imported")
    except Exception as e:
        print(f"❌ DiffusionForcingPipeline failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        from skyreels_v2_infer.pipelines import PromptEnhancer
        print("✅ PromptEnhancer imported")
    except Exception as e:
        print(f"❌ PromptEnhancer failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        from skyreels_v2_infer.pipelines import resizecrop
        print("✅ resizecrop imported")
    except Exception as e:
        print(f"❌ resizecrop failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_gradio_imports():
    """Test Gradio imports."""
    print("\n🔍 Testing Gradio imports...")
    
    try:
        import gradio as gr
        print(f"✅ Gradio: {gr.__version__}")
    except Exception as e:
        print(f"❌ Gradio failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_pipeline_creation():
    """Test if we can create pipeline instances (without loading models)."""
    print("\n🔍 Testing pipeline creation...")
    
    try:
        from skyreels_v2_infer.pipelines import Text2VideoPipeline, Image2VideoPipeline, DiffusionForcingPipeline
        print("✅ All pipeline classes imported successfully")
        
        # Test if we can access the class attributes
        print("✅ Pipeline classes are accessible")
        
    except Exception as e:
        print(f"❌ Pipeline creation test failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_model_configs():
    """Test the model configurations."""
    print("\n🔍 Testing model configurations...")
    
    try:
        from gradio_skyreel import MODEL_CONFIGS
        print("✅ MODEL_CONFIGS imported")
        
        for model_type, models in MODEL_CONFIGS.items():
            print(f"   {model_type}: {len(models)} models")
            for model_id in models.keys():
                print(f"     • {model_id}")
        
    except Exception as e:
        print(f"❌ Model configs test failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def main():
    """Run all tests."""
    print("🧪 SkyReels V2 Debug - Import Testing")
    print("=" * 50)
    
    success = True
    
    # Test basic imports
    if not test_basic_imports():
        success = False
    
    # Test SkyReels modules
    if not test_skyreel_modules():
        success = False
    
    # Test Gradio
    if not test_gradio_imports():
        success = False
    
    # Test pipeline creation
    if not test_pipeline_creation():
        success = False
    
    # Test model configs
    if not test_model_configs():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All import tests passed!")
        print("The issue might be in model loading or pipeline execution.")
        print("\n💡 Suggestions:")
        print("1. Check if models are being downloaded correctly")
        print("2. Verify GPU memory is sufficient")
        print("3. Try with CPU offload enabled")
        print("4. Check the specific error in the Gradio interface")
    else:
        print("❌ Some import tests failed.")
        print("Please fix the import issues before proceeding.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
