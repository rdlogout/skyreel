#!/usr/bin/env python3
"""
Test script to verify Gradio interface can be imported and basic functionality works.
"""

import sys
import traceback

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import gradio as gr
        print("✅ Gradio imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Gradio: {e}")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch imported successfully (version: {torch.__version__})")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ Failed to import PyTorch: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy imported successfully (version: {np.__version__})")
    except ImportError as e:
        print(f"❌ Failed to import NumPy: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ PIL imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import PIL: {e}")
        return False
    
    try:
        import imageio
        print("✅ ImageIO imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import ImageIO: {e}")
        return False
    
    return True

def test_skyreel_imports():
    """Test if SkyReels modules can be imported."""
    print("\nTesting SkyReels imports...")
    
    try:
        from skyreels_v2_infer.modules import download_model
        print("✅ download_model imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import download_model: {e}")
        return False
    
    try:
        from skyreels_v2_infer.pipelines import Text2VideoPipeline, Image2VideoPipeline, PromptEnhancer, resizecrop
        print("✅ Basic pipelines imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import basic pipelines: {e}")
        return False
    
    try:
        from skyreels_v2_infer.pipelines import DiffusionForcingPipeline
        print("✅ DiffusionForcingPipeline imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import DiffusionForcingPipeline: {e}")
        return False
    
    return True

def test_gradio_interface():
    """Test if the Gradio interface can be created."""
    print("\nTesting Gradio interface creation...")
    
    try:
        # Import the gradio interface
        from gradio_skyreel import create_gradio_interface
        print("✅ Gradio interface module imported successfully")
        
        # Try to create the interface (but don't launch it)
        demo = create_gradio_interface()
        print("✅ Gradio interface created successfully")
        
        return True
    except Exception as e:
        print(f"❌ Failed to create Gradio interface: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 SkyReels V2 Gradio Interface Test")
    print("=" * 40)
    
    success = True
    
    # Test basic imports
    if not test_imports():
        success = False
    
    # Test SkyReels imports
    if not test_skyreel_imports():
        success = False
    
    # Test Gradio interface
    if not test_gradio_interface():
        success = False
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 All tests passed! The Gradio interface should work correctly.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
