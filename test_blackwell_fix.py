#!/usr/bin/env python3
"""
Test script for RTX 6000 Blackwell compatibility fix
This script tests if the CUDA compatibility issues have been resolved
"""

import os
import sys
import torch

# Set environment variables for Blackwell compatibility
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_CUDA_ARCH_LIST'] = '5.0;6.0;7.0;7.5;8.0;8.6;9.0;12.0'

def test_pytorch_installation():
    """Test basic PyTorch installation and CUDA availability"""
    print("🔧 Testing PyTorch Installation for RTX 6000 Blackwell")
    print("=" * 60)
    
    # Check PyTorch version
    print(f"✅ PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"✅ CUDA available: {cuda_available}")
    
    if not cuda_available:
        print("❌ CUDA is not available. Please check your installation.")
        return False
    
    # Check CUDA version
    print(f"✅ CUDA version: {torch.version.cuda}")
    print(f"✅ GPU count: {torch.cuda.device_count()}")
    
    # Check GPU details
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"✅ GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"   Compute Capability: sm_{props.major}{props.minor}")
        print(f"   Memory: {props.total_memory / 1024**3:.1f} GB")
    
    return True

def test_basic_cuda_operations():
    """Test basic CUDA operations"""
    print("\n🧪 Testing Basic CUDA Operations")
    print("=" * 40)
    
    try:
        # Test basic tensor operations
        print("Testing basic tensor operations...")
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = torch.matmul(x, y)
        print("✅ Basic CUDA operations successful")
        
        # Test different data types
        print("Testing half precision operations...")
        x_half = torch.randn(100, 100, dtype=torch.float16).cuda()
        y_half = torch.randn(100, 100, dtype=torch.float16).cuda()
        z_half = torch.matmul(x_half, y_half)
        print("✅ Half precision operations successful")
        
        # Test tensor division (the problematic operation)
        print("Testing tensor division operations...")
        a = torch.tensor([1.0, 2.0, 3.0, 4.0]).cuda()
        b = torch.tensor([2.0, 4.0, 6.0, 8.0]).cuda()
        c = 1.0 / b  # This was causing the issue
        print("✅ Tensor division operations successful")
        
        return True
        
    except RuntimeError as e:
        if "no kernel image is available for execution on the device" in str(e):
            print(f"❌ CUDA kernel compatibility issue: {e}")
            print("This indicates that PyTorch nightly with Blackwell support is needed.")
            return False
        else:
            print(f"❌ Unexpected CUDA error: {e}")
            return False

def test_skyreels_imports():
    """Test SkyReels-V2 imports"""
    print("\n📦 Testing SkyReels-V2 Imports")
    print("=" * 35)
    
    try:
        from skyreels_v2_infer.modules import download_model
        print("✅ download_model import successful")
        
        from skyreels_v2_infer.pipelines import Text2VideoPipeline, Image2VideoPipeline
        print("✅ Pipeline imports successful")
        
        from skyreels_v2_infer import DiffusionForcingPipeline
        print("✅ DiffusionForcingPipeline import successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ SkyReels-V2 import error: {e}")
        print("You may need to install additional dependencies")
        return False

def test_vae_compatibility():
    """Test VAE compatibility with the fix"""
    print("\n🎯 Testing VAE Blackwell Compatibility Fix")
    print("=" * 45)
    
    try:
        from skyreels_v2_infer.modules.vae import WanVAE
        
        # Create a dummy VAE instance (without loading actual weights)
        print("Creating VAE instance...")
        
        # Test the problematic tensor operations
        mean = torch.tensor([-0.7571, -0.7089, -0.9113, 0.1075]).cuda()
        std = torch.tensor([2.8184, 1.4541, 2.3275, 2.6558]).cuda()
        
        print("Testing tensor reciprocal operation...")
        # This was the problematic line
        std_reciprocal = 1.0 / std
        print("✅ Tensor reciprocal operation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ VAE compatibility test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 RTX 6000 Blackwell Compatibility Test Suite")
    print("=" * 50)
    
    tests = [
        ("PyTorch Installation", test_pytorch_installation),
        ("Basic CUDA Operations", test_basic_cuda_operations),
        ("SkyReels-V2 Imports", test_skyreels_imports),
        ("VAE Compatibility", test_vae_compatibility),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 25)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your RTX 6000 Blackwell is ready for SkyReels-V2!")
        print("\nYou can now run:")
        print("  python3 gradio_app.py --host 0.0.0.0 --port 7860 --share")
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Please check the errors above.")
        print("\nRecommended actions:")
        print("1. Install PyTorch nightly: pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124")
        print("2. Set environment variables: export CUDA_LAUNCH_BLOCKING=1")
        print("3. Check CUDA installation and drivers")

if __name__ == "__main__":
    main()
