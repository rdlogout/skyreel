#!/usr/bin/env python3
"""
Manual fix script for RTX 6000 Blackwell CUDA compatibility
This script provides step-by-step instructions and automated fixes
"""

import os
import sys
import subprocess
import importlib.util

def run_command(cmd, description=""):
    """Run a shell command and return success status"""
    print(f"🔧 {description}")
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def check_pytorch_installed():
    """Check if PyTorch is installed"""
    spec = importlib.util.find_spec("torch")
    return spec is not None

def get_python_version():
    """Get Python version string"""
    return f"{sys.version_info.major}.{sys.version_info.minor}"

def set_environment_variables():
    """Set required environment variables"""
    print("🔧 Setting environment variables for Blackwell compatibility...")
    
    env_vars = {
        'CUDA_LAUNCH_BLOCKING': '1',
        'TORCH_CUDA_ARCH_LIST': '5.0;6.0;7.0;7.5;8.0;8.6;9.0;12.0'
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"Set {key}={value}")
    
    # Add to bashrc for persistence
    bashrc_path = os.path.expanduser("~/.bashrc")
    try:
        with open(bashrc_path, "a") as f:
            f.write("\n# RTX 6000 Blackwell compatibility\n")
            for key, value in env_vars.items():
                f.write(f"export {key}='{value}'\n")
        print("✅ Environment variables added to ~/.bashrc")
    except Exception as e:
        print(f"⚠️  Could not write to ~/.bashrc: {e}")

def uninstall_pytorch():
    """Uninstall current PyTorch installation"""
    if not check_pytorch_installed():
        print("ℹ️  PyTorch not currently installed")
        return True
    
    print("🔧 Uninstalling current PyTorch...")
    packages = ["torch", "torchvision", "torchaudio"]
    
    for package in packages:
        cmd = f"pip uninstall {package} -y"
        if not run_command(cmd, f"Uninstalling {package}"):
            print(f"⚠️  Failed to uninstall {package}, continuing...")
    
    return True

def install_pytorch_nightly():
    """Install PyTorch nightly with CUDA 12.4"""
    print("🔧 Installing PyTorch nightly with Blackwell support...")
    
    python_version = get_python_version()
    print(f"Python version: {python_version}")
    
    # Try different CUDA versions
    cuda_versions = ["cu124", "cu121", "cu118"]
    
    for cuda_ver in cuda_versions:
        print(f"\n🔧 Trying PyTorch nightly with {cuda_ver}...")
        cmd = f"pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/{cuda_ver}"
        
        if run_command(cmd, f"Installing PyTorch nightly with {cuda_ver}"):
            print(f"✅ Successfully installed PyTorch nightly with {cuda_ver}")
            return True
        else:
            print(f"❌ Failed to install with {cuda_ver}, trying next...")
    
    print("❌ All nightly installations failed, trying stable version...")
    return install_pytorch_stable()

def install_pytorch_stable():
    """Install stable PyTorch with CUDA"""
    print("🔧 Installing stable PyTorch with CUDA...")
    
    cuda_versions = ["cu121", "cu118"]
    
    for cuda_ver in cuda_versions:
        print(f"\n🔧 Trying stable PyTorch with {cuda_ver}...")
        cmd = f"pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/{cuda_ver}"
        
        if run_command(cmd, f"Installing stable PyTorch with {cuda_ver}"):
            print(f"✅ Successfully installed stable PyTorch with {cuda_ver}")
            return True
        else:
            print(f"❌ Failed to install with {cuda_ver}, trying next...")
    
    # Fallback to default PyPI
    print("🔧 Trying default PyPI installation...")
    cmd = "pip install torch torchvision torchaudio"
    return run_command(cmd, "Installing PyTorch from PyPI")

def test_installation():
    """Test the PyTorch installation"""
    print("\n🧪 Testing PyTorch installation...")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"✅ GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"   Compute Capability: sm_{props.major}{props.minor}")
                print(f"   Memory: {props.total_memory / 1024**3:.1f} GB")
            
            # Test basic operations
            print("\n🧪 Testing CUDA operations...")
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.matmul(x, y)
            
            # Test the problematic operation
            a = torch.tensor([1.0, 2.0, 3.0, 4.0]).cuda()
            b = torch.tensor([2.0, 4.0, 6.0, 8.0]).cuda()
            c = 1.0 / b
            
            print("✅ All CUDA operations successful!")
            return True
        else:
            print("⚠️  CUDA not available, but PyTorch is installed")
            return False
            
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False

def main():
    """Main fix process"""
    print("🔧 RTX 6000 Blackwell CUDA Compatibility Fix")
    print("=" * 50)
    
    print("\n📋 This script will:")
    print("1. Set environment variables for Blackwell compatibility")
    print("2. Uninstall current PyTorch")
    print("3. Install PyTorch nightly with CUDA 12.4 support")
    print("4. Test the installation")
    
    response = input("\nProceed with the fix? (y/N): ")
    if response.lower() != 'y':
        print("Fix cancelled.")
        return
    
    # Step 1: Set environment variables
    set_environment_variables()
    
    # Step 2: Uninstall current PyTorch
    uninstall_pytorch()
    
    # Step 3: Install PyTorch nightly
    if not install_pytorch_nightly():
        print("❌ Failed to install PyTorch. Please check your internet connection and try again.")
        return
    
    # Step 4: Test installation
    if test_installation():
        print("\n🎉 Fix completed successfully!")
        print("\nYour RTX 6000 Blackwell is now compatible with SkyReels-V2!")
        print("\nNext steps:")
        print("1. Restart your terminal or run: source ~/.bashrc")
        print("2. Test with: python3 test_blackwell_fix.py")
        print("3. Run SkyReels-V2: python3 gradio_app.py --host 0.0.0.0 --port 7860 --share")
    else:
        print("\n❌ Fix completed but tests failed.")
        print("You may need to:")
        print("1. Check your CUDA drivers")
        print("2. Try a different PyTorch version")
        print("3. Use CPU mode as fallback")

if __name__ == "__main__":
    main()
