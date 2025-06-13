#!/usr/bin/env python3
"""
Jupyter-friendly setup script for SkyReels V2 Gradio interface.
This script works in environments without sudo access.
"""

import subprocess
import sys
import os
import importlib.util

def print_status(msg):
    print(f"🔵 [INFO] {msg}")

def print_success(msg):
    print(f"✅ [SUCCESS] {msg}")

def print_warning(msg):
    print(f"⚠️ [WARNING] {msg}")

def print_error(msg):
    print(f"❌ [ERROR] {msg}")

def check_environment():
    """Check the current environment."""
    print_status("Checking environment...")
    
    # Check Python version
    python_version = sys.version
    print_status(f"Python version: {python_version}")
    
    # Check if we're in Jupyter
    try:
        import IPython
        print_success("Running in Jupyter/IPython environment")
    except ImportError:
        print_warning("Not running in Jupyter environment")
    
    # Check GPU
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        print_success("NVIDIA GPU detected:")
        for line in result.stdout.strip().split('\n'):
            print(f"  {line}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_warning("No NVIDIA GPU detected or nvidia-smi not available")

def install_package(package):
    """Install a Python package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_and_install_dependencies():
    """Check and install required dependencies."""
    print_status("Checking and installing Python dependencies...")
    
    # List of required packages
    required_packages = [
        "torch>=2.0.0",
        "torchvision",
        "opencv-python",
        "diffusers>=0.31.0", 
        "transformers==4.49.0",
        "tokenizers==0.21.1",
        "accelerate==1.6.0",
        "tqdm",
        "imageio",
        "easydict",
        "ftfy",
        "imageio-ffmpeg",
        "numpy>=1.23.5,<2",
        "gradio>=4.0.0",
        "decord",
        "moviepy",
        "safetensors",
        "huggingface_hub",
        "Pillow"
    ]
    
    # Special handling for packages that might need specific installation
    special_packages = {
        "flash_attn": "flash-attn --no-build-isolation",
        "xfuser": "git+https://github.com/xdit-project/xDiT.git"
    }
    
    failed_packages = []
    
    # Install regular packages
    for package in required_packages:
        package_name = package.split('>=')[0].split('==')[0].split('<')[0]
        print_status(f"Installing {package}...")
        
        if install_package(package):
            print_success(f"✅ {package_name} installed successfully")
        else:
            print_error(f"❌ Failed to install {package_name}")
            failed_packages.append(package)
    
    # Install special packages
    for package_name, install_cmd in special_packages.items():
        print_status(f"Installing {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + install_cmd.split())
            print_success(f"✅ {package_name} installed successfully")
        except subprocess.CalledProcessError:
            print_warning(f"⚠️ Failed to install {package_name} - this might be optional")
    
    if failed_packages:
        print_warning(f"Some packages failed to install: {failed_packages}")
        print_warning("You may need to install them manually or they might not be required")
    
    return len(failed_packages) == 0

def verify_installation():
    """Verify that key packages can be imported."""
    print_status("Verifying installation...")
    
    test_imports = [
        ("torch", "PyTorch"),
        ("gradio", "Gradio"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("imageio", "ImageIO"),
        ("transformers", "Transformers"),
        ("diffusers", "Diffusers")
    ]
    
    failed_imports = []
    
    for module_name, display_name in test_imports:
        try:
            importlib.import_module(module_name)
            print_success(f"✅ {display_name} imported successfully")
        except ImportError as e:
            print_error(f"❌ Failed to import {display_name}: {e}")
            failed_imports.append(display_name)
    
    # Special check for PyTorch CUDA
    try:
        import torch
        print_status(f"PyTorch version: {torch.__version__}")
        print_status(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print_status(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print_warning(f"Could not check PyTorch CUDA status: {e}")
    
    return len(failed_imports) == 0

def create_directories():
    """Create necessary output directories."""
    print_status("Creating output directories...")
    
    directories = ["video_out", "diffusion_forcing", "outputs"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print_success(f"✅ Created directory: {directory}")

def main():
    """Main setup function."""
    print("🎬 SkyReels V2 Jupyter Setup")
    print("=" * 40)
    
    # Check environment
    check_environment()
    
    print("\n" + "=" * 40)
    print("📦 Installing Dependencies")
    print("=" * 40)
    
    # Install dependencies
    deps_success = check_and_install_dependencies()
    
    print("\n" + "=" * 40)
    print("🧪 Verifying Installation")
    print("=" * 40)
    
    # Verify installation
    verify_success = verify_installation()
    
    # Create directories
    create_directories()
    
    print("\n" + "=" * 40)
    print("📋 Setup Summary")
    print("=" * 40)
    
    if deps_success and verify_success:
        print_success("🎉 Setup completed successfully!")
        print_status("You can now run the Gradio interface:")
        print("  • In Jupyter: Run the gradio_skyreel_notebook.ipynb")
        print("  • In terminal: python gradio_skyreel.py --share")
    else:
        print_warning("⚠️ Setup completed with some issues.")
        print_warning("Some packages may have failed to install.")
        print_warning("Try running the interface anyway - it might still work.")
    
    print("\n" + "=" * 40)
    print("🚀 Next Steps")
    print("=" * 40)
    print("1. Test the installation: python test_gradio.py")
    print("2. Run the Gradio interface: python gradio_skyreel.py --share")
    print("3. Or use the Jupyter notebook for interactive usage")

if __name__ == "__main__":
    main()
