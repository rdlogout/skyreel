#!/usr/bin/env python3
"""
Complete SkyReels-V2 Setup and Run Script
This script handles everything: system checks, dependency installation, 
Blackwell GPU compatibility, and running the Gradio app.
"""

import os
import sys
import subprocess
import importlib.util
import json
import time
import argparse
from pathlib import Path

# Colors for terminal output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    WHITE = '\033[1;37m'
    NC = '\033[0m'  # No Color

def print_status(msg, color=Colors.BLUE):
    print(f"{color}[INFO]{Colors.NC} {msg}")

def print_success(msg):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {msg}")

def print_warning(msg):
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {msg}")

def print_error(msg):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")

def print_header(msg):
    print(f"\n{Colors.CYAN}{'='*60}{Colors.NC}")
    print(f"{Colors.WHITE}{msg}{Colors.NC}")
    print(f"{Colors.CYAN}{'='*60}{Colors.NC}")

def run_command(cmd, description="", capture_output=True, check=True):
    """Run a shell command with proper error handling"""
    if description:
        print_status(f"{description}")
    
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, check=check, 
                                  capture_output=True, text=True, timeout=300)
            return result.returncode == 0, result.stdout, result.stderr
        else:
            result = subprocess.run(cmd, shell=True, check=check)
            return result.returncode == 0, "", ""
    except subprocess.TimeoutExpired:
        print_error(f"Command timed out: {cmd}")
        return False, "", "Command timed out"
    except subprocess.CalledProcessError as e:
        if check:
            print_error(f"Command failed: {cmd}")
            print_error(f"Error: {e}")
        return False, "", str(e)
    except Exception as e:
        print_error(f"Unexpected error running command: {e}")
        return False, "", str(e)

def check_python_version():
    """Check if Python version is compatible"""
    print_header("🐍 Checking Python Version")
    
    version = sys.version_info
    print_status(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_error("Python 3.8+ is required")
        return False
    
    print_success("Python version is compatible")
    return True

def check_gpu_compatibility():
    """Check GPU and determine compatibility requirements"""
    print_header("🎮 Checking GPU Compatibility")
    
    # Check if nvidia-smi is available
    success, stdout, stderr = run_command("nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits", 
                                        "Checking GPU information")
    
    gpu_info = {
        'has_gpu': False,
        'is_blackwell': False,
        'gpu_name': '',
        'memory_gb': 0,
        'compute_cap': '',
        'needs_nightly': False
    }
    
    if success and stdout.strip():
        lines = stdout.strip().split('\n')
        for line in lines:
            parts = line.split(', ')
            if len(parts) >= 3:
                gpu_name = parts[0].strip()
                memory_mb = float(parts[1].strip())
                compute_cap = parts[2].strip()
                
                gpu_info['has_gpu'] = True
                gpu_info['gpu_name'] = gpu_name
                gpu_info['memory_gb'] = memory_mb / 1024
                gpu_info['compute_cap'] = compute_cap
                
                print_status(f"GPU: {gpu_name}")
                print_status(f"Memory: {gpu_info['memory_gb']:.1f} GB")
                print_status(f"Compute Capability: {compute_cap}")
                
                # Check for Blackwell architecture (sm_120)
                if "RTX PRO 6000 Blackwell" in gpu_name or compute_cap.startswith("12."):
                    gpu_info['is_blackwell'] = True
                    gpu_info['needs_nightly'] = True
                    print_warning("RTX 6000 Blackwell detected - requires PyTorch nightly")
                
                break
    else:
        print_warning("No NVIDIA GPU detected or nvidia-smi not available")
        print_warning("Will install CPU-only version (very slow for video generation)")
    
    return gpu_info

def set_environment_variables(gpu_info):
    """Set required environment variables"""
    print_header("🔧 Setting Environment Variables")
    
    env_vars = {}
    
    if gpu_info['has_gpu']:
        env_vars['CUDA_LAUNCH_BLOCKING'] = '1'
        
        if gpu_info['is_blackwell']:
            env_vars['TORCH_CUDA_ARCH_LIST'] = '5.0;6.0;7.0;7.5;8.0;8.6;9.0;12.0'
            print_status("Setting Blackwell-specific environment variables")
        else:
            env_vars['TORCH_CUDA_ARCH_LIST'] = '5.0;6.0;7.0;7.5;8.0;8.6;9.0'
    
    # Set environment variables for current session
    for key, value in env_vars.items():
        os.environ[key] = value
        print_status(f"Set {key}={value}")
    
    # Save to a file for persistence
    env_file = Path("skyreel_env.sh")
    with open(env_file, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# SkyReels-V2 Environment Variables\n")
        for key, value in env_vars.items():
            f.write(f"export {key}='{value}'\n")
    
    print_success(f"Environment variables saved to {env_file}")
    return env_vars

def install_pytorch(gpu_info):
    """Install appropriate PyTorch version"""
    print_header("🔥 Installing PyTorch")
    
    # Uninstall existing PyTorch first
    print_status("Removing existing PyTorch installation...")
    run_command("pip uninstall torch torchvision torchaudio -y", check=False)
    
    if not gpu_info['has_gpu']:
        # CPU-only installation
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
        success, stdout, stderr = run_command(cmd, "Installing PyTorch CPU version")
    elif gpu_info['needs_nightly']:
        # Blackwell GPU - needs nightly build
        print_status("Installing PyTorch nightly for Blackwell architecture...")
        cmd = "pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124"
        success, stdout, stderr = run_command(cmd, "Installing PyTorch nightly with CUDA 12.4")
        
        if not success:
            print_warning("Nightly build failed, trying CUDA 12.1...")
            cmd = "pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu121"
            success, stdout, stderr = run_command(cmd, "Installing PyTorch nightly with CUDA 12.1")
    else:
        # Regular GPU - stable version
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        success, stdout, stderr = run_command(cmd, "Installing PyTorch with CUDA 12.1")
        
        if not success:
            print_warning("CUDA 12.1 failed, trying CUDA 11.8...")
            cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
            success, stdout, stderr = run_command(cmd, "Installing PyTorch with CUDA 11.8")
    
    if success:
        print_success("PyTorch installation completed")
        return True
    else:
        print_error("PyTorch installation failed")
        print_error(f"Error: {stderr}")
        return False

def install_dependencies():
    """Install all other dependencies"""
    print_header("📦 Installing Dependencies")
    
    # Upgrade pip first
    run_command("pip install --upgrade pip setuptools wheel", "Upgrading pip")
    
    # Install from requirements.txt
    if Path("requirements.txt").exists():
        success, stdout, stderr = run_command("pip install -r requirements.txt", 
                                            "Installing from requirements.txt")
        if not success:
            print_warning("Some requirements failed to install, continuing...")
    
    # Install critical packages individually
    critical_packages = [
        "gradio>=4.0.0",
        "diffusers>=0.31.0", 
        "transformers==4.49.0",
        "accelerate==1.6.0",
        "opencv-python",
        "imageio",
        "imageio-ffmpeg",
        "numpy<2",
        "tqdm",
        "safetensors",
        "huggingface_hub"
    ]
    
    for package in critical_packages:
        success, _, _ = run_command(f"pip install '{package}'", 
                                  f"Installing {package.split('>=')[0].split('==')[0]}", 
                                  check=False)
        if not success:
            print_warning(f"Failed to install {package}, continuing...")
    
    print_success("Dependencies installation completed")

def test_installation(gpu_info):
    """Test the installation"""
    print_header("🧪 Testing Installation")
    
    try:
        # Test basic imports
        import torch
        import gradio as gr
        print_success(f"PyTorch version: {torch.__version__}")
        print_success(f"Gradio version: {gr.__version__}")
        
        # Test CUDA
        if gpu_info['has_gpu']:
            cuda_available = torch.cuda.is_available()
            print_status(f"CUDA available: {cuda_available}")
            
            if cuda_available:
                device_count = torch.cuda.device_count()
                print_status(f"GPU count: {device_count}")
                
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    print_status(f"GPU {i}: {torch.cuda.get_device_name(i)} (sm_{props.major}{props.minor})")
                
                # Test basic CUDA operations
                try:
                    x = torch.randn(10, 10).cuda()
                    y = torch.randn(10, 10).cuda()
                    z = torch.matmul(x, y)
                    
                    # Test the problematic operation for Blackwell
                    if gpu_info['is_blackwell']:
                        a = torch.tensor([1.0, 2.0]).cuda()
                        b = 1.0 / a  # This was causing issues
                    
                    print_success("CUDA operations test passed")
                except Exception as e:
                    print_error(f"CUDA operations test failed: {e}")
                    return False
        
        # Test SkyReels imports
        try:
            from skyreels_v2_infer.modules import download_model
            from skyreels_v2_infer.pipelines import Text2VideoPipeline, Image2VideoPipeline
            from skyreels_v2_infer import DiffusionForcingPipeline
            print_success("SkyReels-V2 imports successful")
        except ImportError as e:
            print_warning(f"SkyReels-V2 import issue: {e}")
            print_warning("This is normal for first-time setup")
        
        print_success("Installation test completed successfully")
        return True
        
    except Exception as e:
        print_error(f"Installation test failed: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print_header("📁 Creating Directories")
    
    directories = [
        "models",
        "result/video_out", 
        "result/diffusion_forcing",
        "uploads",
        "cache"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print_status(f"Created directory: {directory}")
    
    print_success("All directories created")

def run_gradio_app(share=True, port=7860):
    """Run the Gradio application"""
    print_header("🚀 Starting SkyReels-V2 Gradio Interface")
    
    if not Path("gradio_app.py").exists():
        print_error("gradio_app.py not found!")
        return False
    
    # Prepare command
    cmd = f"python3 gradio_app.py --host 0.0.0.0 --port {port}"
    if share:
        cmd += " --share"
        print_status("Starting with public sharing enabled...")
        print_status(f"Local URL: http://localhost:{port}")
        print_status("Public URL will be shown below...")
    else:
        print_status(f"Starting locally at: http://localhost:{port}")
    
    print_status("Press Ctrl+C to stop the server")
    print("")
    
    # Run the app (this will block)
    try:
        subprocess.run(cmd, shell=True, check=True)
    except KeyboardInterrupt:
        print_status("\nShutting down gracefully...")
        return True
    except Exception as e:
        print_error(f"Failed to start Gradio app: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Complete SkyReels-V2 Setup and Run Script")
    parser.add_argument("--no-share", action="store_true", help="Run locally without public sharing")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on (default: 7860)")
    parser.add_argument("--setup-only", action="store_true", help="Only setup, don't run the app")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    
    args = parser.parse_args()
    
    print_header("🎬 SkyReels-V2 Complete Setup and Run")
    print_status("This script will check your system, install dependencies, and run the app")
    print("")
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Check GPU compatibility
    gpu_info = check_gpu_compatibility()
    
    # Step 3: Set environment variables
    set_environment_variables(gpu_info)
    
    # Step 4: Create directories
    create_directories()
    
    if not args.skip_deps:
        # Step 5: Install PyTorch
        if not install_pytorch(gpu_info):
            print_error("PyTorch installation failed. Exiting.")
            sys.exit(1)
        
        # Step 6: Install other dependencies
        install_dependencies()
        
        # Step 7: Test installation
        if not test_installation(gpu_info):
            print_error("Installation test failed. Please check the errors above.")
            sys.exit(1)
    
    if args.setup_only:
        print_success("Setup completed successfully!")
        print_status("To run the app later, use: python3 gradio_app.py --host 0.0.0.0 --port 7860 --share")
        return
    
    # Step 8: Run the Gradio app
    print_success("🎉 Setup completed! Starting the application...")
    time.sleep(2)
    
    if not run_gradio_app(share=not args.no_share, port=args.port):
        print_error("Failed to start the application")
        sys.exit(1)

if __name__ == "__main__":
    main()
