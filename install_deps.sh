#!/bin/bash

# Complete dependency installer for SkyReels-V2
# Optimized for RTX PRO 6000 Blackwell GPU

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check GPU
check_gpu() {
    print_status "Checking GPU availability..."
    
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits)
        echo "$GPU_INFO"
        export GPU_AVAILABLE=true
        print_success "GPU detected and available"
        
        # Check if it's RTX 6000 Blackwell
        if echo "$GPU_INFO" | grep -q "RTX PRO 6000 Blackwell"; then
            print_success "RTX PRO 6000 Blackwell detected - using optimized settings"
            export RTX_6000_BLACKWELL=true
        fi
    else
        print_warning "No GPU detected. Will run on CPU (very slow)"
        export GPU_AVAILABLE=false
        export RTX_6000_BLACKWELL=false
    fi
}

# Function to install PyTorch with correct CUDA version
install_pytorch() {
    print_status "Installing PyTorch..."

    if [ "$GPU_AVAILABLE" = true ]; then
        if [ "$RTX_6000_BLACKWELL" = true ]; then
            print_status "Installing PyTorch nightly for RTX 6000 Blackwell (sm_120 support)..."

            # RTX 6000 Blackwell needs PyTorch nightly for sm_120 support
            python3 -m pip install --upgrade pip
            python3 -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

            print_status "Setting CUDA environment variables for Blackwell..."
            export TORCH_CUDA_ARCH_LIST="5.0;6.0;7.0;7.5;8.0;8.6;9.0;12.0"
            export CUDA_LAUNCH_BLOCKING=1

        else
            print_status "Installing PyTorch with CUDA 12.1 support..."
            python3 -m pip install --upgrade pip
            python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        fi

        # Verify installation
        print_status "Verifying PyTorch installation..."
        python3 -c "
import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'✅ CUDA version: {torch.version.cuda}')
    print(f'✅ GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'✅ GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'   Memory: {props.total_memory / 1024**3:.1f} GB')
        print(f'   Compute Capability: {props.major}.{props.minor}')

    # Test basic CUDA operations
    try:
        x = torch.randn(10, 10).cuda()
        y = torch.randn(10, 10).cuda()
        z = torch.matmul(x, y)
        print('✅ Basic CUDA operations working')
    except Exception as e:
        print(f'❌ CUDA operation failed: {e}')
        print('This may indicate sm_120 compatibility issues')
else:
    print('❌ CUDA not available')
"
    else
        print_status "Installing PyTorch CPU version..."
        python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi

    print_success "PyTorch installation completed"
}

# Function to install core dependencies
install_core_deps() {
    print_status "Installing core dependencies..."
    
    # Install in specific order to avoid conflicts
    python3 -m pip install numpy
    python3 -m pip install opencv-python==4.10.0.84
    python3 -m pip install Pillow
    python3 -m pip install tqdm
    python3 -m pip install psutil
    
    print_success "Core dependencies installed"
}

# Function to install ML dependencies
install_ml_deps() {
    print_status "Installing ML dependencies..."
    
    python3 -m pip install transformers==4.49.0
    python3 -m pip install tokenizers==0.21.1
    python3 -m pip install diffusers>=0.31.0
    python3 -m pip install accelerate==1.6.0
    python3 -m pip install safetensors
    python3 -m pip install huggingface_hub
    
    print_success "ML dependencies installed"
}

# Function to install video processing dependencies
install_video_deps() {
    print_status "Installing video processing dependencies..."
    
    python3 -m pip install decord>=0.6.0
    python3 -m pip install imageio
    python3 -m pip install imageio-ffmpeg
    python3 -m pip install moviepy
    
    print_success "Video processing dependencies installed"
}

# Function to install performance dependencies
install_performance_deps() {
    print_status "Installing performance dependencies..."
    
    # Flash attention for better performance
    if [ "$GPU_AVAILABLE" = true ]; then
        print_status "Installing flash attention..."
        python3 -m pip install flash-attn --no-build-isolation
    fi
    
    python3 -m pip install ninja
    python3 -m pip install packaging
    python3 -m pip install xfuser
    
    print_success "Performance dependencies installed"
}

# Function to install web interface
install_web_deps() {
    print_status "Installing web interface dependencies..."
    
    python3 -m pip install gradio>=4.0.0
    
    print_success "Web interface dependencies installed"
}

# Function to install additional dependencies
install_additional_deps() {
    print_status "Installing additional dependencies..."
    
    python3 -m pip install easydict
    python3 -m pip install ftfy
    python3 -m pip install dashscope
    
    print_success "Additional dependencies installed"
}

# Function to test installation
test_installation() {
    print_status "Testing installation..."
    
    python3 -c "
import sys
print(f'Python version: {sys.version}')

# Test core imports
import torch
import torchvision
import numpy as np
import cv2
import PIL
print('✅ Core libraries imported successfully')

# Test ML imports
import transformers
import diffusers
import accelerate
print('✅ ML libraries imported successfully')

# Test video processing
import decord
import imageio
import moviepy
print('✅ Video processing libraries imported successfully')

# Test web interface
import gradio as gr
print('✅ Gradio imported successfully')

# Test PyTorch CUDA
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')

# Test SkyReels-V2 imports
try:
    from skyreels_v2_infer.modules import download_model
    from skyreels_v2_infer.pipelines import Text2VideoPipeline, Image2VideoPipeline
    from skyreels_v2_infer import DiffusionForcingPipeline
    print('✅ SkyReels-V2 imports successful')
except ImportError as e:
    print(f'⚠️  SkyReels-V2 import issue: {e}')
    print('This may be normal if running for the first time')

print('🎉 Installation test completed!')
"
    
    if [ $? -eq 0 ]; then
        print_success "Installation test passed"
        return 0
    else
        print_error "Installation test failed"
        return 1
    fi
}

# Main function
main() {
    echo "🎬 SkyReels-V2 Complete Dependency Installer"
    echo "==========================================="
    echo "Optimized for RTX PRO 6000 Blackwell GPU"
    echo ""
    
    check_gpu
    
    print_status "Starting dependency installation..."
    
    install_pytorch
    install_core_deps
    install_ml_deps
    install_video_deps
    install_performance_deps
    install_web_deps
    install_additional_deps
    
    print_success "All dependencies installed!"
    
    test_installation
    
    if [ $? -eq 0 ]; then
        print_success "🚀 Installation completed successfully!"
        echo ""
        print_status "You can now run:"
        print_status "  bash simple_setup.sh run-share"
        print_status "  or"
        print_status "  python3 gradio_app.py --host 0.0.0.0 --port 7860 --share"
    else
        print_error "Installation completed but some tests failed"
        print_status "You can still try running the application"
    fi
}

main "$@"
