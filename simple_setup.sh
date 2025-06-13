#!/bin/bash

# Simple SkyReels-V2 Setup Script
# This script focuses on Python dependencies and avoids system package issues

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
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
        export GPU_AVAILABLE=true
        print_success "GPU detected and available"
    else
        print_warning "No GPU detected. Will run on CPU (very slow)"
        export GPU_AVAILABLE=false
    fi
}

# Function to install Python dependencies only
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # Upgrade pip first
    print_status "Upgrading pip..."
    python3 -m pip install --upgrade pip setuptools wheel --user --quiet
    
    # Install PyTorch with proper CUDA support
    if [ "$GPU_AVAILABLE" = true ]; then
        # Check if RTX 6000 Blackwell (needs sm_120 support)
        if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "RTX PRO 6000 Blackwell"; then
            print_status "Installing PyTorch nightly for RTX 6000 Blackwell (sm_120 support)..."
            python3 -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124 --user

            # Set environment variables for Blackwell
            export TORCH_CUDA_ARCH_LIST="5.0;6.0;7.0;7.5;8.0;8.6;9.0;12.0"
            export CUDA_LAUNCH_BLOCKING=1

            print_warning "Using PyTorch nightly build for Blackwell architecture support"
        else
            print_status "Installing PyTorch with CUDA 12.1 support..."
            python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --user --quiet
        fi

        # Verify CUDA installation
        python3 -c "
import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {torch.cuda.get_device_name(i)} (sm_{props.major}{props.minor})')
"
    else
        print_status "Installing PyTorch CPU version..."
        python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --user --quiet
    fi
    
    # Install requirements
    print_status "Installing other dependencies..."
    python3 -m pip install -r requirements.txt --user --quiet
    
    # Install additional packages that might be missing
    print_status "Installing additional packages..."
    python3 -m pip install ninja packaging psutil decord --user --quiet

    print_success "Python dependencies installed"
}

# Function to create directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p models result/video_out result/diffusion_forcing uploads
    print_success "Directories created"
}

# Function to test installation
test_installation() {
    print_status "Testing installation..."
    
    python3 -c "
import sys
import torch
import gradio as gr

print('✅ Basic imports successful')
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Gradio: {gr.__version__}')

try:
    from skyreels_v2_infer.modules import download_model
    from skyreels_v2_infer.pipelines import Text2VideoPipeline, Image2VideoPipeline
    from skyreels_v2_infer import DiffusionForcingPipeline
    print('✅ SkyReels-V2 imports successful')
except ImportError as e:
    print(f'⚠️  SkyReels-V2 import issue: {e}')
    print('This is normal if running for the first time')
"
    
    if [ $? -eq 0 ]; then
        print_success "Installation test completed"
        return 0
    else
        print_error "Installation test failed"
        return 1
    fi
}

# Function to start Gradio
start_gradio() {
    local share_mode=${1:-"share"}
    
    print_status "Starting SkyReels-V2 Gradio interface..."
    
    if [ "$share_mode" = "share" ]; then
        print_status "Starting with public sharing enabled..."
        print_status "The interface will be available at:"
        print_status "- Local: http://localhost:7860"
        print_status "- Public: Check output below for public URL"
        echo ""
        python3 gradio_app.py --host 0.0.0.0 --port 7860 --share
    else
        print_status "Starting in local mode..."
        print_status "Interface available at: http://localhost:7860"
        echo ""
        python3 gradio_app.py --host 0.0.0.0 --port 7860
    fi
}

# Main function
main() {
    echo "🎬 SkyReels-V2 Simple Setup"
    echo "=========================="
    
    case "${1:-full}" in
        "deps")
            check_gpu
            create_directories
            install_python_deps
            print_success "Dependencies installed! Run '$0 test' to verify."
            ;;
        "test")
            test_installation
            ;;
        "run")
            test_installation
            start_gradio "local"
            ;;
        "run-share")
            test_installation
            start_gradio "share"
            ;;
        "full")
            check_gpu
            create_directories
            install_python_deps
            test_installation
            print_success "🚀 Setup completed! Starting interface..."
            echo ""
            start_gradio "share"
            ;;
        "help"|"-h"|"--help")
            echo "Usage: $0 [COMMAND]"
            echo ""
            echo "Commands:"
            echo "  deps        Install Python dependencies only"
            echo "  test        Test the installation"
            echo "  run         Start interface locally"
            echo "  run-share   Start interface with public sharing"
            echo "  full        Complete setup and start (default)"
            echo "  help        Show this help"
            echo ""
            echo "This script avoids system package installation issues"
            echo "and focuses only on Python dependencies."
            ;;
        *)
            print_error "Unknown command: $1"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

main "$@"
