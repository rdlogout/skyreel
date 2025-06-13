#!/bin/bash

# SkyReels-V2 Automated Setup and Launch Script
# This script automatically installs dependencies and starts the Gradio interface

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

# Function to run system requirements check
check_system_requirements() {
    print_status "Running system requirements check..."

    if python3 check_system.py; then
        print_success "System requirements check passed"
        return 0
    else
        print_warning "System requirements check found some issues"
        print_warning "SkyReels-V2 may still work but with limitations"
        return 1
    fi
}

# Function to check if we're in a Jupyter/Colab environment
check_environment() {
    if [ -n "$COLAB_GPU" ]; then
        print_status "Detected Google Colab environment"
        export JUPYTER_ENV="colab"
    elif [ -n "$KAGGLE_KERNEL_RUN_TYPE" ]; then
        print_status "Detected Kaggle environment"
        export JUPYTER_ENV="kaggle"
    elif command -v jupyter &> /dev/null; then
        print_status "Detected Jupyter environment"
        export JUPYTER_ENV="jupyter"
    else
        print_status "Detected standard Linux environment"
        export JUPYTER_ENV="linux"
    fi
}

# Function to check GPU availability
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

# Function to install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."

    if [ "$JUPYTER_ENV" = "colab" ] || [ "$JUPYTER_ENV" = "kaggle" ]; then
        # Colab/Kaggle usually have most dependencies
        apt-get update -qq 2>/dev/null || true
        apt-get install -y -qq ffmpeg libsm6 libxext6 libxrender-dev libglib2.0-0 libgl1-mesa-glx 2>/dev/null || true
    else
        # Check if we need sudo or if we already have admin privileges
        if command -v apt-get &> /dev/null; then
            if [ "$EUID" -eq 0 ] || [ -w /usr/bin ]; then
                # We have admin privileges, no sudo needed
                apt-get update -qq 2>/dev/null || true
                apt-get install -y -qq python3-pip ffmpeg libsm6 libxext6 libxrender-dev libglib2.0-0 libgl1-mesa-glx 2>/dev/null || true
            elif command -v sudo &> /dev/null; then
                # Use sudo if available
                sudo apt-get update -qq 2>/dev/null || true
                sudo apt-get install -y -qq python3-pip ffmpeg libsm6 libxext6 libxrender-dev libglib2.0-0 libgl1-mesa-glx 2>/dev/null || true
            else
                print_warning "Cannot install system dependencies (no sudo available). Continuing anyway..."
            fi
        elif command -v yum &> /dev/null; then
            if command -v sudo &> /dev/null; then
                sudo yum install -y python3-pip ffmpeg 2>/dev/null || true
            else
                yum install -y python3-pip ffmpeg 2>/dev/null || true
            fi
        else
            print_warning "Package manager not found. Skipping system dependencies..."
        fi
    fi

    print_success "System dependencies installation completed"
}

# Function to install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."

    # Upgrade pip first
    python3 -m pip install --upgrade pip setuptools wheel --quiet

    # Install PyTorch with CUDA support if GPU is available
    if [ "$GPU_AVAILABLE" = true ]; then
        print_status "Installing PyTorch with CUDA support..."
        python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
    else
        print_status "Installing PyTorch CPU version..."
        python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
    fi

    # Install other requirements
    print_status "Installing other Python packages..."
    python3 -m pip install -r requirements.txt --quiet

    # Install additional dependencies that might be missing
    python3 -m pip install ninja packaging psutil decord --quiet

    print_success "Python dependencies installed"
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p models
    mkdir -p result/video_out
    mkdir -p result/diffusion_forcing
    mkdir -p uploads
    
    print_success "Directories created"
}

# Function to test the installation
test_installation() {
    print_status "Testing installation..."
    
    python3 -c "
import torch
import gradio as gr
from skyreels_v2_infer.modules import download_model
from skyreels_v2_infer.pipelines import Text2VideoPipeline, Image2VideoPipeline
from skyreels_v2_infer import DiffusionForcingPipeline

print('✅ All imports successful')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Gradio version: {gr.__version__}')
"
    
    if [ $? -eq 0 ]; then
        print_success "Installation test passed"
        return 0
    else
        print_error "Installation test failed"
        return 1
    fi
}

# Function to start Gradio interface
start_gradio() {
    local share_mode=${1:-"local"}
    
    print_status "Starting SkyReels-V2 Gradio interface..."
    
    if [ "$share_mode" = "share" ]; then
        print_status "Starting with public sharing enabled..."
        python3 gradio_app.py --host 0.0.0.0 --port 7860 --share
    else
        print_status "Starting in local mode..."
        python3 gradio_app.py --host 0.0.0.0 --port 7860
    fi
}

# Function to show usage
show_usage() {
    echo "🎬 SkyReels-V2 Automated Setup Script"
    echo "====================================="
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  check       Check system requirements"
    echo "  setup       Install all dependencies and test"
    echo "  run         Start Gradio interface (local access)"
    echo "  run-share   Start Gradio interface with public sharing"
    echo "  full        Complete setup and start with sharing (default)"
    echo "  test        Test the installation"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 full          # Complete automated setup and start"
    echo "  $0 setup         # Just install dependencies"
    echo "  $0 run-share     # Start with public URL"
}

# Main function
main() {
    echo "🎬 SkyReels-V2 Automated Setup"
    echo "=============================="
    
    case "${1:-full}" in
        "check")
            check_system_requirements
            ;;
        "setup")
            check_system_requirements
            check_environment
            check_gpu
            install_system_deps
            create_directories
            install_python_deps
            test_installation
            print_success "Setup completed! Run '$0 run' to start the interface."
            ;;
        "run")
            if ! test_installation; then
                print_error "Installation not complete. Run '$0 setup' first."
                exit 1
            fi
            start_gradio "local"
            ;;
        "run-share")
            if ! test_installation; then
                print_error "Installation not complete. Run '$0 setup' first."
                exit 1
            fi
            start_gradio "share"
            ;;
        "full")
            check_system_requirements
            check_environment
            check_gpu
            install_system_deps
            create_directories
            install_python_deps
            if test_installation; then
                print_success "🚀 Setup completed! Starting Gradio interface with public sharing..."
                echo ""
                print_status "The interface will be available at:"
                print_status "- Local: http://localhost:7860"
                print_status "- Public: Check the output below for the public URL"
                echo ""
                start_gradio "share"
            else
                print_error "Setup failed. Please check the errors above."
                exit 1
            fi
            ;;
        "test")
            test_installation
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            print_error "Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
