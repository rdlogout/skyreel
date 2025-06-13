#!/bin/bash

# SkyReels V2 Setup and Run Script
# This script sets up the environment and launches the Gradio interface

set -e  # Exit on any error

echo "🎬 SkyReels V2 Setup and Launch Script"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Check if running on a server environment
check_environment() {
    print_status "Checking environment..."
    
    # Check if CUDA is available
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    else
        print_warning "No NVIDIA GPU detected. CPU-only mode will be used."
    fi
    
    # Check Python version
    python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    print_status "Python version: $python_version"
    
    # Check if we're in a virtual environment
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        print_success "Virtual environment detected: $VIRTUAL_ENV"
    else
        print_warning "No virtual environment detected. Consider using one."
    fi
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    # Check if we're on Ubuntu/Debian
    if command -v apt-get &> /dev/null; then
        print_status "Detected Debian/Ubuntu system"
        
        # Update package list
        sudo apt-get update -qq
        
        # Install required system packages
        sudo apt-get install -y \
            python3-pip \
            python3-dev \
            python3-venv \
            git \
            wget \
            curl \
            ffmpeg \
            libgl1-mesa-glx \
            libglib2.0-0 \
            libsm6 \
            libxext6 \
            libxrender-dev \
            libgomp1
            
        print_success "System dependencies installed"
    elif command -v yum &> /dev/null; then
        print_status "Detected RHEL/CentOS system"
        
        sudo yum update -y
        sudo yum install -y \
            python3-pip \
            python3-devel \
            git \
            wget \
            curl \
            ffmpeg \
            mesa-libGL \
            glib2 \
            libSM \
            libXext \
            libXrender
            
        print_success "System dependencies installed"
    else
        print_warning "Unknown system. Please install dependencies manually."
    fi
}

# Setup Python environment
setup_python_env() {
    print_status "Setting up Python environment..."
    
    # Upgrade pip
    python3 -m pip install --upgrade pip
    
    # Install wheel and setuptools
    python3 -m pip install wheel setuptools
    
    print_success "Python environment setup complete"
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # Check if requirements.txt exists
    if [[ ! -f "requirements.txt" ]]; then
        print_error "requirements.txt not found!"
        exit 1
    fi
    
    # Install PyTorch with CUDA support if available
    if command -v nvidia-smi &> /dev/null; then
        print_status "Installing PyTorch with CUDA support..."
        python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        print_status "Installing PyTorch CPU version..."
        python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install other requirements
    print_status "Installing other dependencies..."
    python3 -m pip install -r requirements.txt
    
    # Install additional dependencies that might be needed
    python3 -m pip install pillow opencv-python-headless
    
    print_success "Python dependencies installed"
}

# Verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    # Test imports
    python3 -c "
import torch
import gradio as gr
import numpy as np
import PIL
import cv2
print('✅ Core dependencies imported successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
print(f'Gradio version: {gr.__version__}')
"
    
    if [[ $? -eq 0 ]]; then
        print_success "Installation verification passed"
    else
        print_error "Installation verification failed"
        exit 1
    fi
}

# Create output directories
create_directories() {
    print_status "Creating output directories..."
    
    mkdir -p video_out
    mkdir -p diffusion_forcing
    mkdir -p outputs
    
    print_success "Output directories created"
}

# Launch Gradio interface
launch_gradio() {
    print_status "Launching SkyReels V2 Gradio interface..."
    
    # Check if gradio_skyreel.py exists
    if [[ ! -f "gradio_skyreel.py" ]]; then
        print_error "gradio_skyreel.py not found!"
        exit 1
    fi
    
    # Set environment variables for better performance
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    export CUDA_VISIBLE_DEVICES=0  # Use first GPU by default
    
    # Launch with share enabled by default for server environments
    print_success "Starting Gradio interface..."
    print_status "Access the interface at: http://localhost:7860"
    print_status "If running on a server, the public URL will be displayed below."
    
    python3 gradio_skyreel.py --share --server_name 0.0.0.0 --server_port 7860
}

# Main execution
main() {
    echo
    print_status "Starting SkyReels V2 setup process..."
    echo
    
    # Parse command line arguments
    SKIP_DEPS=false
    SKIP_SYSTEM=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-deps)
                SKIP_DEPS=true
                shift
                ;;
            --skip-system)
                SKIP_SYSTEM=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --skip-deps     Skip Python dependency installation"
                echo "  --skip-system   Skip system dependency installation"
                echo "  --help, -h      Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Execute setup steps
    check_environment
    
    if [[ "$SKIP_SYSTEM" != true ]]; then
        install_system_deps
    else
        print_warning "Skipping system dependency installation"
    fi
    
    setup_python_env
    
    if [[ "$SKIP_DEPS" != true ]]; then
        install_python_deps
    else
        print_warning "Skipping Python dependency installation"
    fi
    
    verify_installation
    create_directories
    
    echo
    print_success "Setup complete! Launching Gradio interface..."
    echo
    
    launch_gradio
}

# Run main function
main "$@"
