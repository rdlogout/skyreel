#!/bin/bash

# Fix for RTX 6000 Blackwell CUDA sm_120 compatibility issue
# This script installs PyTorch nightly with Blackwell architecture support

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

echo "🔧 RTX 6000 Blackwell CUDA sm_120 Compatibility Fix"
echo "=================================================="

# Check if we have RTX 6000 Blackwell
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "RTX PRO 6000 Blackwell"; then
    print_status "RTX PRO 6000 Blackwell detected"
else
    print_warning "This script is specifically for RTX 6000 Blackwell GPUs"
    print_warning "Your GPU may not need this fix"
fi

# Set environment variables for Blackwell
print_status "Setting environment variables for Blackwell architecture..."
export CUDA_LAUNCH_BLOCKING=1
export TORCH_CUDA_ARCH_LIST="5.0;6.0;7.0;7.5;8.0;8.6;9.0;12.0"

# Add to shell profile for persistence
echo 'export CUDA_LAUNCH_BLOCKING=1' >> ~/.bashrc
echo 'export TORCH_CUDA_ARCH_LIST="5.0;6.0;7.0;7.5;8.0;8.6;9.0;12.0"' >> ~/.bashrc

print_success "Environment variables set"

# Uninstall current PyTorch
print_status "Removing current PyTorch installation..."
python3 -m pip uninstall torch torchvision torchaudio -y

# Install PyTorch nightly with CUDA 12.4 (has sm_120 support)
print_status "Installing PyTorch nightly with Blackwell support..."
python3 -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# Verify installation
print_status "Verifying PyTorch installation..."
python3 -c "
import torch
import os

# Set environment variables
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'✅ CUDA version: {torch.version.cuda}')
    print(f'✅ GPU count: {torch.cuda.device_count()}')
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'✅ GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'   Compute Capability: sm_{props.major}{props.minor}')
        print(f'   Memory: {props.total_memory / 1024**3:.1f} GB')
    
    # Test basic CUDA operations
    print('\\n🧪 Testing CUDA operations...')
    try:
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = torch.matmul(x, y)
        print('✅ Basic CUDA operations successful')
        
        # Test with different data types
        x_half = torch.randn(100, 100, dtype=torch.float16).cuda()
        y_half = torch.randn(100, 100, dtype=torch.float16).cuda()
        z_half = torch.matmul(x_half, y_half)
        print('✅ Half precision operations successful')
        
        print('🎉 All CUDA tests passed!')
        
    except Exception as e:
        print(f'❌ CUDA operation failed: {e}')
        print('This may indicate ongoing compatibility issues')
        exit(1)
else:
    print('❌ CUDA not available')
    exit(1)
"

if [ $? -eq 0 ]; then
    print_success "🎉 Blackwell compatibility fix applied successfully!"
    echo ""
    print_status "You can now run SkyReels-V2:"
    print_status "  python3 gradio_app.py --host 0.0.0.0 --port 7860 --share"
    echo ""
    print_warning "Note: You're using PyTorch nightly build"
    print_warning "This may have some instability but supports sm_120"
else
    print_error "Fix failed. Please check the errors above."
    exit 1
fi

# Test SkyReels-V2 imports
print_status "Testing SkyReels-V2 imports..."
python3 -c "
try:
    from skyreels_v2_infer.modules import download_model
    from skyreels_v2_infer.pipelines import Text2VideoPipeline, Image2VideoPipeline
    from skyreels_v2_infer import DiffusionForcingPipeline
    print('✅ SkyReels-V2 imports successful')
except ImportError as e:
    print(f'⚠️  SkyReels-V2 import issue: {e}')
    print('You may need to install additional dependencies')
"

print_success "Blackwell fix completed!"
echo ""
echo "🚀 Ready to generate videos with your RTX 6000 Blackwell!"
