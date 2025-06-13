#!/bin/bash

# Quick fix for flash_attn installation issue
# Run this script to resolve the current installation problem

echo "🔧 Quick Fix for flash_attn Installation Issue"
echo "=============================================="

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  No virtual environment detected. Activating venv..."
    if [ -d "venv" ]; then
        source venv/bin/activate
        echo "✅ Virtual environment activated"
    else
        echo "❌ No venv directory found. Please create one first:"
        echo "python3 -m venv venv && source venv/bin/activate"
        exit 1
    fi
fi

# Check if PyTorch is installed
echo "🔍 Checking PyTorch installation..."
if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo "✅ PyTorch $TORCH_VERSION is installed"
else
    echo "❌ PyTorch not found. Installing PyTorch first..."
    
    # Detect CUDA version
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/')
        echo "Detected CUDA version: $CUDA_VERSION"
        
        if [[ "$CUDA_VERSION" == 12.* ]]; then
            pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
        elif [[ "$CUDA_VERSION" == 11.* ]]; then
            pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
        else
            pip install torch==2.5.1 torchvision==0.20.1
        fi
    else
        echo "No CUDA detected, installing CPU version..."
        pip install torch==2.5.1 torchvision==0.20.1
    fi
fi

# Now install flash_attn separately
echo "⚡ Installing flash_attn..."
echo "This may take 5-10 minutes as it needs to compile from source..."

# Try multiple installation methods
if pip install flash_attn --no-build-isolation; then
    echo "✅ flash_attn installed successfully!"
elif FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash_attn --no-build-isolation; then
    echo "✅ flash_attn installed with forced build!"
elif pip install flash_attn --find-links https://github.com/Dao-AILab/flash-attention/releases; then
    echo "✅ flash_attn installed from pre-built wheel!"
else
    echo "❌ flash_attn installation failed with all methods."
    echo ""
    echo "🔧 Troubleshooting options:"
    echo "1. Check if you have build tools installed:"
    echo "   sudo apt-get install build-essential (Ubuntu/Debian)"
    echo "   yum groupinstall 'Development Tools' (CentOS/RHEL)"
    echo ""
    echo "2. Check CUDA compatibility:"
    echo "   flash_attn requires CUDA 11.6+ and compatible GPU"
    echo ""
    echo "3. Try installing without flash_attn for now:"
    echo "   You can comment out flash_attn usage in your code temporarily"
    echo ""
    exit 1
fi

# Install remaining dependencies
echo "📦 Installing remaining dependencies..."
pip install \
    opencv-python==4.10.0.84 \
    "diffusers>=0.31.0" \
    transformers==4.49.0 \
    tokenizers==0.21.1 \
    accelerate==1.6.0 \
    tqdm \
    imageio \
    easydict \
    ftfy \
    dashscope \
    imageio-ffmpeg \
    "numpy>=1.23.5,<2"

# Try to install xfuser
echo "🚀 Installing xfuser..."
if pip install xfuser; then
    echo "✅ xfuser installed successfully!"
else
    echo "⚠️  xfuser installation failed. You may need to install it manually later."
fi

# Verify the installation
echo "🧪 Verifying installation..."
python -c "
try:
    import torch
    import flash_attn
    print('✅ Both PyTorch and flash_attn are working!')
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

echo ""
echo "✅ Installation fix completed!"
echo "You can now run your application or continue with the full setup."
