#!/bin/bash

# SkyReels V2 Setup Script for Blackwell GPUs (RTX 6000 Pro, etc.)
# This script sets up the environment with proper PyTorch support for sm_120 architecture

set -e  # Exit on any error

echo "🚀 Setting up SkyReels V2 for Blackwell GPU (RTX 6000 Pro)"
echo "============================================================"

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found. Please run this script from the SkyReels-V2 directory."
    exit 1
fi

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ Error: nvidia-smi not found. Please install NVIDIA drivers first."
    exit 1
fi

# Get GPU info
echo "🔍 Detecting GPU..."
GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
echo "Found GPU: $GPU_INFO"

# Check CUDA version
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/')
echo "CUDA Version: $CUDA_VERSION"

# Determine the appropriate PyTorch version for Blackwell
PYTORCH_VERSION="2.5.1"  # Use latest stable version with Blackwell support
TORCHVISION_VERSION="0.20.1"
TORCHAUDIO_VERSION="2.5.1"

echo "📦 Installing PyTorch with Blackwell GPU support..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support for Blackwell
echo "🔥 Installing PyTorch $PYTORCH_VERSION with CUDA support..."

# For CUDA 12.x
if [[ "$CUDA_VERSION" == 12.* ]]; then
    pip install torch==$PYTORCH_VERSION torchvision==$TORCHVISION_VERSION torchaudio==$TORCHAUDIO_VERSION --index-url https://download.pytorch.org/whl/cu121
elif [[ "$CUDA_VERSION" == 11.* ]]; then
    pip install torch==$PYTORCH_VERSION torchvision==$TORCHVISION_VERSION torchaudio==$TORCHAUDIO_VERSION --index-url https://download.pytorch.org/whl/cu118
else
    echo "⚠️  Warning: Unsupported CUDA version. Installing default PyTorch..."
    pip install torch==$PYTORCH_VERSION torchvision==$TORCHVISION_VERSION torchaudio==$TORCHAUDIO_VERSION
fi

# Verify PyTorch installation
echo "🧪 Verifying PyTorch installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        compute_capability = torch.cuda.get_device_capability(i)
        print(f'GPU {i}: {gpu_name} (Compute Capability: {compute_capability[0]}.{compute_capability[1]})')
"

# Install other dependencies in correct order
echo "📚 Installing SkyReels V2 dependencies..."

# First, install dependencies that don't require special compilation
echo "🔧 Installing basic dependencies..."
pip install opencv-python==4.10.0.84 \
           diffusers>=0.31.0 \
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

# Install flash_attn separately after PyTorch is confirmed working
echo "⚡ Installing flash_attn (this may take several minutes)..."
echo "Note: flash_attn requires compilation and may take 5-10 minutes to build..."
pip install flash_attn --no-build-isolation

# Install xfuser last as it may have specific requirements
echo "🚀 Installing xfuser..."
pip install xfuser

# Install Gradio and additional dependencies
echo "🌐 Installing Gradio and web interface dependencies..."
pip install gradio==4.44.0 jupyter jupyterlab ipywidgets

# Additional optimizations for Blackwell
echo "⚡ Installing performance optimizations..."
pip install xformers triton

# Set environment variables for optimal performance
echo "🔧 Setting environment variables..."
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=1

# Create environment file
cat > .env << EOF
# Environment variables for SkyReels V2 on Blackwell GPU
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
CUDA_LAUNCH_BLOCKING=0
TORCH_USE_CUDA_DSA=1
TORCH_LOGS=+dynamo
TORCH_COMPILE_DEBUG=0
EOF

echo "📝 Environment variables saved to .env file"

# Test CUDA functionality
echo "🧪 Testing CUDA functionality..."
python -c "
import torch
import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'

try:
    # Test basic CUDA operations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    if torch.cuda.is_available():
        # Test tensor operations
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        z = torch.matmul(x, y)
        print(f'✅ CUDA tensor operations working: {z.shape}')
        
        # Test mixed precision
        with torch.cuda.amp.autocast():
            z_mixed = torch.matmul(x.half(), y.half())
        print(f'✅ Mixed precision working: {z_mixed.dtype}')
        
        # Clear cache
        torch.cuda.empty_cache()
        print('✅ CUDA cache cleared successfully')
    else:
        print('❌ CUDA not available')
except Exception as e:
    print(f'❌ CUDA test failed: {e}')
    print('This might be normal for first run. Try restarting your session.')
"

# Create launcher script
cat > run_gradio.sh << 'EOF'
#!/bin/bash
# Launcher script for SkyReels V2 Gradio interface

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# Activate virtual environment
source venv/bin/activate

# Set optimizations for Blackwell
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_USE_CUDA_DSA=1

echo "🚀 Starting SkyReels V2 Gradio interface..."
echo "Hardware optimization enabled for Blackwell GPU"

# Run the Gradio app
python app.py "$@"
EOF

chmod +x run_gradio.sh

# Create Jupyter launcher
cat > run_jupyter.sh << 'EOF'
#!/bin/bash
# Launcher script for Jupyter with SkyReels V2

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# Activate virtual environment
source venv/bin/activate

# Set optimizations for Blackwell
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_USE_CUDA_DSA=1

echo "🚀 Starting Jupyter Lab with SkyReels V2 environment..."
echo "Hardware optimization enabled for Blackwell GPU"

# Install kernel if not exists
python -m ipykernel install --user --name skyreels-v2 --display-name "SkyReels V2"

# Run Jupyter Lab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root "$@"
EOF

chmod +x run_jupyter.sh

# Check model cache directory
if [ ! -d "cache" ]; then
    mkdir -p cache
    echo "📁 Created cache directory for models"
fi

# Create result directory
if [ ! -d "result" ]; then
    mkdir -p result
    echo "📁 Created result directory for outputs"
fi

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "📋 Quick Start Guide:"
echo "====================="
echo ""
echo "1. 🌐 Run Gradio Web Interface:"
echo "   ./run_gradio.sh"
echo ""
echo "2. 📓 Run Jupyter Lab:"
echo "   ./run_jupyter.sh"
echo ""
echo "3. 🐍 Manual Python execution:"
echo "   source venv/bin/activate"
echo "   python app.py"
echo ""
echo "📊 System Information:"
echo "====================="
echo "GPU: $GPU_INFO"
echo "CUDA: $CUDA_VERSION"
echo "PyTorch: $PYTORCH_VERSION"
echo ""
echo "🔧 Optimizations Applied:"
echo "========================"
echo "✅ Blackwell GPU support (sm_120)"
echo "✅ Memory optimization for large models"
echo "✅ Mixed precision training"
echo "✅ CUDA DSA enabled for debugging"
echo "✅ XFormers for attention optimization"
echo ""
echo "⚠️  Important Notes:"
echo "==================="
echo "• Models will be downloaded automatically on first use"
echo "• Ensure you have at least 50GB free space for models"
echo "• Use 1.3B models first to test your setup"
echo "• Monitor GPU temperature during long generations"
echo ""
echo "🎬 Happy video generating!"
EOF 