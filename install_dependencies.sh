#!/bin/bash

# Manual dependency installation script for SkyReels V2
# Use this if the main setup script fails or for troubleshooting

set -e

echo "🔧 Manual SkyReels V2 Dependency Installation"
echo "=============================================="

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "❌ Error: Virtual environment not activated."
    echo "Please run: source venv/bin/activate"
    exit 1
fi

# Check if PyTorch is installed
echo "🔍 Checking PyTorch installation..."
python -c "import torch; print(f'PyTorch {torch.__version__} found')" 2>/dev/null || {
    echo "❌ PyTorch not found. Please install PyTorch first using setup_blackwell.sh"
    exit 1
}

echo "✅ PyTorch is available"

# Install basic dependencies first
echo "📦 Installing basic dependencies..."
pip install --upgrade pip

echo "🔧 Installing core dependencies..."
pip install \
    "numpy>=1.23.5,<2" \
    opencv-python==4.10.0.84 \
    tqdm \
    imageio \
    imageio-ffmpeg \
    easydict \
    ftfy

echo "🤖 Installing ML/AI libraries..."
pip install \
    "diffusers>=0.31.0" \
    transformers==4.49.0 \
    tokenizers==0.21.1 \
    accelerate==1.6.0 \
    dashscope

# Install flash_attn with special handling
echo "⚡ Installing flash_attn (this will take several minutes)..."
echo "Note: flash_attn requires compilation. Please be patient..."

# Try different installation methods for flash_attn
if ! pip install flash_attn --no-build-isolation; then
    echo "⚠️  Standard flash_attn installation failed. Trying alternative method..."
    
    # Try with specific CUDA architecture
    if ! FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash_attn --no-build-isolation; then
        echo "⚠️  flash_attn installation failed. Trying pre-built wheel..."
        
        # Try to install from pre-built wheels
        if ! pip install flash_attn --find-links https://github.com/Dao-AILab/flash-attention/releases; then
            echo "❌ All flash_attn installation methods failed."
            echo "You may need to:"
            echo "1. Check your CUDA version compatibility"
            echo "2. Install build tools: apt-get install build-essential"
            echo "3. Try installing without flash_attn for now"
            echo ""
            echo "To continue without flash_attn, you can comment it out in your code."
        else
            echo "✅ flash_attn installed from pre-built wheel"
        fi
    else
        echo "✅ flash_attn installed with forced build"
    fi
else
    echo "✅ flash_attn installed successfully"
fi

# Install xfuser
echo "🚀 Installing xfuser..."
if ! pip install xfuser; then
    echo "⚠️  xfuser installation failed. This may be due to compatibility issues."
    echo "You can try installing it manually later if needed."
else
    echo "✅ xfuser installed successfully"
fi

# Verify installations
echo "🧪 Verifying installations..."
python -c "
import sys
modules_to_check = [
    'torch', 'torchvision', 'cv2', 'diffusers', 
    'transformers', 'accelerate', 'numpy'
]

optional_modules = ['flash_attn', 'xfuser']

print('Core modules:')
for module in modules_to_check:
    try:
        __import__(module)
        print(f'  ✅ {module}')
    except ImportError as e:
        print(f'  ❌ {module}: {e}')
        sys.exit(1)

print('\\nOptional modules:')
for module in optional_modules:
    try:
        __import__(module)
        print(f'  ✅ {module}')
    except ImportError:
        print(f'  ⚠️  {module}: Not available (this may be okay)')

print('\\n✅ Core dependencies verification completed!')
"

echo ""
echo "✅ Dependency installation completed!"
echo ""
echo "📋 Next steps:"
echo "1. Test your installation with: python -c 'import torch; print(torch.cuda.is_available())'"
echo "2. Run the main application: python app.py"
echo "3. Or use the Gradio interface: ./run_gradio.sh"
