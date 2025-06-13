#!/bin/bash

# Quick fix for missing dependencies

echo "🔧 Installing missing dependencies for SkyReels-V2..."

# Install missing packages one by one
echo "Installing decord..."
python3 -m pip install decord>=0.6.0

echo "Installing other potentially missing packages..."
python3 -m pip install ninja packaging psutil easydict ftfy

# Ensure PyTorch is properly installed for RTX 6000 Blackwell
echo "Checking PyTorch installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

# Test the installation
echo ""
echo "🧪 Testing SkyReels-V2 imports..."

python3 -c "
try:
    import decord
    print('✅ decord imported successfully')

    import torch
    print(f'✅ PyTorch {torch.__version__} with CUDA: {torch.cuda.is_available()}')

    from skyreels_v2_infer.modules import download_model
    from skyreels_v2_infer.pipelines import Text2VideoPipeline, Image2VideoPipeline
    from skyreels_v2_infer import DiffusionForcingPipeline
    print('✅ All SkyReels-V2 imports successful')
    print('🎉 Ready to run!')
except ImportError as e:
    print(f'❌ Import error: {e}')
    print('Trying to install missing dependency...')

    # Try to identify and install the missing package
    error_str = str(e)
    if 'decord' in error_str:
        import subprocess
        subprocess.run(['python3', '-m', 'pip', 'install', 'decord>=0.6.0'])
    elif 'flash_attn' in error_str:
        print('flash_attn missing - this is optional for basic functionality')
    else:
        print(f'Unknown import error: {error_str}')

    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🚀 Fix applied successfully! You can now run:"
    echo "   python3 gradio_app.py --host 0.0.0.0 --port 7860 --share"
    echo "   or"
    echo "   bash simple_setup.sh run-share"
else
    echo ""
    echo "❌ Some issues remain. Let's try the complete installer:"
    echo "   bash install_deps.sh"
fi
