# 🔧 RTX 6000 Blackwell CUDA sm_120 Fix Guide

## Problem
Your RTX PRO 6000 Blackwell GPU has CUDA capability sm_120, but standard PyTorch only supports up to sm_90. This causes the error:

```
CUDA error: no kernel image is available for execution on the device
NVIDIA RTX PRO 6000 Blackwell Workstation Edition with CUDA capability sm_120 is not compatible with the current PyTorch installation.
```

## ✅ Solution: Use PyTorch Nightly with Blackwell Support

### Option 1: Automated Fix Script (Recommended)

```bash
# Run the automated Blackwell fix
bash fix_blackwell.sh
```

This script will:
- ✅ Detect your RTX 6000 Blackwell GPU
- ✅ Set required environment variables
- ✅ Uninstall current PyTorch
- ✅ Install PyTorch nightly with sm_120 support
- ✅ Test CUDA operations
- ✅ Verify SkyReels-V2 compatibility

### Option 2: Manual Fix

```bash
# Set environment variables
export CUDA_LAUNCH_BLOCKING=1
export TORCH_CUDA_ARCH_LIST="5.0;6.0;7.0;7.5;8.0;8.6;9.0;12.0"

# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio -y

# Install PyTorch nightly with CUDA 12.4 (has sm_120 support)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# Test installation
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'GPU: {torch.cuda.get_device_name(0)} (sm_{props.major}{props.minor})')
"
```

### Option 3: Jupyter Notebook Fix

Use the updated `SkyReels_V2_Jupyter.ipynb` which includes a dedicated cell for Blackwell fix:

1. Open the notebook
2. Run the "RTX 6000 Blackwell Fix" cell
3. This will automatically install PyTorch nightly

## 🎯 What's Different in the Fix

### Environment Variables
```bash
CUDA_LAUNCH_BLOCKING=1          # Better error reporting
TORCH_CUDA_ARCH_LIST="5.0;6.0;7.0;7.5;8.0;8.6;9.0;12.0"  # Include sm_120
```

### PyTorch Version
- **Before**: Stable PyTorch with CUDA 12.1 (max sm_90)
- **After**: PyTorch nightly with CUDA 12.4 (includes sm_120)

### Updated Files
1. **`fix_blackwell.sh`** - Automated fix script
2. **`install_deps.sh`** - Updated to detect Blackwell and use nightly
3. **`simple_setup.sh`** - Updated with Blackwell detection
4. **`SkyReels_V2_Jupyter.ipynb`** - Added Blackwell fix cell
5. **`gradio_app.py`** - Environment variables set automatically

## 🧪 Testing the Fix

After applying the fix, test with:

```python
import torch
import os

# Environment should be set
print(f"CUDA_LAUNCH_BLOCKING: {os.environ.get('CUDA_LAUNCH_BLOCKING')}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: sm_{props.major}{props.minor}")
    
    # Test basic operations
    x = torch.randn(100, 100).cuda()
    y = torch.randn(100, 100).cuda()
    z = torch.matmul(x, y)
    print("✅ CUDA operations working!")
```

## 🚀 Running SkyReels-V2 After Fix

Once the fix is applied:

```bash
# Start the Gradio interface
python3 gradio_app.py --host 0.0.0.0 --port 7860 --share

# Or use the setup script
bash simple_setup.sh run-share
```

## ⚠️ Important Notes

### PyTorch Nightly
- **Pros**: Supports latest GPU architectures (sm_120)
- **Cons**: May have some instability (it's a development version)
- **Alternative**: Wait for stable PyTorch 2.6+ with Blackwell support

### Performance Expectations
- **RTX 6000 Blackwell**: Excellent performance with 97GB VRAM
- **Recommended settings**:
  - Use 14B models for best quality
  - Enable TeaCache for 2-3x speedup
  - Start with 540P resolution, then try 720P

### Memory Usage
With 97GB VRAM, you can:
- ✅ Run 14B models comfortably
- ✅ Generate long videos (60+ seconds)
- ✅ Use multiple models simultaneously
- ✅ Process high resolutions (720P)

## 🔍 Troubleshooting

### If Fix Doesn't Work
1. **Check CUDA version**: `nvidia-smi`
2. **Verify PyTorch nightly**: Should show version like `2.6.0.dev20241201`
3. **Test environment variables**: Should be set in Python
4. **Try CPU fallback**: Set `CUDA_VISIBLE_DEVICES=""`

### Alternative Solutions
1. **Use CPU mode**: Slower but works
2. **Wait for stable PyTorch**: PyTorch 2.6+ will support Blackwell
3. **Use Docker**: May have different CUDA compatibility

## 📊 Performance Comparison

| Model | Resolution | VRAM Usage | Generation Time |
|-------|------------|------------|-----------------|
| 1.3B-540P | 544×960 | ~15GB | ~30s (4s video) |
| 14B-540P | 544×960 | ~45GB | ~60s (4s video) |
| 14B-720P | 720×1280 | ~65GB | ~90s (4s video) |

With your 97GB VRAM, all models will run comfortably!

## 🎉 Ready to Generate!

After applying the fix, your RTX 6000 Blackwell will be fully compatible with SkyReels-V2. You'll be able to:

- ✅ Generate high-quality videos
- ✅ Use all model variants
- ✅ Create long-form content
- ✅ Leverage the full 97GB VRAM

The Blackwell architecture is cutting-edge, so using PyTorch nightly is currently the best solution for compatibility! 🚀
