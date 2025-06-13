# 🎬 SkyReels-V2 Complete Setup Guide

## 🚀 One-Command Setup and Run

This guide provides a comprehensive, automated setup system that handles everything for you:
- ✅ System compatibility checks
- ✅ RTX 6000 Blackwell GPU detection and fixes
- ✅ PyTorch installation (stable or nightly as needed)
- ✅ All dependencies installation
- ✅ Environment configuration
- ✅ Automatic Gradio app launch

## 📋 Quick Start

### Option 1: Complete Setup and Run (Recommended)
```bash
# One command does everything and starts the app with public sharing
./run_skyreel.sh
```

### Option 2: Setup Only (No Auto-Run)
```bash
# Setup everything but don't start the app
./run_skyreel.sh setup
```

### Option 3: Local Only (No Public Sharing)
```bash
# Setup and run locally without public URL
./run_skyreel.sh local
```

## 🎯 Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `run` | Complete setup + run with sharing (default) | `./run_skyreel.sh run` |
| `local` | Complete setup + run locally only | `./run_skyreel.sh local` |
| `setup` | Setup only, don't run the app | `./run_skyreel.sh setup` |
| `deps` | Install dependencies only | `./run_skyreel.sh deps` |
| `test` | Test current installation | `./run_skyreel.sh test` |
| `help` | Show help information | `./run_skyreel.sh help` |

## ⚙️ Advanced Options

```bash
# Run on custom port
./run_skyreel.sh run --port 8080

# Skip dependency installation (if already installed)
./run_skyreel.sh run --skip-deps

# Setup only without running
./run_skyreel.sh setup

# Test installation without reinstalling
./run_skyreel.sh test
```

## 🔧 What the Script Does

### 1. System Checks
- ✅ Python version compatibility (3.8+)
- ✅ GPU detection and compatibility analysis
- ✅ RTX 6000 Blackwell architecture detection

### 2. Environment Setup
- ✅ Sets CUDA environment variables
- ✅ Configures Blackwell-specific settings if needed
- ✅ Creates necessary directories

### 3. PyTorch Installation
- ✅ **RTX 6000 Blackwell**: Installs PyTorch nightly with sm_120 support
- ✅ **Other NVIDIA GPUs**: Installs stable PyTorch with appropriate CUDA version
- ✅ **No GPU**: Installs CPU-only version

### 4. Dependencies
- ✅ Installs all required packages from requirements.txt
- ✅ Handles version conflicts automatically
- ✅ Installs critical packages individually if batch install fails

### 5. Testing
- ✅ Verifies PyTorch and CUDA functionality
- ✅ Tests GPU operations (including Blackwell-specific fixes)
- ✅ Validates SkyReels-V2 imports

### 6. App Launch
- ✅ Starts Gradio interface
- ✅ Provides both local and public URLs
- ✅ Handles graceful shutdown

## 🎮 GPU-Specific Handling

### RTX 6000 Blackwell (sm_120)
```bash
# Automatically detected and handled
./run_skyreel.sh
```
**What happens:**
- Installs PyTorch nightly with CUDA 12.4
- Sets `TORCH_CUDA_ARCH_LIST="5.0;6.0;7.0;7.5;8.0;8.6;9.0;12.0"`
- Applies Blackwell compatibility fixes in VAE code
- Tests tensor operations for compatibility

### Other NVIDIA GPUs
```bash
# Standard setup works for most GPUs
./run_skyreel.sh
```
**What happens:**
- Installs stable PyTorch with CUDA 12.1
- Standard CUDA architecture list
- Regular compatibility testing

### CPU-Only Systems
```bash
# Automatically falls back to CPU mode
./run_skyreel.sh
```
**What happens:**
- Installs CPU-only PyTorch
- Warns about slow performance
- Skips GPU-specific configurations

## 📊 Expected Output

### Successful Setup
```
🎬 SkyReels-V2 Complete Setup and Run
============================================================
[INFO] Python version: 3.10.12
[SUCCESS] Python version is compatible

🎮 Checking GPU Compatibility
============================================================
[INFO] GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
[INFO] Memory: 97.0 GB
[INFO] Compute Capability: 12.0
[WARNING] RTX 6000 Blackwell detected - requires PyTorch nightly

🔧 Setting Environment Variables
============================================================
[INFO] Setting Blackwell-specific environment variables
[INFO] Set CUDA_LAUNCH_BLOCKING=1
[INFO] Set TORCH_CUDA_ARCH_LIST=5.0;6.0;7.0;7.5;8.0;8.6;9.0;12.0

🔥 Installing PyTorch
============================================================
[INFO] Installing PyTorch nightly for Blackwell architecture...
[SUCCESS] PyTorch installation completed

📦 Installing Dependencies
============================================================
[SUCCESS] Dependencies installation completed

🧪 Testing Installation
============================================================
[SUCCESS] PyTorch version: 2.7.0.dev20250310+cu124
[SUCCESS] CUDA operations test passed
[SUCCESS] SkyReels-V2 imports successful

🚀 Starting SkyReels-V2 Gradio Interface
============================================================
[INFO] Starting with public sharing enabled...
[INFO] Local URL: http://localhost:7860
[INFO] Public URL will be shown below...

Running on local URL:  http://0.0.0.0:7860
Running on public URL: https://abc123def456.gradio.live
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. Permission Errors
```bash
# Make scripts executable
chmod +x run_skyreel.sh setup_and_run_complete.py
```

#### 2. Python Not Found
```bash
# Check Python installation
python3 --version
# or try
python --version
```

#### 3. CUDA Issues
```bash
# Check NVIDIA drivers
nvidia-smi

# Test CUDA after setup
python3 -c "import torch; print(torch.cuda.is_available())"
```

#### 4. Network Issues (PyTorch Installation)
```bash
# Try with different index URLs
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

#### 5. Memory Issues
```bash
# For systems with limited RAM, install packages one by one
./run_skyreel.sh deps
```

### Manual Fallback

If the automated script fails, you can run components manually:

```bash
# 1. Install PyTorch manually
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables (for Blackwell)
export CUDA_LAUNCH_BLOCKING=1
export TORCH_CUDA_ARCH_LIST="5.0;6.0;7.0;7.5;8.0;8.6;9.0;12.0"

# 4. Run the app
python3 gradio_app.py --host 0.0.0.0 --port 7860 --share
```

## 📁 Files Created

The setup script creates several files:

- `skyreel_env.sh` - Environment variables for persistence
- `models/` - Downloaded model cache
- `result/video_out/` - Generated videos
- `result/diffusion_forcing/` - Diffusion forcing outputs
- `uploads/` - Uploaded images
- `cache/` - Various cache files

## 🎉 Success Indicators

You'll know the setup worked when you see:
- ✅ All system checks pass
- ✅ PyTorch installs without errors
- ✅ CUDA operations test successfully
- ✅ Gradio interface starts and shows URLs
- ✅ You can access the web interface

## 🚀 Next Steps

Once setup is complete:
1. **Access the interface** at the provided URL
2. **Select a model** (14B models recommended for RTX 6000)
3. **Enter a prompt** and generate your first video
4. **Experiment with settings** like TeaCache for faster generation

## 💡 Pro Tips

- **RTX 6000 Users**: You have 97GB VRAM - use the largest models!
- **Enable TeaCache**: 2-3x faster generation with minimal quality loss
- **Start with 540P**: Then try 720P once you're comfortable
- **Long videos**: Use Diffusion Forcing mode for 60+ second videos

---

**Need help?** Check the troubleshooting section or run `./run_skyreel.sh help` for more options.
