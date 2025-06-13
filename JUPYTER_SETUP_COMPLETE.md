# 🎬 SkyReels-V2 Jupyter Setup - Complete!

## 📁 Files Created for Your GitHub Repository

I've created a complete Jupyter-friendly setup for SkyReels-V2. Here are the files you should push to your GitHub repository at `https://github.com/rdlogout/skyreel`:

### Core Files:
1. **`setup_and_run.sh`** - Main automated setup script
2. **`gradio_app.py`** - Full-featured Gradio web interface
3. **`SkyReels_V2_Jupyter.ipynb`** - Ready-to-use Jupyter notebook
4. **`check_system.py`** - System requirements checker
5. **`requirements.txt`** - Updated Python dependencies
6. **`README_JUPYTER.md`** - Documentation for your repo

## 🚀 How Users Will Use Your Repository

### Step 1: Clone Your Repository
```bash
git clone https://github.com/rdlogout/skyreel.git
cd skyreel
```

### Step 2: One-Command Setup and Launch
```bash
# Complete automated setup and start with public sharing
bash setup_and_run.sh full
```

That's it! The script automatically:
- ✅ Detects environment (Colab/Kaggle/Jupyter/Linux)
- ✅ Checks system requirements (GPU, memory, disk space)
- ✅ Installs all dependencies
- ✅ Downloads and configures models
- ✅ Starts Gradio interface with public URL
- ✅ Provides both local and public access

## 🎯 Key Features

### Automated Environment Detection
- **Google Colab**: Automatically detected and configured
- **Kaggle**: Optimized for Kaggle notebooks
- **Jupyter**: Works with any Jupyter environment
- **Linux**: Standard Linux server support

### Smart Dependency Management
- **PyTorch**: Automatically installs CUDA or CPU version
- **System Packages**: Installs required system dependencies
- **Python Packages**: Handles all Python requirements
- **GPU Support**: Detects and configures GPU acceleration

### Full Gradio Interface
- **Text-to-Video**: Generate videos from text descriptions
- **Image-to-Video**: Animate static images
- **Diffusion Forcing**: Create long videos (30-60+ seconds)
- **Model Selection**: 540P/720P, 1.3B/14B variants
- **Advanced Controls**: Duration, quality, performance settings

### Public Sharing
- **Gradio Share**: Creates public `https://xxxxx.gradio.live` URLs
- **Port Exposure**: Direct access via `http://server-ip:7860`
- **No Configuration**: Works out of the box

## 📱 Usage Options

### Option 1: Jupyter Notebook (Recommended)
1. Open `SkyReels_V2_Jupyter.ipynb`
2. Run the cells in order
3. Access the Gradio interface

### Option 2: Command Line
```bash
# Check system requirements
bash setup_and_run.sh check

# Complete setup and start
bash setup_and_run.sh full

# Or step by step
bash setup_and_run.sh setup     # Install dependencies
bash setup_and_run.sh run-share # Start with public URL
```

### Option 3: Direct Python
```python
# After setup, start directly
python gradio_app.py --host 0.0.0.0 --port 7860 --share
```

## 🔧 Available Commands

```bash
bash setup_and_run.sh check      # Check system requirements
bash setup_and_run.sh setup      # Install dependencies only
bash setup_and_run.sh run        # Start locally (http://localhost:7860)
bash setup_and_run.sh run-share  # Start with public URL
bash setup_and_run.sh full       # Complete setup and start (default)
bash setup_and_run.sh test       # Test installation
bash setup_and_run.sh help       # Show help
```

## 💻 System Requirements

### Minimum (1.3B Models)
- **GPU**: 16GB VRAM
- **RAM**: 16GB
- **Storage**: 50GB

### Recommended (14B Models)
- **GPU**: 24GB+ VRAM
- **RAM**: 32GB+
- **Storage**: 100GB+

### Supported Platforms
- ✅ Google Colab (Free/Pro)
- ✅ Kaggle Notebooks
- ✅ Jupyter Lab/Notebook
- ✅ Local Linux/Ubuntu
- ✅ Any Python 3.8+ environment

## 🎨 Generation Examples

### Text-to-Video
```
Prompt: "A majestic eagle soaring over snow-capped mountains at sunset"
Model: SkyReels-V2-T2V-14B-540P
Duration: 4 seconds (~97 frames)
```

### Image-to-Video
```
Upload: Portrait image
Prompt: "The person smiling and waving at the camera"
Model: SkyReels-V2-I2V-14B-540P
Duration: 4 seconds (~97 frames)
```

### Long Video (Diffusion Forcing)
```
Prompt: "A cinematic motorcycle ride through a desert highway at sunset"
Model: SkyReels-V2-DF-14B-540P
Duration: 30 seconds (~737 frames)
```

## 🌐 Access URLs

After running the setup:
- **Local**: http://localhost:7860
- **Public**: https://xxxxx.gradio.live (auto-generated)
- **Network**: http://YOUR_SERVER_IP:7860

## 🛠️ Troubleshooting

### Common Issues
- **Out of Memory**: Use 1.3B models, enable offloading
- **Slow Generation**: Enable TeaCache, reduce inference steps
- **GPU Issues**: Check CUDA installation, restart runtime
- **Installation Errors**: Run setup script again

### Performance Tips
1. Start with 1.3B models for testing
2. Enable TeaCache for 2-3x speedup
3. Use shorter durations initially (4-10 seconds)
4. Monitor GPU memory usage

## 📋 Files to Push to Your GitHub

Make sure to include these files in your repository:

```
skyreel/
├── setup_and_run.sh           # Main setup script
├── gradio_app.py              # Gradio interface
├── SkyReels_V2_Jupyter.ipynb  # Jupyter notebook
├── check_system.py            # System checker
├── requirements.txt           # Python dependencies
├── README_JUPYTER.md          # Documentation
└── skyreels_v2_infer/         # Original SkyReels code
```

## 🎉 Ready to Deploy!

Your repository will provide users with:
- **One-click setup** for any environment
- **Automatic configuration** for Colab/Kaggle/Jupyter
- **Public sharing** with Gradio URLs
- **Full feature access** to all SkyReels-V2 capabilities
- **Comprehensive documentation** and examples

Users can simply clone your repo and run one command to get a fully functional video generation interface! 🚀
