# 🎬 SkyReels-V2 Jupyter Setup

Easy-to-use Jupyter notebook setup for SkyReels-V2 video generation with full Gradio interface.

## 🚀 One-Click Setup

### For Jupyter Notebooks / Google Colab / Kaggle

```bash
# Clone this repository
git clone https://github.com/rdlogout/skyreel.git
cd skyreel

# Run automated setup and launch
bash setup_and_run.sh full
```

That's it! The script will:
- ✅ Detect your environment (Colab/Kaggle/Jupyter)
- ✅ Check GPU availability
- ✅ Install all dependencies automatically
- ✅ Start Gradio interface with public sharing
- ✅ Provide both local and public URLs

## 📱 Quick Access

### Option 1: Use the Jupyter Notebook
1. Open `SkyReels_V2_Jupyter.ipynb`
2. Run all cells
3. Access the Gradio interface

### Option 2: Command Line
```bash
# Complete setup and start
bash setup_and_run.sh full

# Or step by step
bash setup_and_run.sh setup    # Install dependencies
bash setup_and_run.sh run-share # Start with public URL
```

## 🎯 Features

### Generation Modes
- **Text-to-Video**: Create videos from text descriptions
- **Image-to-Video**: Animate static images into videos
- **Diffusion Forcing**: Generate long videos (30-60+ seconds)

### Model Options
- **540P Models**: 544×960 resolution, faster generation
- **720P Models**: 720×1280 resolution, higher quality
- **Size Variants**: 1.3B (faster) and 14B (better quality)

### Advanced Features
- **Public Sharing**: Automatic Gradio public URLs
- **Prompt Enhancement**: AI-powered prompt improvement
- **TeaCache**: 2-3x faster inference
- **Custom Duration**: 4 seconds to 60+ seconds
- **Quality Controls**: Guidance scale, inference steps, etc.

## 💻 System Requirements

### Minimum (1.3B Models)
- **GPU**: 16GB VRAM
- **RAM**: 16GB
- **Storage**: 50GB

### Recommended (14B Models)
- **GPU**: 24GB+ VRAM (RTX 4090, A100, etc.)
- **RAM**: 32GB+
- **Storage**: 100GB+

### Supported Environments
- ✅ Google Colab (Free/Pro)
- ✅ Kaggle Notebooks
- ✅ Jupyter Lab/Notebook
- ✅ Local Linux/Ubuntu
- ✅ Any environment with Python 3.8+

## 🎨 Usage Examples

### Text-to-Video
```
Mode: Text-to-Video
Model: SkyReels-V2-T2V-14B-540P
Prompt: "A majestic eagle soaring over snow-capped mountains at sunset"
Duration: 4 seconds (~97 frames)
```

### Image-to-Video
```
Mode: Image-to-Video
Upload: Your image file
Prompt: "The person in the image walking through a magical forest"
Duration: 4 seconds (~97 frames)
```

### Long Video Generation
```
Mode: Diffusion Forcing
Prompt: "A cinematic sequence of a motorcycle ride through a desert highway"
Duration: 30 seconds (~737 frames)
```

## 🔧 Available Commands

```bash
# Complete automated setup and start
bash setup_and_run.sh full

# Install dependencies only
bash setup_and_run.sh setup

# Start interface (local access)
bash setup_and_run.sh run

# Start interface with public sharing
bash setup_and_run.sh run-share

# Test installation
bash setup_and_run.sh test

# Show help
bash setup_and_run.sh help
```

## 🌐 Access Methods

After running the setup:

- **Local**: http://localhost:7860
- **Public**: https://xxxxx.gradio.live (auto-generated)
- **Network**: http://YOUR_IP:7860 (if accessible)

The public URL is perfect for:
- 📱 Sharing with others
- 🔗 Accessing from different devices
- 🎯 Demos and presentations

## 🛠️ Troubleshooting

### Common Issues

**Out of Memory**
- Use 1.3B models instead of 14B
- Enable model offloading in interface
- Reduce frame count/duration

**Slow Generation**
- Enable TeaCache (2-3x speedup)
- Reduce inference steps to 20-30
- Use smaller models

**Installation Errors**
- Run `bash setup_and_run.sh setup` again
- Check internet connection
- Ensure sufficient disk space

**GPU Not Detected**
- Check CUDA installation
- Restart runtime (in Colab/Kaggle)
- Will fallback to CPU (very slow)

## 📊 Performance Tips

1. **Start Small**: Begin with 4-second videos and 1.3B models
2. **Use TeaCache**: Enable for 2-3x speedup with minimal quality loss
3. **Optimize Settings**: Lower inference steps (20-30) for faster generation
4. **Batch Processing**: Generate multiple short videos rather than one long video
5. **Monitor Resources**: Keep an eye on GPU memory usage

## 🔗 Links

- **Original SkyReels-V2**: https://github.com/SkyworkAI/SkyReels-V2
- **Models on Hugging Face**: https://huggingface.co/collections/Skywork/skyreels-v2-6801b1b93df627d441d0d0d9
- **Technical Paper**: https://arxiv.org/pdf/2504.13074
- **Discord Community**: https://discord.gg/PwM6NYtccQ

## 📄 License

This project follows the original SkyReels-V2 license terms.

---

**Ready to create amazing videos? Just run the setup script and start generating! 🚀**
