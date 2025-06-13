# 🎬 SkyReels V2 Gradio Web Interface

A comprehensive web interface for SkyReels V2 video generation models, built with Gradio. This interface exposes all the powerful features of SkyReels V2 through an easy-to-use web UI.

## 🚀 Quick Start

### Automatic Setup and Launch

The easiest way to get started is using the provided setup script:

```bash
# Make the script executable (if not already)
chmod +x setup_and_run.sh

# Run the setup and launch script
./setup_and_run.sh
```

This script will:
- ✅ Check your environment (GPU, Python, etc.)
- 📦 Install system dependencies
- 🐍 Set up Python environment
- 📚 Install all required packages
- 🧪 Verify the installation
- 🚀 Launch the Gradio interface with sharing enabled

### Manual Setup

If you prefer manual setup:

```bash
# Install dependencies
pip install -r requirements.txt

# Test the installation
python test_gradio.py

# Launch the interface
python gradio_skyreel.py --share
```

## 🎯 Features

### 📝 Text-to-Video (T2V)
Generate videos from text descriptions using state-of-the-art T2V models.

**Available Models:**
- `Skywork/SkyReels-V2-T2V-14B-540P` (540P resolution)
- `Skywork/SkyReels-V2-T2V-14B-720P` (720P resolution)

**Key Parameters:**
- **Prompt**: Detailed text description
- **Frames**: 25-300 frames (97 for 540P, 121 for 720P recommended)
- **FPS**: 8-30 frames per second
- **Guidance Scale**: 6.0 recommended for T2V
- **Shift**: 8.0 recommended for T2V

### 🖼️ Image-to-Video (I2V)
Animate images with text prompts to create dynamic videos.

**Available Models:**
- `Skywork/SkyReels-V2-I2V-1.3B-540P` (540P, lightweight)
- `Skywork/SkyReels-V2-I2V-14B-540P` (540P, high quality)
- `Skywork/SkyReels-V2-I2V-14B-720P` (720P, high quality)

**Key Parameters:**
- **Input Image**: Upload your source image
- **Prompt**: Describe the desired animation
- **Guidance Scale**: 5.0 recommended for I2V
- **Shift**: 3.0 recommended for I2V

### 🎯 Diffusion Forcing (DF)
Advanced video generation with long video support, video extension, and frame control.

**Available Models:**
- `Skywork/SkyReels-V2-DF-1.3B-540P` (540P, lightweight)
- `Skywork/SkyReels-V2-DF-14B-540P` (540P, high quality)
- `Skywork/SkyReels-V2-DF-14B-720P` (720P, high quality)

**Advanced Features:**
- **Long Video Generation**: Create videos up to 30+ seconds
- **Video Extension**: Extend existing videos
- **Frame Control**: Control start and end frames
- **Sync/Async Modes**: Choose generation strategy

**Key Parameters:**
- **AR Step**: 0 for synchronous (10s videos), 5+ for asynchronous (30s+ videos)
- **Base Frames**: 97 for 540P, 121 for 720P
- **Overlap History**: 17 recommended for smooth transitions
- **Add Noise Condition**: 20 recommended for consistency

## 💡 Tips for Best Results

### Text-to-Video Tips
- Use detailed, descriptive prompts
- Mention camera movements, lighting, and atmosphere
- Example: "A majestic eagle soaring through mountain peaks at golden hour with cinematic camera movement"

### Image-to-Video Tips
- Use high-quality input images (preferably 1024x1024 or higher)
- Describe the desired motion clearly
- Portrait images work well for character animation
- Example: "The person in the image turns their head slowly and smiles"

### Diffusion Forcing Tips
- **For 10-second videos**: Use synchronous mode (ar_step=0)
- **For 30+ second videos**: Use asynchronous mode (ar_step=5)
- Set overlap_history=17 for smooth long videos
- Use addnoise_condition=20 for better consistency
- Base frames should match your model: 97 for 540P, 121 for 720P

## 🔧 Configuration Options

### Command Line Arguments

```bash
python gradio_skyreel.py [OPTIONS]

Options:
  --share              Enable Gradio sharing (creates public URL)
  --server_name HOST   Server hostname (default: 0.0.0.0)
  --server_port PORT   Server port (default: 7860)
  --debug              Enable debug mode
```

### Setup Script Options

```bash
./setup_and_run.sh [OPTIONS]

Options:
  --skip-deps     Skip Python dependency installation
  --skip-system   Skip system dependency installation
  --help, -h      Show help message
```

## 📊 Model Specifications

| Model Type | Resolution | Recommended Frames | Guidance | Shift | Use Case |
|------------|------------|-------------------|----------|-------|----------|
| T2V-14B-540P | 960x544 | 97 | 6.0 | 8.0 | Text-to-video |
| T2V-14B-720P | 1280x720 | 121 | 6.0 | 8.0 | High-res text-to-video |
| I2V-14B-540P | 960x544 | 97 | 5.0 | 3.0 | Image animation |
| I2V-14B-720P | 1280x720 | 121 | 5.0 | 3.0 | High-res image animation |
| DF-14B-540P | 960x544 | 97+ | 6.0 | 8.0 | Long videos, extensions |
| DF-14B-720P | 1280x720 | 121+ | 6.0 | 8.0 | High-res long videos |

## 🚨 System Requirements

### Minimum Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **RAM**: 16GB+ system RAM
- **Storage**: 50GB+ free space for models
- **Python**: 3.8+

### Recommended Requirements
- **GPU**: NVIDIA RTX 4090 or A100 with 24GB+ VRAM
- **RAM**: 32GB+ system RAM
- **Storage**: 100GB+ SSD storage
- **Python**: 3.10+

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Enable "CPU Offload" option
   - Reduce number of frames
   - Use smaller models (1.3B instead of 14B)

2. **Slow Generation**
   - Ensure CUDA is properly installed
   - Use CPU offload if GPU memory is limited
   - Consider using lower resolution models

3. **Import Errors**
   - Run `python test_gradio.py` to diagnose issues
   - Ensure all dependencies are installed
   - Check Python version compatibility

4. **Model Download Issues**
   - Ensure stable internet connection
   - Check Hugging Face Hub access
   - Verify sufficient storage space

### Getting Help

If you encounter issues:
1. Run the test script: `python test_gradio.py`
2. Check the console output for error messages
3. Ensure your system meets the requirements
4. Try the setup script with `--skip-deps` if dependencies are already installed

## 📝 Example Prompts

### Text-to-Video Examples
- "A graceful white swan swimming in a serene lake at dawn with mist rising from the water"
- "Ocean waves crashing against rocky cliffs in slow motion during golden hour"
- "A bustling city street with people walking and cars passing by, shot from above"
- "Cherry blossoms falling gently in a peaceful Japanese garden with soft lighting"

### Image-to-Video Examples
- "The person in the image turns their head and smiles warmly"
- "The landscape comes alive with gentle wind moving through the trees"
- "The character in the image waves hello with a friendly expression"
- "The scene transforms with magical sparkles and soft movement"

## 🎬 Output

Generated videos are saved in the current directory with timestamps:
- `output_t2v_[timestamp].mp4` for Text-to-Video
- `output_i2v_[timestamp].mp4` for Image-to-Video  
- `output_df_[timestamp].mp4` for Diffusion Forcing

Enjoy creating amazing videos with SkyReels V2! 🎉
