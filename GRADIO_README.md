# SkyReels V2 Gradio Web Interface

This Gradio web application provides a user-friendly interface to access all features of SkyReels V2 through your web browser with public link sharing enabled.

## 🚀 Quick Start

### Prerequisites

1. **GPU Requirements**: CUDA-compatible GPU with at least 16GB VRAM (24GB+ recommended for larger models)
2. **Python**: Python 3.10 or higher
3. **CUDA**: CUDA toolkit installed and properly configured

### Installation

1. **Clone the repository**:

```bash
git clone https://github.com/SkyworkAI/SkyReels-V2
cd SkyReels-V2
```

2. **Install dependencies**:

```bash
# Install base SkyReels-V2 dependencies
pip install -r requirements.txt

# Install additional Gradio dependencies
pip install -r gradio_requirements.txt
```

3. **Launch the Gradio app**:

```bash
python app.py
```

The app will start on `http://localhost:7860` and automatically generate a public shareable link that you can access from anywhere.

## 🎬 Features

### 📝 Text-to-Video Generation

- Generate videos directly from text descriptions
- Multiple model options (1.3B, 14B parameters)
- Resolution options: 540P, 720P
- Customizable parameters: frames, guidance scale, inference steps
- Optional prompt enhancement for better results

### 🖼️ Image-to-Video Generation

- Animate static images with text prompts
- Upload any image and describe desired motion
- Automatic image preprocessing and aspect ratio handling
- Same quality controls as text-to-video

### 🔄 Diffusion Forcing (Long Video Generation)

- Generate very long videos (up to infinite length theoretically)
- AutoRegressive approach for seamless continuity
- Optional start and end frame control
- Advanced parameters for controlling generation flow

### ➕ Video Extension

- Extend existing videos with new content
- Upload a video and describe how to continue it
- Maintains consistency with original content
- Perfect for creating longer sequences

### 💬 Video Captioning

- Generate detailed captions for videos using SkyCaptioner-V1
- Structural analysis including shot types, camera movements
- Professional film terminology and descriptions

## ⚙️ Configuration Options

### Model Selection

- **1.3B Models**: Faster, lower memory usage, good quality
- **14B Models**: Best quality, higher memory requirements
- **540P vs 720P**: Choose based on your GPU memory and quality needs

### Performance Optimization

- **TeaCache**: Speeds up generation (2-3x faster) with minimal quality loss
- **Offloading**: Automatically manages GPU memory for large models
- **Batch Processing**: Efficient handling of multiple generations

### Advanced Parameters

- **Guidance Scale**: Controls adherence to prompt (1.0-20.0)
- **Shift**: Fine-tunes the generation process (1.0-20.0)
- **Inference Steps**: Quality vs speed trade-off (10-100)
- **Seeds**: Reproducible results with specific seed values

## 🔧 Hardware Requirements

### Minimum Requirements

- GPU: 16GB VRAM (GTX 4090, RTX A4000, etc.)
- RAM: 32GB system memory
- Storage: 50GB free space for models

### Recommended Requirements

- GPU: 24GB+ VRAM (RTX 4090, A5000, H100)
- RAM: 64GB+ system memory
- Storage: 100GB+ SSD storage for optimal performance

### Model Memory Usage

- **1.3B Models**: ~8-12GB VRAM
- **14B Models**: ~20-24GB VRAM
- **540P Resolution**: Lower memory usage
- **720P Resolution**: ~30% more VRAM required

## 🚀 Usage Tips

### For Best Results

1. **Use descriptive prompts**: Include details about lighting, camera angles, actions
2. **Enable prompt enhancement**: For automated prompt optimization
3. **Start with shorter videos**: Test with 97 frames before generating longer content
4. **Use appropriate resolution**: Match your hardware capabilities

### Optimizing Performance

1. **Enable TeaCache**: Significant speed improvement with minimal quality loss
2. **Use smaller models first**: Test with 1.3B before moving to 14B
3. **Close other applications**: Free up GPU memory for better performance
4. **Monitor GPU temperature**: Ensure adequate cooling during long generations

### Troubleshooting

- **Out of Memory**: Reduce resolution, frames, or use smaller model
- **Slow Generation**: Enable TeaCache, reduce inference steps
- **Poor Quality**: Increase inference steps, try prompt enhancement
- **Model Loading Issues**: Check internet connection for model downloads

## 🌐 Public Link Sharing

The Gradio app automatically generates a public link (like `https://xxxxx.gradio.live`) that you can:

- Share with others to access your interface remotely
- Use from different devices and locations
- Keep active for up to 72 hours per session

**Security Note**: The public link allows anyone with the URL to access your interface. Only share with trusted users.

## 📊 Model Information

### Available Models

| Model Type    | Size | Resolution | Performance  | Memory  |
| ------------- | ---- | ---------- | ------------ | ------- |
| T2V-1.3B-540P | 1.3B | 544×960    | Fast         | 8-12GB  |
| T2V-14B-540P  | 14B  | 544×960    | Best         | 20-24GB |
| T2V-14B-720P  | 14B  | 720×1280   | Best+HQ      | 24-28GB |
| I2V-1.3B-540P | 1.3B | 544×960    | Fast         | 8-12GB  |
| I2V-14B-540P  | 14B  | 544×960    | Best         | 20-24GB |
| I2V-14B-720P  | 14B  | 720×1280   | Best+HQ      | 24-28GB |
| DF-1.3B-540P  | 1.3B | 544×960    | Fast+Long    | 8-12GB  |
| DF-14B-540P   | 14B  | 544×960    | Best+Long    | 20-24GB |
| DF-14B-720P   | 14B  | 720×1280   | Best+HQ+Long | 24-28GB |

## 🔗 Links

- [SkyReels V2 GitHub](https://github.com/SkyworkAI/SkyReels-V2)
- [Technical Paper](https://arxiv.org/pdf/2504.13074)
- [HuggingFace Models](https://huggingface.co/collections/Skywork/skyreels-v2-6801b1b93df627d441d0d0d9)
- [Official Playground](https://www.skyreels.ai/home)

## 📄 License

This project follows the same license as SkyReels V2. Please refer to the main repository for licensing information.

## 🆘 Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the main SkyReels V2 documentation
3. Open an issue on the GitHub repository
4. Join the Discord community for real-time help

---

**Happy Video Generating! 🎬✨**
