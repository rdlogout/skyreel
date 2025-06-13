#!/usr/bin/env python3
"""
Example usage of SkyReels V2 Gradio interface functions.
This script demonstrates how to use the video generation functions programmatically.
"""

import os
import sys
from PIL import Image

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gradio_skyreel import (
    generate_text_to_video,
    generate_image_to_video, 
    generate_diffusion_forcing_video,
    MODEL_CONFIGS
)

def example_text_to_video():
    """Example of Text-to-Video generation."""
    print("🎬 Text-to-Video Example")
    print("-" * 30)
    
    prompt = "A majestic eagle soaring through mountain peaks at golden hour"
    model_id = "Skywork/SkyReels-V2-T2V-14B-540P"
    
    print(f"Prompt: {prompt}")
    print(f"Model: {model_id}")
    print("Generating video...")
    
    try:
        video_path, info = generate_text_to_video(
            prompt=prompt,
            model_id=model_id,
            num_frames=97,
            fps=24,
            guidance_scale=6.0,
            shift=8.0,
            inference_steps=30,
            seed=42,
            use_prompt_enhancer=True,
            offload=True
        )
        
        if video_path:
            print(f"✅ Video generated successfully: {video_path}")
            print(f"Info: {info}")
        else:
            print(f"❌ Generation failed: {info}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def example_image_to_video():
    """Example of Image-to-Video generation."""
    print("\n🖼️ Image-to-Video Example")
    print("-" * 30)
    
    # Create a simple test image (you can replace this with your own image)
    image = Image.new('RGB', (512, 512), color='lightblue')
    prompt = "The image comes to life with gentle movement and natural motion"
    model_id = "Skywork/SkyReels-V2-I2V-14B-540P"
    
    print(f"Prompt: {prompt}")
    print(f"Model: {model_id}")
    print("Generating video from image...")
    
    try:
        video_path, info = generate_image_to_video(
            prompt=prompt,
            image=image,
            model_id=model_id,
            num_frames=97,
            fps=24,
            guidance_scale=5.0,
            shift=3.0,
            inference_steps=30,
            seed=42,
            use_prompt_enhancer=True,
            offload=True
        )
        
        if video_path:
            print(f"✅ Video generated successfully: {video_path}")
            print(f"Info: {info}")
        else:
            print(f"❌ Generation failed: {info}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def example_diffusion_forcing():
    """Example of Diffusion Forcing generation."""
    print("\n🎯 Diffusion Forcing Example")
    print("-" * 30)
    
    prompt = "A graceful white swan swimming in a serene lake at dawn"
    model_id = "Skywork/SkyReels-V2-DF-14B-540P"
    
    print(f"Prompt: {prompt}")
    print(f"Model: {model_id}")
    print("Generating video with Diffusion Forcing...")
    
    try:
        video_path, info = generate_diffusion_forcing_video(
            prompt=prompt,
            model_id=model_id,
            num_frames=257,  # Longer video
            fps=24,
            guidance_scale=6.0,
            shift=8.0,
            inference_steps=30,
            seed=42,
            use_prompt_enhancer=True,
            offload=True,
            # DF specific parameters
            ar_step=0,  # Synchronous mode
            base_num_frames=97,
            overlap_history=17,
            addnoise_condition=20,
            causal_block_size=5,
            # Optional inputs
            input_image=None,
            end_image=None,
            video_path=None
        )
        
        if video_path:
            print(f"✅ Video generated successfully: {video_path}")
            print(f"Info: {info}")
        else:
            print(f"❌ Generation failed: {info}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def list_available_models():
    """List all available models."""
    print("\n📋 Available Models")
    print("-" * 30)
    
    for model_type, models in MODEL_CONFIGS.items():
        print(f"\n{model_type} Models:")
        for model_id, config in models.items():
            print(f"  • {model_id}")
            print(f"    Resolution: {config['resolution']}")
            print(f"    Frames: {config['frames']}")
            print(f"    Guidance: {config['guidance']}")
            print(f"    Shift: {config['shift']}")

def main():
    """Run example usage."""
    print("🎬 SkyReels V2 Gradio Interface - Example Usage")
    print("=" * 50)
    
    # List available models
    list_available_models()
    
    # Check if user wants to run examples
    print("\n" + "=" * 50)
    print("⚠️  WARNING: The following examples will download models and generate videos.")
    print("This requires significant GPU memory and time.")
    print("Make sure you have:")
    print("  • NVIDIA GPU with 8GB+ VRAM")
    print("  • Stable internet connection for model downloads")
    print("  • Sufficient disk space (models are several GB each)")
    
    response = input("\nDo you want to run the generation examples? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        print("\n🚀 Running examples...")
        
        # Run examples (uncomment the ones you want to test)
        # example_text_to_video()
        # example_image_to_video()
        # example_diffusion_forcing()
        
        print("\n💡 Examples are commented out by default.")
        print("Uncomment the example functions in main() to run them.")
        print("Start with Text-to-Video as it's the most straightforward.")
    else:
        print("\n✅ Examples skipped. You can run them later by modifying this script.")
    
    print("\n🎉 To use the web interface, run:")
    print("  python gradio_skyreel.py --share")
    print("\nOr use the setup script:")
    print("  ./setup_and_run.sh")

if __name__ == "__main__":
    main()
