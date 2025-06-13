#!/usr/bin/env python3
"""
SkyReels-V2 Gradio Web Interface
A comprehensive web interface for video generation using SkyReels-V2 models
"""

import argparse
import gc
import os
import random
import time
import tempfile
from typing import Optional, Tuple, List

import gradio as gr
import imageio
import torch
from diffusers.utils import load_image
from PIL import Image

from skyreels_v2_infer import DiffusionForcingPipeline
from skyreels_v2_infer.modules import download_model
from skyreels_v2_infer.pipelines import Image2VideoPipeline, Text2VideoPipeline, PromptEnhancer, resizecrop

# Model configurations
MODEL_CONFIGS = {
    "Text-to-Video": {
        "SkyReels-V2-T2V-14B-540P": "Skywork/SkyReels-V2-T2V-14B-540P",
        "SkyReels-V2-T2V-14B-720P": "Skywork/SkyReels-V2-T2V-14B-720P",
    },
    "Image-to-Video": {
        "SkyReels-V2-I2V-1.3B-540P": "Skywork/SkyReels-V2-I2V-1.3B-540P",
        "SkyReels-V2-I2V-14B-540P": "Skywork/SkyReels-V2-I2V-14B-540P",
        "SkyReels-V2-I2V-14B-720P": "Skywork/SkyReels-V2-I2V-14B-720P",
    },
    "Diffusion Forcing": {
        "SkyReels-V2-DF-1.3B-540P": "Skywork/SkyReels-V2-DF-1.3B-540P",
        "SkyReels-V2-DF-14B-540P": "Skywork/SkyReels-V2-DF-14B-540P",
        "SkyReels-V2-DF-14B-720P": "Skywork/SkyReels-V2-DF-14B-720P",
    }
}

# Resolution configurations
RESOLUTION_CONFIGS = {
    "540P": {"height": 544, "width": 960, "default_frames": 97},
    "720P": {"height": 720, "width": 1280, "default_frames": 121}
}

# Duration presets (in frames)
DURATION_PRESETS = {
    "4 seconds (~97 frames)": 97,
    "5 seconds (~121 frames)": 121,
    "10 seconds (~257 frames)": 257,
    "15 seconds (~377 frames)": 377,
    "30 seconds (~737 frames)": 737,
    "60 seconds (~1457 frames)": 1457,
    "Custom": -1
}

class SkyReelsInterface:
    def __init__(self):
        self.current_pipeline = None
        self.current_model_id = None
        self.current_mode = None
        
    def get_model_choices(self, mode: str) -> List[str]:
        """Get available models for the selected mode"""
        return list(MODEL_CONFIGS.get(mode, {}).keys())
    
    def get_resolution_from_model(self, model_name: str) -> str:
        """Extract resolution from model name"""
        if "720P" in model_name:
            return "720P"
        return "540P"
    
    def load_pipeline(self, mode: str, model_name: str, offload: bool = True):
        """Load the appropriate pipeline based on mode and model"""
        if model_name not in MODEL_CONFIGS.get(mode, {}):
            raise ValueError(f"Invalid model {model_name} for mode {mode}")
            
        model_id = MODEL_CONFIGS[mode][model_name]
        
        # Check if we need to reload the pipeline
        if self.current_pipeline is None or self.current_model_id != model_id or self.current_mode != mode:
            # Clear previous pipeline
            if self.current_pipeline is not None:
                del self.current_pipeline
                gc.collect()
                torch.cuda.empty_cache()
            
            # Download model if needed
            model_path = download_model(model_id)
            
            # Load appropriate pipeline
            if mode == "Text-to-Video":
                self.current_pipeline = Text2VideoPipeline(
                    model_path=model_path, 
                    dit_path=model_path, 
                    offload=offload
                )
            elif mode == "Image-to-Video":
                self.current_pipeline = Image2VideoPipeline(
                    model_path=model_path, 
                    dit_path=model_path, 
                    offload=offload
                )
            elif mode == "Diffusion Forcing":
                self.current_pipeline = DiffusionForcingPipeline(
                    model_path=model_path,
                    dit_path=model_path,
                    offload=offload
                )
            
            self.current_model_id = model_id
            self.current_mode = mode
            
        return self.current_pipeline

# Global interface instance
interface = SkyReelsInterface()

def update_model_choices(mode: str):
    """Update model dropdown based on selected mode"""
    choices = interface.get_model_choices(mode)
    if choices:
        return gr.Dropdown(choices=choices, value=choices[0], interactive=True)
    return gr.Dropdown(choices=[], value=None, interactive=False)

def update_resolution_and_frames(model_name: str):
    """Update resolution and default frames based on selected model"""
    if not model_name:
        return "540P", 97
    
    resolution = interface.get_resolution_from_model(model_name)
    default_frames = RESOLUTION_CONFIGS[resolution]["default_frames"]
    return resolution, default_frames

def update_custom_frames_visibility(duration_preset: str):
    """Show/hide custom frames input based on duration preset"""
    return gr.Number(visible=(duration_preset == "Custom"))

def enhance_prompt(prompt: str, use_enhancer: bool) -> str:
    """Enhance prompt if requested"""
    if not use_enhancer or not prompt.strip():
        return prompt

    try:
        enhancer = PromptEnhancer()
        enhanced = enhancer(prompt)
        del enhancer
        gc.collect()
        torch.cuda.empty_cache()
        return enhanced
    except Exception as e:
        print(f"Prompt enhancement failed: {e}")
        return prompt

def generate_video(
    mode: str,
    model_name: str,
    prompt: str,
    image: Optional[Image.Image],
    resolution: str,
    duration_preset: str,
    custom_frames: int,
    guidance_scale: float,
    shift: float,
    inference_steps: int,
    fps: int,
    seed: int,
    use_prompt_enhancer: bool,
    use_teacache: bool,
    teacache_thresh: float,
    # Diffusion Forcing specific parameters
    ar_step: int,
    base_num_frames: int,
    overlap_history: int,
    addnoise_condition: int,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """Generate video based on the provided parameters"""

    try:
        progress(0.1, desc="Initializing...")

        # Validate inputs
        if not prompt.strip():
            return None, "❌ Please provide a prompt"

        if mode == "Image-to-Video" and image is None:
            return None, "❌ Please provide an image for Image-to-Video mode"

        if not model_name:
            return None, "❌ Please select a model"

        # Determine number of frames
        if duration_preset == "Custom":
            num_frames = custom_frames
        else:
            num_frames = DURATION_PRESETS[duration_preset]

        if num_frames <= 0:
            return None, "❌ Invalid number of frames"

        # Get resolution config
        res_config = RESOLUTION_CONFIGS[resolution]
        height, width = res_config["height"], res_config["width"]

        # Set random seed
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        progress(0.2, desc="Loading model...")

        # Load pipeline
        pipeline = interface.load_pipeline(mode, model_name, offload=True)

        progress(0.3, desc="Enhancing prompt...")

        # Enhance prompt if requested
        enhanced_prompt = enhance_prompt(prompt, use_prompt_enhancer)

        # Prepare negative prompt
        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

        progress(0.4, desc="Preparing generation...")

        # Configure teacache if enabled
        if use_teacache and hasattr(pipeline, 'transformer'):
            pipeline.transformer.initialize_teacache(
                enable_teacache=True,
                num_steps=inference_steps,
                teacache_thresh=teacache_thresh,
                use_ret_steps=True,
                ckpt_dir=interface.current_model_id
            )

        # Prepare generation parameters
        generator = torch.Generator(device="cuda").manual_seed(seed)

        progress(0.5, desc="Generating video...")

        # Generate video based on mode
        if mode == "Text-to-Video":
            with torch.cuda.amp.autocast(dtype=pipeline.transformer.dtype), torch.no_grad():
                video_frames = pipeline(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    num_frames=num_frames,
                    num_inference_steps=inference_steps,
                    guidance_scale=guidance_scale,
                    shift=shift,
                    generator=generator,
                    height=height,
                    width=width,
                )[0]

        elif mode == "Image-to-Video":
            # Process input image
            if image.height > image.width:
                height, width = width, height
            processed_image = resizecrop(image, height, width)

            with torch.cuda.amp.autocast(dtype=pipeline.transformer.dtype), torch.no_grad():
                video_frames = pipeline(
                    image=processed_image.convert("RGB"),
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    num_frames=num_frames,
                    num_inference_steps=inference_steps,
                    guidance_scale=guidance_scale,
                    shift=shift,
                    generator=generator,
                    height=height,
                    width=width,
                )[0]

        elif mode == "Diffusion Forcing":
            # Process input image if provided
            processed_image = None
            if image is not None:
                if image.height > image.width:
                    height, width = width, height
                processed_image = resizecrop(image, height, width).convert("RGB")

            with torch.cuda.amp.autocast(dtype=pipeline.transformer.dtype), torch.no_grad():
                video_frames = pipeline(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    image=processed_image,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=inference_steps,
                    shift=shift,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    overlap_history=overlap_history if num_frames > base_num_frames else None,
                    addnoise_condition=addnoise_condition,
                    base_num_frames=base_num_frames,
                    ar_step=ar_step,
                    causal_block_size=1,
                    fps=fps,
                )[0]

        progress(0.9, desc="Saving video...")

        # Save video
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{mode.lower().replace('-', '_')}_{timestamp}_{seed}.mp4"
        output_path = os.path.join("/app/result/video_out", filename)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        imageio.mimwrite(output_path, video_frames, fps=fps, quality=8, output_params=["-loglevel", "error"])

        progress(1.0, desc="Complete!")

        # Prepare info message
        info_msg = f"""
✅ **Video Generated Successfully!**

**Settings Used:**
- Mode: {mode}
- Model: {model_name}
- Resolution: {resolution} ({height}x{width})
- Frames: {num_frames} ({num_frames/fps:.1f}s at {fps}fps)
- Seed: {seed}
- Enhanced Prompt: {"Yes" if use_prompt_enhancer else "No"}

**Prompt:** {enhanced_prompt[:200]}{"..." if len(enhanced_prompt) > 200 else ""}
        """

        return output_path, info_msg.strip()

    except Exception as e:
        error_msg = f"❌ **Generation Failed:** {str(e)}"
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return None, error_msg

def create_interface():
    """Create the Gradio interface"""

    with gr.Blocks(
        title="SkyReels-V2 Video Generator",
        theme=gr.themes.Soft(),
        css="""
        .main-header { text-align: center; margin-bottom: 2rem; }
        .section-header { margin-top: 1.5rem; margin-bottom: 1rem; font-weight: bold; }
        .parameter-group { border: 1px solid #e0e0e0; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; }
        """
    ) as demo:

        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🎬 SkyReels-V2 Video Generator</h1>
            <p>Generate high-quality videos using state-of-the-art AI models</p>
            <p><em>Supports Text-to-Video, Image-to-Video, and Diffusion Forcing for long videos</em></p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                # Mode and Model Selection
                gr.HTML('<div class="section-header">🎯 Model Configuration</div>')

                mode = gr.Dropdown(
                    choices=list(MODEL_CONFIGS.keys()),
                    value="Text-to-Video",
                    label="Generation Mode",
                    info="Choose the type of video generation"
                )

                model_name = gr.Dropdown(
                    choices=interface.get_model_choices("Text-to-Video"),
                    value=interface.get_model_choices("Text-to-Video")[0] if interface.get_model_choices("Text-to-Video") else None,
                    label="Model",
                    info="Select the model to use for generation"
                )

                # Input Section
                gr.HTML('<div class="section-header">📝 Input</div>')

                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the video you want to generate...",
                    lines=3,
                    value="A serene lake surrounded by towering mountains, with a few swans gracefully gliding across the water and sunlight dancing on the surface."
                )

                image = gr.Image(
                    label="Input Image (for Image-to-Video and Diffusion Forcing)",
                    type="pil",
                    visible=False
                )

                use_prompt_enhancer = gr.Checkbox(
                    label="Use Prompt Enhancer",
                    value=False,
                    info="Automatically enhance your prompt for better results"
                )

                # Video Configuration
                gr.HTML('<div class="section-header">🎥 Video Configuration</div>')

                resolution = gr.Dropdown(
                    choices=list(RESOLUTION_CONFIGS.keys()),
                    value="540P",
                    label="Resolution",
                    info="Video resolution (determined by model)"
                )

                duration_preset = gr.Dropdown(
                    choices=list(DURATION_PRESETS.keys()),
                    value="4 seconds (~97 frames)",
                    label="Duration",
                    info="Select video duration"
                )

                custom_frames = gr.Number(
                    label="Custom Frame Count",
                    value=97,
                    minimum=1,
                    maximum=2000,
                    step=1,
                    visible=False,
                    info="Number of frames for custom duration"
                )

                fps = gr.Slider(
                    minimum=12,
                    maximum=30,
                    value=24,
                    step=1,
                    label="FPS (Frames Per Second)",
                    info="Video frame rate"
                )

            with gr.Column(scale=1):
                # Advanced Parameters
                gr.HTML('<div class="section-header">⚙️ Generation Parameters</div>')

                with gr.Group():
                    guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=15.0,
                        value=6.0,
                        step=0.1,
                        label="Guidance Scale",
                        info="Controls adherence to prompt (6.0 for T2V, 5.0 for I2V)"
                    )

                    shift = gr.Slider(
                        minimum=1.0,
                        maximum=15.0,
                        value=8.0,
                        step=0.1,
                        label="Shift",
                        info="Flow matching parameter (8.0 for T2V, 5.0 for I2V)"
                    )

                    inference_steps = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=30,
                        step=1,
                        label="Inference Steps",
                        info="Number of denoising steps"
                    )

                    seed = gr.Number(
                        label="Seed",
                        value=-1,
                        info="Random seed (-1 for random)"
                    )

                # Performance Options
                gr.HTML('<div class="section-header">🚀 Performance Options</div>')

                with gr.Group():
                    use_teacache = gr.Checkbox(
                        label="Enable TeaCache",
                        value=False,
                        info="Faster inference with slight quality trade-off"
                    )

                    teacache_thresh = gr.Slider(
                        minimum=0.1,
                        maximum=0.5,
                        value=0.2,
                        step=0.05,
                        label="TeaCache Threshold",
                        info="Higher values = faster but lower quality",
                        visible=False
                    )

                # Diffusion Forcing Parameters (Advanced)
                gr.HTML('<div class="section-header">🔄 Diffusion Forcing (Advanced)</div>')

                with gr.Group():
                    ar_step = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=0,
                        step=1,
                        label="AR Step",
                        info="Autoregressive steps (0 for synchronous, >0 for asynchronous)"
                    )

                    base_num_frames = gr.Number(
                        label="Base Num Frames",
                        value=97,
                        minimum=50,
                        maximum=200,
                        step=1,
                        info="Base frame count for long video generation"
                    )

                    overlap_history = gr.Number(
                        label="Overlap History",
                        value=17,
                        minimum=5,
                        maximum=50,
                        step=1,
                        info="Frame overlap for smooth transitions"
                    )

                    addnoise_condition = gr.Slider(
                        minimum=0,
                        maximum=50,
                        value=20,
                        step=1,
                        label="Add Noise Condition",
                        info="Noise for smooth long video generation"
                    )

        # Output Section
        with gr.Row():
            with gr.Column():
                gr.HTML('<div class="section-header">🎬 Generated Video</div>')

                generate_btn = gr.Button(
                    "🚀 Generate Video",
                    variant="primary",
                    size="lg"
                )

                video_output = gr.Video(
                    label="Generated Video",
                    height=400
                )

                info_output = gr.Markdown(
                    label="Generation Info",
                    value="Click 'Generate Video' to start!"
                )

        # Examples Section
        gr.HTML('<div class="section-header">💡 Examples</div>')

        gr.Examples(
            examples=[
                [
                    "Text-to-Video",
                    "SkyReels-V2-T2V-14B-540P",
                    "A graceful white swan with a curved neck and delicate feathers swimming in a serene lake at dawn, its reflection perfectly mirrored in the still water as mist rises from the surface.",
                    None,
                    "540P",
                    "4 seconds (~97 frames)",
                    97,
                    6.0,
                    8.0,
                    30,
                    24,
                    42,
                    False,
                    False,
                    0.2,
                    0,
                    97,
                    17,
                    20
                ],
                [
                    "Diffusion Forcing",
                    "SkyReels-V2-DF-14B-540P",
                    "A woman in a leather jacket and sunglasses riding a vintage motorcycle through a desert highway at sunset, her hair blowing wildly in the wind.",
                    None,
                    "540P",
                    "10 seconds (~257 frames)",
                    257,
                    6.0,
                    8.0,
                    30,
                    24,
                    123,
                    False,
                    True,
                    0.3,
                    0,
                    97,
                    17,
                    20
                ]
            ],
            inputs=[
                mode, model_name, prompt, image, resolution, duration_preset, custom_frames,
                guidance_scale, shift, inference_steps, fps, seed, use_prompt_enhancer,
                use_teacache, teacache_thresh, ar_step, base_num_frames, overlap_history, addnoise_condition
            ]
        )

        # Event Handlers
        mode.change(
            fn=update_model_choices,
            inputs=[mode],
            outputs=[model_name]
        )

        model_name.change(
            fn=update_resolution_and_frames,
            inputs=[model_name],
            outputs=[resolution, base_num_frames]
        )

        duration_preset.change(
            fn=update_custom_frames_visibility,
            inputs=[duration_preset],
            outputs=[custom_frames]
        )

        use_teacache.change(
            fn=lambda x: gr.Slider(visible=x),
            inputs=[use_teacache],
            outputs=[teacache_thresh]
        )

        mode.change(
            fn=lambda x: gr.Image(visible=(x in ["Image-to-Video", "Diffusion Forcing"])),
            inputs=[mode],
            outputs=[image]
        )

        # Main generation event
        generate_btn.click(
            fn=generate_video,
            inputs=[
                mode, model_name, prompt, image, resolution, duration_preset, custom_frames,
                guidance_scale, shift, inference_steps, fps, seed, use_prompt_enhancer,
                use_teacache, teacache_thresh, ar_step, base_num_frames, overlap_history, addnoise_condition
            ],
            outputs=[video_output, info_output]
        )

        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 2rem; padding: 1rem; border-top: 1px solid #e0e0e0;">
            <p><strong>SkyReels-V2</strong> - Infinite-Length Film Generative Model</p>
            <p>🔗 <a href="https://github.com/SkyworkAI/SkyReels-V2" target="_blank">GitHub</a> |
               📄 <a href="https://arxiv.org/pdf/2504.13074" target="_blank">Paper</a> |
               🤗 <a href="https://huggingface.co/collections/Skywork/skyreels-v2-6801b1b93df627d441d0d0d9" target="_blank">Models</a></p>
        </div>
        """)

    return demo

def main():
    """Main function to launch the Gradio interface"""
    parser = argparse.ArgumentParser(description="SkyReels-V2 Gradio Interface")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Create and launch interface
    demo = create_interface()

    print("🚀 Starting SkyReels-V2 Gradio Interface...")
    print(f"📡 Server will be available at: http://{args.host}:{args.port}")

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=True,
        favicon_path=None,
        ssl_verify=False
    )

if __name__ == "__main__":
    main()
