import argparse
import gc
import os
import random
import time
import tempfile
from typing import Optional, Tuple, List
import json

import gradio as gr
import imageio
import torch
import numpy as np
from diffusers.utils import load_image
from PIL import Image

from skyreels_v2_infer.modules import download_model
from skyreels_v2_infer.pipelines import Image2VideoPipeline, Text2VideoPipeline, PromptEnhancer, resizecrop, DiffusionForcingPipeline

# Model configurations
MODEL_CONFIGS = {
    "T2V": {
        "Skywork/SkyReels-V2-T2V-14B-540P": {"resolution": "540P", "frames": 97, "guidance": 6.0, "shift": 8.0},
        "Skywork/SkyReels-V2-T2V-14B-720P": {"resolution": "720P", "frames": 121, "guidance": 6.0, "shift": 8.0},
    },
    "I2V": {
        "Skywork/SkyReels-V2-I2V-1.3B-540P": {"resolution": "540P", "frames": 97, "guidance": 5.0, "shift": 3.0},
        "Skywork/SkyReels-V2-I2V-14B-540P": {"resolution": "540P", "frames": 97, "guidance": 5.0, "shift": 3.0},
        "Skywork/SkyReels-V2-I2V-14B-720P": {"resolution": "720P", "frames": 121, "guidance": 5.0, "shift": 3.0},
    },
    "DF": {
        "Skywork/SkyReels-V2-DF-1.3B-540P": {"resolution": "540P", "frames": 97, "base_frames": 97, "guidance": 6.0, "shift": 8.0},
        "Skywork/SkyReels-V2-DF-14B-540P": {"resolution": "540P", "frames": 97, "base_frames": 97, "guidance": 6.0, "shift": 8.0},
        "Skywork/SkyReels-V2-DF-14B-720P": {"resolution": "720P", "frames": 121, "base_frames": 121, "guidance": 6.0, "shift": 8.0},
    }
}

RESOLUTION_CONFIGS = {
    "540P": {"height": 544, "width": 960},
    "720P": {"height": 720, "width": 1280}
}

# Global variables for loaded models
loaded_models = {}
prompt_enhancer = None

def get_resolution_info(resolution: str) -> Tuple[int, int]:
    """Get height and width for resolution."""
    config = RESOLUTION_CONFIGS[resolution]
    return config["height"], config["width"]

def load_model(model_type: str, model_id: str, offload: bool = True) -> object:
    """Load and cache models."""
    global loaded_models
    
    cache_key = f"{model_type}_{model_id}_{offload}"
    if cache_key in loaded_models:
        return loaded_models[cache_key]
    
    print(f"Loading model: {model_id}")
    model_path = download_model(model_id)
    
    if model_type == "T2V":
        pipeline = Text2VideoPipeline(
            model_path=model_path, 
            dit_path=model_path, 
            use_usp=False, 
            offload=offload
        )
    elif model_type == "I2V":
        pipeline = Image2VideoPipeline(
            model_path=model_path, 
            dit_path=model_path, 
            use_usp=False, 
            offload=offload
        )
    elif model_type == "DF":
        pipeline = DiffusionForcingPipeline(
            model_path=model_path, 
            dit_path=model_path, 
            use_usp=False, 
            offload=offload
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    loaded_models[cache_key] = pipeline
    return pipeline

def get_prompt_enhancer():
    """Load and cache prompt enhancer."""
    global prompt_enhancer
    if prompt_enhancer is None:
        prompt_enhancer = PromptEnhancer()
    return prompt_enhancer

def enhance_prompt(prompt: str, use_enhancer: bool) -> str:
    """Enhance prompt if requested."""
    if not use_enhancer or not prompt.strip():
        return prompt
    
    try:
        enhancer = get_prompt_enhancer()
        enhanced = enhancer(prompt)
        return enhanced
    except Exception as e:
        print(f"Prompt enhancement failed: {e}")
        return prompt

def generate_text_to_video(
    prompt: str,
    model_id: str,
    num_frames: int,
    fps: int,
    guidance_scale: float,
    shift: float,
    inference_steps: int,
    seed: Optional[int],
    use_prompt_enhancer: bool,
    offload: bool,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """Generate video from text prompt."""
    try:
        progress(0.1, desc="Loading model...")
        
        # Get model config
        config = MODEL_CONFIGS["T2V"][model_id]
        resolution = config["resolution"]
        height, width = get_resolution_info(resolution)
        
        # Load model
        pipeline = load_model("T2V", model_id, offload)
        
        progress(0.2, desc="Enhancing prompt...")
        enhanced_prompt = enhance_prompt(prompt, use_prompt_enhancer)
        
        progress(0.3, desc="Generating video...")
        
        # Set seed
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        # Generate video
        video_frames = pipeline(
            prompt=enhanced_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            shift=shift,
            generator=generator,
        )[0]
        
        progress(0.9, desc="Saving video...")
        
        # Save video
        output_path = f"output_t2v_{int(time.time())}.mp4"
        imageio.mimsave(output_path, video_frames, fps=fps)
        
        progress(1.0, desc="Complete!")
        
        info = f"Generated T2V video with {len(video_frames)} frames at {fps} FPS\n"
        info += f"Model: {model_id}\n"
        info += f"Resolution: {resolution} ({width}x{height})\n"
        info += f"Seed: {seed}\n"
        if use_prompt_enhancer:
            info += f"Enhanced prompt: {enhanced_prompt}\n"
        
        return output_path, info
        
    except Exception as e:
        error_msg = f"Error generating T2V video: {str(e)}"
        print(error_msg)
        return None, error_msg

def generate_image_to_video(
    prompt: str,
    image: Image.Image,
    model_id: str,
    num_frames: int,
    fps: int,
    guidance_scale: float,
    shift: float,
    inference_steps: int,
    seed: Optional[int],
    use_prompt_enhancer: bool,
    offload: bool,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """Generate video from image and text prompt."""
    try:
        if image is None:
            return None, "Please provide an input image for I2V generation."
        
        progress(0.1, desc="Loading model...")
        
        # Get model config
        config = MODEL_CONFIGS["I2V"][model_id]
        resolution = config["resolution"]
        height, width = get_resolution_info(resolution)
        
        # Load model
        pipeline = load_model("I2V", model_id, offload)
        
        progress(0.2, desc="Processing image...")
        
        # Process image
        image_width, image_height = image.size
        if image_height > image_width:
            height, width = width, height
        processed_image = resizecrop(image, height, width)
        
        progress(0.3, desc="Enhancing prompt...")
        enhanced_prompt = enhance_prompt(prompt, use_prompt_enhancer)
        
        progress(0.4, desc="Generating video...")
        
        # Set seed
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        # Generate video
        video_frames = pipeline(
            image=processed_image,
            prompt=enhanced_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            shift=shift,
            generator=generator,
        )[0]
        
        progress(0.9, desc="Saving video...")
        
        # Save video
        output_path = f"output_i2v_{int(time.time())}.mp4"
        imageio.mimsave(output_path, video_frames, fps=fps)
        
        progress(1.0, desc="Complete!")
        
        info = f"Generated I2V video with {len(video_frames)} frames at {fps} FPS\n"
        info += f"Model: {model_id}\n"
        info += f"Resolution: {resolution} ({width}x{height})\n"
        info += f"Seed: {seed}\n"
        if use_prompt_enhancer:
            info += f"Enhanced prompt: {enhanced_prompt}\n"
        
        return output_path, info
        
    except Exception as e:
        error_msg = f"Error generating I2V video: {str(e)}"
        print(error_msg)
        return None, error_msg

def generate_diffusion_forcing_video(
    prompt: str,
    model_id: str,
    num_frames: int,
    fps: int,
    guidance_scale: float,
    shift: float,
    inference_steps: int,
    seed: Optional[int],
    use_prompt_enhancer: bool,
    offload: bool,
    # DF specific parameters
    ar_step: int,
    base_num_frames: int,
    overlap_history: Optional[int],
    addnoise_condition: int,
    causal_block_size: int,
    # Optional inputs
    input_image: Optional[Image.Image],
    end_image: Optional[Image.Image],
    video_path: Optional[str],
    progress=gr.Progress()
) -> Tuple[str, str]:
    """Generate video using Diffusion Forcing."""
    try:
        progress(0.1, desc="Loading model...")

        # Get model config
        config = MODEL_CONFIGS["DF"][model_id]
        resolution = config["resolution"]
        height, width = get_resolution_info(resolution)

        # Load model
        pipeline = load_model("DF", model_id, offload)

        progress(0.2, desc="Enhancing prompt...")
        enhanced_prompt = enhance_prompt(prompt, use_prompt_enhancer)

        progress(0.3, desc="Generating video...")

        # Set seed
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        generator = torch.Generator(device="cuda").manual_seed(seed)

        # Process optional inputs
        processed_image = None
        processed_end_image = None

        if input_image is not None:
            image_width, image_height = input_image.size
            if image_height > image_width:
                height, width = width, height
            processed_image = resizecrop(input_image, height, width)

        if end_image is not None:
            image_width, image_height = end_image.size
            if image_height > image_width:
                height, width = width, height
            processed_end_image = resizecrop(end_image, height, width)

        # Set default overlap_history if needed for long videos
        if num_frames > base_num_frames and overlap_history is None:
            overlap_history = 17

        # Generate video based on mode
        if video_path and os.path.exists(video_path):
            # Video extension mode
            video_frames = pipeline.extend_video(
                prompt=enhanced_prompt,
                negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                prefix_video_path=video_path,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=inference_steps,
                shift=shift,
                guidance_scale=guidance_scale,
                generator=generator,
                overlap_history=overlap_history,
                addnoise_condition=addnoise_condition,
                base_num_frames=base_num_frames,
                ar_step=ar_step,
                causal_block_size=causal_block_size,
                fps=fps,
            )[0]
        else:
            # Regular generation or start/end frame control
            video_frames = pipeline(
                prompt=enhanced_prompt,
                negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                image=processed_image,
                end_image=processed_end_image,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=inference_steps,
                shift=shift,
                guidance_scale=guidance_scale,
                generator=generator,
                overlap_history=overlap_history,
                addnoise_condition=addnoise_condition,
                base_num_frames=base_num_frames,
                ar_step=ar_step,
                causal_block_size=causal_block_size,
                fps=fps,
            )[0]

        progress(0.9, desc="Saving video...")

        # Save video
        output_path = f"output_df_{int(time.time())}.mp4"
        imageio.mimsave(output_path, video_frames, fps=fps)

        progress(1.0, desc="Complete!")

        info = f"Generated DF video with {len(video_frames)} frames at {fps} FPS\n"
        info += f"Model: {model_id}\n"
        info += f"Resolution: {resolution} ({width}x{height})\n"
        info += f"Seed: {seed}\n"
        info += f"AR Step: {ar_step} ({'Asynchronous' if ar_step > 0 else 'Synchronous'})\n"
        info += f"Base Frames: {base_num_frames}, Total Frames: {num_frames}\n"
        if overlap_history:
            info += f"Overlap History: {overlap_history}\n"
        if addnoise_condition > 0:
            info += f"Add Noise Condition: {addnoise_condition}\n"
        if use_prompt_enhancer:
            info += f"Enhanced prompt: {enhanced_prompt}\n"

        return output_path, info

    except Exception as e:
        error_msg = f"Error generating DF video: {str(e)}"
        print(error_msg)
        return None, error_msg

def update_model_defaults(model_type: str, model_id: str):
    """Update default values based on selected model."""
    if model_type not in MODEL_CONFIGS or model_id not in MODEL_CONFIGS[model_type]:
        return 97, 6.0, 8.0, 30, 24

    config = MODEL_CONFIGS[model_type][model_id]
    return (
        config["frames"],
        config["guidance"],
        config["shift"],
        30,  # inference_steps
        24   # fps
    )

def create_gradio_interface():
    """Create the main Gradio interface."""

    with gr.Blocks(title="SkyReels V2 - Video Generation", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎬 SkyReels V2 - Advanced Video Generation

        Generate high-quality videos using state-of-the-art AI models. Choose from Text-to-Video, Image-to-Video, or Diffusion Forcing modes.

        **Features:**
        - 📝 **Text-to-Video**: Generate videos from text descriptions
        - 🖼️ **Image-to-Video**: Animate images with text prompts
        - 🎯 **Diffusion Forcing**: Advanced control with long video generation, video extension, and frame control
        - 🎨 **Multiple Resolutions**: 540P and 720P support
        - ⚡ **Prompt Enhancement**: Automatic prompt optimization
        """)

        with gr.Tabs() as tabs:
            # Text-to-Video Tab
            with gr.Tab("📝 Text-to-Video", id="t2v"):
                with gr.Row():
                    with gr.Column(scale=2):
                        t2v_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe the video you want to generate...",
                            lines=3,
                            value="A serene lake surrounded by towering mountains, with a few swans gracefully gliding across the water and sunlight dancing on the surface."
                        )

                        with gr.Row():
                            t2v_model = gr.Dropdown(
                                label="Model",
                                choices=list(MODEL_CONFIGS["T2V"].keys()),
                                value="Skywork/SkyReels-V2-T2V-14B-540P"
                            )
                            t2v_enhance = gr.Checkbox(label="Enhance Prompt", value=True)

                        with gr.Row():
                            t2v_frames = gr.Slider(label="Number of Frames", minimum=25, maximum=300, value=97, step=1)
                            t2v_fps = gr.Slider(label="FPS", minimum=8, maximum=30, value=24, step=1)

                        with gr.Row():
                            t2v_guidance = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=15.0, value=6.0, step=0.1)
                            t2v_shift = gr.Slider(label="Shift", minimum=1.0, maximum=15.0, value=8.0, step=0.1)

                        with gr.Row():
                            t2v_steps = gr.Slider(label="Inference Steps", minimum=10, maximum=100, value=30, step=1)
                            t2v_seed = gr.Number(label="Seed (optional)", value=None, precision=0)

                        t2v_offload = gr.Checkbox(label="CPU Offload (saves GPU memory)", value=True)

                        t2v_generate = gr.Button("🎬 Generate Video", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        t2v_output = gr.Video(label="Generated Video", height=400)
                        t2v_info = gr.Textbox(label="Generation Info", lines=8, interactive=False)

                # Update defaults when model changes
                t2v_model.change(
                    fn=lambda model_id: update_model_defaults("T2V", model_id),
                    inputs=[t2v_model],
                    outputs=[t2v_frames, t2v_guidance, t2v_shift, t2v_steps, t2v_fps]
                )

                # Generate button click
                t2v_generate.click(
                    fn=generate_text_to_video,
                    inputs=[
                        t2v_prompt, t2v_model, t2v_frames, t2v_fps,
                        t2v_guidance, t2v_shift, t2v_steps, t2v_seed,
                        t2v_enhance, t2v_offload
                    ],
                    outputs=[t2v_output, t2v_info]
                )

            # Image-to-Video Tab
            with gr.Tab("🖼️ Image-to-Video", id="i2v"):
                with gr.Row():
                    with gr.Column(scale=2):
                        i2v_image = gr.Image(
                            label="Input Image",
                            type="pil",
                            height=300
                        )

                        i2v_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe how you want the image to be animated...",
                            lines=3,
                            value="The image comes to life with gentle movement and natural motion."
                        )

                        with gr.Row():
                            i2v_model = gr.Dropdown(
                                label="Model",
                                choices=list(MODEL_CONFIGS["I2V"].keys()),
                                value="Skywork/SkyReels-V2-I2V-14B-540P"
                            )
                            i2v_enhance = gr.Checkbox(label="Enhance Prompt", value=True)

                        with gr.Row():
                            i2v_frames = gr.Slider(label="Number of Frames", minimum=25, maximum=300, value=97, step=1)
                            i2v_fps = gr.Slider(label="FPS", minimum=8, maximum=30, value=24, step=1)

                        with gr.Row():
                            i2v_guidance = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=15.0, value=5.0, step=0.1)
                            i2v_shift = gr.Slider(label="Shift", minimum=1.0, maximum=15.0, value=3.0, step=0.1)

                        with gr.Row():
                            i2v_steps = gr.Slider(label="Inference Steps", minimum=10, maximum=100, value=30, step=1)
                            i2v_seed = gr.Number(label="Seed (optional)", value=None, precision=0)

                        i2v_offload = gr.Checkbox(label="CPU Offload (saves GPU memory)", value=True)

                        i2v_generate = gr.Button("🎬 Generate Video", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        i2v_output = gr.Video(label="Generated Video", height=400)
                        i2v_info = gr.Textbox(label="Generation Info", lines=8, interactive=False)

                # Update defaults when model changes
                i2v_model.change(
                    fn=lambda model_id: update_model_defaults("I2V", model_id),
                    inputs=[i2v_model],
                    outputs=[i2v_frames, i2v_guidance, i2v_shift, i2v_steps, i2v_fps]
                )

                # Generate button click
                i2v_generate.click(
                    fn=generate_image_to_video,
                    inputs=[
                        i2v_prompt, i2v_image, i2v_model, i2v_frames, i2v_fps,
                        i2v_guidance, i2v_shift, i2v_steps, i2v_seed,
                        i2v_enhance, i2v_offload
                    ],
                    outputs=[i2v_output, i2v_info]
                )

            # Diffusion Forcing Tab
            with gr.Tab("🎯 Diffusion Forcing", id="df"):
                gr.Markdown("""
                **Diffusion Forcing** enables advanced video generation with:
                - 🔄 **Long Video Generation**: Create videos up to 30+ seconds
                - 📹 **Video Extension**: Extend existing videos
                - 🎬 **Frame Control**: Control start and end frames
                - ⚡ **Sync/Async Modes**: Choose generation strategy
                """)

                with gr.Row():
                    with gr.Column(scale=2):
                        df_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe the video you want to generate...",
                            lines=3,
                            value="A graceful white swan with a curved neck and delicate feathers swimming in a serene lake at dawn, its reflection perfectly mirrored in the still water as mist rises from the surface."
                        )

                        with gr.Row():
                            df_model = gr.Dropdown(
                                label="Model",
                                choices=list(MODEL_CONFIGS["DF"].keys()),
                                value="Skywork/SkyReels-V2-DF-14B-540P"
                            )
                            df_enhance = gr.Checkbox(label="Enhance Prompt", value=True)

                        with gr.Accordion("📊 Basic Settings", open=True):
                            with gr.Row():
                                df_frames = gr.Slider(label="Total Frames", minimum=25, maximum=800, value=97, step=1)
                                df_fps = gr.Slider(label="FPS", minimum=8, maximum=30, value=24, step=1)

                            with gr.Row():
                                df_guidance = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=15.0, value=6.0, step=0.1)
                                df_shift = gr.Slider(label="Shift", minimum=1.0, maximum=15.0, value=8.0, step=0.1)

                            with gr.Row():
                                df_steps = gr.Slider(label="Inference Steps", minimum=10, maximum=100, value=30, step=1)
                                df_seed = gr.Number(label="Seed (optional)", value=None, precision=0)

                        with gr.Accordion("⚙️ Diffusion Forcing Settings", open=True):
                            with gr.Row():
                                df_ar_step = gr.Slider(
                                    label="AR Step (0=Sync, >0=Async)",
                                    minimum=0, maximum=10, value=0, step=1,
                                    info="0 for synchronous, 5+ for asynchronous generation"
                                )
                                df_base_frames = gr.Slider(
                                    label="Base Frames",
                                    minimum=25, maximum=200, value=97, step=1,
                                    info="Base frame count (97 for 540P, 121 for 720P)"
                                )

                            with gr.Row():
                                df_overlap = gr.Slider(
                                    label="Overlap History",
                                    minimum=0, maximum=50, value=17, step=1,
                                    info="Frames to overlap for smooth transitions (17 recommended)"
                                )
                                df_addnoise = gr.Slider(
                                    label="Add Noise Condition",
                                    minimum=0, maximum=60, value=20, step=1,
                                    info="Improves consistency (20 recommended)"
                                )

                            df_causal_block = gr.Slider(
                                label="Causal Block Size",
                                minimum=1, maximum=10, value=5, step=1,
                                info="Used with async generation (ar_step > 0)"
                            )

                        with gr.Accordion("🎬 Advanced Controls", open=False):
                            df_start_image = gr.Image(
                                label="Start Frame Image (optional)",
                                type="pil",
                                height=200
                            )
                            df_end_image = gr.Image(
                                label="End Frame Image (optional)",
                                type="pil",
                                height=200
                            )
                            df_video_path = gr.File(
                                label="Video to Extend (optional) - Upload a video to extend it",
                                file_types=[".mp4", ".avi", ".mov"]
                            )

                        df_offload = gr.Checkbox(label="CPU Offload (saves GPU memory)", value=True)

                        df_generate = gr.Button("🎬 Generate Video", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        df_output = gr.Video(label="Generated Video", height=400)
                        df_info = gr.Textbox(label="Generation Info", lines=10, interactive=False)

                # Update defaults when model changes
                df_model.change(
                    fn=lambda model_id: update_model_defaults("DF", model_id),
                    inputs=[df_model],
                    outputs=[df_frames, df_guidance, df_shift, df_steps, df_fps]
                )

                # Generate button click
                df_generate.click(
                    fn=generate_diffusion_forcing_video,
                    inputs=[
                        df_prompt, df_model, df_frames, df_fps,
                        df_guidance, df_shift, df_steps, df_seed,
                        df_enhance, df_offload,
                        df_ar_step, df_base_frames, df_overlap, df_addnoise, df_causal_block,
                        df_start_image, df_end_image, df_video_path
                    ],
                    outputs=[df_output, df_info]
                )

        # Examples section
        with gr.Accordion("📚 Examples & Tips", open=False):
            gr.Markdown("""
            ### 💡 Tips for Better Results:

            **Text-to-Video:**
            - Use detailed, descriptive prompts
            - Mention camera movements, lighting, and atmosphere
            - 540P models: Use guidance=6.0, shift=8.0
            - 720P models: Use guidance=6.0, shift=8.0

            **Image-to-Video:**
            - Use high-quality input images
            - Describe the desired motion clearly
            - Use guidance=5.0, shift=3.0 for best results
            - Portrait images work well for character animation

            **Diffusion Forcing:**
            - For 10s videos: Use synchronous mode (ar_step=0)
            - For 30s+ videos: Use asynchronous mode (ar_step=5)
            - Set overlap_history=17 for smooth long videos
            - Use addnoise_condition=20 for consistency
            - Base frames: 97 for 540P, 121 for 720P

            ### 🎬 Example Prompts:
            - "A majestic eagle soaring through mountain peaks at golden hour"
            - "Ocean waves crashing against rocky cliffs in slow motion"
            - "A bustling city street with people walking and cars passing by"
            - "Cherry blossoms falling gently in a peaceful Japanese garden"
            """)

        return demo

def main():
    """Main function to run the Gradio interface."""
    parser = argparse.ArgumentParser(description="SkyReels V2 Gradio Interface")
    parser.add_argument("--share", action="store_true", help="Enable Gradio sharing")
    parser.add_argument("--server_name", type=str, default="0.0.0.0", help="Server name")
    parser.add_argument("--server_port", type=int, default=7860, help="Server port")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    print("🎬 Starting SkyReels V2 Gradio Interface...")
    print(f"Server: {args.server_name}:{args.server_port}")
    print(f"Share: {args.share}")

    # Create and launch the interface
    demo = create_gradio_interface()

    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        debug=args.debug,
        show_error=True,
        favicon_path=None,
        ssl_verify=False
    )

if __name__ == "__main__":
    main()
