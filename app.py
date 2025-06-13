import gradio as gr
import os
import sys
import torch
import time
import random
import gc
import tempfile
import shutil
from typing import Optional, Tuple, Dict, Any
import imageio
from diffusers.utils import load_image
from moviepy.editor import VideoFileClip
import numpy as np

# Add the repository to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from skyreels_v2_infer.modules import download_model
from skyreels_v2_infer.pipelines import (
    Image2VideoPipeline, 
    Text2VideoPipeline, 
    PromptEnhancer,
    resizecrop
)
from skyreels_v2_infer import DiffusionForcingPipeline

# Global variables to store loaded models
loaded_models = {}
prompt_enhancer = None

# Model configurations
MODEL_ID_CONFIG = {
    "text2video": [
        "Skywork/SkyReels-V2-T2V-14B-540P",
        "Skywork/SkyReels-V2-T2V-14B-720P",
    ],
    "image2video": [
        "Skywork/SkyReels-V2-I2V-1.3B-540P",
        "Skywork/SkyReels-V2-I2V-14B-540P",
        "Skywork/SkyReels-V2-I2V-14B-720P",
    ],
    "diffusion_forcing": [
        "Skywork/SkyReels-V2-DF-1.3B-540P",
        "Skywork/SkyReels-V2-DF-14B-540P",
        "Skywork/SkyReels-V2-DF-14B-720P",
    ]
}

def get_resolution_params(resolution: str) -> Tuple[int, int]:
    """Get height and width for the given resolution."""
    if resolution == "540P":
        return 544, 960
    elif resolution == "720P":
        return 720, 1280
    else:
        raise ValueError(f"Invalid resolution: {resolution}")

def load_model(model_id: str, model_type: str) -> Any:
    """Load and cache models to avoid reloading."""
    global loaded_models
    
    if model_id in loaded_models:
        return loaded_models[model_id]
    
    try:
        # Download model if needed
        model_path = download_model(model_id)
        
        if model_type == "text2video":
            pipeline = Text2VideoPipeline(
                model_path=model_path, 
                dit_path=model_path, 
                use_usp=False, 
                offload=True
            )
        elif model_type == "image2video":
            pipeline = Image2VideoPipeline(
                model_path=model_path, 
                dit_path=model_path, 
                use_usp=False, 
                offload=True
            )
        elif model_type == "diffusion_forcing":
            pipeline = DiffusionForcingPipeline(
                model_path,
                dit_path=model_path,
                device=torch.device("cuda"),
                weight_dtype=torch.bfloat16,
                use_usp=False,
                offload=True,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        loaded_models[model_id] = pipeline
        return pipeline
    except Exception as e:
        raise gr.Error(f"Failed to load model {model_id}: {str(e)}")

def enhance_prompt(prompt: str) -> str:
    """Enhance prompt using PromptEnhancer."""
    global prompt_enhancer
    
    try:
        if prompt_enhancer is None:
            prompt_enhancer = PromptEnhancer()
        
        enhanced = prompt_enhancer(prompt)
        return enhanced
    except Exception as e:
        print(f"Failed to enhance prompt: {e}")
        return prompt

def text_to_video(
    prompt: str,
    model_id: str,
    resolution: str,
    num_frames: int,
    guidance_scale: float,
    shift: float,
    inference_steps: int,
    fps: int,
    seed: Optional[int],
    use_prompt_enhancer: bool,
    use_teacache: bool,
    teacache_thresh: float
) -> str:
    """Generate video from text prompt."""
    try:
        # Set random seed
        if seed is None or seed == -1:
            seed = random.randint(0, 2**32 - 1)
        
        # Get resolution parameters
        height, width = get_resolution_params(resolution)
        
        # Enhance prompt if requested
        prompt_input = enhance_prompt(prompt) if use_prompt_enhancer else prompt
        
        # Load model
        pipe = load_model(model_id, "text2video")
        
        # Configure teacache if enabled
        if use_teacache:
            pipe.transformer.initialize_teacache(
                enable_teacache=True, 
                num_steps=inference_steps,
                teacache_thresh=teacache_thresh, 
                use_ret_steps=True,
                ckpt_dir=model_id
            )
        
        # Negative prompt
        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        
        # Generate video
        with torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
            video_frames = pipe(
                prompt=prompt_input,
                negative_prompt=negative_prompt,
                num_frames=num_frames,
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                shift=shift,
                generator=torch.Generator(device="cuda").manual_seed(seed),
                height=height,
                width=width,
            )[0]
        
        # Save video
        output_path = tempfile.mktemp(suffix=".mp4")
        imageio.mimwrite(output_path, video_frames, fps=fps, quality=8, output_params=["-loglevel", "error"])
        
        return output_path
        
    except Exception as e:
        raise gr.Error(f"Failed to generate video: {str(e)}")

def image_to_video(
    prompt: str,
    image: str,
    model_id: str,
    resolution: str,
    num_frames: int,
    guidance_scale: float,
    shift: float,
    inference_steps: int,
    fps: int,
    seed: Optional[int],
    use_prompt_enhancer: bool,
    use_teacache: bool,
    teacache_thresh: float
) -> str:
    """Generate video from image and text prompt."""
    try:
        if image is None:
            raise gr.Error("Please provide an input image")
        
        # Set random seed
        if seed is None or seed == -1:
            seed = random.randint(0, 2**32 - 1)
        
        # Get resolution parameters
        height, width = get_resolution_params(resolution)
        
        # Enhance prompt if requested
        prompt_input = enhance_prompt(prompt) if use_prompt_enhancer else prompt
        
        # Load and process image
        input_image = load_image(image).convert("RGB")
        image_width, image_height = input_image.size
        if image_height > image_width:
            height, width = width, height
        input_image = resizecrop(input_image, height, width)
        
        # Load model
        pipe = load_model(model_id, "image2video")
        
        # Configure teacache if enabled
        if use_teacache:
            pipe.transformer.initialize_teacache(
                enable_teacache=True, 
                num_steps=inference_steps,
                teacache_thresh=teacache_thresh, 
                use_ret_steps=True,
                ckpt_dir=model_id
            )
        
        # Negative prompt
        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        
        # Generate video
        with torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
            video_frames = pipe(
                prompt=prompt_input,
                negative_prompt=negative_prompt,
                image=input_image,
                num_frames=num_frames,
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                shift=shift,
                generator=torch.Generator(device="cuda").manual_seed(seed),
                height=height,
                width=width,
            )[0]
        
        # Save video
        output_path = tempfile.mktemp(suffix=".mp4")
        imageio.mimwrite(output_path, video_frames, fps=fps, quality=8, output_params=["-loglevel", "error"])
        
        return output_path
        
    except Exception as e:
        raise gr.Error(f"Failed to generate video: {str(e)}")

def diffusion_forcing_generation(
    prompt: str,
    model_id: str,
    resolution: str,
    num_frames: int,
    base_num_frames: int,
    overlap_history: int,
    ar_step: int,
    addnoise_condition: int,
    guidance_scale: float,
    shift: float,
    inference_steps: int,
    fps: int,
    seed: Optional[int],
    image: Optional[str],
    end_image: Optional[str],
    use_prompt_enhancer: bool,
    use_teacache: bool,
    teacache_thresh: float
) -> str:
    """Generate long video using Diffusion Forcing."""
    try:
        # Set random seed
        if seed is None or seed == -1:
            seed = random.randint(0, 2**32 - 1)
        
        # Get resolution parameters
        height, width = get_resolution_params(resolution)
        
        # Enhance prompt if requested
        prompt_input = enhance_prompt(prompt) if use_prompt_enhancer else prompt
        
        # Load model
        pipe = load_model(model_id, "diffusion_forcing")
        
        # Configure teacache if enabled
        if use_teacache:
            if ar_step > 0:
                num_steps = inference_steps + (((base_num_frames - 1) // 4 + 1) // 1 - 1) * ar_step
            else:
                num_steps = inference_steps
            pipe.transformer.initialize_teacache(
                enable_teacache=True, 
                num_steps=num_steps,
                teacache_thresh=teacache_thresh, 
                use_ret_steps=True,
                ckpt_dir=model_id
            )
        
        # Process input images if provided
        input_image = None
        input_end_image = None
        
        if image:
            input_image = load_image(image).convert("RGB")
            image_width, image_height = input_image.size
            if image_height > image_width:
                height, width = width, height
            input_image = resizecrop(input_image, height, width)
        
        if end_image:
            input_end_image = load_image(end_image).convert("RGB")
            input_end_image = resizecrop(input_end_image, height, width)
        
        # Negative prompt
        negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        
        # Generate video
        with torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
            video_frames = pipe(
                prompt=prompt_input,
                negative_prompt=negative_prompt,
                image=input_image,
                end_image=input_end_image,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=inference_steps,
                shift=shift,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device="cuda").manual_seed(seed),
                overlap_history=overlap_history,
                addnoise_condition=addnoise_condition,
                base_num_frames=base_num_frames,
                ar_step=ar_step,
                causal_block_size=1,
            )[0]
        
        # Save video
        output_path = tempfile.mktemp(suffix=".mp4")
        imageio.mimwrite(output_path, video_frames, fps=fps, quality=8, output_params=["-loglevel", "error"])
        
        return output_path
        
    except Exception as e:
        raise gr.Error(f"Failed to generate video: {str(e)}")

def extend_video(
    prompt: str,
    video_path: str,
    model_id: str,
    resolution: str,
    num_frames: int,
    base_num_frames: int,
    overlap_history: int,
    ar_step: int,
    addnoise_condition: int,
    guidance_scale: float,
    shift: float,
    inference_steps: int,
    fps: int,
    seed: Optional[int],
    use_prompt_enhancer: bool,
    use_teacache: bool,
    teacache_thresh: float
) -> str:
    """Extend an existing video."""
    try:
        if video_path is None:
            raise gr.Error("Please provide an input video")
        
        # Set random seed
        if seed is None or seed == -1:
            seed = random.randint(0, 2**32 - 1)
        
        # Get resolution parameters
        height, width = get_resolution_params(resolution)
        
        # Enhance prompt if requested
        prompt_input = enhance_prompt(prompt) if use_prompt_enhancer else prompt
        
        # Check video compatibility
        def get_video_num_frames_moviepy(video_path):
            with VideoFileClip(video_path) as clip:
                num_frames = 0
                for _ in clip.iter_frames():
                    num_frames += 1
                return clip.size, num_frames
        
        (v_width, v_height), input_num_frames = get_video_num_frames_moviepy(video_path)
        
        if input_num_frames < overlap_history:
            raise gr.Error("The input video is too short.")
        
        if v_height > v_width:
            width, height = height, width
        
        # Load model
        pipe = load_model(model_id, "diffusion_forcing")
        
        # Configure teacache if enabled
        if use_teacache:
            if ar_step > 0:
                num_steps = inference_steps + (((base_num_frames - 1) // 4 + 1) // 1 - 1) * ar_step
            else:
                num_steps = inference_steps
            pipe.transformer.initialize_teacache(
                enable_teacache=True, 
                num_steps=num_steps,
                teacache_thresh=teacache_thresh, 
                use_ret_steps=True,
                ckpt_dir=model_id
            )
        
        # Negative prompt
        negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        
        # Extend video
        video_frames = pipe.extend_video(
            prompt=prompt_input,
            negative_prompt=negative_prompt,
            prefix_video_path=video_path,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=inference_steps,
            shift=shift,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            overlap_history=overlap_history,
            addnoise_condition=addnoise_condition,
            base_num_frames=base_num_frames,
            ar_step=ar_step,
            causal_block_size=1,
            fps=fps,
        )[0]
        
        # Save video
        output_path = tempfile.mktemp(suffix=".mp4")
        imageio.mimwrite(output_path, video_frames, fps=fps, quality=8, output_params=["-loglevel", "error"])
        
        return output_path
        
    except Exception as e:
        raise gr.Error(f"Failed to extend video: {str(e)}")

def caption_video(video_path: str) -> str:
    """Caption video using SkyCaptioner-V1 (placeholder implementation)."""
    try:
        if video_path is None:
            raise gr.Error("Please provide a video to caption")
        
        # This is a placeholder - in practice you would integrate the SkyCaptioner-V1 model
        # For now, return a sample caption
        return "A sample video caption generated by SkyCaptioner-V1. This would normally analyze the video content and provide detailed structural descriptions including subjects, actions, shot types, camera movements, and environmental details."
        
    except Exception as e:
        raise gr.Error(f"Failed to caption video: {str(e)}")

# Create Gradio interface
def create_gradio_app():
    with gr.Blocks(
        title="SkyReels V2: Infinite-Length Film Generative Model",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .tab-nav {
            background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        }
        """
    ) as demo:
        
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1>🎬 SkyReels V2: Infinite-Length Film Generative Model</h1>
            <p>Generate stunning videos with state-of-the-art AI models</p>
        </div>
        """)
        
        with gr.Tabs():
            # Text-to-Video Tab
            with gr.TabItem("📝 Text-to-Video"):
                with gr.Row():
                    with gr.Column():
                        t2v_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="A serene lake surrounded by towering mountains...",
                            lines=3
                        )
                        t2v_model = gr.Dropdown(
                            label="Model",
                            choices=MODEL_ID_CONFIG["text2video"],
                            value=MODEL_ID_CONFIG["text2video"][0]
                        )
                        t2v_resolution = gr.Dropdown(
                            label="Resolution",
                            choices=["540P", "720P"],
                            value="540P"
                        )
                        
                        with gr.Row():
                            t2v_num_frames = gr.Slider(1, 200, value=97, step=1, label="Number of Frames")
                            t2v_fps = gr.Slider(8, 30, value=24, step=1, label="FPS")
                        
                        with gr.Row():
                            t2v_guidance_scale = gr.Slider(1.0, 20.0, value=6.0, step=0.1, label="Guidance Scale")
                            t2v_shift = gr.Slider(1.0, 20.0, value=8.0, step=0.1, label="Shift")
                        
                        with gr.Row():
                            t2v_inference_steps = gr.Slider(10, 100, value=30, step=1, label="Inference Steps")
                            t2v_seed = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                        
                        with gr.Row():
                            t2v_enhance_prompt = gr.Checkbox(label="Enhance Prompt", value=False)
                            t2v_teacache = gr.Checkbox(label="Use TeaCache", value=True)
                            t2v_teacache_thresh = gr.Slider(0.1, 0.5, value=0.2, step=0.1, label="TeaCache Threshold")
                        
                        t2v_generate_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg")
                    
                    with gr.Column():
                        t2v_output = gr.Video(label="Generated Video")
                
                t2v_generate_btn.click(
                    text_to_video,
                    inputs=[
                        t2v_prompt, t2v_model, t2v_resolution, t2v_num_frames,
                        t2v_guidance_scale, t2v_shift, t2v_inference_steps, t2v_fps,
                        t2v_seed, t2v_enhance_prompt, t2v_teacache, t2v_teacache_thresh
                    ],
                    outputs=t2v_output
                )
            
            # Image-to-Video Tab
            with gr.TabItem("🖼️ Image-to-Video"):
                with gr.Row():
                    with gr.Column():
                        i2v_image = gr.Image(label="Input Image", type="filepath")
                        i2v_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe the motion you want in the video...",
                            lines=3
                        )
                        i2v_model = gr.Dropdown(
                            label="Model",
                            choices=MODEL_ID_CONFIG["image2video"],
                            value=MODEL_ID_CONFIG["image2video"][0]
                        )
                        i2v_resolution = gr.Dropdown(
                            label="Resolution",
                            choices=["540P", "720P"],
                            value="540P"
                        )
                        
                        with gr.Row():
                            i2v_num_frames = gr.Slider(1, 200, value=97, step=1, label="Number of Frames")
                            i2v_fps = gr.Slider(8, 30, value=24, step=1, label="FPS")
                        
                        with gr.Row():
                            i2v_guidance_scale = gr.Slider(1.0, 20.0, value=6.0, step=0.1, label="Guidance Scale")
                            i2v_shift = gr.Slider(1.0, 20.0, value=8.0, step=0.1, label="Shift")
                        
                        with gr.Row():
                            i2v_inference_steps = gr.Slider(10, 100, value=30, step=1, label="Inference Steps")
                            i2v_seed = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                        
                        with gr.Row():
                            i2v_enhance_prompt = gr.Checkbox(label="Enhance Prompt", value=False)
                            i2v_teacache = gr.Checkbox(label="Use TeaCache", value=True)
                            i2v_teacache_thresh = gr.Slider(0.1, 0.5, value=0.2, step=0.1, label="TeaCache Threshold")
                        
                        i2v_generate_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg")
                    
                    with gr.Column():
                        i2v_output = gr.Video(label="Generated Video")
                
                i2v_generate_btn.click(
                    image_to_video,
                    inputs=[
                        i2v_prompt, i2v_image, i2v_model, i2v_resolution, i2v_num_frames,
                        i2v_guidance_scale, i2v_shift, i2v_inference_steps, i2v_fps,
                        i2v_seed, i2v_enhance_prompt, i2v_teacache, i2v_teacache_thresh
                    ],
                    outputs=i2v_output
                )
            
            # Diffusion Forcing Tab
            with gr.TabItem("🔄 Diffusion Forcing (Long Video)"):
                with gr.Row():
                    with gr.Column():
                        df_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="A woman in a leather jacket riding a motorcycle...",
                            lines=3
                        )
                        df_model = gr.Dropdown(
                            label="Model",
                            choices=MODEL_ID_CONFIG["diffusion_forcing"],
                            value=MODEL_ID_CONFIG["diffusion_forcing"][0]
                        )
                        df_resolution = gr.Dropdown(
                            label="Resolution",
                            choices=["540P", "720P"],
                            value="540P"
                        )
                        
                        with gr.Row():
                            df_image = gr.Image(label="Start Image (Optional)", type="filepath")
                            df_end_image = gr.Image(label="End Image (Optional)", type="filepath")
                        
                        with gr.Row():
                            df_num_frames = gr.Slider(97, 500, value=257, step=1, label="Total Frames")
                            df_base_num_frames = gr.Slider(50, 200, value=97, step=1, label="Base Frames")
                        
                        with gr.Row():
                            df_overlap_history = gr.Slider(10, 50, value=17, step=1, label="Overlap History")
                            df_ar_step = gr.Slider(0, 20, value=0, step=1, label="AR Steps")
                        
                        with gr.Row():
                            df_addnoise_condition = gr.Slider(0, 60, value=20, step=1, label="Add Noise Condition")
                            df_fps = gr.Slider(8, 30, value=24, step=1, label="FPS")
                        
                        with gr.Row():
                            df_guidance_scale = gr.Slider(1.0, 20.0, value=6.0, step=0.1, label="Guidance Scale")
                            df_shift = gr.Slider(1.0, 20.0, value=8.0, step=0.1, label="Shift")
                        
                        with gr.Row():
                            df_inference_steps = gr.Slider(10, 100, value=30, step=1, label="Inference Steps")
                            df_seed = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                        
                        with gr.Row():
                            df_enhance_prompt = gr.Checkbox(label="Enhance Prompt", value=False)
                            df_teacache = gr.Checkbox(label="Use TeaCache", value=True)
                            df_teacache_thresh = gr.Slider(0.1, 0.5, value=0.3, step=0.1, label="TeaCache Threshold")
                        
                        df_generate_btn = gr.Button("🎬 Generate Long Video", variant="primary", size="lg")
                    
                    with gr.Column():
                        df_output = gr.Video(label="Generated Video")
                
                df_generate_btn.click(
                    diffusion_forcing_generation,
                    inputs=[
                        df_prompt, df_model, df_resolution, df_num_frames, df_base_num_frames,
                        df_overlap_history, df_ar_step, df_addnoise_condition, df_guidance_scale,
                        df_shift, df_inference_steps, df_fps, df_seed, df_image, df_end_image,
                        df_enhance_prompt, df_teacache, df_teacache_thresh
                    ],
                    outputs=df_output
                )
            
            # Video Extension Tab
            with gr.TabItem("➕ Video Extension"):
                with gr.Row():
                    with gr.Column():
                        ve_video = gr.Video(label="Input Video")
                        ve_prompt = gr.Textbox(
                            label="Extension Prompt",
                            placeholder="Describe how you want to extend the video...",
                            lines=3
                        )
                        ve_model = gr.Dropdown(
                            label="Model",
                            choices=MODEL_ID_CONFIG["diffusion_forcing"],
                            value=MODEL_ID_CONFIG["diffusion_forcing"][0]
                        )
                        ve_resolution = gr.Dropdown(
                            label="Resolution",
                            choices=["540P", "720P"],
                            value="540P"
                        )
                        
                        with gr.Row():
                            ve_num_frames = gr.Slider(97, 500, value=257, step=1, label="Total Frames")
                            ve_base_num_frames = gr.Slider(50, 200, value=97, step=1, label="Base Frames")
                        
                        with gr.Row():
                            ve_overlap_history = gr.Slider(10, 50, value=17, step=1, label="Overlap History")
                            ve_ar_step = gr.Slider(0, 20, value=0, step=1, label="AR Steps")
                        
                        with gr.Row():
                            ve_addnoise_condition = gr.Slider(0, 60, value=20, step=1, label="Add Noise Condition")
                            ve_fps = gr.Slider(8, 30, value=24, step=1, label="FPS")
                        
                        with gr.Row():
                            ve_guidance_scale = gr.Slider(1.0, 20.0, value=6.0, step=0.1, label="Guidance Scale")
                            ve_shift = gr.Slider(1.0, 20.0, value=8.0, step=0.1, label="Shift")
                        
                        with gr.Row():
                            ve_inference_steps = gr.Slider(10, 100, value=30, step=1, label="Inference Steps")
                            ve_seed = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                        
                        with gr.Row():
                            ve_enhance_prompt = gr.Checkbox(label="Enhance Prompt", value=False)
                            ve_teacache = gr.Checkbox(label="Use TeaCache", value=True)
                            ve_teacache_thresh = gr.Slider(0.1, 0.5, value=0.3, step=0.1, label="TeaCache Threshold")
                        
                        ve_extend_btn = gr.Button("➕ Extend Video", variant="primary", size="lg")
                    
                    with gr.Column():
                        ve_output = gr.Video(label="Extended Video")
                
                ve_extend_btn.click(
                    extend_video,
                    inputs=[
                        ve_prompt, ve_video, ve_model, ve_resolution, ve_num_frames,
                        ve_base_num_frames, ve_overlap_history, ve_ar_step, ve_addnoise_condition,
                        ve_guidance_scale, ve_shift, ve_inference_steps, ve_fps, ve_seed,
                        ve_enhance_prompt, ve_teacache, ve_teacache_thresh
                    ],
                    outputs=ve_output
                )
            
            # Video Captioning Tab
            with gr.TabItem("💬 Video Captioning"):
                with gr.Row():
                    with gr.Column():
                        cap_video = gr.Video(label="Input Video")
                        cap_button = gr.Button("📝 Generate Caption", variant="primary", size="lg")
                    
                    with gr.Column():
                        cap_output = gr.Textbox(
                            label="Generated Caption",
                            lines=5,
                            placeholder="Video caption will appear here..."
                        )
                
                cap_button.click(
                    caption_video,
                    inputs=cap_video,
                    outputs=cap_output
                )
        
        # Information section
        gr.HTML("""
        <div style="margin-top: 20px; padding: 15px; background: #f0f0f0; border-radius: 10px;">
            <h3>ℹ️ About SkyReels V2</h3>
            <p>SkyReels V2 is the first open-source video generative model employing AutoRegressive Diffusion-Forcing architecture. 
            It supports infinite-length video generation with state-of-the-art performance.</p>
            
            <h4>Features:</h4>
            <ul>
                <li><strong>Text-to-Video:</strong> Generate videos from text descriptions</li>
                <li><strong>Image-to-Video:</strong> Animate images with text prompts</li>
                <li><strong>Diffusion Forcing:</strong> Generate very long videos (up to infinite length)</li>
                <li><strong>Video Extension:</strong> Extend existing videos seamlessly</li>
                <li><strong>Video Captioning:</strong> Generate detailed captions for videos</li>
            </ul>
            
            <p><strong>Note:</strong> Generation may take several minutes depending on the length and complexity of the video.</p>
        </div>
        """)
    
    return demo

if __name__ == "__main__":
    # Create and launch the Gradio app
    demo = create_gradio_app()
    
    # Launch with public link sharing enabled
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # This enables public link sharing
        show_error=True,
        debug=True
    ) 