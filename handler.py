"""
TRELLIS.2 RunPod Serverless Handler
API compatible with mockupWebsite frontend

Usage:
    POST /run with JSON body:
    {
        "input": {
            "image": "base64...",
            "resolution": 512 | 1024 | 1536,
            "texture_size": 1024 | 2048 | 4096,
            "output_format": "glb" | "obj" | "ply",
            "seed": 42,
            "sparse_structure_steps": 20,
            "slat_sampler_steps": 20
        }
    }
"""

import runpod
import os
import sys
import base64
import tempfile
import time
import torch
import requests
from io import BytesIO
from PIL import Image

# Add TRELLIS.2 to path
TRELLIS_PATH = os.environ.get("TRELLIS_PATH", "/app/TRELLIS.2")
sys.path.insert(0, TRELLIS_PATH)

# RunPod model cache configuration
CACHE_DIR = "/runpod-volume/huggingface-cache/hub"
HF_MODEL_ID = "microsoft/TRELLIS.2-4B"

# Global pipeline - initialized once at worker startup
pipeline = None


def find_cached_model_path(model_name: str) -> str | None:
    """Find model path in RunPod's cache directory."""
    cache_name = model_name.replace("/", "--")
    snapshots_dir = os.path.join(CACHE_DIR, f"models--{cache_name}", "snapshots")

    if os.path.exists(snapshots_dir):
        snapshots = os.listdir(snapshots_dir)
        if snapshots:
            return os.path.join(snapshots_dir, snapshots[0])
    return None


def load_model():
    """Load TRELLIS.2 model into GPU memory."""
    global pipeline

    if pipeline is not None:
        return pipeline

    print("Loading TRELLIS.2 model...")
    start_time = time.time()

    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    # Check RunPod's cache first
    cached_path = find_cached_model_path(HF_MODEL_ID)

    if cached_path:
        print(f"Loading from RunPod cache: {cached_path}")
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(cached_path)
    else:
        print(f"Model not in cache, downloading: {HF_MODEL_ID}")
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(HF_MODEL_ID)

    pipeline.cuda()

    elapsed = time.time() - start_time
    print(f"Model loaded in {elapsed:.2f}s")

    return pipeline


def decode_base64_image(data: str) -> Image.Image:
    """Decode base64 image string."""
    # Remove data URL prefix if present
    if "," in data:
        data = data.split(",", 1)[1]
    image_data = base64.b64decode(data)
    return Image.open(BytesIO(image_data)).convert("RGBA")


def download_image(url: str) -> Image.Image:
    """Download image from URL."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGBA")


def handler(job):
    """
    RunPod serverless handler for TRELLIS.2 image-to-3D.
    API compatible with mockupWebsite frontend.

    Input parameters:
    - image: str - Base64 encoded image (required)
    - resolution: int - 512, 1024, or 1536 (default: 1024)
    - texture_size: int - 1024, 2048, or 4096 (default: 2048)
    - output_format: str - "glb", "obj", or "ply" (default: "glb")
    - seed: int - Random seed (default: random)
    - sparse_structure_steps: int - Steps for sparse structure (default: 20)
    - slat_sampler_steps: int - Steps for SLAT sampler (default: 20)

    Returns (matching Trellis2GenerateResponse):
    - model: str - Base64 encoded GLB/OBJ/PLY
    - thumbnail: str - Base64 encoded PNG thumbnail
    - metadata: object - Generation metadata
    """
    job_input = job["input"]
    start_time = time.time()

    # Get image - support both 'image' (website) and 'input_image' (direct API)
    image_data = job_input.get("image") or job_input.get("input_image")
    if not image_data:
        return {"error": "image is required"}

    # Load model (cached after first call)
    runpod.serverless.progress_update(job, "Loading model...")
    pipe = load_model()

    # Load image
    runpod.serverless.progress_update(job, "Processing image...")
    try:
        if image_data.startswith(("http://", "https://")):
            image = download_image(image_data)
        else:
            image = decode_base64_image(image_data)
    except Exception as e:
        return {"error": f"Failed to load image: {str(e)}"}

    print(f"Image size: {image.size}")

    # Extract parameters with defaults
    resolution = job_input.get("resolution", 1024)
    texture_size = job_input.get("texture_size", 2048)
    output_format = job_input.get("output_format", "glb")
    seed = job_input.get("seed")
    sparse_structure_steps = job_input.get("sparse_structure_steps", 20)
    slat_sampler_steps = job_input.get("slat_sampler_steps", 20)

    if seed is None:
        import random
        seed = random.randint(0, 2**32 - 1)

    # Map resolution to simplify target
    simplify_targets = {512: 4194304, 1024: 16777216, 1536: 16777216}
    simplify_target = simplify_targets.get(resolution, 16777216)

    # Map resolution to decimation target
    decimation_targets = {512: 100000, 1024: 500000, 1536: 1000000}
    decimation_target = decimation_targets.get(resolution, 500000)

    # Map resolution to pipeline_type (TRELLIS.2 API requirement)
    pipeline_types = {512: "512", 1024: "1024_cascade", 1536: "1536_cascade"}
    pipeline_type = pipeline_types.get(resolution, "1024_cascade")

    # Run inference
    runpod.serverless.progress_update(job, "Generating 3D model...")
    print(f"Running inference: resolution={resolution}, pipeline_type={pipeline_type}, seed={seed}")

    try:
        mesh = pipe.run(
            image,
            seed=seed,
            pipeline_type=pipeline_type,
            sparse_structure_sampler_params={"steps": sparse_structure_steps},
            shape_slat_sampler_params={"steps": slat_sampler_steps},
            tex_slat_sampler_params={"steps": slat_sampler_steps},
        )[0]
        mesh.simplify(simplify_target)
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}

    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.2f}s")

    # Export mesh
    runpod.serverless.progress_update(job, "Exporting 3D model...")
    try:
        import o_voxel

        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation_target,
            texture_size=texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=True,
        )

        # Export to temp file
        with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as f:
            model_path = f.name

        if output_format == "glb":
            glb.export(model_path, extension_webp=True)
        else:
            glb.export(model_path)

        with open(model_path, "rb") as f:
            model_base64 = base64.b64encode(f.read()).decode("utf-8")

        os.unlink(model_path)

        # Generate thumbnail
        runpod.serverless.progress_update(job, "Generating thumbnail...")
        thumbnail_base64 = ""
        try:
            # Render a simple thumbnail from the mesh
            # For now, use input image as placeholder thumbnail
            thumb_buffer = BytesIO()
            image.thumbnail((256, 256))
            image.save(thumb_buffer, format="PNG")
            thumbnail_base64 = base64.b64encode(thumb_buffer.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"Thumbnail generation failed: {e}")

    except Exception as e:
        return {"error": f"Export failed: {str(e)}"}

    total_time = time.time() - start_time
    model_size_mb = len(model_base64) * 3 / 4 / 1024 / 1024
    print(f"Export completed. Model size: ~{model_size_mb:.2f}MB, Total time: {total_time:.2f}s")

    # Return in format expected by website
    return {
        "model": model_base64,
        "thumbnail": thumbnail_base64,
        "metadata": {
            "triangle_count": decimation_target,
            "texture_resolution": f"{texture_size}x{texture_size}",
            "generation_time_ms": int(total_time * 1000),
            "voxel_resolution": str(resolution),
        },
    }


# Initialize model at worker startup (RunPod best practice)
print("=" * 50)
print("TRELLIS.2 RunPod Worker Starting...")
print("=" * 50)

try:
    load_model()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Model pre-load failed: {e}")
    print("Model will be loaded on first request")

# Start RunPod serverless worker
runpod.serverless.start({"handler": handler})
