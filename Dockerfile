# TRELLIS.2 RunPod Serverless
# - Minimal image (no model weights baked in)
# - Uses RunPod's built-in model caching

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV HF_HOME=/runpod-volume/huggingface-cache
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
# Set CUDA architectures for compilation (8.0=A100, 9.0=H100)
ENV TORCH_CUDA_ARCH_LIST="8.0;9.0"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libjpeg-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Remove conflicting distutils packages
RUN rm -rf /usr/lib/python3/dist-packages/blinker* || true

# Install build tools first (required for nvdiffrast)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel ninja

# Install Python dependencies
RUN pip install --no-cache-dir \
    runpod \
    huggingface_hub \
    accelerate \
    safetensors \
    imageio \
    imageio-ffmpeg \
    pillow \
    opencv-python-headless \
    trimesh \
    easydict \
    einops \
    kornia \
    timm \
    omegaconf \
    xformers \
    tqdm \
    lpips \
    transformers \
    plyfile \
    pymeshlab \
    open3d \
    git+https://github.com/EasternJournalist/utils3d.git

# Install nvdiffrast (CUDA extension - requires setuptools, wheel, ninja)
RUN pip install --no-cache-dir --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git

# Install flash-attn (optional - improves performance)
RUN pip install --no-cache-dir flash-attn==2.7.3 || true

# Clone TRELLIS.2
RUN git clone --recursive https://github.com/microsoft/TRELLIS.2.git /app/TRELLIS.2

# Add TRELLIS.2 to Python path (it's not a pip package)
ENV PYTHONPATH="/app/TRELLIS.2:${PYTHONPATH}"
ENV TRELLIS_PATH=/app/TRELLIS.2

# Copy handler
COPY handler.py /app/handler.py

CMD ["python3", "-u", "handler.py"]
