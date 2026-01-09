# TRELLIS.2 RunPod Serverless
# - Minimal image (no model weights baked in)
# - Uses RunPod's built-in model caching

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV OPENCV_IO_ENABLE_OPENEXR=1
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
    git \
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

# Install nvdiffrast v0.4.0 (CUDA extension - requires setuptools, wheel, ninja)
RUN git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast \
    && pip install --no-cache-dir --no-build-isolation /tmp/nvdiffrast \
    && rm -rf /tmp/nvdiffrast

# Install nvdiffrec (renderutils) - using JeffreyXiang's fork as required by TRELLIS.2
RUN git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/nvdiffrec \
    && pip install --no-cache-dir --no-build-isolation /tmp/nvdiffrec \
    && rm -rf /tmp/nvdiffrec

# Install cumesh (CUDA mesh operations)
RUN git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/CuMesh --recursive \
    && pip install --no-cache-dir --no-build-isolation /tmp/CuMesh \
    && rm -rf /tmp/CuMesh

# Install FlexGEMM (CUDA GEMM operations)
RUN git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/FlexGEMM --recursive \
    && pip install --no-cache-dir --no-build-isolation /tmp/FlexGEMM \
    && rm -rf /tmp/FlexGEMM

# Install flash-attn (improves performance)
RUN pip install --no-cache-dir --no-build-isolation flash-attn==2.7.3

# Clone TRELLIS.2
RUN git clone --recursive https://github.com/microsoft/TRELLIS.2.git /app/TRELLIS.2

# Install o-voxel from TRELLIS.2
RUN pip install --no-cache-dir --no-build-isolation /app/TRELLIS.2/o-voxel

# Add TRELLIS.2 to Python path (it's not a pip package)
ENV PYTHONPATH="/app/TRELLIS.2:${PYTHONPATH}"
ENV TRELLIS_PATH=/app/TRELLIS.2

# Copy handler
COPY handler.py /app/handler.py

CMD ["python3", "-u", "handler.py"]
