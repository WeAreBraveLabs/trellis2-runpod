# TRELLIS.2 RunPod Serverless Deployment

Deploy Microsoft's TRELLIS.2 image-to-3D model on RunPod Serverless.

## Architecture (Following RunPod Best Practices)

- **Minimal Docker image** - No model weights baked in (~5GB vs 30GB+)
- **Model caching** - RunPod caches model on first run, subsequent starts are fast
- **FlashBoot compatible** - Enable for faster cold starts
- **Model loaded at startup** - Not reloaded per request

## Deployment Steps

### 1. Build Docker Image

Since we're on Mac, use GitHub Actions or a Linux VM to build:

**Option A: Build locally (if Docker works)**
```bash
docker build --platform linux/amd64 -t wearebravelabs/trellis2-runpod:latest .
docker push wearebravelabs/trellis2-runpod:latest
```

**Option B: Use GitHub Actions**
Push to GitHub and the workflow will build and push automatically.

### 2. Create RunPod Serverless Endpoint

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click **New Endpoint**
3. Configure:

| Setting | Value |
|---------|-------|
| **Container Image** | `wearebravelabs/trellis2-runpod:latest` |
| **Model** | `microsoft/TRELLIS.2-4B` (enables caching) |
| **GPU** | H100 PRO (80GB) - fastest |
| **Active Workers** | 0 (pay only when used) |
| **Max Workers** | 3 |
| **Idle Timeout** | 5 seconds |
| **Execution Timeout** | 300000ms (5 min) |
| **FlashBoot** | Enabled (recommended) |

### 3. Test the Endpoint

```bash
export RUNPOD_API_KEY="your-api-key"
export ENDPOINT_ID="your-endpoint-id"

# Submit job
curl -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/run" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "input_image": "https://example.com/your-image.png",
      "seed": 42,
      "pipeline_type": "512"
    }
  }'

# Check status
curl "https://api.runpod.ai/v2/${ENDPOINT_ID}/status/JOB_ID" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}"
```

## API Reference

### Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_image` | string | required | URL or base64 encoded image |
| `seed` | int | random | Random seed for reproducibility |
| `pipeline_type` | string | "512" | "512", "1024_cascade", "1536_cascade" |
| `texture_size` | int | 1024 | Output texture resolution |
| `decimation_target` | int | 20000 | Target triangle count |

### Output

```json
{
  "glb": "base64-encoded-glb-file",
  "format": "glb",
  "seed": 42,
  "pipeline_type": "512",
  "inference_time": 15.5,
  "glb_size_mb": 2.5
}
```

## Cost Estimates

| GPU | Resolution | Time | Cost/Generation |
|-----|------------|------|-----------------|
| H100 | 512³ | ~3 sec | ~$0.003 |
| H100 | 1024³ | ~17 sec | ~$0.02 |
| H100 | 1536³ | ~60 sec | ~$0.07 |
| A6000 | 512³ | ~10 sec | ~$0.003 |
| A6000 | 1024³ | ~60 sec | ~$0.02 |

**First cold start**: Model download adds ~5 min (one-time, cached after)

## Troubleshooting

### Out of Memory
- Use `pipeline_type: "512"`
- Reduce `texture_size` to 512

### Slow Cold Starts
- Enable FlashBoot
- Set 1 Active Worker (costs more but no cold starts)

### Model Loading Errors
- Check GPU has 24GB+ VRAM
- Verify model is specified in endpoint settings for caching

