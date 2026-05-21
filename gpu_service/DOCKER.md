# GPU Transcription API — Docker (for CTO / GPU server)

Hand off the **`gpu_service/`** folder. CTO runs these commands on a Linux machine with NVIDIA GPU + Docker.

## Prerequisites (one-time on GPU server)

1. **NVIDIA driver** installed (`nvidia-smi` works).
2. **Docker** + **Docker Compose v2**.
3. **NVIDIA Container Toolkit** so containers can use the GPU:
   - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
4. Verify GPU in Docker:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```
5. **Hugging Face token** with access to Pyannote models (see `.env.example` links).

## Quick start

```bash
cd gpu_service

cp .env.example .env
# Edit .env — set HUGGINGFACE_TOKEN (required for diarization)

docker compose up -d --build
docker compose logs -f
```

First start downloads Whisper + Pyannote weights (can take several minutes). Wait until logs show Uvicorn running.

## Health check

```bash
curl http://localhost:8000/health
```

Expected:

```json
{"status":"ok","diarization_enabled":true}
```

If `diarization_enabled` is `false`, check `HUGGINGFACE_TOKEN` in `.env` and HF model license acceptance.

## Stop / restart

```bash
docker compose down
docker compose up -d
```

## Laptop connection

Use the GPU server IP instead of `localhost`:

```text
http://<GPU_SERVER_IP>:8000
```

Example from your laptop:

```powershell
python laptop_client\verify_setup.py --api-url "http://10.0.0.50:8000"
```

Open firewall port **8000** on the GPU server if the laptop is on another machine.

## Smaller GPU / faster startup

Edit `.env` before build:

```env
WHISPER_MODEL_SIZE=medium
WHISPER_COMPUTE_TYPE=int8_float16
```

Then rebuild:

```bash
docker compose up -d --build
```

## Verify API

After the container is running:

```bash
pip install requests   # if needed on the machine running the check
python verify_setup.py --api-url http://localhost:8000
```

## Troubleshooting

| Issue | Fix |
|--------|-----|
| `could not select device driver` | Install NVIDIA Container Toolkit, restart Docker |
| `CUDA out of memory` | Use smaller `WHISPER_MODEL_SIZE` or `WHISPER_COMPUTE_TYPE=int8_float16` |
| Pyannote 401 / gated model | Set token in `.env`, accept HF model terms |
| `torchvision` / `extension` AttributeError on start | Rebuild after pulling latest `requirements-docker.txt` (pins torchvision to match base image). If you change the `FROM pytorch/...` tag, update the `torchvision` / `torchaudio` pins in that file too. |
| Slow first request | Normal — models load on container start |

## Files in this package

| File | Purpose |
|------|---------|
| `Dockerfile` | CUDA image + FFmpeg + API |
| `docker-compose.yml` | One-command run with GPU + model cache volume |
| `.env.example` | Config template |
| `app.py` | FastAPI Whisper + Pyannote service |
| `verify_setup.py` | Quick `/health` + `/transcribe` check |
