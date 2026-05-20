# Sales Call Batch Transcription (GPU API + Laptop Client)

This setup is split into two parts:

1. `gpu_service/` runs Whisper + Pyannote on your GPU machine and exposes an HTTP endpoint.
2. `laptop_client/` scans `E:\Google Drive`, creates a manifest, sends calls in batches, and stores per-call JSON outputs.

## 1) GPU Machine Setup

### Recommended: Docker (`gpu_service/`)

See **`gpu_service/DOCKER.md`** for full steps. Summary:

```bash
cd gpu_service
cp .env.example .env   # set HUGGINGFACE_TOKEN
docker compose up -d --build
curl http://localhost:8000/health
```

Laptop uses: `http://<GPU_SERVER_IP>:8000`

### Alternative: bare Python on GPU server

### Files used
- `gpu_service/app.py`
- `gpu_service/requirements.txt`

### Commands to run on GPU server
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r gpu_service/requirements.txt

# Required for Pyannote diarization:
export HUGGINGFACE_TOKEN="your_hf_token"

# Optional model settings:
export WHISPER_MODEL_SIZE="large-v3"
export WHISPER_DEVICE="cuda"
export WHISPER_COMPUTE_TYPE="float16"

uvicorn gpu_service.app:app --host 0.0.0.0 --port 8000
```

### API endpoints
- `GET /health`
- `POST /transcribe` (multipart form)
  - file: audio file
  - call_id: unique id for call
  - language: e.g. `en`, `hi`

## 2) Laptop Setup

### Files used
- `laptop_client/build_manifest.py`
- `laptop_client/process_batch.py`
- `laptop_client/check_api.py`
- `laptop_client/requirements.txt`

### Commands to run on laptop (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r laptop_client\requirements.txt

# 1) Build manifest from SSD root (recursive + zip members)
python laptop_client\build_manifest.py --root "E:\Google Drive" --out ".\data\manifest.jsonl"

# 2) Check GPU API connectivity
python laptop_client\check_api.py --api-url "http://<GPU_IP>:8000"

# 3) Process first batch (example: 100 calls)
python laptop_client\process_batch.py `
  --manifest ".\data\manifest.jsonl" `
  --state ".\data\progress.json" `
  --out-root ".\data\transcripts" `
  --batch-size 100 `
  --api-url "http://<GPU_IP>:8000" `
  --language "en" `
  --timeout 3600
```

Run step 3 repeatedly for next batches until pending calls are done.

## Output Folder Structure

For each call:

```text
data/
  transcripts/
    <call_id>/
      transcript.json   # whisper + diarization structured output
      source.json       # original source metadata (file or zip member)
```

## Notes

- Manifest includes:
  - regular audio files in nested subfolders
  - audio files inside `.zip` archives
- `progress.json` prevents reprocessing already completed calls.
- For zip members, audio is extracted to temp only during processing, then deleted.
- You can safely delete original recordings only after you verify desired call folders in `data/transcripts`.
