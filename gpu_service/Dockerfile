# CUDA + PyTorch runtime (needs NVIDIA Container Toolkit)
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WHISPER_MODEL_SIZE=large-v3 \
    WHISPER_DEVICE=cuda \
    WHISPER_COMPUTE_TYPE=float16 \
    DIARIZATION_MODEL=pyannote/speaker-diarization-3.1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
# Install torchvision/torchaudio first so pyannote/lightning do not pull incompatible wheels
RUN pip install --upgrade pip \
    && pip install -r requirements-docker.txt

COPY app.py transcription_utils.py ./

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
