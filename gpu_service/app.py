import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

from transcription_utils import (
    attach_speakers_to_segments,
    build_transcript_response,
    diarization_turns_from_annotation,
    segments_to_dict,
)


MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3")
DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
DIARIZATION_MODEL = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
SKIP_MODEL_LOAD = os.getenv("SKIP_MODEL_LOAD") == "1"

whisper_model: Optional[WhisperModel] = None
diarization_pipeline: Optional[Pipeline] = None


def ensure_models_loaded() -> None:
    global whisper_model, diarization_pipeline
    if SKIP_MODEL_LOAD or whisper_model is not None:
        return

    whisper_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    if HF_TOKEN:
        try:
            diarization_pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=HF_TOKEN)
        except TypeError:
            diarization_pipeline = Pipeline.from_pretrained(
                DIARIZATION_MODEL, use_auth_token=HF_TOKEN
            )


def _diarize_file(audio_path: str) -> List[Dict[str, Any]]:
    if diarization_pipeline is None:
        return []
    diarization = diarization_pipeline(audio_path)
    return diarization_turns_from_annotation(diarization)


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_models_loaded()
    yield


app = FastAPI(title="Sales Call Transcription API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "models_loaded": whisper_model is not None,
        "diarization_enabled": diarization_pipeline is not None,
        "diarization_configured": bool(HF_TOKEN),
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    call_id: str = Form(...),
    language: str = Form("en"),
) -> Dict[str, Any]:
    ensure_models_loaded()
    if whisper_model is None:
        raise HTTPException(status_code=503, detail="Models are not loaded")

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        content = await file.read()
        tmp.write(content)

    try:
        segments, info = whisper_model.transcribe(
            tmp_path,
            language=language,
            vad_filter=True,
            beam_size=5,
        )
        whisper_segments = segments_to_dict(segments)
        diarization_turns = _diarize_file(tmp_path)
        attach_speakers_to_segments(whisper_segments, diarization_turns)

        return build_transcript_response(
            call_id=call_id,
            language=info.language,
            duration_seconds=info.duration,
            source_filename=file.filename,
            diarization_enabled=diarization_pipeline is not None,
            whisper_segments=whisper_segments,
            diarization_turns=diarization_turns,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.post("/transcribe/json")
async def transcribe_json(payload: Dict[str, Any]) -> Dict[str, Any]:
    raise HTTPException(
        status_code=405,
        detail="Use multipart endpoint /transcribe with file upload and fields call_id, language",
    )
