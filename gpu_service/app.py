import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline


MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3")
DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
DIARIZATION_MODEL = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")

app = FastAPI(title="Sales Call Transcription API", version="1.0.0")

whisper_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
if HF_TOKEN:
    try:
        diarization_pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=HF_TOKEN)
    except TypeError:
        diarization_pipeline = Pipeline.from_pretrained(
            DIARIZATION_MODEL, use_auth_token=HF_TOKEN
        )
else:
    diarization_pipeline = None


def _segments_to_dict(segments: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for segment in segments:
        out.append(
            {
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "text": segment.text.strip(),
            }
        )
    return out


def _diarize_file(audio_path: str) -> List[Dict[str, Any]]:
    if diarization_pipeline is None:
        return []

    diarization = diarization_pipeline(audio_path)
    turns: List[Dict[str, Any]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append(
            {
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "speaker": speaker,
            }
        )
    return turns


def _best_speaker_for_segment(start: float, end: float, turns: List[Dict[str, Any]]) -> str:
    best_speaker = "UNKNOWN"
    best_overlap = 0.0
    for turn in turns:
        overlap = max(0.0, min(end, turn["end"]) - max(start, turn["start"]))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn["speaker"]
    return best_speaker


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "diarization_enabled": diarization_pipeline is not None}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    call_id: str = Form(...),
    language: str = Form("en"),
) -> Dict[str, Any]:
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
        whisper_segments = _segments_to_dict(segments)
        diarization_turns = _diarize_file(tmp_path)

        for segment in whisper_segments:
            segment["speaker"] = _best_speaker_for_segment(
                segment["start"], segment["end"], diarization_turns
            )

        transcript_text = " ".join([s["text"] for s in whisper_segments if s["text"]])
        speakers = sorted({turn["speaker"] for turn in diarization_turns}) if diarization_turns else []

        return {
            "call_id": call_id,
            "language": info.language,
            "duration_seconds": round(info.duration, 3),
            "source_filename": file.filename,
            "diarization_enabled": diarization_pipeline is not None,
            "speakers": speakers,
            "segments": whisper_segments,
            "diarization_turns": diarization_turns,
            "full_text": transcript_text,
        }
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
