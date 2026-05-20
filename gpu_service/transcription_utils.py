from typing import Any, Dict, List, Optional


def segments_to_dict(segments: Any) -> List[Dict[str, Any]]:
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


def diarization_turns_from_annotation(diarization: Any) -> List[Dict[str, Any]]:
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


def best_speaker_for_segment(start: float, end: float, turns: List[Dict[str, Any]]) -> str:
    best_speaker = "UNKNOWN"
    best_overlap = 0.0
    for turn in turns:
        overlap = max(0.0, min(end, turn["end"]) - max(start, turn["start"]))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn["speaker"]
    return best_speaker


def attach_speakers_to_segments(
    whisper_segments: List[Dict[str, Any]], diarization_turns: List[Dict[str, Any]]
) -> None:
    for segment in whisper_segments:
        segment["speaker"] = best_speaker_for_segment(
            segment["start"], segment["end"], diarization_turns
        )


def build_transcript_response(
    *,
    call_id: str,
    language: str,
    duration_seconds: float,
    source_filename: Optional[str],
    diarization_enabled: bool,
    whisper_segments: List[Dict[str, Any]],
    diarization_turns: List[Dict[str, Any]],
) -> Dict[str, Any]:
    transcript_text = " ".join([s["text"] for s in whisper_segments if s["text"]])
    speakers = sorted({turn["speaker"] for turn in diarization_turns}) if diarization_turns else []
    return {
        "call_id": call_id,
        "language": language,
        "duration_seconds": round(duration_seconds, 3),
        "source_filename": source_filename,
        "diarization_enabled": diarization_enabled,
        "speakers": speakers,
        "segments": whisper_segments,
        "diarization_turns": diarization_turns,
        "full_text": transcript_text,
    }
