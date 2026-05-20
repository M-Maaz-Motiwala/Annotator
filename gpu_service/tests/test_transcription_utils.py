from types import SimpleNamespace
from unittest.mock import MagicMock

from transcription_utils import (
    attach_speakers_to_segments,
    best_speaker_for_segment,
    build_transcript_response,
    diarization_turns_from_annotation,
    segments_to_dict,
)


def test_segments_to_dict_rounds_and_strips():
    segments = [
        SimpleNamespace(start=0.1234, end=1.9876, text="  hello  "),
        SimpleNamespace(start=2.0, end=3.0, text="world"),
    ]
    out = segments_to_dict(segments)
    assert out == [
        {"start": 0.123, "end": 1.988, "text": "hello"},
        {"start": 2.0, "end": 3.0, "text": "world"},
    ]


def test_best_speaker_for_segment_picks_max_overlap():
    turns = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 5.0, "speaker": "SPEAKER_01"},
    ]
    assert best_speaker_for_segment(0.5, 1.5, turns) == "SPEAKER_00"
    assert best_speaker_for_segment(2.1, 4.0, turns) == "SPEAKER_01"
    assert best_speaker_for_segment(9.0, 10.0, turns) == "UNKNOWN"


def test_attach_speakers_to_segments():
    whisper_segments = [{"start": 0.0, "end": 2.0, "text": "hi"}]
    turns = [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]
    attach_speakers_to_segments(whisper_segments, turns)
    assert whisper_segments[0]["speaker"] == "SPEAKER_00"


def test_diarization_turns_from_annotation():
    turn_a = SimpleNamespace(start=0.0, end=1.0)
    turn_b = SimpleNamespace(start=1.0, end=2.5)
    diarization = MagicMock()
    diarization.itertracks.return_value = [
        (turn_a, None, "SPEAKER_00"),
        (turn_b, None, "SPEAKER_01"),
    ]
    turns = diarization_turns_from_annotation(diarization)
    assert turns == [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 2.5, "speaker": "SPEAKER_01"},
    ]


def test_build_transcript_response_shape():
    response = build_transcript_response(
        call_id="call_1",
        language="en",
        duration_seconds=10.5,
        source_filename="demo.wav",
        diarization_enabled=True,
        whisper_segments=[{"start": 0.0, "end": 1.0, "text": "hello", "speaker": "SPEAKER_00"}],
        diarization_turns=[{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}],
    )
    assert response["call_id"] == "call_1"
    assert response["full_text"] == "hello"
    assert response["speakers"] == ["SPEAKER_00"]
    assert response["duration_seconds"] == 10.5
