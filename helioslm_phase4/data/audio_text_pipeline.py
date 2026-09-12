"""Audio-text alignment for audio-language pretraining."""
import json
from pathlib import Path
from typing import Iterator, Dict, List, Optional
import torch


class AudioTextAligner:
    """
    Align audio segments with text transcripts.

    Sources:
      - ASR transcripts (Whisper, etc.)
      - Music captions (MusicCaps)
      - Sound event descriptions (AudioSet)
    """

    def __init__(self, max_audio_length_sec: float = 30.0, sample_rate: int = 16000):
        self.max_audio_length = max_audio_length_sec
        self.sample_rate = sample_rate

    def align_transcript(self, audio_path: str, transcript: str, timestamps: Optional[List] = None) -> List[Dict]:
        """
        Align transcript with audio using timestamps.

        Returns segments: [{"audio_start": 0, "audio_end": 10, "text": "..."}]
        """
        if timestamps:
            segments = []
            for start, end, text in timestamps:
                segments.append({
                    "audio_path": audio_path,
                    "audio_start": start,
                    "audio_end": end,
                    "text": text,
                })
            return segments
        else:
            # No timestamps: use entire audio with full transcript
            return [{
                "audio_path": audio_path,
                "audio_start": 0,
                "audio_end": self.max_audio_length,
                "text": transcript,
            }]


class SpeechCaptionDataset:
    """Dataset for speech-to-text and audio captioning."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def stream(self) -> Iterator[Dict]:
        """Stream audio-text samples."""
        for jsonl_file in self.data_dir.glob("*.jsonl"):
            with open(jsonl_file, "r") as f:
                for line in f:
                    try:
                        sample = json.loads(line)

                        # Validate
                        if "audio_path" not in sample or "text" not in sample:
                            continue

                        yield sample
                    except Exception:
                        continue
