"""Audio preprocessing utilities."""
import torch
import torchaudio
from typing import Union, List


class AudioProcessor:
    """Preprocess audio for encoder input."""

    def __init__(self, sample_rate: int = 16000, max_length_sec: float = 30.0):
        self.sample_rate = sample_rate
        self.max_length = int(max_length_sec * sample_rate)

    def load(self, audio_path: str) -> torch.Tensor:
        """Load audio file and resample."""
        waveform, sr = torchaudio.load(audio_path)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Pad or truncate
        if waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
        else:
            pad = self.max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        return waveform.squeeze(0)  # [T]

    def load_batch(self, paths: List[str]) -> torch.Tensor:
        """Load batch of audio files."""
        waveforms = [self.load(p) for p in paths]
        return torch.stack(waveforms)  # [B, T]
