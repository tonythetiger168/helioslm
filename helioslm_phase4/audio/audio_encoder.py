"""Audio encoder using mel-spectrogram + transformer."""
import torch
import torch.nn as nn


class SpectrogramExtractor(nn.Module):
    """Extract mel-spectrogram from raw audio."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Mel filterbank (learnable or fixed)
        self.mel_scale = nn.Parameter(
            self._create_mel_filterbank(sample_rate, n_fft, n_mels),
            requires_grad=False,
        )

    def _create_mel_filterbank(self, sr, n_fft, n_mels):
        """Create mel filterbank matrix."""
        import numpy as np

        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        f_min, f_max = 0, sr // 2
        mel_min, mel_max = hz_to_mel(f_min), hz_to_mel(f_max)
        mels = np.linspace(mel_min, mel_max, n_mels + 2)
        hz = mel_to_hz(mels)

        bins = np.floor((n_fft + 1) * hz / sr).astype(int)

        fbank = np.zeros((n_mels, n_fft // 2 + 1))
        for i in range(n_mels):
            for j in range(bins[i], bins[i + 1]):
                fbank[i, j] = (j - bins[i]) / (bins[i + 1] - bins[i])
            for j in range(bins[i + 1], bins[i + 2]):
                fbank[i, j] = (bins[i + 2] - j) / (bins[i + 2] - bins[i + 1])

        return torch.from_numpy(fbank).float()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [B, T] raw audio

        Returns:
            mel_spec: [B, n_mels, time_frames]
        """
        # STFT
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True,
        )

        # Power spectrogram
        spec = stft.abs() ** 2  # [B, freq, time]

        # Apply mel filterbank
        mel_spec = torch.matmul(self.mel_scale.to(spec.device), spec)  # [B, n_mels, time]

        # Log compression
        mel_spec = torch.log(mel_spec + 1e-9)

        return mel_spec


class AudioEncoder(nn.Module):
    """
    Audio transformer encoder.

    Similar to Whisper encoder: mel-spectrogram -> conv -> transformer.
    """

    def __init__(
        self,
        n_mels: int = 80,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
    ):
        super().__init__()

        # Spectrogram extractor
        self.spec_extractor = SpectrogramExtractor(n_mels=n_mels)

        # Conv frontend (like Whisper)
        self.conv1 = nn.Conv1d(n_mels, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 1500, embed_dim))  # Max 30s audio

        # Transformer
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Project to language model
        self.proj_to_lm = nn.Linear(embed_dim, 12288)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [B, T] raw audio waveform

        Returns:
            features: [B, seq_len, lm_hidden_size]
        """
        # Extract mel spectrogram
        mel_spec = self.spec_extractor(waveform)  # [B, n_mels, time]

        # Conv frontend
        x = self.conv1(mel_spec)  # [B, embed_dim, time]
        x = torch.relu(x)
        x = self.conv2(x)  # [B, embed_dim, time/2]
        x = torch.relu(x)

        # Transpose for transformer
        x = x.transpose(1, 2)  # [B, time/2, embed_dim]

        # Add positional embedding
        x = x + self.pos_embed[:, :x.shape[1], :]

        # Transformer
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Project to LM dimension
        return self.proj_to_lm(x)
