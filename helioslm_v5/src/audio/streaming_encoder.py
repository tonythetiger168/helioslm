"""Streaming Audio Encoder - Real-time Speech Processing

Chunk-based processing with causal convolution and causal transformer
for low-latency streaming encoding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class StreamingAudioEncoder(nn.Module):
    """
    Streaming audio encoder with chunk-based processing.

    Features:
      - Causal convolution (no future dependency), with per-layer conv
        input-tail state carried across chunks.
      - Causal chunk-level transformer. Cross-chunk context is provided
        by prefixing each transformer layer with the cached *inputs* of
        that layer from previous chunks (memory prefix on the
        key/value side; queries are only computed for the current
        chunk). The attention mask is causal within the current chunk
        and makes the whole memory prefix visible, so streaming output
        is numerically identical (up to float error) to a single
        causal forward over the concatenated chunks.
      - Bounded memory: the per-layer memory prefix is a sliding window
        capped at ``max_memory_frames`` frames (config:
        ``multimodal.audio_max_memory_frames``, default 1000); older
        frames are dropped from the OLDEST side. While the total number
        of frames processed so far fits inside the window, streaming
        output is numerically identical to an unbounded/one-shot causal
        forward; once the stream exceeds the window, history older than
        the window is forgotten (sliding-window attention
        approximation), so long-range outputs may differ from the
        unbounded encoder.
      - Input is a *precomputed* mel spectrogram [1, n_mels, frames];
        this module does NOT compute mel spectrograms from waveform.

    State handling:
      - ``conv_state_i``: buffer of shape [1, C_in_i, kernel-1] holding
        the tail of the *input* of conv layer i (C_in_0 = n_mels,
        C_in_1 = hidden_size // 2). Registered as separate buffers
        because the per-layer input channel counts differ.
      - ``_memory``: python list of per-layer cached layer inputs,
        each [1, past_frames, hidden] with
        ``past_frames <= max_memory_frames`` (sliding window, see
        above). Call ``reset_state()`` between utterances.
    """

    def __init__(self, config):
        super().__init__()
        mm = config.multimodal
        self.n_mels = mm.audio_n_mels
        self.hidden_size = mm.audio_hidden_size
        self.hop_length = getattr(mm, "audio_hop_length", 160)
        self.sample_rate = getattr(mm, "audio_sample_rate", 16000)
        nhead = getattr(mm, "audio_num_heads", 8)
        num_layers = getattr(mm, "audio_num_layers", 4)
        # Sliding-window cap (frames) on the per-layer memory prefix.
        self.max_memory_frames = int(getattr(mm, "audio_max_memory_frames", 1000))

        # Causal convolution layers (no padding; left context from state)
        self.kernel_size = 3
        self.conv1 = nn.Conv1d(self.n_mels, self.hidden_size // 2,
                               kernel_size=self.kernel_size, padding=0, bias=False)
        self.conv2 = nn.Conv1d(self.hidden_size // 2, self.hidden_size,
                               kernel_size=self.kernel_size, padding=0, bias=False)
        self.convs = [self.conv1, self.conv2]

        # Per-layer causal conv state: [1, C_in, kernel-1], C_in = layer input channels
        self.register_buffer("conv_state_0", torch.zeros(1, self.n_mels, self.kernel_size - 1))
        self.register_buffer("conv_state_1", torch.zeros(1, self.hidden_size // 2, self.kernel_size - 1))

        # Causal chunk-level transformer (layers run one by one so that
        # per-layer memory prefixes can be applied; see class docstring).
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=nhead,
                dim_feedforward=self.hidden_size * 4,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(self.hidden_size)

        # Cross-chunk memory: list of per-layer cached inputs [1, T, H].
        # Plain attribute (not a buffer) because shapes grow over time.
        self._memory = None

    def reset_state(self):
        """Reset internal state for a new utterance."""
        self.conv_state_0.zero_()
        self.conv_state_1.zero_()
        self._memory = None

    def forward(self, mel_chunk: torch.Tensor):
        """
        Process a single chunk of precomputed mel spectrogram.

        Args:
            mel_chunk: [1, n_mels, chunk_frames] mel spectrogram chunk

        Returns:
            features: [1, chunk_frames, hidden]
        """
        x = mel_chunk

        # Causal convs with input-tail state carry
        for idx, conv in enumerate(self.convs):
            state = getattr(self, f"conv_state_{idx}").to(x.device)
            x = torch.cat([state, x], dim=-1)
            # Store the tail of the conv *input* for the next chunk
            # (channel count matches this layer's input channels).
            setattr(self, f"conv_state_{idx}", x[:, :, -(self.kernel_size - 1):].detach())
            x = F.gelu(conv(x))

        # Transpose for transformer
        x = x.transpose(1, 2)  # [1, frames, hidden]

        # Per-layer causal attention with memory prefix (previous chunks'
        # layer inputs on the key/value side, queries only for current
        # chunk). Mask: memory fully visible, current chunk causal.
        if self._memory is None or self._memory[0].device != x.device:
            self._memory = [torch.zeros(1, 0, self.hidden_size, device=x.device, dtype=x.dtype)
                            for _ in self.layers]

        for layer_idx, layer in enumerate(self.layers):
            mem = self._memory[layer_idx]
            num_mem = mem.shape[1]
            seq = torch.cat([mem, x], dim=1)
            total = seq.shape[1]
            # -inf above the diagonal: causal over [memory | current]
            mask = torch.triu(
                torch.full((total, total), float("-inf"), device=x.device, dtype=x.dtype),
                diagonal=1,
            )
            seq_out = layer(seq, src_mask=mask)
            # Cache this layer's *input* for future chunks (key/value
            # source), capped at max_memory_frames: when the window
            # overflows, the OLDEST frames are dropped (sliding window;
            # forgotten history no longer contributes to attention).
            new_mem = torch.cat([mem, x.detach()], dim=1)
            m = self.max_memory_frames
            if m <= 0:
                new_mem = new_mem[:, :0]
            elif new_mem.shape[1] > m:
                new_mem = new_mem[:, -m:]
            self._memory[layer_idx] = new_mem
            # Queries only for the current chunk
            x = seq_out[:, num_mem:]

        return self.norm(x)

    def process_stream(self, audio_stream: torch.Tensor, chunk_ms: int = 500):
        """
        Process a continuous precomputed mel spectrogram stream.

        Args:
            audio_stream: [1, n_mels, total_frames] mel spectrogram
            chunk_ms: chunk size in milliseconds. Frames per chunk are
                derived from sample_rate / hop_length
                (chunk_ms * sample_rate // (hop_length * 1000);
                500 ms = 50 frames at 16 kHz / hop 160).

        Yields:
            features: [1, chunk_frames, hidden] for each chunk
        """
        self.reset_state()

        frames_per_chunk = max(1, chunk_ms * self.sample_rate // (self.hop_length * 1000))
        total_frames = audio_stream.shape[-1]

        for start in range(0, total_frames, frames_per_chunk):
            end = min(start + frames_per_chunk, total_frames)
            chunk = audio_stream[:, :, start:end]
            yield self.forward(chunk)
