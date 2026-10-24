"""HeliosLM v5 test cases — streaming audio + NaViT vision.

Split out of the monolithic test_v5.py on 2026-10-11; the test_v5.py
runner imports every case module and executes the tests in the
historical order with unchanged exit-code semantics. The case_*.py
filename deliberately avoids pytest's default test_*.py collection
pattern so the CI pytest step does not double-run these.
"""

import torch

from evastar.configs.config_v5 import EvastarV5Config
from evastar.tests.case_common import (_expect_raises, _pass)


# ----------------------------------------------------------------------
# Streaming audio: chunked == one-shot; causality; process_stream chunking
# ----------------------------------------------------------------------
def test_streaming_audio():
    from evastar.src.audio.streaming_encoder import StreamingAudioEncoder

    config = EvastarV5Config(size="lite")
    config.multimodal.audio_hidden_size = 128
    config.multimodal.audio_num_layers = 2
    torch.manual_seed(112)
    enc = StreamingAudioEncoder(config).eval()
    n_mels = config.multimodal.audio_n_mels
    mel = torch.randn(1, n_mels, 60)

    with torch.no_grad():
        enc.reset_state()
        full = enc(mel)
        enc.reset_state()
        chunks = [enc(mel[:, :, i:i + 20]) for i in range(0, 60, 20)]
        streamed = torch.cat(chunks, dim=1)
    diff = (full - streamed).abs().max().item()
    assert full.shape == streamed.shape == (1, 60, 128)
    assert diff < 1e-3, f"streaming != one-shot: max|diff|={diff:.3e}"

    # Causality: changing frames >= 40 must not change outputs < 40.
    mel2 = mel.clone()
    mel2[:, :, 40:] = torch.randn(1, n_mels, 20)
    with torch.no_grad():
        enc.reset_state()
        full2 = enc(mel2)
    assert torch.allclose(full[:, :40], full2[:, :40], atol=1e-6), \
        "future audio frames leaked into past outputs"

    # process_stream: 500 ms chunks at 16 kHz / hop 160 -> 50 frames per chunk.
    mel3 = torch.randn(1, n_mels, 120)
    with torch.no_grad():
        outs = list(enc.process_stream(mel3, chunk_ms=500))
    assert [o.shape[1] for o in outs] == [50, 50, 20], \
        f"chunk frame counts {[o.shape[1] for o in outs]} != [50, 50, 20]"
    _pass("test_streaming_audio",
          f"3-chunk max|diff|={diff:.2e}, causal, 500ms->50 frames/chunk")


# ----------------------------------------------------------------------
# v5.2: streaming audio sliding-window memory cap
# ----------------------------------------------------------------------
def test_audio_sliding_window():
    from evastar.src.audio.streaming_encoder import StreamingAudioEncoder

    config = EvastarV5Config(size="lite")
    config.multimodal.audio_hidden_size = 128
    config.multimodal.audio_num_layers = 2
    config.multimodal.audio_max_memory_frames = 90
    torch.manual_seed(122)
    enc = StreamingAudioEncoder(config).eval()
    assert enc.max_memory_frames == 90
    n_mels = config.multimodal.audio_n_mels
    chunk = 20

    # 1) Within the window (4 chunks x 20 = 80 frames <= 90): streaming is
    #    identical to a one-shot forward and memory is NOT truncated.
    mel = torch.randn(1, n_mels, 4 * chunk)
    with torch.no_grad():
        enc.reset_state()
        full = enc(mel)
        enc.reset_state()
        streamed = torch.cat(
            [enc(mel[:, :, i:i + chunk]) for i in range(0, 80, chunk)], dim=1)
        in_window_lens = [m.shape[1] for m in enc._memory]
    diff = (full - streamed).abs().max().item()
    assert diff < 1e-4, f"in-window streaming != one-shot: max|diff|={diff:.3e}"
    assert all(l == 80 for l in in_window_lens), \
        f"in-window memory must be uncapped (80), got {in_window_lens}"

    # 2) Beyond the window: 10 chunks x 20 = 200 frames; every layer's
    #    memory must stay capped at 90 (oldest frames dropped).
    mel_long = torch.randn(1, n_mels, 10 * chunk)
    with torch.no_grad():
        enc.reset_state()
        for i in range(0, 200, chunk):
            out = enc(mel_long[:, :, i:i + chunk])
            lens = [m.shape[1] for m in enc._memory]
            assert all(l <= 90 for l in lens), \
                f"memory exceeded window at frame {i + chunk}: {lens}"
        assert out.shape == (1, chunk, 128) and torch.isfinite(out).all()
        assert all(m.shape[1] == 90 for m in enc._memory), \
            f"memory should be pinned at the window: " \
            f"{[m.shape[1] for m in enc._memory]}"

    # 3) reset_state restarts cleanly: a fresh forward reproduces the
    #    pre-overflow one-shot output.
    enc.reset_state()
    assert enc._memory is None, "reset_state must clear the memory prefix"
    with torch.no_grad():
        again = enc(mel)
    restart_diff = (again - full).abs().max().item()
    assert restart_diff < 1e-4, \
        f"post-reset forward diverged: {restart_diff:.3e}"
    _pass("test_audio_sliding_window",
          f"memory capped at 90 over 200 frames, in-window "
          f"stream-vs-oneshot max|diff|={diff:.2e}, reset_state OK")


# ----------------------------------------------------------------------
# NaViT: 224x224 works, over-grid raises, forward_packed mixed sizes
# ----------------------------------------------------------------------
def test_navit():
    from evastar.src.vision.navit import NaViTEncoder

    config = EvastarV5Config(size="lite")
    config.multimodal.vision_hidden_size = 128
    config.multimodal.vision_num_layers = 2
    config.multimodal.vision_num_heads = 4
    torch.manual_seed(113)
    enc = NaViTEncoder(config).eval()
    assert enc.max_grid == 32  # lite default
    # Factorized 2D position embedding: separate row/col tables.
    assert enc.row_embed.weight.shape == (32, 128)
    assert enc.col_embed.weight.shape == (32, 128)

    img = torch.randn(1, 3, 224, 224)  # 16x16 = 256 patches, within grid
    with torch.no_grad():
        out = enc(img)
    assert out.shape == (1, 256, 128) and torch.isfinite(out).all()

    # 512x512 -> 36x36 grid > max_grid=32 -> ValueError.
    big = torch.randn(1, 3, 512, 512)
    _expect_raises(ValueError, lambda: enc(big), "over-grid image")

    # forward_packed: mixed sizes padded to the longest sequence with a
    # boolean mask; the un-padded row must match a solo forward.
    imgs = [torch.randn(3, 224, 224), torch.randn(3, 112, 112)]  # 256 / 64 patches
    with torch.no_grad():
        feats, mask = enc.forward_packed(imgs)
    assert feats.shape == (2, 256, 128)
    assert mask.dtype == torch.bool
    assert int(mask[0].sum()) == 256 and bool(mask[0].all())
    assert int(mask[1].sum()) == 64 and not bool(mask[1, 64:].any())
    with torch.no_grad():
        solo = enc(imgs[0].unsqueeze(0))
    pack_diff = (feats[0][mask[0]] - solo[0]).abs().max().item()
    assert pack_diff < 1e-4, \
        f"padded batching changed unpadded row outputs: {pack_diff:.3e}"
    _pass("test_navit",
          f"224px OK (256 patches), 512px@grid32 raises, packed mask exact, "
          f"packed-vs-solo max|diff|={pack_diff:.2e}")


# ----------------------------------------------------------------------
# v5.4 (LOW): NaViT rejects degenerate inputs with a clear ValueError
# ----------------------------------------------------------------------
def test_navit_input_validation():
    from evastar.src.vision.navit import NaViTEncoder

    config = EvastarV5Config(size="lite")
    config.multimodal.vision_hidden_size = 128
    config.multimodal.vision_num_layers = 2
    config.multimodal.vision_num_heads = 4
    torch.manual_seed(135)
    enc = NaViTEncoder(config).eval()
    p = enc.patch_size  # 14

    # Image smaller than the patch size produces zero patches; this must
    # raise a clear ValueError (not a cryptic Conv2d shape error).
    _expect_raises(ValueError, lambda: enc(torch.randn(1, 3, p - 1, 224)),
                   "forward with H < patch_size")
    _expect_raises(ValueError, lambda: enc(torch.randn(1, 3, 224, p - 1)),
                   "forward with W < patch_size")

    # forward_packed: empty list and sub-patch member images raise ValueError
    # (previously: IndexError on all_patches[0] / cryptic Conv2d error).
    _expect_raises(ValueError, lambda: enc.forward_packed([]),
                   "forward_packed([])")
    _expect_raises(ValueError,
                   lambda: enc.forward_packed(
                       [torch.randn(3, 224, 224), torch.randn(3, p - 1, 224)]),
                   "forward_packed with a sub-patch image")
    _pass("test_navit_input_validation",
          "small image raises in forward/forward_packed, empty list raises")
