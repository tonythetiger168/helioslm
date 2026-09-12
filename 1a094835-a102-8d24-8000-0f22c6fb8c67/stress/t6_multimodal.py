"""T6: multimodal boundary tests (vision/audio prefixes, NaViT aspect ratios)."""
import sys
sys.path.insert(0, "/mnt/agents/output")
import torch
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5
from helioslm_v5.src.vision.navit import NaViTEncoder
from helioslm_v5.src.audio.streaming_encoder import StreamingAudioEncoder

def banner(s): print(f"\n=== {s} ===")

# lite config with small multimodal encoders enabled (lite turns them off by default)
cfg = HeliosLMv5Config(size="lite")
cfg.multimodal.enabled = True
cfg.multimodal.vision_patch_size = 4
cfg.multimodal.vision_hidden_size = 32
cfg.multimodal.vision_num_layers = 1
cfg.multimodal.vision_num_heads = 4
cfg.multimodal.vision_max_grid = 8
cfg.multimodal.audio_n_mels = 16
cfg.multimodal.audio_hidden_size = 32
cfg.multimodal.audio_num_layers = 1

torch.manual_seed(0)
model = HeliosLMv5(cfg).eval()

banner("1. vision prefix + forward vs text-only")
ids = torch.tensor([[3, 4, 5]])
img = torch.randn(1, 3, 8, 8)  # grid 2x2 = 4 patches
with torch.no_grad():
    logits_v, _, past_v = model(ids, images=img, use_cache=True)
    logits_t, _, _ = model(ids)
print("vision logits shape:", tuple(logits_v.shape), "(expect [1, 4+3, V])")
print("text  logits shape:", tuple(logits_t.shape))
print("prefix shifts text predictions (logits differ):",
      not torch.allclose(logits_v[:, -1], logits_t[:, -1]))

banner("2. vision prefix + cached decode continuation")
with torch.no_grad():
    cont = torch.tensor([[11, 12, 13]])
    # cached: decode cont after prefix
    outs = []
    past = past_v
    for t in range(3):
        lg, _, past = model(cont[:, t:t+1], past_key_values=past, use_cache=True)
        outs.append(lg[:, -1])
    cached = torch.stack(outs, 1)
    # full recompute reference: images + ids + cont in one forward
    full_ids = torch.cat([ids, cont], dim=1)
    lg_full, _, _ = model(full_ids, images=img)
    d = (cached - lg_full[:, -3:]).abs().max().item()
print(f"cached continuation vs full recompute max diff: {d:.3e} {'OK' if d < 1e-4 else 'FAIL'}")

banner("3. audio prefix + forward")
aud = torch.randn(1, 16, 20)  # [1, n_mels, frames]
with torch.no_grad():
    logits_a, _, past_a = model(ids, audio_features=aud, use_cache=True)
print("audio logits shape:", tuple(logits_a.shape), "finite:", torch.isfinite(logits_a).all().item())

banner("4. vision+audio together")
with torch.no_grad():
    logits_both, _, _ = model(ids, images=img, audio_features=aud)
print("both prefixes shape:", tuple(logits_both.shape), "(expect 1, 4+19+3, V) finite:",
      torch.isfinite(logits_both).all().item())

banner("5. NaViT extreme aspect ratios")
enc = model.vision_encoder
for h, w in [(4, 32), (32, 4), (4, 8), (8, 4), (4, 4)]:
    try:
        with torch.no_grad():
            f = enc(torch.randn(1, 3, h, w))
        print(f"  {h}x{w}: OK grid=({h//4},{w//4}) out={tuple(f.shape)}")
    except Exception as e:
        print(f"  {h}x{w}: EXC {type(e).__name__}: {e}")

banner("6. NaViT grid beyond max_grid raises clean error")
try:
    enc(torch.randn(1, 3, 36, 4))  # 9x1 > max_grid 8
    print("  NO ERROR (unexpected)")
except ValueError as e:
    print("  ValueError (expected):", str(e)[:80])

banner("7. NaViT image smaller than patch (H=2 < patch=4)")
try:
    f = enc(torch.randn(1, 3, 2, 8))
    print("  out:", tuple(f.shape), "finite:", torch.isfinite(f).all().item())
except Exception as e:
    print(f"  EXC {type(e).__name__}: {e}")

banner("8. forward_packed mixed sizes")
try:
    feats, mask = enc.forward_packed([torch.randn(3, 8, 8), torch.randn(3, 4, 32)])
    print("  packed:", tuple(feats.shape), "mask:", mask.sum(1).tolist(), "finite:", torch.isfinite(feats).all().item())
except Exception as e:
    print(f"  EXC {type(e).__name__}: {e}")

banner("9. audio streaming: chunked == one-shot causal")
torch.manual_seed(1)
mel = torch.randn(1, 16, 37)
enc_a = model.audio_encoder
enc_a.reset_state()
with torch.no_grad():
    one_shot = enc_a(mel)
    enc_a.reset_state()
    chunks = [enc_a(mel[:, :, :10]), enc_a(mel[:, :, 10:25]), enc_a(mel[:, :, 25:])]
    streamed = torch.cat(chunks, dim=1)
d = (one_shot - streamed).abs().max().item()
print(f"one-shot vs 3-chunk max diff: {d:.3e} {'OK' if d < 1e-4 else 'FAIL'}")
# without reset between utterances: state leaks?
enc_a.reset_state()
with torch.no_grad():
    a1 = enc_a(mel)
    a2 = enc_a(mel)   # NO reset: memory/conv state carries over
print("no-reset second call differs (state carried):",
      not torch.allclose(a1, a2), "(documented: reset_state() required)")

banner("10. audio chunk of 1 frame")
enc_a.reset_state()
with torch.no_grad():
    f1 = enc_a(torch.randn(1, 16, 1))
print("1-frame chunk:", tuple(f1.shape), "finite:", torch.isfinite(f1).all().item())
print("DONE")
