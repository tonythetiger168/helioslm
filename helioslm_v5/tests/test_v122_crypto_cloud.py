"""T64 - v1.22: crypto/benchmark-v2/cloud/math-v2 plugins."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.harness_core import Context
from harness.harness_plugins import (BenchmarkV2Plugin, CloudPlugin,
                                     CryptoPlugin, MathV2Plugin, PresetPlugin)


def test_crypto_roundtrip():
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    ctx.use(CryptoPlugin())
    h = ctx.crypto.hash("hello")
    assert h == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    tok = ctx.crypto.xor_encrypt("secret", "key")
    assert ctx.crypto.xor_decrypt(tok, "key") == "secret"
    sig = ctx.crypto.hmac_sign("msg", "k")
    assert ctx.crypto.hmac_verify("msg", "k", sig)
    assert not ctx.crypto.hmac_verify("tampered", "k", sig)
    print("PASS test_crypto_roundtrip")


def test_bench_v2_trend():
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    ctx.use(BenchmarkV2Plugin())
    t = ctx.bench.trend({"a": 0.8, "b": 0.6}, {"a": 0.7, "b": 0.65})
    assert t["a"]["trend"] == "up"
    assert t["b"]["trend"] == "down"
    print("PASS test_bench_v2_trend")


def test_cloud_stubs():
    ctx = Context()
    ctx.use(CloudPlugin())
    for fn in (ctx.cloud.aws_s3_list, ctx.cloud.gcp_storage_list):
        r = fn("test-bucket")
        assert "error" in r or "objects" in r or "blobs" in r
    print("PASS test_cloud_stubs")


def test_math_v2():
    ctx = Context()
    ctx.use(MathV2Plugin())
    m = ctx.linalg.matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]])
    assert m == [[19, 22], [43, 50]]
    s = ctx.stats.summary([1, 2, 3, 4, 5])
    assert s["mean"] == 3.0
    f = ctx.signal.fft_magnitudes([0, 1, 0, -1] * 4)
    assert len(f) == 16
    print("PASS test_math_v2")


if __name__ == "__main__":
    test_crypto_roundtrip()
    test_bench_v2_trend()
    test_cloud_stubs()
    test_math_v2()
    print("\nv1.22 tests done")
