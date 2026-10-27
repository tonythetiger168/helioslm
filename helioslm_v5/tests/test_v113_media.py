"""T55 - v1.13: video/audio plugins."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.harness_core import Context
from harness.harness_plugins import (AudioPlugin, NanoVideoPlugin,
                                     PresetPlugin, VideoPlugin)


def test_video_stub():
    ctx = Context()
    ctx.use(VideoPlugin(provider="runway", api_key_env="RUNWAY_KEY"))
    r = ctx.video.generate("a cat dancing")
    assert "error" in r or "stub" in r
    print("PASS test_video_stub")


def test_audio_tts_local_or_error():
    ctx = Context()
    ctx.use(AudioPlugin(provider="local"))
    r = ctx.audio.tts("hello world")
    assert "error" in r or "ok" in r or "spoken" in r
    print("PASS test_audio_tts_local_or_error")


def test_audio_stt_or_error():
    ctx = Context()
    ctx.use(AudioPlugin())
    r = ctx.audio.stt("/nonexistent.wav")
    assert "error" in r
    print("PASS test_audio_stt_or_error")


def test_nanovideo_or_ffmpeg_missing():
    ctx = Context()
    ctx.use(NanoVideoPlugin())
    r = ctx.nanovideo.from_images(["/nonexistent.png"])
    assert "error" in r or "ok" in r
    print("PASS test_nanovideo_or_ffmpeg_missing")


if __name__ == "__main__":
    test_video_stub()
    test_audio_tts_local_or_error()
    test_audio_stt_or_error()
    test_nanovideo_or_ffmpeg_missing()
    print("\nvideo/audio plugin tests done")
