"""Video frame extraction and captioning for video-language pretraining."""
import json
from pathlib import Path
from typing import Iterator, List, Dict, Optional
import torch
import cv2
import numpy as np


class VideoFrameExtractor:
    """
    Extract representative frames from videos for training.

    Strategies:
      - Uniform sampling: N frames evenly spaced
      - Scene detection: extract key frames at scene changes
      - Motion-based: sample frames with significant motion
    """

    def __init__(
        self,
        frames_per_video: int = 8,
        frame_size: Tuple[int, int] = (224, 224),
        sampling_strategy: str = "uniform",
    ):
        self.frames_per_video = frames_per_video
        self.frame_size = frame_size
        self.sampling_strategy = sampling_strategy

    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract frames from a video file."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            return []

        frame_indices = self._select_frame_indices(total_frames)

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.frame_size)
                frames.append(frame)

        cap.release()
        return frames

    def _select_frame_indices(self, total_frames: int) -> List[int]:
        """Select frame indices based on strategy."""
        if self.sampling_strategy == "uniform":
            return [int(i * total_frames / self.frames_per_video) for i in range(self.frames_per_video)]
        elif self.sampling_strategy == "scene":
            # Simplified: just use uniform for now
            # Real implementation would use scene detection
            return [int(i * total_frames / self.frames_per_video) for i in range(self.frames_per_video)]
        else:
            return [0]  # Fallback


class VideoCaptionDataset:
    """
    Dataset of video-caption pairs.

    Format:
    {
      "video_path": "path/to/video.mp4",
      "caption": "A person walking in a park",
      "frame_captions": ["frame1 cap", "frame2 cap", ...],
      "metadata": {"duration": 10.5, "fps": 30}
    }
    """

    def __init__(self, data_dir: str, frame_extractor: Optional[VideoFrameExtractor] = None):
        self.data_dir = Path(data_dir)
        self.frame_extractor = frame_extractor or VideoFrameExtractor()

    def stream(self) -> Iterator[Dict]:
        """Stream video-caption samples."""
        for jsonl_file in self.data_dir.glob("*.jsonl"):
            with open(jsonl_file, "r") as f:
                for line in f:
                    try:
                        sample = json.loads(line)

                        # Extract frames if needed
                        if "frames" not in sample:
                            video_path = sample.get("video_path", "")
                            if Path(video_path).exists():
                                frames = self.frame_extractor.extract_frames(video_path)
                                sample["frames"] = frames

                        yield sample
                    except Exception:
                        continue

    def create_interleaved_format(self, sample: Dict) -> List[Dict]:
        """
        Convert video sample to interleaved image-text format.

        Output: [{"type": "image", "content": frame1}, {"type": "text", "content": cap1}, ...]
        """
        result = []
        frames = sample.get("frames", [])
        captions = sample.get("frame_captions", [])

        for i, frame in enumerate(frames):
            result.append({"type": "image", "content": frame})
            if i < len(captions):
                result.append({"type": "text", "content": captions[i]})

        return result
