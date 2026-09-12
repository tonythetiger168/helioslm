"""Video scene detection using content-aware analysis.

Replaces uniform sampling with actual scene change detection.

Installation:
    pip install scenedetect[opencv]  # or pip install scenedetect

Usage:
    detector = SceneDetector(threshold=30.0)
    scenes = detector.detect_scenes("video.mp4")
    frames = detector.extract_keyframes("video.mp4", scenes, num_frames=8)
"""
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np


class SceneDetector:
    """
    Detect scene changes in video using histogram comparison.

    Alternative to pyscenedetect when not available.
    """

    def __init__(self, threshold: float = 30.0, min_scene_len: float = 2.0):
        """
        Args:
            threshold: Histogram difference threshold (higher = fewer scenes)
            min_scene_len: Minimum scene length in seconds
        """
        self.threshold = threshold
        self.min_scene_len = min_scene_len

    def detect_scenes(self, video_path: str) -> List[Tuple[float, float]]:
        """
        Detect scene boundaries.

        Returns:
            scenes: List of (start_time, end_time) in seconds
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            return []

        scenes = []
        scene_start = 0
        prev_hist = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Compute histogram
            hist = self._compute_histogram(frame)

            if prev_hist is not None:
                # Compare with previous frame
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)

                # Check if scene change
                time_since_start = (frame_idx - scene_start) / fps

                if diff > self.threshold and time_since_start >= self.min_scene_len:
                    scenes.append((scene_start / fps, frame_idx / fps))
                    scene_start = frame_idx

            prev_hist = hist
            frame_idx += 1

        # Add final scene
        if scene_start < total_frames:
            scenes.append((scene_start / fps, total_frames / fps))

        cap.release()
        return scenes

    def _compute_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Compute color histogram for frame comparison."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    def extract_keyframes(
        self,
        video_path: str,
        scenes: Optional[List[Tuple[float, float]]] = None,
        num_frames: int = 8,
    ) -> List[np.ndarray]:
        """
        Extract representative keyframes from video.

        Args:
            video_path: Path to video file
            scenes: Pre-detected scenes (auto-detect if None)
            num_frames: Number of frames to extract

        Returns:
            frames: List of keyframe images
        """
        if scenes is None:
            scenes = self.detect_scenes(video_path)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_duration = sum(end - start for start, end in scenes)

        # Select frame timestamps
        if len(scenes) >= num_frames:
            # One frame per scene, select most significant scenes
            selected_scenes = self._select_diverse_scenes(scenes, num_frames)
            timestamps = [(start + end) / 2 for start, end in selected_scenes]
        else:
            # Multiple frames per scene
            timestamps = []
            for start, end in scenes:
                scene_duration = end - start
                n_frames = max(1, int(num_frames * scene_duration / total_duration))
                for i in range(n_frames):
                    t = start + scene_duration * (i + 0.5) / n_frames
                    timestamps.append(t)
            timestamps = timestamps[:num_frames]

        # Extract frames
        frames = []
        for t in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        cap.release()
        return frames

    def _select_diverse_scenes(self, scenes: List[Tuple[float, float]], num: int) -> List[Tuple[float, float]]:
        """Select most diverse/different scenes."""
        # Sort by duration and pick longest ones (usually most content-rich)
        sorted_scenes = sorted(scenes, key=lambda s: s[1] - s[0], reverse=True)
        return sorted_scenes[:num]


# Integration with VideoFrameExtractor
class SceneAwareFrameExtractor:
    """Drop-in replacement for VideoFrameExtractor with scene detection."""

    def __init__(self, frames_per_video: int = 8, frame_size: Tuple[int, int] = (224, 224)):
        self.frames_per_video = frames_per_video
        self.frame_size = frame_size
        self.scene_detector = SceneDetector()

    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract frames using scene-aware detection."""
        frames = self.scene_detector.extract_keyframes(
            video_path,
            num_frames=self.frames_per_video,
        )

        # Resize frames
        resized = []
        for frame in frames:
            frame = cv2.resize(frame, self.frame_size)
            resized.append(frame)

        return resized
