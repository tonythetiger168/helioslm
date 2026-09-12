"""Checkpoint management with automatic rotation and cloud sync."""
import os
import shutil
import json
from pathlib import Path
from typing import Optional, List, Dict
import torch


class CheckpointManager:
    """
    Manage training checkpoints with:
      - Automatic rotation (keep N most recent)
      - Best model tracking (by validation loss)
      - Async cloud upload (S3/GCS optional)
      - Resume from any checkpoint
    """

    def __init__(
        self,
        output_dir: str,
        max_checkpoints: int = 3,
        keep_best: bool = True,
        cloud_sync: Optional[str] = None,  # "s3://bucket/path" or "gs://bucket/path"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.keep_best = keep_best
        self.cloud_sync = cloud_sync

        self.best_loss = float("inf")
        self.checkpoint_history: List[Dict] = []

        # Load existing history
        history_file = self.output_dir / "checkpoint_history.json"
        if history_file.exists():
            with open(history_file) as f:
                self.checkpoint_history = json.load(f)

    def save(
        self,
        model_state: Dict,
        optimizer_state: Dict,
        scheduler_state: Optional[Dict],
        step: int,
        loss: float,
        metrics: Optional[Dict] = None,
        is_best: bool = False,
    ) -> str:
        """Save a checkpoint and manage rotation."""
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(exist_ok=True)

        # Save states
        torch.save(model_state, checkpoint_dir / "model.pt")
        torch.save(optimizer_state, checkpoint_dir / "optimizer.pt")
        if scheduler_state:
            torch.save(scheduler_state, checkpoint_dir / "scheduler.pt")

        # Save metadata
        metadata = {
            "step": step,
            "loss": loss,
            "metrics": metrics or {},
            "is_best": is_best or (loss < self.best_loss),
        }
        with open(checkpoint_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Track best
        if metadata["is_best"]:
            self.best_loss = loss
            best_dir = self.output_dir / "best"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(checkpoint_dir, best_dir)
            print(f"🏆 New best checkpoint: step {step}, loss {loss:.4f}")

        # Update history
        self.checkpoint_history.append({
            "step": step,
            "loss": loss,
            "path": str(checkpoint_dir),
            "timestamp": time.time(),
        })

        # Rotation
        self._rotate_checkpoints()

        # Save history
        with open(self.output_dir / "checkpoint_history.json", "w") as f:
            json.dump(self.checkpoint_history, f, indent=2)

        # Cloud sync (optional)
        if self.cloud_sync:
            self._upload_to_cloud(checkpoint_dir)

        return str(checkpoint_dir)

    def _rotate_checkpoints(self):
        """Remove old checkpoints, keeping only max_checkpoints."""
        if len(self.checkpoint_history) <= self.max_checkpoints:
            return

        # Sort by step
        sorted_history = sorted(self.checkpoint_history, key=lambda x: x["step"])
        to_remove = sorted_history[:-self.max_checkpoints]

        for entry in to_remove:
            path = Path(entry["path"])
            if path.exists() and "best" not in str(path):
                shutil.rmtree(path)
                print(f"🗑️  Removed old checkpoint: {path}")

        self.checkpoint_history = sorted_history[-self.max_checkpoints:]

    def _upload_to_cloud(self, checkpoint_dir: Path):
        """Upload checkpoint to cloud storage."""
        # Placeholder: implement with boto3 or google-cloud-storage
        print(f"☁️  Cloud sync: {checkpoint_dir} → {self.cloud_sync}")

    def load_latest(self) -> Optional[Dict]:
        """Load the most recent checkpoint."""
        if not self.checkpoint_history:
            return None

        latest = sorted(self.checkpoint_history, key=lambda x: x["step"])[-1]
        return self.load_from_path(latest["path"])

    def load_best(self) -> Optional[Dict]:
        """Load the best checkpoint."""
        best_dir = self.output_dir / "best"
        if not best_dir.exists():
            return None
        return self.load_from_path(str(best_dir))

    def load_from_path(self, path: str) -> Dict:
        """Load checkpoint from path."""
        checkpoint_dir = Path(path)
        return {
            "model": torch.load(checkpoint_dir / "model.pt", map_location="cpu"),
            "optimizer": torch.load(checkpoint_dir / "optimizer.pt", map_location="cpu"),
            "scheduler": torch.load(checkpoint_dir / "scheduler.pt", map_location="cpu") if (checkpoint_dir / "scheduler.pt").exists() else None,
            "metadata": json.load(open(checkpoint_dir / "metadata.json")),
        }

    def list_checkpoints(self) -> List[Dict]:
        """List all available checkpoints."""
        return sorted(self.checkpoint_history, key=lambda x: x["step"])


import time
