"""Quality classifier training pipeline."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Optional
import json


class QualityDataset(Dataset):
    """Dataset of labeled documents for quality classifier training."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(data_path) as f:
            for line in f:
                data = json.loads(line)
                self.samples.append(data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]
        labels = sample["labels"]  # Dict of dimension scores

        tokens = self.tokenizer.encode(text, max_length=self.max_length, truncation=True)
        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.ones(len(tokens), dtype=torch.long)

        # Labels
        label_tensor = torch.tensor([
            labels.get("grammar", 0.5),
            labels.get("knowledge", 0.5),
            labels.get("coherence", 0.5),
            labels.get("toxicity", 0.5),
            labels.get("diversity", 0.5),
            labels.get("composite", 0.5),
        ], dtype=torch.float)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_tensor,
        }


def collate_fn(batch):
    """Collate with padding."""
    max_len = max(len(b["input_ids"]) for b in batch)

    input_ids = []
    attention_masks = []
    labels = []

    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        input_ids.append(torch.cat([b["input_ids"], torch.zeros(pad_len, dtype=torch.long)]))
        attention_masks.append(torch.cat([b["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        labels.append(b["labels"])

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.stack(labels),
    }


class QualityClassifierTrainer:
    """Trainer for quality classifier."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.criterion = nn.MSELoss()
        self.global_step = 0

    def compute_loss(self, predictions: Dict, targets: torch.Tensor) -> torch.Tensor:
        """Multi-task loss across all dimensions."""
        # targets: [batch, 6] — [grammar, knowledge, coherence, toxicity, diversity, composite]

        dim_loss = (
            self.criterion(predictions["grammar"], targets[:, 0]) +
            self.criterion(predictions["knowledge"], targets[:, 1]) +
            self.criterion(predictions["coherence"], targets[:, 2]) +
            self.criterion(predictions["toxicity"], targets[:, 3]) +
            self.criterion(predictions["diversity"], targets[:, 4])
        ) / 5.0

        composite_loss = self.criterion(predictions["composite"], targets[:, 5])

        return dim_loss + composite_loss

    def train_step(self, batch: Dict) -> Dict:
        self.model.train()
        self.optimizer.zero_grad()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        predictions = self.model(input_ids, attention_mask)
        loss = self.compute_loss(predictions, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.global_step += 1

        return {
            "loss": loss.item(),
            "step": self.global_step,
        }

    def train_epoch(self, dataloader: DataLoader, log_interval: int = 100):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            metrics = self.train_step(batch)
            total_loss += metrics["loss"]

            if batch_idx % log_interval == 0:
                avg = total_loss / (batch_idx + 1)
                print(f"QC Step {metrics['step']} | Loss: {avg:.4f}")

        return total_loss / len(dataloader)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f"💾 Quality classifier saved: {path}")

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"📂 Quality classifier loaded: {path}")
