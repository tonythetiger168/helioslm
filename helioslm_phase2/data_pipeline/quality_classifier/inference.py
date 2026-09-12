"""Batch inference for data quality scoring."""
import json
import torch
from typing import Iterator, Dict, List, Callable
from .classifier_model import QualityClassifier, QualityDimensions


def batch_score_documents(
    documents: Iterator[Dict],
    model: QualityClassifier,
    tokenizer,
    batch_size: int = 32,
    device: str = "cuda",
) -> Iterator[Dict]:
    """
    Score documents in batches.

    Yields documents with added 'quality_scores' field.
    """
    model.eval()
    model.to(device)

    batch = []

    for doc in documents:
        batch.append(doc)

        if len(batch) >= batch_size:
            yield from _score_batch(batch, model, tokenizer, device)
            batch = []

    if batch:
        yield from _score_batch(batch, model, tokenizer, device)


def _score_batch(batch: List[Dict], model, tokenizer, device):
    """Score a single batch."""
    # Tokenize
    texts = [doc["text"] for doc in batch]
    max_len = 2048

    input_ids_list = []
    attention_mask_list = []

    for text in texts:
        tokens = tokenizer.encode(text, max_length=max_len, truncation=True)
        pad_len = max_len - len(tokens)
        input_ids_list.append(tokens + [0] * pad_len)
        attention_mask_list.append([1] * len(tokens) + [0] * pad_len)

    input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention_mask_list, dtype=torch.long, device=device)

    with torch.no_grad():
        scores = model(input_ids, attention_mask)

    for i, doc in enumerate(batch):
        doc["quality_scores"] = QualityDimensions(
            grammar=scores["grammar"][i].item(),
            knowledge=scores["knowledge"][i].item(),
            coherence=scores["coherence"][i].item(),
            toxicity=scores["toxicity"][i].item(),
            diversity=scores["diversity"][i].item(),
        )
        doc["quality_composite"] = doc["quality_scores"].composite
        yield doc


def filter_by_score(
    documents: Iterator[Dict],
    threshold: float = 0.6,
    top_k_percent: Optional[float] = None,
    min_keep: int = 1000,
) -> Iterator[Dict]:
    """
    Filter documents by quality score.

    Args:
        threshold: Minimum composite score to keep
        top_k_percent: If set, keep top K% instead of threshold
        min_keep: Minimum number of documents to keep
    """
    if top_k_percent is not None:
        # Buffer and sort
        docs = list(documents)
        docs.sort(key=lambda d: d.get("quality_composite", 0), reverse=True)
        keep_count = max(int(len(docs) * top_k_percent), min_keep)
        for doc in docs[:keep_count]:
            yield doc
    else:
        count = 0
        for doc in documents:
            score = doc.get("quality_composite", 0)
            if score >= threshold or count < min_keep:
                yield doc
                count += 1
