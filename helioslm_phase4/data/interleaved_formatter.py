"""Interleaved document formatting (Flamingo/Gato style).

Creates training sequences with mixed image and text tokens:
  <image> <text> <image> <text> ...

This format enables in-context learning with multiple images.
"""
import random
from typing import List, Dict, Iterator


class InterleavedDocumentFormatter:
    """
    Format multimodal data as interleaved sequences.

    Similar to Flamingo's M3W (Multimodal MassiveWeb) dataset.
    """

    def __init__(
        self,
        max_seq_length: int = 4096,
        max_images_per_sequence: int = 8,
        image_token: str = "<image>",
        text_tokenizer=None,
    ):
        self.max_seq_length = max_seq_length
        self.max_images = max_images_per_sequence
        self.image_token = image_token
        self.tokenizer = text_tokenizer

    def format_sequence(self, items: List[Dict]) -> Dict:
        """
        Format a list of interleaved items into a training sequence.

        Args:
            items: [{"type": "image|text", "content": ...}, ...]

        Returns:
            {
                "input_ids": [...],
                "image_indices": [positions where images occur],
                "image_tensors": [image data],
                "attention_mask": [...],
            }
        """
        input_ids = []
        image_indices = []
        image_tensors = []
        attention_mask = []

        for item in items:
            if item["type"] == "text":
                text = item["content"]
                tokens = self.tokenizer.encode(text, add_special_tokens=False) if self.tokenizer else [ord(c) for c in text]
                input_ids.extend(tokens)
                attention_mask.extend([1] * len(tokens))

            elif item["type"] == "image":
                # Insert image token
                img_token_id = self.tokenizer.encode(self.image_token, add_special_tokens=False)[0] if self.tokenizer else 99999
                input_ids.append(img_token_id)
                image_indices.append(len(input_ids) - 1)
                image_tensors.append(item["content"])
                attention_mask.append(1)

        # Truncate if too long
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]
            # Remove image indices that are out of bounds
            image_indices = [i for i in image_indices if i < self.max_seq_length]
            image_tensors = image_tensors[:len(image_indices)]

        return {
            "input_ids": input_ids,
            "image_indices": image_indices,
            "image_tensors": image_tensors,
            "attention_mask": attention_mask,
        }

    def create_interleaved_documents(self, text_corpus: Iterator[str], image_pool: List[str], num_docs: int = 10000) -> Iterator[Dict]:
        """
        Create synthetic interleaved documents by inserting random images into text.

        This is the Flamingo M3W approach: take web text and randomly
        replace some text spans with nearby images.
        """
        for _ in range(num_docs):
            text = next(text_corpus, "")
            if not text:
                break

            # Split text into segments
            words = text.split()
            num_images = random.randint(1, min(self.max_images, len(words) // 20))

            items = []
            last_idx = 0

            for _ in range(num_images):
                insert_pos = random.randint(last_idx + 10, len(words) - 10)

                # Add text before image
                text_segment = " ".join(words[last_idx:insert_pos])
                items.append({"type": "text", "content": text_segment})

                # Add random image
                image_path = random.choice(image_pool)
                items.append({"type": "image", "content": image_path})

                last_idx = insert_pos

            # Add remaining text
            if last_idx < len(words):
                items.append({"type": "text", "content": " ".join(words[last_idx:])})

            yield self.format_sequence(items)
