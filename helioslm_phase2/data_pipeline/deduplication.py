"""Fuzzy and exact deduplication using MinHash LSH."""
import hashlib
import re
from typing import Iterator, Dict, Set, Optional, List
from dataclasses import dataclass


@dataclass
class DocumentSignature:
    doc_id: str
    minhash: List[int]
    text_hash: str


class ExactDeduplicator:
    """Exact deduplication using MD5 hashes."""

    def __init__(self):
        self.seen_hashes: Set[str] = set()

    def __call__(self, doc: Dict) -> bool:
        text = doc.get("text", "")
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        if h in self.seen_hashes:
            return False
        self.seen_hashes.add(h)
        return True

    def reset(self):
        self.seen_hashes.clear()


class MinHashDeduplicator:
    """
    Fuzzy deduplication using MinHash + LSH.

    References:
      - Broder, A.Z. "On the resemblance and containment of documents"
      - datasketch library for production use
    """

    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.85,
        shingle_size: int = 5,
        max_buckets: int = 1_000_000,
    ):
        self.num_perm = num_perm
        self.threshold = threshold
        self.shingle_size = shingle_size
        self.max_buckets = max_buckets
        self.bands = self._compute_bands()
        self.lsh_buckets: Dict[int, Set[str]] = {}
        self.signatures: Dict[str, List[int]] = {}

    def _compute_bands(self) -> int:
        """Compute optimal number of bands for given threshold."""
        # r * b = num_perm, threshold ≈ (1/b)^(1/r)
        # Simple heuristic: use ~16 bands
        return 16

    def _get_shingles(self, text: str) -> Set[str]:
        """Generate word shingles."""
        words = re.findall(r"\w+", text.lower())
        if len(words) < self.shingle_size:
            return set()
        return set(
            " ".join(words[i:i + self.shingle_size])
            for i in range(len(words) - self.shingle_size + 1)
        )

    def _compute_minhash(self, shingles: Set[str]) -> List[int]:
        """Compute MinHash signature."""
        if not shingles:
            return [0] * self.num_perm

        signature = []
        for i in range(self.num_perm):
            min_hash = float("inf")
            for shingle in shingles:
                # Use multiple hash functions via seeded MD5
                h = int(hashlib.md5(f"{shingle}:{i}".encode()).hexdigest(), 16)
                min_hash = min(min_hash, h)
            signature.append(min_hash)
        return signature

    def _get_band_hashes(self, signature: List[int]) -> List[int]:
        """Compute LSH band hashes."""
        rows_per_band = len(signature) // self.bands
        band_hashes = []
        for b in range(self.bands):
            start = b * rows_per_band
            end = start + rows_per_band
            band_str = ",".join(map(str, signature[start:end]))
            band_hashes.append(hash(band_str))
        return band_hashes

    def __call__(self, doc: Dict) -> bool:
        text = doc.get("text", "")
        doc_id = hashlib.md5(text[:1000].encode()).hexdigest()[:16]

        shingles = self._get_shingles(text)
        if not shingles:
            return True

        signature = self._compute_minhash(shingles)
        band_hashes = self._get_band_hashes(signature)

        # Check for candidates in LSH buckets
        candidates: Set[str] = set()
        for band_hash in band_hashes:
            bucket_key = band_hash % self.max_buckets
            if bucket_key in self.lsh_buckets:
                candidates.update(self.lsh_buckets[bucket_key])

        # Check Jaccard similarity with candidates
        for cand_id in candidates:
            if self._jaccard(signature, self.signatures[cand_id]) >= self.threshold:
                return False  # Duplicate found

        # Add to index
        for band_hash in band_hashes:
            bucket_key = band_hash % self.max_buckets
            if bucket_key not in self.lsh_buckets:
                self.lsh_buckets[bucket_key] = set()
            self.lsh_buckets[bucket_key].add(doc_id)

        self.signatures[doc_id] = signature
        return True

    def _jaccard(self, sig1: List[int], sig2: List[int]) -> float:
        """Estimate Jaccard similarity from MinHash signatures."""
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)

    def reset(self):
        self.lsh_buckets.clear()
        self.signatures.clear()
