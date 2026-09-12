"""Data ingestion from multiple sources for pre-training."""
import json
import gzip
import bz2
import lzma
from pathlib import Path
from typing import Iterator, Dict, Optional, Callable, List
from dataclasses import dataclass
import requests


@dataclass
class DataSource:
    """Base class for data sources."""
    name: str
    target_tokens: int
    weight: float = 1.0

    def stream(self) -> Iterator[Dict]:
        raise NotImplementedError


class CommonCrawlSource(DataSource):
    """Common Crawl WARC ingestion with quality heuristics."""

    def __init__(self, data_dir: str, target_tokens: int = 9_000_000_000_000, weight: float = 0.60):
        super().__init__("common_crawl", target_tokens, weight)
        self.data_dir = Path(data_dir)

    def stream(self) -> Iterator[Dict]:
        """Stream documents from WET files."""
        for wet_file in self.data_dir.glob("*.wet.gz"):
            with gzip.open(wet_file, "rt", encoding="utf-8", errors="ignore") as f:
                doc = {"text": "", "url": "", "metadata": {}}
                for line in f:
                    line = line.strip()
                    if line.startswith("WARC-Target-URI:"):
                        doc["url"] = line.split(":", 1)[1].strip()
                    elif line == "":
                        if doc["text"]:
                            yield doc
                            doc = {"text": "", "url": doc.get("url", ""), "metadata": {}}
                    else:
                        doc["text"] += line + "\n"


class GitHubSource(DataSource):
    """GitHub code ingestion with language filtering."""

    def __init__(self, data_dir: str, target_tokens: int = 3_000_000_000_000, 
                 weight: float = 0.20, allowed_langs: Optional[List[str]] = None):
        super().__init__("github", target_tokens, weight)
        self.data_dir = Path(data_dir)
        self.allowed_langs = allowed_langs or ["py", "js", "ts", "java", "cpp", "c", "go", "rs"]

    def stream(self) -> Iterator[Dict]:
        for jsonl_file in self.data_dir.glob("*.jsonl*"):
            opener = gzip.open if str(jsonl_file).endswith(".gz") else open
            with opener(jsonl_file, "rt", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        lang = data.get("language", "").lower()
                        if lang in self.allowed_langs:
                            yield {
                                "text": data.get("content", ""),
                                "language": lang,
                                "repo": data.get("repo_name", ""),
                                "metadata": {"source": "github", "lang": lang},
                            }
                    except json.JSONDecodeError:
                        continue


class ArxivSource(DataSource):
    """arXiv LaTeX ingestion with math formula preservation."""

    def __init__(self, data_dir: str, target_tokens: int = 1_500_000_000_000, weight: float = 0.10):
        super().__init__("arxiv", target_tokens, weight)
        self.data_dir = Path(data_dir)

    def stream(self) -> Iterator[Dict]:
        for tex_file in self.data_dir.rglob("*.tex"):
            try:
                text = tex_file.read_text(encoding="utf-8", errors="ignore")
                # Preserve LaTeX math environments
                yield {
                    "text": text,
                    "metadata": {"source": "arxiv", "file": str(tex_file)},
                }
            except Exception:
                continue


class BookSource(DataSource):
    """Book and long-form text ingestion."""

    def __init__(self, data_dir: str, target_tokens: int = 1_500_000_000_000, weight: float = 0.10):
        super().__init__("books", target_tokens, weight)
        self.data_dir = Path(data_dir)

    def stream(self) -> Iterator[Dict]:
        for txt_file in self.data_dir.rglob("*.txt"):
            try:
                text = txt_file.read_text(encoding="utf-8", errors="ignore")
                yield {
                    "text": text,
                    "metadata": {"source": "books", "file": str(txt_file)},
                }
            except Exception:
                continue
