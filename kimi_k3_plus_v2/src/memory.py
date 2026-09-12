"""
Long-Term Memory - Phase 3
外部向量存储 + 压缩摘要 + 会话管理
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any
import time


class MemoryCompressor(nn.Module):
    """记忆压缩器 - 10:1 压缩比"""
    def __init__(self, hidden_size, compression_ratio=0.1):
        super().__init__()
        self.compression_ratio = compression_ratio
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=16, batch_first=True),
            num_layers=2
        )
        self.compress_proj = nn.Linear(hidden_size, int(hidden_size * compression_ratio))

    def compress(self, memory_sequence):
        """压缩长序列记忆"""
        encoded = self.encoder(memory_sequence)
        compressed = self.compress_proj(encoded.mean(dim=1, keepdim=True))
        return compressed

    def decompress(self, compressed_memory):
        """解压缩 (有损)"""
        return compressed_memory  # 简化实现


class LongTermMemory:
    """长期记忆系统"""
    def __init__(self, config):
        self.max_entries = config.agentic.memory_max_entries
        self.compression_ratio = config.agentic.memory_compression_ratio
        self.hidden_size = config.hidden_size

        self.memories = []  # [(key, value, importance, timestamp)]
        self.compressor = MemoryCompressor(self.hidden_size, self.compression_ratio)

        # 尝试导入 FAISS
        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.hidden_size)
            self.use_faiss = True
        except ImportError:
            self.use_faiss = False
            self.index = None

    def add(self, key, value, importance=1.0):
        """添加记忆"""
        if len(self.memories) >= self.max_entries:
            # 移除最不重要的记忆
            min_idx = min(range(len(self.memories)), key=lambda i: self.memories[i][2])
            self.memories.pop(min_idx)

        # 压缩长记忆
        if isinstance(value, torch.Tensor) and value.numel() > 1000:
            value = self.compressor.compress(value.unsqueeze(0)).squeeze(0)

        self.memories.append({
            "key": key.detach() if isinstance(key, torch.Tensor) else key,
            "value": value.detach() if isinstance(value, torch.Tensor) else value,
            "importance": importance,
            "timestamp": time.time()
        })

        # 添加到向量索引
        if self.use_faiss and isinstance(key, torch.Tensor):
            key_np = key.detach().cpu().numpy().reshape(1, -1)
            self.index.add(key_np)

    def retrieve(self, query, top_k=5):
        """检索相关记忆"""
        if not self.memories:
            return []

        if self.use_faiss and isinstance(query, torch.Tensor):
            query_np = query.detach().cpu().numpy().reshape(1, -1)
            distances, indices = self.index.search(query_np, min(top_k, len(self.memories)))
            return [self.memories[i] for i in indices[0] if i < len(self.memories)]

        # 回退到线性搜索
        return sorted(self.memories, key=lambda m: m["importance"], reverse=True)[:top_k]

    def summarize_session(self, session_history):
        """会话摘要生成"""
        if len(session_history) == 0:
            return None

        # 提取关键信息 (简化实现)
        key_points = session_history[::max(1, len(session_history) // 5)]
        return {
            "key_points": key_points,
            "session_length": len(session_history),
            "timestamp": time.time()
        }
