"""Long-Term Memory - P3"""
import torch, time

class LongTermMemory:
    def __init__(self,config):
        self.max_entries=config.agentic.memory_max_entries; self.compression_ratio=config.agentic.memory_compression_ratio
        self.hidden_size=config.hidden_size; self.memories=[]
        try:
            import faiss; self.index=faiss.IndexFlatIP(self.hidden_size); self.use_faiss=True
        except ImportError: self.use_faiss=False
    def add(self,key,value,importance=1.0):
        if len(self.memories)>=self.max_entries:
            min_idx=min(range(len(self.memories)),key=lambda i:self.memories[i]["importance"])
            self.memories.pop(min_idx)
        self.memories.append({"key":key.detach() if isinstance(key,torch.Tensor) else key,"value":value,"importance":importance,"timestamp":time.time()})
        if self.use_faiss and isinstance(key,torch.Tensor): self.index.add(key.detach().cpu().numpy().reshape(1,-1))
    def retrieve(self,query,top_k=5):
        if not self.memories: return []
        if self.use_faiss and isinstance(query,torch.Tensor):
            _,indices=self.index.search(query.detach().cpu().numpy().reshape(1,-1),min(top_k,len(self.memories)))
            return [self.memories[i] for i in indices[0] if i<len(self.memories)]
        return sorted(self.memories,key=lambda m:m["importance"],reverse=True)[:top_k]
