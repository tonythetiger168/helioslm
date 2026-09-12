"""Speculative Decoding + Continuous Batching - P0"""
import torch, torch.nn as nn, time

class DraftModel(nn.Module):
    def __init__(self,config):
        super().__init__(); h=config.speculative.draft_hidden_size
        self.embed=nn.Embedding(config.vocab_size,h)
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(d_model=h,nhead=32,dim_feedforward=16384,batch_first=True) for _ in range(config.speculative.draft_layers)])
        self.norm=nn.LayerNorm(h); self.lm_head=nn.Linear(h,config.vocab_size,bias=False)
    def forward(self,input_ids):
        x=self.embed(input_ids)
        for l in self.layers: x=l(x)
        return self.lm_head(self.norm(x))
    def generate_draft(self,input_ids,max_tokens=5):
        self.eval()
        with torch.no_grad():
            for _ in range(max_tokens):
                logits=self.forward(input_ids)
                next_token=logits[:,-1,:].argmax(dim=-1,keepdim=True)
                input_ids=torch.cat([input_ids,next_token],dim=1)
        return input_ids

class TreeAttentionVerifier:
    def __init__(self,target_model): self.target_model=target_model
    def verify(self,input_ids,draft_tokens):
        self.target_model.eval()
        with torch.no_grad():
            tree_input=torch.cat([input_ids,draft_tokens],dim=1)
            logits,_=self.target_model(tree_input)
            accepted=[]; draft_len=draft_tokens.shape[1]
            for i in range(draft_len):
                pos=input_ids.shape[1]+i
                draft_t=draft_tokens[0,i]; target_t=logits[0,pos-1,:].argmax()
                if draft_t==target_t: accepted.append(draft_t.item())
                else: accepted.append(target_t.item()); break
            return accepted,len(accepted)

class SpeculativeDecoder:
    def __init__(self,target_model,draft_model,config):
        self.target_model=target_model; self.draft_model=draft_model
        self.max_draft=config.speculative.max_draft_tokens
        self.threshold=config.speculative.acceptance_threshold
        self.verifier=TreeAttentionVerifier(target_model)
        # P0: 连续批处理
        self.continuous_batching=config.speculative.continuous_batching
        self.bucket_sizes=config.speculative.batch_bucket_sizes

    def generate(self,input_ids,max_new_tokens,temperature=0.7):
        generated=input_ids.clone(); total_drafted=0; total_accepted=0; start_time=time.time()
        while generated.shape[1]<input_ids.shape[1]+max_new_tokens:
            remaining=input_ids.shape[1]+max_new_tokens-generated.shape[1]
            draft_len=min(self.max_draft,remaining)
            draft_output=self.draft_model.generate_draft(generated,draft_len)
            draft_tokens=draft_output[:,generated.shape[1]:]; total_drafted+=draft_len
            accepted,num_acc=self.verifier.verify(generated,draft_tokens); total_accepted+=num_acc
            accepted_t=torch.tensor([accepted],device=generated.device)
            generated=torch.cat([generated,accepted_t],dim=1)
            if num_acc==0:
                with torch.no_grad():
                    logits,_=self.target_model(generated)
                    next_t=logits[:,-1,:].argmax(dim=-1,keepdim=True)
                    generated=torch.cat([generated,next_t],dim=1)
        elapsed=time.time()-start_time; speed=(generated.shape[1]-input_ids.shape[1])/elapsed
        return generated,total_accepted/max(total_drafted,1),speed

    def continuous_batch_generate(self,requests):
        """P0: 连续批处理 - 动态长度分桶"""
        if not self.continuous_batching:
            return [self.generate(r["input_ids"],r.get("max_tokens",100)) for r in requests]
        buckets={}
        for req in requests:
            l=len(req["input_ids"][0]); bucket=min([b for b in self.bucket_sizes if b>=l],default=self.bucket_sizes[-1])
            buckets.setdefault(bucket,[]).append(req)
        results=[]
        for bucket,reqs in buckets.items():
            for i in range(0,len(reqs),32):
                batch=reqs[i:i+32]
                # 并行处理 (简化)
                for r in batch: results.append(self.generate(r["input_ids"],r.get("max_tokens",100)))
        return results
