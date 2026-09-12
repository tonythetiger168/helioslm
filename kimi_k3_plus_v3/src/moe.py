"""Stable LatentMoE+ - P0: 动态稀疏度 + P2: 领域专家分群"""
import torch, torch.nn as nn, torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self,dim,eps=1e-6): super().__init__(); self.eps=eps; self.weight=nn.Parameter(torch.ones(dim))
    def forward(self,x): return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)*self.weight

class SiTUGLU(nn.Module):
    def __init__(self,h,eh): super().__init__(); self.gate_proj=nn.Linear(h,eh,bias=False); self.up_proj=nn.Linear(h,eh,bias=False); self.down_proj=nn.Linear(eh,h,bias=False)
    def forward(self,x): return self.down_proj(torch.sigmoid(self.gate_proj(x))*torch.tanh(self.up_proj(x)))

class StableLatentMoE(nn.Module):
    def __init__(self,config):
        super().__init__(); self.config=config; self.hidden_size=config.hidden_size
        self.num_experts=config.moe.num_experts; self.num_shared=config.moe.num_shared_experts
        self.top_k=config.moe.num_activated_experts; self.min_k=config.moe.min_experts; self.max_k=config.moe.max_experts
        self.expert_hidden=config.moe.expert_hidden_size
        self.router=nn.Linear(self.hidden_size,self.num_experts,bias=False)
        self.experts=nn.ModuleList([SiTUGLU(self.hidden_size,self.expert_hidden) for _ in range(self.num_experts)])
        self.shared_experts=nn.ModuleList([SiTUGLU(self.hidden_size,self.expert_hidden) for _ in range(self.num_shared)])
        self.difficulty_estimator=nn.Sequential(nn.Linear(self.hidden_size,256),nn.GELU(),nn.Linear(256,1),nn.Sigmoid())
        self.load_balance_loss_coef=config.moe.load_balance_loss_coef
        self.input_norm=RMSNorm(self.hidden_size,eps=config.rms_norm_eps)
        self.domain_groups=config.moe.domain_groups if config.moe.domain_grouping else {}

    def forward(self,hidden_states,task_type=None):
        B,seq,h=hidden_states.shape; flat=hidden_states.view(-1,h); normed=self.input_norm(flat)
        # P0: 动态稀疏度
        difficulty=self.difficulty_estimator(normed).squeeze(-1)
        dynamic_k=(self.min_k+(self.max_k-self.min_k)*difficulty).round().long().clamp(self.min_k,self.max_k)
        router_logits=self.router(normed)
        # P2: 领域专家分群偏置
        if task_type and self.domain_groups:
            for domain,(start,end) in self.domain_groups.items():
                if domain==task_type and start<self.num_experts and end<self.num_experts:
                    router_logits[:,start:end+1]+=2.0
        top_k_values,top_k_indices=torch.topk(router_logits,self.top_k,dim=-1)
        routing_weights=F.softmax(top_k_values,dim=-1)
        router_prob=F.softmax(router_logits,dim=-1)
        aux_loss=self.num_experts*(router_prob.mean(dim=0)**2).sum()*self.load_balance_loss_coef
        output=torch.zeros_like(flat)
        for i in range(self.top_k):
            ei=top_k_indices[:,i]; ew=routing_weights[:,i:i+1]
            for eid in range(self.num_experts):
                mask=(ei==eid)
                if mask.any(): output[mask]+=ew[mask]*self.experts[eid](normed[mask])
        for s in self.shared_experts: output+=s(normed)
        return output.view(B,seq,h),aux_loss
