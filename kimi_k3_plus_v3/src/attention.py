"""Kimi Delta Attention - P0: KDA + Gated MLA + AttnRes"""
import math, torch, torch.nn as nn, torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.eps=eps; self.weight=nn.Parameter(torch.ones(dim))
    def forward(self,x): return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)*self.weight

class KimiDeltaAttention(nn.Module):
    def __init__(self, config):
        super().__init__(); self.config=config; self.hidden_size=config.hidden_size
        self.num_heads=config.attention.num_attention_heads
        self.num_kv_heads=config.attention.num_key_value_heads
        self.head_dim=self.hidden_size//self.num_heads; self.residual_depth=config.attention.residual_depth
        self.q_proj=nn.Linear(self.hidden_size,self.num_heads*self.head_dim,bias=False)
        self.k_proj=nn.Linear(self.hidden_size,self.num_kv_heads*self.head_dim,bias=False)
        self.v_proj=nn.Linear(self.hidden_size,self.num_kv_heads*self.head_dim,bias=False)
        self.o_proj=nn.Linear(self.num_heads*self.head_dim,self.hidden_size,bias=False)
        self.gate=nn.Linear(self.hidden_size,self.num_heads,bias=False)
        self.linear_q_kernel=nn.Linear(self.head_dim,self.head_dim,bias=False)
        self.linear_k_kernel=nn.Linear(self.head_dim,self.head_dim,bias=False)
        if config.attention.attn_res_enabled:
            self.residual_gates=nn.Parameter(torch.zeros(config.num_hidden_layers,self.residual_depth))
            self.residual_proj=nn.ModuleList([nn.Linear(self.hidden_size,self.hidden_size,bias=False) for _ in range(self.residual_depth)])
        self.input_norm=RMSNorm(self.hidden_size,eps=config.rms_norm_eps)

    def linear_attention(self,q,k,v):
        q=F.elu(self.linear_q_kernel(q))+1; k=F.elu(self.linear_k_kernel(k))+1
        kv_state=torch.einsum('bhsk,bhsv->bhkv',k,v)
        z=torch.einsum('bhqd,bhkd->bhq',q,k.sum(dim=2))+1e-6
        return torch.einsum('bhqd,bhkd,bhkv->bhqv',q,k,kv_state)/z.unsqueeze(-1)

    def global_attention(self,q,k,v,mask=None):
        scores=torch.einsum('bhqd,bhkd->bhqk',q,k)/math.sqrt(self.head_dim)
        if mask is not None: scores=scores.masked_fill(mask==0,float('-inf'))
        return torch.einsum('bhqk,bhkv->bhqv',F.softmax(scores,dim=-1),v)

    def forward(self,hidden_states,layer_idx,past_residuals=None,attention_mask=None):
        B,seq,_=hidden_states.shape; normed=self.input_norm(hidden_states)
        q=self.q_proj(normed).view(B,seq,self.num_heads,self.head_dim).transpose(1,2)
        k=self.k_proj(normed).view(B,seq,self.num_kv_heads,self.head_dim).transpose(1,2)
        v=self.v_proj(normed).view(B,seq,self.num_kv_heads,self.head_dim).transpose(1,2)
        gate=torch.sigmoid(self.gate(normed)).unsqueeze(-1).transpose(1,2)
        attn_out=self.global_attention(q,k,v,attention_mask) if layer_idx%4==0 else self.linear_attention(q,k,v)
        attn_out=attn_out*gate
        if self.config.attention.attn_res_enabled and past_residuals is not None:
            res_out=torch.zeros_like(attn_out)
            for i,(res,proj) in enumerate(zip(past_residuals[-self.residual_depth:],self.residual_proj)):
                res_out+=torch.sigmoid(self.residual_gates[layer_idx,i])*proj(res).view_as(attn_out)
            attn_out=attn_out+res_out
        return self.o_proj(attn_out.transpose(1,2).contiguous().view(B,seq,self.hidden_size)),hidden_states
