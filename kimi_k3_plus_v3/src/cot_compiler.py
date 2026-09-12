"""Chain-of-Thought Compiler - P2"""
import torch, torch.nn as nn

class CoTCompiler(nn.Module):
    """P2: 结构化推理步骤编译器 + 工具使用"""
    def __init__(self,config):
        super().__init__(); self.config=config; self.hidden_size=config.hidden_size
        self.max_reasoning_tokens=config.cot.max_reasoning_tokens
        self.force_steps=config.cot.force_steps
        self.tools=config.cot.tools if config.cot.tool_use_enabled else []
        # 推理步骤生成器
        self.step_encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.hidden_size,nhead=16,batch_first=True),num_layers=4)
        self.step_classifier=nn.Linear(self.hidden_size,3)  # [继续推理, 使用工具, 输出答案]
        self.tool_selector=nn.Linear(self.hidden_size,len(self.tools)) if self.tools else None

    def compile_reasoning(self,hidden_states,task_difficulty="medium"):
        """编译结构化推理步骤"""
        encoded=self.step_encoder(hidden_states)
        step_logits=self.step_classifier(encoded[:,-1,:])
        step_type=torch.argmax(step_logits,dim=-1)
        # 根据任务难度控制推理预算
        if task_difficulty=="simple": max_steps=3
        elif task_difficulty=="medium": max_steps=8
        else: max_steps=15
        return {"step_type":step_type,"max_steps":max_steps,"encoded":encoded}

    def select_tool(self,hidden_states):
        """P2: 自动选择工具 (Wolfram/Calculator/Python)"""
        if self.tool_selector is None: return None
        scores=self.tool_selector(hidden_states[:,-1,:])
        tool_idx=torch.argmax(scores,dim=-1)
        return self.tools[tool_idx.item()] if tool_idx.item()<len(self.tools) else None
