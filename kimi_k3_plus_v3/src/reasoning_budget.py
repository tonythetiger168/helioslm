"""Reasoning Budget Controller - P4"""
import torch, torch.nn as nn

class ReasoningBudgetController(nn.Module):
    """P4: 推理预算控制 + 投机推理 + 并行子任务"""
    def __init__(self,config):
        super().__init__(); self.config=config
        self.max_ratio=config.reasoning_budget.max_reasoning_ratio
        self.adaptive=config.reasoning_budget.adaptive_depth
        self.budgets={"simple":config.reasoning_budget.simple_task_budget,"medium":config.reasoning_budget.medium_task_budget,"hard":config.reasoning_budget.hard_task_budget}
        self.speculative_reasoning=config.reasoning_budget.speculative_reasoning
        self.parallel_subtasks=config.reasoning_budget.parallel_subtasks
        # 任务难度分类器
        self.difficulty_classifier=nn.Sequential(nn.Linear(config.hidden_size,256),nn.GELU(),nn.Linear(256,3))  # [简单,中等,困难]

    def classify_difficulty(self,hidden_states):
        logits=self.difficulty_classifier(hidden_states[:,-1,:])
        return torch.argmax(logits,dim=-1)  # 0=simple,1=medium,2=hard

    def get_budget(self,hidden_states,total_token_budget):
        """P4: 动态推理预算分配"""
        diff=self.classify_difficulty(hidden_states)
        diff_map={0:"simple",1:"medium",2:"hard"}
        task_type=diff_map[diff.item()]
        reasoning_budget=min(self.budgets[task_type],int(total_token_budget*self.max_ratio))
        return reasoning_budget,task_type

    def speculative_reason(self,draft_model,hidden_states,budget):
        """P4: 投机推理 - 草稿模型预推理"""
        if not self.speculative_reasoning or draft_model is None:
            return hidden_states,budget
        # 使用草稿模型快速预推理
        with torch.no_grad():
            draft_out=draft_model(hidden_states)
        # 验证并裁剪
        return draft_out,budget

    def split_parallel_subtasks(self,task_embedding):
        """P4: 将任务拆分为并行子任务"""
        if not self.parallel_subtasks:
            return [task_embedding]
        # 简化为2-4个子任务
        num_subtasks=min(4,max(2,task_embedding.shape[1]//256))
        chunk_size=task_embedding.shape[1]//num_subtasks
        return [task_embedding[:,i*chunk_size:(i+1)*chunk_size,:] for i in range(num_subtasks)]
