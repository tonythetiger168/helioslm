
"""
Kimi K3+ 四階段訓練流程
Phase 1: 預訓練 → Phase 2: 持續預訓練 → Phase 3: SFT → Phase 4: RLHF
"""

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup
import wandb
from tqdm import tqdm
import os

# ============================================
# Phase 1: 預訓練 (Pre-training)
# ============================================

class PretrainingTrainer:
    """
    預訓練階段：15T tokens，漸進式上下文擴展
    """
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.phases[0].learning_rate,
            weight_decay=config.training.phases[0].weight_decay,
            betas=(0.9, 0.95)
        )

        self.scaler = GradScaler()
        self.global_step = 0

    def train(self, train_dataloader, num_steps):
        model = self.model
        model.train()

        # Progressive context length schedule
        context_schedule = self.config.training.phases[0].context_length_progression

        for step in range(num_steps):
            # Determine current context length
            if step < num_steps * 0.3:
                max_len = context_schedule[0]  # 32K
            elif step < num_steps * 0.6:
                max_len = context_schedule[1]  # 128K
            else:
                max_len = context_schedule[2]  # 1M

            batch = next(iter(train_dataloader))
            input_ids = batch['input_ids'][:, :max_len].to(self.device)
            labels = batch['labels'][:, :max_len].to(self.device)

            with autocast(dtype=torch.bfloat16):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )

            self.scaler.scale(loss).backward()

            if (step + 1) % self.config.training.phases[0].gradient_clipping == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            self.global_step += 1

            if step % 100 == 0:
                print(f"Step {step}/{num_steps} | Loss: {loss.item():.4f} | Context: {max_len}")
                wandb.log({"pretrain/loss": loss.item(), "pretrain/context_length": max_len})


# ============================================
# Phase 2: 持續預訓練 (Continual Pre-training)
# ============================================

class ContinualPretrainingTrainer:
    """
    持續預訓練：3T 高品質推理與程式碼數據
    強化專家負載平衡與量化感知訓練
    """
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device

        # Lower learning rate for continual pretraining
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.phases[1].learning_rate,
            weight_decay=0.1
        )

        self.scaler = GradScaler()

    def train(self, train_dataloader, num_steps):
        model = self.model
        model.train()

        for step in range(num_steps):
            batch = next(iter(train_dataloader))
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            task_type = batch.get('task_type', None)  # programming, math, science, creative

            with autocast(dtype=torch.bfloat16):
                logits = model(input_ids, task_type=task_type)

                # Standard cross-entropy loss
                ce_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )

                # Add MoE load balancing loss
                # (aux_loss computed inside StableLatentMoE)
                total_loss = ce_loss  # + aux_loss (handled internally)

            self.scaler.scale(total_loss).backward()

            if (step + 1) % 4 == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            if step % 100 == 0:
                print(f"Continual Pretrain Step {step} | Loss: {total_loss.item():.4f} | Task: {task_type}")


# ============================================
# Phase 3: 監督微調 (SFT)
# ============================================

class SFTTrainer:
    """
    監督微調：多輪對話 + Agent 軌跡 + 多模態指令
    使用 LoRA 微調，凍結 AttnRes
    """
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device

        # Apply LoRA
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=config.training.phases[2].lora.r,
            lora_alpha=config.training.phases[2].lora.alpha,
            target_modules=config.training.phases[2].lora.target_modules,
            lora_dropout=config.training.phases[2].lora.dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(model, lora_config)

        # Freeze AttnRes if specified
        if config.training.phases[2].freeze_attnres:
            for name, param in self.model.named_parameters():
                if "residual" in name:
                    param.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.training.phases[2].learning_rate
        )

    def train(self, train_dataloader, num_epochs):
        self.model.train()

        for epoch in range(num_epochs):
            total_loss = 0

            for batch in tqdm(train_dataloader, desc=f"SFT Epoch {epoch+1}"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                images = batch.get('images', None)
                videos = batch.get('videos', None)

                if images is not None:
                    images = images.to(self.device)
                if videos is not None:
                    videos = videos.to(self.device)

                logits = self.model(input_ids, images=images, videos=videos)

                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_dataloader)
            print(f"SFT Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.4f}")
            wandb.log({"sft/epoch_loss": avg_loss})


# ============================================
# Phase 4: RLHF (PPO + DPO Hybrid)
# ============================================

class RLHFTrainer:
    """
    RLHF 訓練：PPO + DPO 混合
    整合 Constitutional AI 原則
    """
    def __init__(self, policy_model, ref_model, reward_model, config, device):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.config = config
        self.device = device

        self.ppo_optimizer = torch.optim.AdamW(
            policy_model.parameters(),
            lr=1e-5
        )

        self.kl_penalty = config.training.phases[3].kl_penalty
        self.ppo_epochs = config.training.phases[3].ppo_epochs
        self.dpo_beta = config.training.phases[3].dpo_beta

    def compute_rewards(self, prompts, responses):
        """計算獎勵分數"""
        with torch.no_grad():
            rewards = self.reward_model(prompts, responses)
        return rewards

    def ppo_step(self, old_logprobs, rewards, advantages):
        """PPO 更新步驟"""
        for _ in range(self.ppo_epochs):
            # Compute new log probs
            new_logprobs = self.policy_model.get_logprobs(old_logprobs.input_ids)

            # PPO clipped objective
            ratio = torch.exp(new_logprobs - old_logprobs)
            clipped_ratio = torch.clamp(ratio, 0.8, 1.2)

            policy_loss = -torch.min(
                ratio * advantages,
                clipped_ratio * advantages
            ).mean()

            # KL penalty
            with torch.no_grad():
                ref_logprobs = self.ref_model.get_logprobs(old_logprobs.input_ids)
            kl_loss = self.kl_penalty * (new_logprobs - ref_logprobs).mean()

            total_loss = policy_loss + kl_loss

            total_loss.backward()
            self.ppo_optimizer.step()
            self.ppo_optimizer.zero_grad()

    def dpo_step(self, chosen_ids, rejected_ids):
        """DPO (Direct Preference Optimization) 步驟"""
        policy_chosen_logps = self.policy_model.get_logprobs(chosen_ids)
        policy_rejected_logps = self.policy_model.get_logprobs(rejected_ids)

        with torch.no_grad():
            ref_chosen_logps = self.ref_model.get_logprobs(chosen_ids)
            ref_rejected_logps = self.ref_model.get_logprobs(rejected_ids)

        # DPO loss
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps

        logits = pi_logratios - ref_logratios
        dpo_loss = -F.logsigmoid(self.dpo_beta * logits).mean()

        dpo_loss.backward()
        self.ppo_optimizer.step()
        self.ppo_optimizer.zero_grad()

    def train(self, preference_dataloader, num_steps):
        """
        混合訓練流程：
        - 70% PPO (on-policy reinforcement learning)
        - 30% DPO (preference optimization)
        """
        for step in range(num_steps):
            batch = next(iter(preference_dataloader))

            if step % 10 < 7:  # 70% PPO
                prompts = batch['prompts'].to(self.device)
                responses = batch['responses'].to(self.device)
                rewards = batch['rewards'].to(self.device)

                old_logprobs = self.policy_model.get_logprobs(responses)
                advantages = rewards - rewards.mean()

                self.ppo_step(old_logprobs, rewards, advantages)

            else:  # 30% DPO
                chosen = batch['chosen'].to(self.device)
                rejected = batch['rejected'].to(self.device)

                self.dpo_step(chosen, rejected)

            if step % 100 == 0:
                print(f"RLHF Step {step}/{num_steps}")


# ============================================
# 主訓練入口
# ============================================

def main():
    """四階段訓練主入口"""

    # Load config
    with open("kimi_k3_plus_config.json", "r") as f:
        config = json.load(f)

    # Initialize model
    model = KimiK3Plus(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Phase 1: Pre-training
    print("=" * 50)
    print("Phase 1: Pre-training (15T tokens)")
    print("=" * 50)
    pretrain_trainer = PretrainingTrainer(model, config, device)
    # pretrain_dataloader = load_pretrain_data(config)
    # pretrain_trainer.train(pretrain_dataloader, num_steps=...)

    # Save checkpoint
    torch.save(model.state_dict(), "checkpoints/kimi_k3_plus_pretrained.pt")

    # Phase 2: Continual Pre-training
    print("\n" + "=" * 50)
    print("Phase 2: Continual Pre-training (3T tokens)")
    print("=" * 50)
    continual_trainer = ContinualPretrainingTrainer(model, config, device)
    # continual_dataloader = load_continual_data(config)
    # continual_trainer.train(continual_dataloader, num_steps=...)

    torch.save(model.state_dict(), "checkpoints/kimi_k3_plus_continual.pt")

    # Phase 3: SFT
    print("\n" + "=" * 50)
    print("Phase 3: Supervised Fine-Tuning")
    print("=" * 50)
    sft_trainer = SFTTrainer(model, config, device)
    # sft_dataloader = load_sft_data(config)
    # sft_trainer.train(sft_dataloader, num_epochs=3)

    torch.save(model.state_dict(), "checkpoints/kimi_k3_plus_sft.pt")

    # Phase 4: RLHF
    print("\n" + "=" * 50)
    print("Phase 4: RLHF (PPO + DPO)")
    print("=" * 50)
    ref_model = KimiK3Plus(config).to(device)
    ref_model.load_state_dict(model.state_dict())

    # reward_model = load_reward_model(config)
    rlhf_trainer = RLHFTrainer(model, ref_model, reward_model, config, device)
    # preference_dataloader = load_preference_data(config)
    # rlhf_trainer.train(preference_dataloader, num_steps=...)

    # Final save
    torch.save(model.state_dict(), "checkpoints/kimi_k3_plus_final.pt")
    print("\n✅ 訓練完成！最終模型已保存。")

if __name__ == "__main__":
    main()
