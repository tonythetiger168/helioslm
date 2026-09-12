# Kimi K3+ 技術白皮書

> **版本**: v1.0.0  
> **基於**: Kimi K3 (Moonshot AI)  
> **日期**: 2026-08-31  
> **定位**: 開源前沿智能 LLM 框架

---

## 目錄

1. [專案概述](#1-專案概述)
2. [前 5 名 LLM 對比分析](#2-前-5-名-llm-對比分析)
3. [Kimi K3 框架解析](#3-kimi-k3-框架解析)
4. [Kimi K3+ 架構設計](#4-kimi-k3-架構設計)
5. [核心模塊實現](#5-核心模塊實現)
6. [四階段訓練流程](#6-四階段訓練流程)
7. [部署與量化方案](#7-部署與量化方案)
8. [性能目標與基準](#8-性能目標與基準)
9. [文件清單](#9-文件清單)

---

## 1. 專案概述

### 1.1 設計動機

當前 LLM 領域呈現**封閉源碼主導**的格局：Anthropic 的 Claude 系列與 OpenAI 的 GPT 系列佔據品質榜首，但均為閉源商業模型。Kimi K3 是**唯一進入全球前 5 的開源權重模型**，證明了開源路線的可行性。

Kimi K3+ 的設計目標是：
- **繼承 K3 的開源優勢**：公開權重、可自架、透明可控
- **補強 K3 的已知短板**：推理速度、硬體門檻、專業領域精度
- **追趕閉源巨頭的品質**：在關鍵基準上達到或超越 GPT-5.6 Sol / Claude Opus 5

### 1.2 核心創新

| 創新點 | 技術方案 | 預期效果 |
|:---|:---|:---|
| 動態稀疏度 | 根據任務難度自動調整激活專家數 (8~16) | 簡單任務提速 40% |
| 推測解碼 | 30B 草稿模型 + 樹注意力驗證 | 輸出速度提升 2~3 倍 |
| 領域專家分群 | 程式/數學/科學/創意各 224 專家 | 專業任務精度 +5~8% |
| 三級量化 | 雲端 MXFP4/8 → 邊緣 AWQ → 手機 GPTQ-3bit | 部署成本降低 50% |
| 長期記憶 | 外部 FAISS 向量庫 + 壓縮摘要 | 跨會話一致性提升 |

---

## 2. 前 5 名 LLM 對比分析

### 2.1 排名總覽（2026-08）

| 排名 | 模型 | 開發商 | 總參數 | 開源 | 定價 (Input/Output) |
|:---:|:---|:---|:---:|:---:|:---|
| 🥇 | Claude Fable 5 | Anthropic | 未公開 | ❌ | $10.00 / $50.00 |
| 🥈 | Claude Mythos 5 | Anthropic | 未公開 | ❌ | 未公開 |
| 🥉 | Claude Opus 5 | Anthropic | 未公開 | ❌ | $5.00 / $25.00 |
| 4 | GPT-5.6 Sol | OpenAI | 未公開 | ❌ | $5.00 / $30.00 |
| 5 | **Kimi K3** | **Moonshot AI** | **2.8T** | **✅** | **$3.00 / $15.00** |

### 2.2 各模型優缺點詳細分析

#### 🥇 Claude Fable 5（Anthropic）

**優點：**
- **絕對品質第一**：Swfte 綜合品質指數 100/100
- **常識推理最強**：SimpleBench 81.9%，顯著領先其他模型
- **電腦使用能力頂尖**：OSWorld 85%，可執行複雜 GUI 操作
- **長上下文支援**：1M tokens，文件分析能力卓越
- **安全性與對齊**：Anthropic 的 Constitutional AI 確保輸出安全

**缺點：**
- **價格極高**：$10/$50 per 1M tokens，是 Kimi K3 的 3.3 倍
- **完全閉源**：無法自架、無法微調、無法審計
- **性價比低**：對於大多數應用場景，品質提升邊際遞減
- **延遲較高**：深度推理導致首 token 延遲顯著

#### 🥈 Claude Mythos 5（Anthropic）

**優點：**
- **基準測試王者**：BenchAlign v5.2 得分 83.04，全榜第一
- **深度推理與知識整合**：在需要多步推理的任務上表現最佳
- **Agentic 能力強**：長期任務規劃與執行能力突出

**缺點：**
- **技術細節極少**：公開論文與技術報告有限
- **定價不透明**：企業級定價需單獨洽談
- **無法定制**：不支援領域適配或風格調整

#### 🥉 Claude Opus 5（Anthropic）

**優點：**
- **品質接近 Fable**：品質指數 99/100，差距極小
- **科學推理頂尖**：GPQA Diamond 96.2%（通過 Sonnet 5 實現）
- **長期 Agentic 工作**：適合數小時級別的複雜任務
- **性價比優於 Fable**：價格僅為 Fable 的一半

**缺點：**
- **仍屬高價位**：$5/$25 對中小企業仍有壓力
- **閉源限制**：與 Fable 相同的生態封閉問題
- **硬體資源要求高**：API 調用延遲不穩定

#### 4. GPT-5.6 Sol（OpenAI）

**優點：**
- **推理與編碼雙冠**：GPQA Diamond 94.6% + SWE-Bench 96.2%
- **瀏覽能力最強**：BrowseComp 92.2%，即時資訊檢索無敵
- **終端操作能力**：Terminal-Bench 88.8%，DevOps 場景首選
- **生態最完善**：ChatGPT、API、插件生態成熟

**缺點：**
- **閉源且政策多變**：歷史上有多次模型降級爭議
- **多模態能力弱於 K3**：圖像/影片理解非其強項
- **定價策略激進**：高頻調用成本累積迅速
- **數據隱私疑慮**：訓練數據使用政策不透明

#### 5. Kimi K3（Moonshot AI）⭐ 開源之光

**優點：**
- **開源權重唯一前 5**：Kimi K3 License 允許商業使用
- **GPQA Diamond 93.5%**：開源模型中推理能力第一
- **1M token 長上下文**：與 Claude 系列同等級
- **極高性價比**：$3/$15，僅為 Claude Fable 的 30%
- **Agentic 能力驗證**：48 小時自主設計 AI 晶片
- **原生多模態**：統一處理文字/圖像/影片

**缺點：**
- **推理速度較慢**：2.8T MoE 的稀疏激活 overhead
- **硬體需求極高**：需 64+ 加速器才能全量部署
- **部署門檻高**：2.8T 參數對基礎設施要求苛刻
- **訓練數據未公開**：數據來源與清洗流程不透明

---

## 3. Kimi K3 框架解析

### 3.1 三大創新軸

Kimi K3 的架構創新圍繞三個維度展開：

```
┌─────────────────────────────────────────────────────────────┐
│                    Kimi K3 三軸創新                          │
├─────────────────────────────────────────────────────────────┤
│  序列長度 (Sequence Length)                                  │
│  ├── Kimi Delta Attention (KDA)                              │
│  │   └── 3:1 比例交替線性注意力與全局注意力                   │
│  ├── Gated Multi-Layer Attention (Gated MLA)                 │
│  │   └── 門控機制動態調整注意力強度                          │
│  └── NoPE (No Positional Encoding)                           │
│      └── 無需位置編碼即可外推至 1M tokens                    │
├─────────────────────────────────────────────────────────────┤
│  模型深度 (Model Depth)                                      │
│  └── Attention Residuals (AttnRes)                           │
│      └── 跨層殘差連接，每層可學習引用前 N 層輸出             │
│      └── 突破標準 Transformer 的單一路徑限制                  │
├─────────────────────────────────────────────────────────────┤
│  模型寬度 (Model Width)                                      │
│  └── Stable LatentMoE                                        │
│      ├── 896 個專家，每 token 激活 16 個                     │
│      ├── 稀疏度達 56× (2.8T / 104B)                          │
│      ├── 量化平衡機制穩定訓練                                │
│      └── SiTU-GLU 激活函數                                   │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 關鍵規格

| 參數 | 數值 | 說明 |
|:---|:---:|:---|
| 總參數 | 2.8T | Mixture-of-Experts |
| 激活參數 | 104B | 每 forward pass |
| 層數 | 93 | 8 blocks × 12 layers + 1 dense |
| 隱藏維度 | 7,168 | |
| 注意力頭數 | 96 | |
| KV 頭數 | 8 | GQA 分組查詢注意力 |
| 上下文長度 | 1,048,576 | 1M tokens |
| 詞表大小 | 160K | |
| 專家數 | 896 | |
| 激活專家數 | 16 | 每 token |

### 3.3 量化策略

K3 在**訓練階段即引入量化感知**：
- **權重量化**：MXFP4 (Microscaling FP4)
- **激活量化**：MXFP8 (Microscaling FP8)
- **優勢**：避免訓練後量化的精度損失，推理時無需額外轉換

---

## 4. Kimi K3+ 架構設計

### 4.1 整體架構

```
┌─────────────────────────────────────────────────────────────┐
│                    Kimi K3+ 概念架構                          │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: 輸入層 (Multimodal Input)                          │
│  ├── 文字編碼器 (160K 詞表 Embedding)                        │
│  ├── 圖像編碼器 (ViT-Large, 448×448)                         │
│  ├── 影片編碼器 (TimeSformer, 16幀採樣)                      │
│  ├── 音訊編碼器 (Whisper-Large, 16kHz)                       │
│  └── 動態解析度適配器 (Adaptive Token Sampler)               │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: 混合注意力層 (Hybrid Attention Stack)              │
│  ├── KDA 層 (3:1 比例) ──→ O(n) 線性複雜度                  │
│  ├── Gated MLA 層 ──→ 全局關鍵資訊提取                       │
│  └── 跨層 AttnRes ──→ 深度資訊流優化 (殘差深度: 4)          │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: 稀疏專家層 (Stable LatentMoE+)                     │
│  ├── 896 路由專家 + 4 共享專家 (K3 為 2 個)                  │
│  ├── 動態專家激活 (8~16 可調稀疏度)                          │
│  │   └── 簡單任務: 8 專家 (提速 40%)                         │
│  │   └── 複雜任務: 16 專家 (保持精度)                        │
│  └── 領域專家分群:                                           │
│      ├── Group 0-223:   程式設計 (Code/Debug)                │
│      ├── Group 224-447: 數學推理 (Math/Symbolic)             │
│      ├── Group 448-671: 科學文獻 (Science/Literature)        │
│      └── Group 672-895: 創意寫作 (Writing/Design)            │
├─────────────────────────────────────────────────────────────┤
│  Layer 4: 推理與工具層 (Reasoning & Agent Layer)             │
│  ├── Chain-of-Thought 編譯器 (XML 結構化輸出)                │
│  ├── 工具調用路由器 (MCP Protocol)                           │
│  │   ├── 最大 8 工具/輪次                                    │
│  │   └── 30 秒超時保護                                       │
│  └── 長期記憶緩存:                                           │
│      ├── 外部 KV Cache (FAISS 向量庫)                        │
│      ├── 記憶壓縮 (壓縮比 10:1)                              │
│      └── 會話摘要觸發閾值: 16K tokens                        │
├─────────────────────────────────────────────────────────────┤
│  Layer 5: 輸出層 (Output Layer)                              │
│  ├── 多模態解碼器                                            │
│  └── 推測解碼加速 (Speculative Decoding)                     │
│      ├── 草稿模型: Kimi-K3-Plus-Draft-30B                    │
│      ├── 樹注意力驗證 (Tree Attention)                       │
│      └── 預期提速: 2~3×                                     │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 相較 K3 的關鍵改進

| 改進項目 | K3 現狀 | K3+ 改進 | 預期效果 |
|:---|:---|:---|:---|
| 共享專家數 | 2 個 | **4 個** | 通用知識表達更穩定 |
| 專家激活 | 固定 16 個 | **動態 8~16 個** | 簡單任務提速 40% |
| 推測解碼 | 無 | **30B 草稿模型** | 輸出速度 2~3× |
| 領域分群 | 無 | **4 大領域群組** | 專業任務 +5~8% |
| 長期記憶 | 無 | **FAISS + 壓縮** | 跨會話一致性提升 |
| 量化部署 | 單一方案 | **三級量化** | 成本降低 50% |
| 多模態 | 文字+圖像 | **+影片+音訊** | 影片理解 SOTA |

---

## 5. 核心模塊實現

### 5.1 模塊清單

| 模塊 | 文件 | 功能 |
|:---|:---|:---|
| 配置定義 | `kimi_k3_plus_config.json` | 完整模型超參數與訓練配置 |
| 核心架構 | `kimi_k3_plus_core_modules.py` | Attention / MoE / 多模態 / 記憶 |
| 訓練流程 | `kimi_k3_plus_training.py` | 四階段訓練框架 |
| 部署方案 | `kimi_k3_plus_deploy.py` | 量化 / 服務 / 監控 |

### 5.2 Kimi Delta Attention 核心邏輯

```python
# 偽代碼核心邏輯
class KimiDeltaAttention:
    def forward(self, hidden_states, layer_idx):
        # 1. 3:1 比例決定注意力類型
        use_global = (layer_idx % 4 == 0)  # 每第4層用全局注意力

        # 2. Gated MLA: 門控動態調整
        gate = sigmoid(self.gate(hidden_states))

        # 3. 線性注意力 (O(n) 複雜度)
        if not use_global:
            q = elu(linear_q(q)) + 1  # 核函數變換
            k = elu(linear_k(k)) + 1
            kv_state = einsum(k, v)   # KV 狀態累積
            output = einsum(q, kv_state) / sum(k)

        # 4. 全局注意力 (O(n²) 複雜度)
        else:
            scores = einsum(q, k) / sqrt(dim)
            weights = softmax(scores)
            output = einsum(weights, v)

        # 5. AttnRes: 跨層殘差
        for i, residual in enumerate(past_residuals[-4:]):
            gate = sigmoid(residual_gates[layer_idx, i])
            output += gate * proj(residual)

        return output * gate
```

### 5.3 Stable LatentMoE+ 核心邏輯

```python
# 偽代碼核心邏輯
class StableLatentMoE:
    def forward(self, hidden_states, task_type=None):
        # 1. 動態稀疏度估計
        difficulty = difficulty_estimator(hidden_states)
        k = 8 + (16 - 8) * difficulty  # 動態調整 8~16

        # 2. 路由
        router_logits = router(hidden_states)

        # 3. 領域偏置
        if task_type == "programming":
            router_logits[0:224] += 2.0  # 提升程式專家

        # 4. Top-k 選擇 + 負載平衡
        top_k_values, top_k_indices = topk(router_logits, k)
        weights = softmax(top_k_values)
        aux_loss = load_balance_loss(router_prob)

        # 5. 專家計算 (SiTU-GLU)
        output = sum(weights[i] * situ_glu(expert[i](hidden_states)))

        # 6. 共享專家 (始終激活)
        for shared in shared_experts:
            output += situ_glu(shared(hidden_states))

        return output, aux_loss
```

---

## 6. 四階段訓練流程

### 6.1 訓練階段概覽

```
Phase 1: Pre-training          Phase 2: Continual Pre-training
├── Data: 15T tokens           ├── Data: 3T tokens (high-quality)
├── Context: 32K → 128K → 1M   ├── Focus: Reasoning + Code
├── LR: 1.5e-4                 ├── LR: 5e-5
├── Batch: 4096                ├── Batch: 2048
└── Target: Foundation         └── Target: Expert Balancing
         │                              │
         ▼                              ▼
Phase 3: SFT                   Phase 4: RLHF
├── Data: 500K samples         ├── Algorithm: PPO + DPO Hybrid
├── LR: 2e-5                   ├── KL Penalty: 0.02
├── Epochs: 3                  ├── Constitutional AI: Enabled
├── LoRA: r=256, α=512         ├── Principles: Helpful/Harmless/
├── Freeze AttnRes: True       │              Honest/Creative
└── Target: Instruction        └── Target: Value Alignment
              │
              ▼
       Final Model: Kimi K3+
```

### 6.2 各階段詳細配置

#### Phase 1: 預訓練

| 參數 | 數值 |
|:---|:---|
| 數據量 | 15T tokens |
| 數據組成 | 多語言 (60%) + 程式碼 (20%) + 科學文獻 (15%) + 影片字幕 (5%) |
| 上下文長度 | 漸進式：32K (30%) → 128K (30%) → 1M (40%) |
| 學習率 | 1.5e-4 (cosine with restarts) |
| Batch Size | 4096 |
| 優化器 | AdamW (β1=0.9, β2=0.95) |
| 權重衰減 | 0.1 |
| 梯度裁剪 | 1.0 |
| 精度 | bfloat16 |
| 預計時間 | ~3 個月 (1024 H100) |

#### Phase 2: 持續預訓練

| 參數 | 數值 |
|:---|:---|
| 數據量 | 3T tokens |
| 數據重點 | 高品質推理鏈 (Chain-of-Thought) + 程式碼執行軌跡 |
| 學習率 | 5e-5 |
| 專家負載平衡 | 啟用 (loss_coef=0.01) |
| 量化感知訓練 | 啟用 (MXFP4/8) |
| 預計時間 | ~2 週 (1024 H100) |

#### Phase 3: SFT

| 參數 | 數值 |
|:---|:---|
| 數據量 | 500K 對話樣本 |
| 數據組成 | 多輪對話 (40%) + Agent 軌跡 (30%) + 多模態指令 (30%) |
| LoRA 配置 | r=256, α=512, dropout=0.05 |
| 目標模塊 | q_proj, v_proj, o_proj, gate_proj |
| AttnRes 凍結 | 是 (保持預訓練學到的跨層模式) |
| 學習率 | 2e-5 |
| Epochs | 3 |
| 預計時間 | ~3 天 (256 H100) |

#### Phase 4: RLHF

| 參數 | 數值 |
|:---|:---|
| 算法 | PPO (70%) + DPO (30%) 混合 |
| 獎勵模型 | Kimi-K3-Plus-Reward (獨立訓練) |
| KL 懲罰 | 0.02 |
| PPO Epochs | 4 |
| DPO Beta | 0.1 |
| Constitutional AI | 啟用 (4 大原則) |
| 預計時間 | ~1 週 (256 H100) |

---

## 7. 部署與量化方案

### 7.1 三級部署架構

```
┌─────────────────────────────────────────────────────────────┐
│                      部署目標選擇                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   🌩️ 雲端部署                    🖥️ 邊緣部署               │
│   ├─ 硬體: 64+ H100/A100        ├─ 硬體: 單卡 48GB+        │
│   ├─ 量化: MXFP4/8              ├─ 量化: AWQ INT4          │
│   ├─ 框架: vLLM / TGI           ├─ 框架: vLLM / TGI        │
│   ├─ 吞吐量: 最高               ├─ 吞吐量: 中等            │
│   ├─ 延遲: <100ms               ├─ 延遲: <500ms            │
│   └─ 成本: $1.5/$8 per 1M       └─ 成本: 電費為主         │
│                                                             │
│   📱 手機部署                                               │
│   ├─ 硬體: 通過 API 或極小模型                              │
│   ├─ 量化: GPTQ-3bit / GGUF                                 │
│   ├─ 框架: llama.cpp / mlc-llm                              │
│   ├─ 吞吐量: 受限                                            │
│   └─ 延遲: 本地即時                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 量化對比

| 部署級別 | 權重位元 | 激活位元 | 格式 | 模型大小 | 精度損失 |
|:---|:---:|:---:|:---|:---:|:---:|
| 雲端 (原始) | BF16 | BF16 | - | ~5.6 TB | 0% |
| 雲端 (量化) | 4 | 8 | MXFP | ~350 GB | <1% |
| 邊緣 | 4 | 4 | AWQ | ~175 GB | ~2% |
| 手機 | 3 | 4 | GPTQ | ~131 GB | ~5% |

### 7.3 推理優化策略

| 優化技術 | 實現方式 | 預期效果 |
|:---|:---|:---|
| 推測解碼 | 30B 草稿模型 + 樹注意力驗證 | 速度 2~3× |
| KV Cache 壓縮 | 均值池化舊 tokens | 記憶體節省 50% |
| 連續批處理 | 動態長度分桶 | 吞吐量 +30% |
| 專家並行 | 8 GPU 各負責 112 專家 | 延遲降低 40% |

---

## 8. 性能目標與基準

### 8.1 目標基準對比

| 基準測試 | 當前 SOTA | K3+ 目標 | 提升策略 |
|:---|:---:|:---:|:---|
| GPQA Diamond | GPT-5.6 Sol (94.6%) | **95.0%+** | 領域專家分群 + 推理數據強化 |
| SWE-Bench Verified | GPT-5.6 Sol (96.2%) | **95.0%+** | 程式專家群 + 執行軌跡訓練 |
| MMLU-Pro | Claude Fable 5 | **92.0%+** | 科學專家群 + 知識檢索增強 |
| SimpleBench | Claude Fable 5 (81.9%) | **80.0%+** | 常識推理數據擴充 |
| OSWorld | Claude Fable 5 (85%) | **82.0%+** | GUI 操作軌跡訓練 |
| 輸出速度 | GPT-4.1 (529 tok/s) | **200+ tok/s** | 推測解碼 + 動態稀疏度 |
| 上下文長度 | Grok-4 (2M) | **2M tokens** | KDA 線性注意力擴展 |
| 部署成本 | Kimi K3 ($3/$15) | **$1.5/$8** | 三級量化 + 專家裁剪 |

### 8.2 與競品對比矩陣

| 維度 | Claude Fable 5 | GPT-5.6 Sol | Kimi K3 | **Kimi K3+ (目標)** |
|:---|:---:|:---:|:---:|:---:|
| 品質 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐⭐ |
| 開源性 | ❌ | ❌ | ✅ | ✅ |
| 性價比 | ⭐⭐☆☆☆ | ⭐⭐⭐☆☆ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 速度 | ⭐⭐⭐☆☆ | ⭐⭐⭐⭐☆ | ⭐⭐☆☆☆ | ⭐⭐⭐⭐☆ |
| 多模態 | ⭐⭐⭐⭐☆ | ⭐⭐⭐☆☆ | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐⭐ |
| 可部署性 | ⭐⭐☆☆☆ | ⭐⭐☆☆☆ | ⭐⭐⭐☆☆ | ⭐⭐⭐⭐⭐ |
| Agentic | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐⭐ |

---

## 9. 文件清單

| 文件 | 說明 | 大小 |
|:---|:---|:---:|
| `kimi_k3_plus_config.json` | 完整模型配置 (JSON) | ~15 KB |
| `kimi_k3_plus_core_modules.py` | 核心模塊偽代碼 (Python) | ~12 KB |
| `kimi_k3_plus_training.py` | 四階段訓練框架 (Python) | ~10 KB |
| `kimi_k3_plus_deploy.py` | 部署與量化方案 (Python) | ~11 KB |
| `Kimi_K3_Plus_Technical_Whitepaper.md` | 本技術白皮書 | ~20 KB |

---

## 附錄：術語表

| 術語 | 解釋 |
|:---|:---|
| **MoE** | Mixture-of-Experts，混合專家模型 |
| **KDA** | Kimi Delta Attention，Kimi 差分注意力 |
| **Gated MLA** | Gated Multi-Layer Attention，門控多層注意力 |
| **AttnRes** | Attention Residuals，注意力殘差 |
| **SiTU-GLU** | Sigmoid-Tanh Unit Gated Linear Unit，激活函數 |
| **NoPE** | No Positional Encoding，無位置編碼 |
| **MXFP** | Microscaling Floating Point，NVIDIA 微縮放浮點格式 |
| **AWQ** | Activation-aware Weight Quantization，激活感知權重量化 |
| **GPTQ** | General-purpose Post-Training Quantization，通用訓練後量化 |
| **PPO** | Proximal Policy Optimization，近端策略優化 |
| **DPO** | Direct Preference Optimization，直接偏好優化 |
| **RLHF** | Reinforcement Learning from Human Feedback，人類反饋強化學習 |
| **Constitutional AI** | 憲法 AI，Anthropic 提出的安全對齊方法 |
| **Speculative Decoding** | 推測解碼，使用草稿模型加速生成 |
| **KV Cache** | Key-Value Cache，注意力機制的鍵值緩存 |

---

> **聲明**：本文檔為基於 Kimi K3 框架的概念性技術設計方案。Kimi K3 為 Moonshot AI 的商標與知識產權。Kimi K3+ 為本文作者提出的增強概念，非官方產品。
