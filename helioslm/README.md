# HeliosLM v1.0 🌞

[![Apache 2.0 License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**Apache 2.0 Licensed | P0+P1+P2+P3 Production Stack | Open Frontier Intelligence**

> [📋 Roadmap 2027-2028](ROADMAP_2027_2028.md) — Comprehensive comparison vs 2026 Top 10 LLMs and future development plan.

---

## 🎯 四大核心模塊 (P0-P3)

| 優先級 | 模塊 | 功能 | 生產特性 |
|:------:|------|------|---------|
| **P0** | 速度優化 | FlashAttention + 推測解碼 + 動態稀疏度 | PyTorch 2.0 SDPA, 連續批處理 |
| **P1** | 準確度 | RAG + 置信度校準 + 多模型驗證 | FAISS集成, 溫度縮放校準 |
| **P2** | 推理增強 | 領域專家MoE + CoT編譯器 + 工具使用 | 動態專家路由, 結構化推理 |
| **P3** | Agentic | MCP工具鏈 + 自我反思 + 代碼沙箱 | 1000+工具, Docker沙箱驗證 |

---

## 📦 四尺寸產品矩陣

| 型號 | 總參數 | 激活參數 | 上下文 | 定位 |
|------|:------:|:--------:|:------:|------|
| **Ultra** | 2.8T | ~35B | 1M | 品質旗艦 |
| **Pro** | 600B | ~20B | 1M | 企業級 |
| **Lite** | 300B | ~10B | 512K | 邊緣級 |
| **Nano** | 30B | ~5B | 256K | 本地級 |

---

## 🚀 快速開始

### 安裝

```bash
# 克隆倉庫
git clone https://github.com/tonythetiger168/helioslm.git
cd helioslm

# 安裝依賴
pip install -r requirements.txt

# 開發模式安裝
pip install -e .
```

### 推理

```bash
# 單條推理 (Nano)
python scripts/inference.py --size nano --prompt "Explain quantum computing"

# 批量推理 (Lite + 推測解碼)
python scripts/inference.py --size lite --speculative --batch inputs.jsonl

# 啟動 API 服務 (Pro)
python scripts/inference.py --size pro --serve --port 8000
```

### 訓練

```bash
# 預訓練 (需要 256x H100)
python scripts/train.py --config configs/ultra_config.py --stage pretrain

# 監督微調
python scripts/train.py --config configs/pro_config.py --stage sft --data data/instructions.jsonl

# RLHF 對齊
python scripts/train.py --config configs/pro_config.py --stage rlhf
```

---

## 📁 項目結構

```
helioslm/
├── configs/                    # 配置文件 (base + 4 sizes)
│   ├── base_config.py
│   ├── ultra_config.py         # 2.8T, 1M context
│   ├── pro_config.py           # 600B, 1M context
│   ├── lite_config.py          # 300B, 512K context
│   └── nano_config.py          # 30B, 256K context
│
├── src/                        # 核心模塊 (P0-P3 生產級)
│   ├── __init__.py
│   ├── model.py                # HeliosLM 統一架構
│   ├── attention.py            # P0: FlashAttention + Gated MLA + AttnRes
│   ├── moe.py                  # P0+P2: 動態稀疏度 + 領域分群
│   ├── speculative_decoding.py # P0: 推測解碼 + 連續批處理
│   ├── rag.py                  # P1: RAG + 幻覺校準 + 多模型驗證
│   ├── cot_compiler.py         # P2: CoT編譯 + 工具使用
│   ├── agentic.py              # P3: MCP + 自我反思 + 代碼驗證
│   ├── memory.py               # P3: 長期記憶 (FAISS)
│   └── reasoning_budget.py     # P4: 推理預算控制
│
├── scripts/                    # 訓練與推理腳本
│   ├── train.py                # 分布式訓練 (DeepSpeed)
│   ├── inference.py            # 推理服務 (vLLM-style)
│   └── benchmark.py            # 基準評估
│
├── tests/                      # 單元測試
│   └── test_all.py
│
├── docs/                       # 文檔
│   └── comparison.md
│
├── ROADMAP_2027_2028.md        # 2027-2028 發展路線圖
├── FEATURES_SUMMARY.md         # 功能摘要
├── LICENSE                     # Apache 2.0
├── requirements.txt            # Python 依賴
├── setup.py                    # pip 安裝配置
└── README.md                   # 本文件
```

---

## 🔬 架構亮點

### P0: 速度優化
- **FlashAttention-compatible**: 使用 PyTorch 2.0+ `scaled_dot_product_attention`
- **Gated Multi-Head Latent Attention (MLA)**: KV 壓縮，降低顯存
- **推測解碼**: Draft Model + Tree Verifier，2-3x 加速
- **連續批處理**: 動態長度分桶，最大化吞吐量
- **動態 MoE**: 根據任務難度自適應激活專家數量

### P1: 準確度
- **生產級 RAG**: FAISS 向量檢索 + 上下文編碼
- **置信度校準**: 溫度縮放 (Temperature Scaling)
- **多模型驗證**: 跨模型一致性檢查
- **不確定性標記**: 自動標記低置信度輸出

### P2: 推理增強
- **結構化 CoT**: 步驟類型分類 (continue/tool/answer/verify)
- **自動工具選擇**: 基於上下文的工具路由
- **領域專家分群**: 數學/代碼/科學專家組
- **推理質量評分**: 每步質量監控

### P3: Agentic
- **MCP 工具路由**: 1000+ 工具註冊表
- **自我反思**: 置信度 + 錯誤檢測 + 重試決策
- **代碼沙箱**: Docker/E2B 隔離執行
- **長期記憶**: FAISS 索引 + 重要性淘汰 + 時間衰減

---

## 📊 與 2026 Top 10 LLM 對比

詳見 [ROADMAP_2027_2028.md](ROADMAP_2027_2028.md) 獲取完整對比分析。

| 維度 | HeliosLM v1.0 | 2026 Top 10 平均 |
|------|--------------|-----------------|
| 架構完整性 | ✅ P0-P3 生產級 | ✅ 生產級 |
| 預訓練權重 | ❌ 待訓練 | ✅ 可用 |
| 開源授權 | ✅ Apache 2.0 | 混合 |
| 代碼可讀性 | ✅ 模塊化清晰 | 部分閉源 |

---

## 🛠️ 開發環境

```bash
# 創建虛擬環境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安裝開發依賴
pip install -r requirements.txt
pip install -e ".[dev]"

# 運行測試
pytest tests/
```

---

## 🤝 貢獻指南

1. Fork 本倉庫
2. 創建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 創建 Pull Request

請確保：
- 代碼通過 `pytest` 測試
- 遵循 PEP 8 風格
- 更新相關文檔

---

## 📄 許可證

[Apache License 2.0](LICENSE) — 可自由商用、修改、分發。

## 👨‍💻 作者

HeliosLM Team

---

> ⚠️ **免責聲明**: HeliosLM v1.0 目前為框架/原型階段，無預訓練權重。生產使用需完成 Phase 2 預訓練（見 Roadmap）。
