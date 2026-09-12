# Kimi K3+ v2.0 🚀

> **一个架构，四个尺寸，全场景覆盖**

基于 Kimi K3 框架的增强版本，融合 DeepSeek 的性价比策略与 Qwen 的速度优化。

## 四尺寸产品矩阵

| 型号 | 总参数 | 激活参数 | 上下文 | 定位 | 部署场景 |
|:---|:---:|:---:|:---:|:---|:---|
| **Ultra** | 2.8T | 104B | 2M | 云端旗舰 | 数据中心 / 超算 |
| **Pro** | 600B | 24B | 1M | 企业级 | 8x A100 节点 |
| **Lite** | 300B | 12B | 512K | 边缘级 | 单卡 48GB |
| **Nano** | 30B | 30B | 128K | 本地级 | 单卡 24GB / 边缘设备 |

## 三大阶段增强

### Phase 1: 速度 + 成本 ⚡
- **推测解码**: 30B 草稿模型 + 树注意力验证，提速 2.5~3.5x
- **动态稀疏度**: 根据任务难度调整 8~16 专家，简单任务提速 40%
- **三重量化**: 云端 MXFP4/8 → 边缘 AWQ → 本地 GPTQ-3bit

### Phase 2: 多模态 + 部署 🎬
- **统一多模态编码器**: ViT-Giant + TimeSformer + Whisper
- **视频时序感知**: 运动感知注意力，32帧时序分辨率
- **音频情感识别**: 支持 8 种情感 + 说话人识别

### Phase 3: Agentic + 生态 🤖
- **MCP 工具链**: 支持 1000+ 工具，8 工具/轮次
- **长期记忆**: FAISS 向量库 + 10:1 压缩 + 会话摘要
- **自我反思**: 置信度评估 + 错误检测 + 自动重试

## 快速开始

```bash
# 安装依赖
pip install torch numpy tqdm

# 运行测试
python tests/test_all.py

# 训练 (选择尺寸)
python scripts/train.py --size ultra --phase pretrain
python scripts/train.py --size nano --phase distill

# 推理
python scripts/inference.py --size lite --prompt "解释量子计算" --speculative

# 基准测试
python scripts/benchmark.py
```

## 项目结构

```
kimi_k3_plus_v2/
├── configs/              # 配置
│   ├── base_config.py    # 统一基础配置
│   ├── ultra_config.py   # 2.8T 云端
│   ├── pro_config.py     # 600B 企业
│   ├── lite_config.py    # 300B 边缘
│   └── nano_config.py    # 30B 本地
├── src/                  # 核心源码
│   ├── attention.py      # KDA + Gated MLA + AttnRes
│   ├── moe.py            # 动态稀疏度 MoE
│   ├── speculative_decoding.py  # Phase 1
│   ├── multimodal.py     # Phase 2
│   ├── agentic.py        # Phase 3
│   ├── memory.py         # 长期记忆
│   ├── model.py          # 统一架构
│   └── trainer.py        # 训练器
├── scripts/              # 脚本
│   ├── train.py
│   ├── inference.py
│   └── benchmark.py
└── tests/
    └── test_all.py
```

## 性能目标

| 指标 | 当前 K3 | K3+ 目标 | 对标 |
|:---|:---:|:---:|:---|
| 速度 | 35 tok/s | **200+ tok/s** | Qwen 3.7 (206 tok/s) |
| 输出成本 | $15/M | **$3/M** | DeepSeek Flash ($0.66/M) |
| 缓存成本 | $0.30/M | **$0.03/M** | DeepSeek Flash ($0.007/M) |
| 多模态 | 文字+图像 | **全模态 SOTA** | Qwen 3.8-Max |
| 部署门槛 | 64 GPU | **单卡 48GB** | DeepSeek Flash |

## 许可证

Kimi K3 License
