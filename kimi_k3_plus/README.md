# Kimi K3+

基于 Kimi K3 框架的开源前沿智能 LLM 增强版本。

## 特性

- **Kimi Delta Attention (KDA)**: 3:1 比例线性/全局注意力，支持 1M 上下文
- **Gated MLA**: 门控多层级注意力机制
- **Attention Residuals (AttnRes)**: 跨层残差连接，深度 4
- **Stable LatentMoE+**: 896 专家，动态稀疏度 8~16
- **领域专家分群**: 程式/数学/科学/创意四大领域
- **多模态**: 原生支持文字/图像/视频/音频

## 快速开始

```bash
pip install torch numpy tqdm
python tests/test_model.py
python scripts/train.py
python scripts/inference.py
```

## 项目结构

```
kimi_k3_plus/
├── configs/         # 配置文件
├── src/             # 核心源代码
│   ├── attention.py # KDA + Gated MLA + AttnRes
│   ├── moe.py       # Stable LatentMoE+
│   ├── model.py     # 主模型
│   └── trainer.py   # 四阶段训练器
├── scripts/         # 训练与推理脚本
└── tests/           # 单元测试
```

## 许可证

Kimi K3 License (基于 Moonshot AI 开源协议)
