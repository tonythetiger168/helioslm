# HeliosLM P0-P2 設計文件（v5.23-v5.25）

## 原則：每個能力附可執行 oracle
Agent 層最難驗證，故：ground truth by construction；decision gate 繞不過
（v5.22 routing monotonicity gate）；bitwise 精神延伸到行為層；toy-first。

## P0 — Agent layer（v5.23）
- schema.py：嚴格 tool-call wire format（16 錯誤類、深度/大小上限、並行介面）
- tools.py：確定性工具（AST 白名單 calc、str_op、sandbox 檔案 I/O、finish）
- trajectory.py：serde + verify_replay（T5：同 ids ⇒ 同 observations，bitwise）
- envs/：calc/str/compose（ground truth by construction，T9）
- gate.py：Fixed/Oracle/ThresholdGate(τ) + routing monotonicity gate（T6）
- loop.py：plan→act→observe；PARSE_ERROR 恢復（T8）；三模式同題集（T7）
- benchmark.py：三模式對比 + score_stream JSONL 匯出

## P1 — 注意力變體（v5.24）
- dsa.py：sparse top-k decode；兩層 oracle（fp32 證書：pseudo-max gap ⇒
  dropped < 2^-40；fp64 gate：≤1e-9 vs dense）。不宣稱 fp32 bitwise；
  tie-break 確定性（低 index 優先）
- attn_res.py：x_{l+1} = x_l + y_l + Σ α_{l,i}·y_i；α 零初始化 ⇒ 與 vanilla
  residual bitwise 一致（遷移 gate，T12a）；訓練後 ⇒ 確定性 gate（T12b）

## P2 — Disagg evolver 模組（v5.25）
- disagg.py：Mooncake-style 分離的分析模型；monotonicity gate：固定
  workload 指紋上加 worker 不得增加 modeled makespan（T13）
- 三軸 Pareto：makespan · n_workers · worker-seconds

## 測試總表
T1-T9（agent）、T11（DSA）、T12（AttnRes）、T13（disagg）。
T5/T6/T12a/T13 為硬性 oracle gate：違反即 bug，不是 trade-off。
