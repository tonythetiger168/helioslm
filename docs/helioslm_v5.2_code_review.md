# helioslm v5.2 — 代碼審查報告(含 README features 審計)

**日期:** 2026-09-12
**範圍:** v5.2 全部 14 個 Python 模組 + tests(15)+ README/CHANGELOG
**方法:** 4 位獨立 reviewer 並行精讀 + 全部關鍵 findings 實證(含 orchestrator 抽查復現)
**對比基線:** v5.0 審查發現 14 CRITICAL / 27 MAJOR;v5.2 為重寫後重新審查

---

## Executive Summary

**總體結論:v5.2 的核心數學與架構聲稱基本屬實,質量較 v5.0 有質的飛躍** —— absorption 雙模式等價(≤3.6e-7)、GPTQ 與獨立參考實現逐元素一致(5.5e-7)、MTP batch/回滾/引擎合批全部與 greedy 逐 token 一致、DualPipe 梯度逐位相等、FP8 grid 與原生 float8 十萬值零失配。

但本輪發現 **1 CRITICAL + 10 MAJOR + ~20 MINOR**,主要集中在邊界場景(left-padding、採樣路徑、fp16/CUDA、校准質量)與一處 README 數字誤標。

| 嚴重度 | 數量 | 分佈 |
|---|---|---|
| CRITICAL | 1 | absorbed MLA left-padding NaN 跨層污染 |
| MAJOR | 10 | 推理採樣 1、量化 3、訓練 3、模型核心 3(不含 CRITICAL) |
| MINOR | ~20 | 各模組 |

---

## CRITICAL

### C1. Absorbed 路徑 left-padding 產生 NaN 並跨層污染真實 token
**`src/attention/mla.py:385-398`**(orchestrator 已獨立復現)
全 mask 的 pad query 行 → `softmax(-inf…)` = NaN → 進入 residual → 下一層 `c_kv` 在 pad 位置變 NaN → 真實 query `0 × NaN = NaN` → **整行真實 token logits 全 NaN**。left-padding 是 batched generation 標準做法;expanded 模式無此問題(SDPA 全 mask 行返回 0),兩模式在此場景不等價。
→ 修復:`probs = torch.nan_to_num(F.softmax(scores, -1))`。

---

## MAJOR

### 推理
- **M-I1. MTP 採樣 correction token 雙重 temperature** — `mtp.py:455-456`:`p` 已是概率向量,`_sample_from` 內部再次 `softmax(p/T)`。實測分佈從 `[0.763,…]` 被平坦化為 `[0.468,…]`。**預設路徑(temperature=0.7)即命中**;greedy 不受影響。→ `multinomial(p)`;另:嚴格 speculative sampling 應從 `norm((p−q)₊)` 重採,當前為近似,建議修正或文檔化。
- **M-I2. `_apply_rope` 對文檔承諾的 `[S,D]` 形狀廣播錯誤** — `paged_attention.py:232-238`:B≠S 崩潰、B==S **靜默錯誤**。倉內調用均傳 None 故無現時影響,屬 API 陷阱。→ reshape `[1,S,1,D]` 或對 2-D 輸入 raise。
- **M-I3. BlockManager 賬簿不含 prompt_pad** — `vllm_engine.py:165-168`:watermark 越大,塊會計與真實 KV 內存脫節越嚴重,OOM 防護失真。

### 量化 / README
- **M-Q1. GPTQLinear fp16/bf16 輸入崩潰** — `standard_quant.py:409`:bias 未 cast(orchestrator 已復現 RuntimeError)。AWQ/FP8 都有 cast,GPTQ 漏了。
- **M-Q2. GPTQ 校准路徑 CUDA 崩潰** — `standard_quant.py:286,316`:`err_blk` 硬編碼 CPU,in_features>128 且權重在 GPU 時跨設備 matmul 報錯(靜態判定,置信度高)。
- **M-Q3. AWQ 校准路徑(activations=)在所有測試分佈下都比純 RTN 差** — `standard_quant.py:80-85`:尖峰激活下輸出誤差 ~100%(RTN 6.6%),而 AWQ 恰恰為尖峰激活而生;α=1 全幅度 scaling、無 clip grid-search;**零測試覆蓋**。
- **M-Q4. README/CHANGELOG「GPTQ 10.1%→3.4% @ real lite lm_head」數字誤標** — 真實 lm_head 激活復算為 10.06%→**7.07%**;3.4% 來自合成低秩測試數據。算法是真的,數字貼錯了標籤。→ 改為實測 ~7% 或註明合成數據。

### 訓練
- **M-T1. GRPO 全程在 eval 模式訓練** — `grpo.py:230` + `model_v5.py:222`:generate 內 `self.eval()` 不恢復;當前 dropout=0 無數值影響,一旦配置 dropout>0 訓練將靜默失效。→ train_step 開頭 `self.model.train()`。
- **M-T2. FP8 amax history 被 NaN 永久污染** — `fp8_trainer.py:110-116`:一個 NaN batch 即可讓 scale 與 master 權重永久 NaN。→ `nan_to_num`/`isfinite` 守衛。
- **M-T3. GRPO reward 子串匹配假陽性** — `grpo.py:137`:`"25"` 匹配答案 `"2"`。→ 抽取最終答案後相等比較。

### 模型核心
- **M-C1. `_build_attn_mask` 比較 key 索引與 query 位置,單位不一致** — `mla.py:415-417`:默認單調位置下正確;非單調 position_ids(packed sequences)下静默錯誤。→ position 空間比較或斷言單調。
- **M-C2. MoE `expert_load` 在 eval/推理時也累計** — `sigmoid_moe.py:114-119`:RL rollout 會污染 aux-free bias 更新依據。→ `if self.training` 門控。
- **M-C3. 分布式 update_bias 統計失真 + 各 rank bias 發散** — `sigmoid_moe.py:149-165`:遠端 expert 負載恆 0、無 all-reduce、route_bias 非 DDP 同步。
- **M-C4. RoPE 惰性增長破壞 checkpoint 重載** — `mla.py:75-85`:persistent buffer 擴容後 strict load 失敗。→ `persistent=False`。

---

## MINOR(摘要)

- 測試:`test_v5.py:299-300` 存在永真断言(EOS 檢查永遠不會失敗);test_quantization 的「GPTQ」實走 RTN fallback(標籤誤導);weight property/MTP 重綁定/GPTQ 奇數維/AWQ 校准路徑無單測覆蓋。
- GRPO:B×G 全圖同時存活的內存風險(m1);序列級 ratio 溢出無 clamp(m3);k3 在 |logr|~1e-6 有災難性抵消,建議 `expm1`(m4);generate 續寫/全序列啟發式判別有誤判路徑(m2)。
- 引擎:`_pad_caches` memo 無上限(F4);MTP 路徑靜默丟棄 top_p(F5,model_v5.py:218)。
- 模型:cos/sin 未 cast 到 x.dtype(bf16 CUDA 隱患,m1);config 死字段 batch_size/lr;**版本漂移:docstring/model_name 仍寫 "v5.0"**;NaViT 邊界輸入拋裸異常。
- DualPipe:`num_micro_batches` 死參數;`run_dual([])` 裸 IndexError;recompute 不保存 RNG(dropout 層分離路徑不自洽)。

---

## README / CHANGELOG 審計表

| 宣稱 | 結論 | 說明 |
|---|---|---|
| MLA absorbed −97.7%(full)/ −71.9%(lite) | ✅ SUPPORTED | 復算成立 |
| absorbed ≡ expanded <1e-4 | ✅ SUPPORTED(有例外) | C1:left-padding 場景不成立,需修復後補測試 |
| 真 GPTQ(Hessian/OBS) | ✅ SUPPORTED | 與獨立參考實現差 5.5e-7 |
| GPTQ 10.1%→3.4%「real lm_head」 | ❌ **UNSUPPORTED(誤標)** | 真實激活實測 ~7.1%;3.4% 是合成數據(M-Q4) |
| AWQ 4-bit ~8% 誤差、bias 保留 | ✅ SUPPORTED(限無校准路徑) | 校准路徑見 M-Q3,README 缺警告 |
| FP8 native float8 / fallback | ✅ SUPPORTED | torch 2.8 實測 |
| MTP batch>1 == greedy | ✅ SUPPORTED | 但測試有永真断言需修 |
| 引擎合批 == 逐條 greedy、無洩漏 | ✅ SUPPORTED | |
| audio 滑窗/窗內一致 ~1e-6 | ✅ SUPPORTED | |
| NaViT row/col、邊界 ValueError | ✅ SUPPORTED | |
| 15 tests / exit code 語義 / 9/9 集成 | ✅ SUPPORTED | 親跑復核 |

---

## 修復優先級建議

**P0(正確性,影響常用路徑):**
1. C1 absorbed NaN 污染(nan_to_num 一行修復 + 補 left-padding 測試)
2. M-I1 MTP correction 雙重 temperature(預設路徑命中)
3. M-Q1 GPTQ bias dtype cast(一行)
4. M-Q4 README 數字更正(10.1%→7.1% @ real lite lm_head)

**P1(穩健性):** M-T1 model.train()、M-T2 NaN 守衛、M-T3 reward 比較、M-C4 RoPE persistent=False、M-C2 eval 門控、M-Q3 AWQ 校准修復或加警告+測試、M-I2 rope 形狀。

**P2:** M-C1(文檔限制或斷言)、M-C3(all-reduce)、M-I3(賬簿含 pad)、測試永真断言與覆蓋缺口、版本字符串 v5.0→v5.2。

## 已驗證無問題(重點清單)

absorption 切分/結合律/scale/RoPE 作用域;雙模式在 prefill/decode/padding/自定義位置下等價;cache dim-2 截斷;MoE 排序 dispatch 與 index_add 逐位一致、梯度流通;generate EOS/確定性;MTP 分組不變量、回滾無 off-by-one;PagedAttention CoW 全邊界;引擎 pad 前綴中性(4.2e-7);DualPipe 梯度逐位相等;FP8 grid 十萬值零失配、STE、E5M2 hook;GRPO logprob 切分無 off-by-one、on-policy ratio=1.0;GPTQ 全鏈路數學;audio 窗內等價;NaViT packed≡solo。
