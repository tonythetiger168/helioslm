# HeliosLM v5.3 壓力與邊界驗證報告

環境:CPU,PyTorch 2.8.0,`HeliosLMv5Config(size="lite")`(vocab=1024, hidden=256, 2 層, MLA absorbed cache, MoE 8 experts, MTP 1 module)。所有測試以獨立腳本實跑(scripts in `stress/t1..t8`),未修改項目任何文件。專案自帶測試基線:`python -m helioslm_v5.tests.test_v5` → 19/19 PASS。

---

## 發現的問題清單

### BUG-1 [MEDIUM] Engine `step()` 對 `max_new_tokens=0` 的 request 仍生成一個 token
- **位置**: `src/inference/vllm_engine.py:341-361`(`step()` 的 retire 循環在 `schedule()` 之前執行,新 admitted 的已完成 request 直接進 prefill 循環)
- **現象**: `run()` 路徑正確(返回 `{0: []}`),但手動 `step()` 路徑會為 `max_new_tokens=0` 的 request prefill 並 append 一個 token,違反 max_new_tokens 契約。
- **最小復現**:
```python
eng = VLLMEngine(model, cfg, block_size=4, max_num_blocks=2000)
eng.add_request([1, 2, 3], max_new_tokens=0)
outs = eng.step()            # 期望 {},實際 {0: 345}
# eng.finished_requests[0].generated_token_ids == [345]  ← 多生成了 1 個 token
```
- **對照**: `eng.run()` 路徑正確(`run()` 在 schedule 後再次 retire,vllm_engine.py:386-390)。

### BUG-2 [MEDIUM] `FP8Trainer` 轉換後 MTP 的 weight tying 斷裂
- **位置**: `src/training/fp8_trainer.py:222-244`(`_convert_to_fp8` 無 re-bind 邏輯;對照 `src/quantization/standard_quant.py:625-639` 的 QuantizationManager 有顯式 re-bind 修復)
- **現象**: `named_modules()` 按 identity 去重,共享的 `lm_head` 只以 `lm_head` 名字 yield 一次;替換後 `model.lm_head` 是新的 `FP8Linear`,而 `mtp_modules[0].lm_head` 仍指向**舊的未轉換 nn.Linear**(實測:轉換後 lite 模型剩 1 個未轉換 Linear,即 `mtp_modules.0.lm_head`)。兩者成為獨立對象,FP8 訓練中 master weights 各自獨立更新 → 靜默發散、MTP 與主模型輸出頭脫鉤。embed_tokens 是 Embedding 不被轉換,故不受影響。
- **最小復現**:
```python
m = HeliosLMv5(HeliosLMv5Config(size="lite"))
FP8Trainer(m, cfg)   # cfg 只需 fp8_format/lr 屬性
assert m.mtp_modules[0].lm_head is m.lm_head          # FAIL:False
type(m.mtp_modules[0].lm_head)   # nn.Linear(未轉換的舊對象)
type(m.lm_head)                  # FP8Linear(新對象)
```

### BUG-3 [LOW] 空序列輸入崩潰且錯誤訊息無指向性
- **位置**: `src/attention/mla.py:99`(`position_ids.max()` 對空 tensor);入口 `src/model_v5.py:132-149` 無 seq==0 校驗
- **現象**: `model(torch.zeros(1,0,dtype=long))`、`generate(空 prompt)`、`engine.generate([[]])`、GRPO `train_step([""])` 均以 `RuntimeError: max(): Expected reduction dim to be specified for input.numel() == 0` 崩潰,而非清晰的 ValueError。
- **最小復現**:
```python
model.generate(torch.zeros(1, 0, dtype=torch.long), max_new_tokens=2)
# RuntimeError: max(): Expected reduction dim ... (來自 RoPE _ensure_capacity)
```

### BUG-4 [LOW] AWQ calibration 激活形狀錯誤時靜默接受
- **位置**: `src/quantization/standard_quant.py:161-162`
- **現象**: `AWQLinear.from_linear(lin(16→8), activations=randn(10, 32))` 不報錯——`act.reshape(-1, 16)` 把 320 個元素靜默重排成 [20,16],校準數據語義全錯。對照 `GPTQLinear.from_linear`(standard_quant.py:426-429)有顯式 shape 檢查會 ValueError。
- **最小復現**:
```python
lin = nn.Linear(16, 8)
AWQLinear.from_linear(lin, group_size=8, activations=torch.randn(10, 32))  # 應報錯,實際靜默通過
```

### BUG-5 [LOW] `generate()` 把模型永久切到 eval mode(模式洩漏)
- **位置**: `src/model_v5.py:248`(`self.eval()` 無恢復);MTP 路徑 `src/inference/mtp.py:376-379` 同樣
- **現象**: train mode 下調用 `generate()` 後 `model.training == False` 且不恢復。GRPO 內部已有 M-T1 workaround(grpo.py:210-292),但其他 caller(如自定義訓練循環先 train() 再 generate 再 train forward)會靜默丟失 train mode(dropout 等失效)。實測:`model.train(); model.generate(...); model.training` → `False`。
- **最小復現**:
```python
model.train()
model.generate(torch.tensor([[1,2,3]]), max_new_tokens=2)
assert model.training  # FAIL
```

### BUG-6 [LOW-MEDIUM] 多模態 audio prefix 路徑有狀態洩漏,`model.forward` 不冪等
- **位置**: `src/model_v5.py:163-166` 調用有狀態的 `StreamingAudioEncoder`(`src/audio/streaming_encoder.py:111-150` 的 `conv_state_*`/`_memory`),model 層無 reset
- **現象**: 同一 `audio_features` 連續兩次 `model.forward`,text 位置 logits 不同(實測 max diff = 0.1364),因為 encoder 的 streaming 狀態跨調用累積。vision 路徑無此問題(diff = 0)。調用者必須自行 `model.audio_encoder.reset_state()`,但 model API 文檔未說明。
- **最小復現**:
```python
la, _, _ = mm(ids, audio_features=aud)
lb, _, _ = mm(ids, audio_features=aud)   # 無 reset
(la[:, -1] - lb[:, -1]).abs().max()      # 0.1364,應為 0
```

### BUG-7 [LOW] NaViT 輸入小於 patch_size 時裸 RuntimeError
- **位置**: `src/vision/navit.py:91`(`patch_embed` conv 前無尺寸校驗)
- **現象**: 2x8 圖(patch=4)→ `RuntimeError: Calculated padded input size ... Kernel size can't be greater than actual input size`,而非清晰的 ValueError。超 max_grid 則有清晰 ValueError(navt.py:65-71,正確)。
- **最小復現**: `NaViTEncoder(cfg)(torch.randn(1, 3, 2, 8))`(patch_size=4)

### BUG-8 [LOW] `quantize_model` 對裸 `nn.Linear` 模型靜默 no-op
- **位置**: `src/quantization/standard_quant.py:603-606`(`name == ""` 直接 skip,無警告)
- **最小復現**:
```python
lin = nn.Linear(16, 8)
QuantizationManager("awq").quantize_model(lin)
type(lin)  # 仍是 nn.Linear,無任何提示
```

---

## 壓力通過清單(實跑驗證)

| 維度 | 結果 |
|---|---|
| 長度 1 prompt generate / forward | PASS |
| `max_new_tokens=0/1`(普通路徑) | PASS(0 時原樣返回) |
| 全同 token batch | PASS |
| vocab 邊界 id(0 與 vocab_size-1) | PASS |
| 非法 id(≥vocab、負數) | PASS — `IndexError: index out of range`(響亮報錯,可接受) |
| temperature=1e-8 | PASS — 無 NaN/overflow,輸出與 greedy 完全一致 |
| temperature=100 | PASS — finite |
| top_p=0 | PASS — 退化為 greedy(與 temperature=0 輸出一致) |
| top_p=1.0 | PASS |
| batch=8 forward vs 逐行 solo | PASS — max diff 7.5e-07 |
| batch=8 greedy generate vs 逐行 solo | PASS — 8/8 行完全一致 |
| left-pad batch(attention_mask)generate | PASS — pad 行與 solo 一致 |
| 全 batch 第一步即 EOS(強制) | PASS — EOS 後凍結為 PAD,普通與 MTP 路徑均正確 |
| cached decode vs 全序列 forward(prompt 1/2/7/64 × 8 decode 步) | PASS — max diff ≤ 8.6e-07 |
| 切分 prefill(prompt 7 拆 1+6、64 拆 63+1) | PASS — ≤ 9.5e-07 |
| MTP greedy == 非 MTP greedy(batch 1/2/4、prompt 1/7/64) | PASS — bitwise 一致 |
| MTP 採樣分佈 vs 非 MTP(temp 1.0 / 0.7+top_p 0.9 / 0.5,加尖峰 logits,N=600) | PASS — 各位置 TV 距離均在噪聲基線內(投機採樣分佈正確) |
| MTP `max_new=0/1`、`temperature=None` | PASS |
| MTP batch 內單行提前 EOS | PASS — 該行 EOS 後右側 PAD,其餘行繼續 |
| Engine: 單 request / 8 條不等長(1..21)vs solo greedy | PASS — 全部一致(含 watermark pad 路徑) |
| Engine: 長 prompt(40)中途加入超過 watermark | PASS — r1/r2 均與 solo 一致 |
| Engine: 全 request 同時 EOS(強制) | PASS — 各輸出恰為 [EOS] |
| Engine: 連續兩波 run→add→run | PASS — 一致且 block 無洩漏(2000/2000) |
| Engine: block OOM | PASS — 響亮 `RuntimeError: Out of memory` |
| Engine: `max_batch_size=1` 串行 | PASS |
| Engine: `max_new=0` 走 `run()` | PASS — `{0: []}`(但 `step()` 路徑見 BUG-1) |
| AWQ/GPTQ/FP8 全模型量化後 generate(含 MTP) | PASS |
| 量化冪等(連續兩次 quantize_model) | PASS — packed buffers 不變、輸出一致、無殘留 nn.Linear |
| 量化奇數維(in=33,out=7)、group_size>in_features(128>16)、tiny in(16<blocksize 128) | PASS |
| GPTQ 全零校準激活 / FP8 全零權重 | PASS — 無 NaN |
| 量化後 state_dict round-trip | PASS |
| 未知量化方法 | PASS — ValueError |
| vision prefix + cached 解碼 vs 全量重算 | PASS — 7.5e-07 |
| audio prefix / vision+audio 雙 prefix | PASS — shape 正確、finite |
| NaViT 極端寬高比(1×8、8×1 grid) | PASS;超 max_grid 清晰 ValueError |
| NaViT `forward_packed` 混合尺寸 | PASS — mask 正確、finite |
| audio streaming: 3-chunk == one-shot | PASS — 9.5e-07;1-frame chunk PASS |
| GRPO group_size=1、batch=1 | PASS — 無崩潰(advantage 全零為算法固有,代碼有注釋) |
| GRPO eval-mode 調用後模式恢復 | PASS(finally 恢復) |
| FP8Trainer 全零輸入 / 單參數模型 | PASS |
| FP8Linear NaN 輸入 | PASS — amax 歷史不被污染(nan→0 防護生效) |
| DualPipe M=1 / M=3 梯度 vs 直接反向 | PASS — grad diff 0 / 7.5e-09;M=0 建構與調用均 ValueError |
| 連續兩次 generate 無 cache 殘留;穿插不同 batch 後仍一致;採樣同種子可復現 | PASS |
| 訓練 forward 累積 MoE expert_load,eval/generate 不污染 | PASS |
| MTP generate 後再普通 generate 無交叉污染 | PASS |
| RoPE buffer 增長後 forward 確定性 | PASS |

## 未覆蓋範圍
- 僅 CPU / fp32;CUDA、bf16/fp16 路徑未驗。
- `torch.distributed` 相關分支(MoE device-limited routing、ExpertParallelism.all_to_all)單進程環境不可驗。
- Full-size config(48 層 / 4096 hidden)未跑(計算量);lite 與 full 共享全部代碼路徑,但 full 的 `vision_max_grid=32` 等參數差異未覆蓋。
- 引擎 `run(max_steps=...)` 中斷後續跑、GPUMonitor 的 CUDA 分支未驗。
