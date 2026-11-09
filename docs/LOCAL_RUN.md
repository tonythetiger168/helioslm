# 在本機下載並執行羲和（HeliosLM）

下載頁：**https://helioslm.pages.dev/**

模型完全在你的電腦上執行，不需要網路，對話內容不會傳出去。
三個步驟：選檔、下載解壓、用 LM Studio、Ollama 或 llama.cpp 執行。

> **預覽版**：目前的套件都是以 Google Gemma 4 為基礎的早期試作，尚未通過正式評測，僅供試用。

## 1. 選哪個檔

先看你的電腦有多少記憶體：

- **NVIDIA / AMD 顯示卡**：看顯示卡記憶體（VRAM）。
  Windows：工作管理員 → 效能 → GPU →「專用 GPU 記憶體」；Linux：`nvidia-smi`。
- **Mac（Apple M 系列）**：看「關於這台 Mac」的記憶體。GPU 約可用其中 2/3 到 3/4。
- **Intel Mac、沒有獨立顯示卡的電腦**：用 L-Edge，只用 CPU（至少 8 GB RAM）。
- **系統記憶體**：用顯示卡跑時，除了 VRAM，載入時還要約 3–4 GB 可用的系統記憶體（RAM）。
  16 GB RAM 的電腦請先關掉瀏覽器、Docker / WSL 等程式（2026-11-05 在 RTX 4060 8 GB、16 GB RAM 的
  Windows 桌機上實測：只剩約 1 GB 可用 RAM 時，Ollama 載入 L8 失敗）。

| 層級 | 顯示卡記憶體 | Mac 統一記憶體 | 類型 | zip 內的模型檔 | 檔案大小 | 上下文 |
|---|---|---|---|---|---:|---|
| **L-Edge** | 無（CPU） | 8 GB | 基底模型 | `helioslm-tw-e4b-l-edge-Q4_K.gguf` | 4.94 GB | 8K |
| **L8** | 8 GB | 16 GB | 基底模型 | `helioslm-tw-e4b-l8-Q4_K.gguf` | 4.94 GB | 32K |
| **L16** | 12–16 GB | 24 GB | 基底模型 | `helioslm-tw-e4b-l16-Q8_0.gguf` | 7.43 GB | 32K |
| **L24** | 24–32 GB | 32–48 GB | 對話模型 | `helioslm-tw-26b-a4b-l24-Q4_K.gguf` | 15.64 GB | 32K |
| **L48+** | 48–96 GB | 64–96 GB | 對話模型 | `helioslm-tw-26b-a4b-l48plus-Q8_0.gguf` | 25.02 GB | 128K |
| **L128** | — | 128 GB 以上 | 對話模型 | `helioslm-tw-26b-a4b-l128-F16.gguf` | 47.04 GB | 128K |

**模型差別**：

- **基底模型**（L-Edge、L8、L16，HeliosLM TW E4B）只做過繁體中文續訓，沒有對話微調。
  給它一段開頭，它會接著寫；不適合一問一答。
- **對話模型**（L24 起，HeliosLM TW 26B-A4B）可以直接問答，預設直接回答、不先思考。
  它是混合專家模型，每個 token 約 4B 參數啟用。
- 只有 12–16 GB 顯示卡也想對話：可以跑 L24，加上 `--cpu-moe`，把專家權重放在系統記憶體
  （建議 32 GB RAM 以上，速度較慢，見第 5 節）。
- 只有 8 GB 顯示卡（RTX 4060、3060 Ti、3070 等）也想對話：下載 L24 的 26B 檔，加上 `--n-cpu-moe 24`，
  顯示卡只用約 5.7 GB，實測每秒約 34 個 token；系統記憶體要 32 GB 以上（見第 5 節）。

## 2. 下載、驗證、解壓

在下載頁的表格上按右鍵複製下載連結。硬碟要留約兩倍檔案大小的空間，解壓後可以刪掉 zip。
算出的 SHA-256 要和下載頁 `packages.json` 裡該檔的 `sha256` 相同，不同就重新下載。

以下用 L24 的檔名當例子，換成你下載的檔名即可。

**Windows**（PowerShell，Windows 10 以上內建 `curl.exe` 和 `tar`）：

```powershell
curl.exe -L -O <下載連結>
Get-FileHash .\helioslm-tw-26b-a4b-l24-Q4_K-tw-preview-6.zip -Algorithm SHA256
tar -xf .\helioslm-tw-26b-a4b-l24-Q4_K-tw-preview-6.zip
```

**Linux**：

```bash
curl -L -O <下載連結>
sha256sum helioslm-tw-26b-a4b-l24-Q4_K-tw-preview-6.zip
unzip helioslm-tw-26b-a4b-l24-Q4_K-tw-preview-6.zip
```

**macOS**：

```bash
curl -L -O <下載連結>
shasum -a 256 helioslm-tw-26b-a4b-l24-Q4_K-tw-preview-6.zip
unzip helioslm-tw-26b-a4b-l24-Q4_K-tw-preview-6.zip
```

每個 zip 內有：`.gguf` 模型檔、Ollama 用的 `Modelfile`、該層級的 `README.md`（完整指令）和 `manifest.json`。

## 3. 安裝執行工具（三選一）

| 工具 | 適合 | 安裝 |
|---|---|---|
| **LM Studio** | 想用圖形介面、最簡單 | 從 lmstudio.ai 下載安裝（Windows / macOS / Linux） |
| **Ollama** | 想用指令或給其他程式呼叫 | Windows / macOS：從 ollama.com 下載安裝；Linux：`curl -fsSL https://ollama.com/install.sh \| sh` |
| **llama.cpp** | 想自己調參數、最省記憶體 | Windows：`winget install llama.cpp`；macOS / Linux：`brew install llama.cpp`；或下載官方預編譯版（NVIDIA 選 CUDA 版，AMD / Intel 顯示卡選 Vulkan 版，Mac 已內建 Metal） |

## 4. 執行

先在終端機 `cd` 到解壓後的資料夾。

### LM Studio

用「匯入模型」選 zip 裡的 `.gguf`（或命令列 `lms import <模型檔>.gguf`），
載入時把 GPU Offload 拉到最大，然後開始對話。

### Ollama

```bash
ollama create helioslm-tw-26b-a4b-l24 -f Modelfile
ollama run helioslm-tw-26b-a4b-l24
```

名稱就是模型檔名去掉最後的量化標記（例如 `helioslm-tw-e4b-l8`）。
基底模型可以直接給開頭：`ollama run helioslm-tw-e4b-l8 "台灣的夜市文化"`。

### llama.cpp

對話模型（L24 起）：

```bash
llama-server -m helioslm-tw-26b-a4b-l24-Q4_K.gguf -c 32768 -ngl 99 --reasoning-budget 0 --port 8080
```

啟動後用瀏覽器開 <http://127.0.0.1:8080> 就能對話；
其他程式可以用 OpenAI 相容 API `http://127.0.0.1:8080/v1`。
`--reasoning-budget 0` 是預設的直接回答；拿掉它模型會先思考再回答。

基底模型（L-Edge、L8、L16），把 `-p` 換成你的開頭：

```bash
llama-completion -m helioslm-tw-e4b-l8-Q4_K.gguf -c 32768 -ngl 99 -n 256 -p "台灣的夜市文化"
```

L-Edge 只用 CPU，不加 `-ngl`。舊版 llama.cpp 沒有 `llama-completion` 時，改用
`llama-cli -no-cnv` 加同樣的參數。

各層級的指令：

| 層級 | 指令 |
|---|---|
| L-Edge | `llama-completion -m helioslm-tw-e4b-l-edge-Q4_K.gguf -c 8192 -n 256 -p "…"` |
| L8 | `llama-completion -m helioslm-tw-e4b-l8-Q4_K.gguf -c 32768 -ngl 99 -n 256 -p "…"` |
| L16 | `llama-completion -m helioslm-tw-e4b-l16-Q8_0.gguf -c 32768 -ngl 99 -n 256 -p "…"` |
| L24 | `llama-server -m helioslm-tw-26b-a4b-l24-Q4_K.gguf -c 32768 -ngl 99 --reasoning-budget 0 --port 8080` |
| L48+ | `llama-server -m helioslm-tw-26b-a4b-l48plus-Q8_0.gguf -c 131072 -ngl 99 --reasoning-budget 0 --port 8080` |
| L128 | `llama-server -m helioslm-tw-26b-a4b-l128-F16.gguf -c 131072 -ngl 99 --reasoning-budget 0 --port 8080` |

## 5. 記憶體不夠時

- **`unable to allocate CUDA_Host buffer` 或 `ptxas fatal: Memory allocation failure`**：
  是系統記憶體（RAM）不夠，不是顯示卡。關掉其他程式再試；這塊約 2.9 GB 的緩衝區跟上下文長度無關，調小 `-c` 沒有用。
- **把 `-c` 調小**（例如 `-c 8192`）：上下文越長，KV 快取越吃記憶體。
- **KV 快取量化**：加 `-fa on -ctk q8_0 -ctv q8_0`，幾乎不影響品質，KV 快取減半。
  Ollama 則在啟動 `ollama serve` 前設 `OLLAMA_FLASH_ATTENTION=1` 和 `OLLAMA_KV_CACHE_TYPE=q8_0`。
- **混合專家模型（L24 起）放不進顯示卡**：加 `--cpu-moe`，專家權重放在系統記憶體、其他放顯示卡；
  再試 `--n-cpu-moe N`（只放前 N 層的專家），在放得下的前提下把 N 調小會比較快。
  統一記憶體的電腦（Mac、DGX Spark、Ryzen AI Max+ 395）不需要。

  ```bash
  llama-server -m helioslm-tw-26b-a4b-l24-Q4_K.gguf -c 32768 -ngl 99 --cpu-moe --reasoning-budget 0 --port 8080
  ```

  8 GB 顯示卡（2026-11-09 實測，16K 上下文、KV 快取 q8_0）：`--n-cpu-moe 24` 用約 5.7 GB 顯示卡記憶體
  （32K 上下文約 5.9 GB），每秒約 34 個 token；`--cpu-moe` 約 2.7 GB、每秒約 30 個。
  速度主要看 CPU 和系統記憶體頻寬。開 MTP 只快約 4%，這種用法不建議。Ollama 不能這樣分配，請用 llama.cpp
  或 LM Studio（載入設定裡把 MoE 專家權重放到 CPU）。

  ```bash
  llama-server -m helioslm-tw-26b-a4b-l24-Q4_K.gguf -c 16384 -ngl 99 --n-cpu-moe 24 -fa on -ctk q8_0 -ctv q8_0 --reasoning-budget 0 --port 8080
  ```

- **Windows 上變得很慢**：關掉其他吃顯示卡記憶體的程式（瀏覽器硬體加速、遊戲、Docker / WSL）。

## 6. 速度參考（實測）

llama.cpp（build 12098）、`llama-server -ngl 99 -c 8192 -fa on`、單一請求、產生 512 個 token，
8 個對話提示的中位數，2026-11-04 實測。單位 token/秒，越高越快。
其他顯示卡（例如 RTX 4090、3090）沒有實測。

| 層級 | 檔案 | 顯示卡 | 速度 | 開 MTP | 首字延遲 | VRAM |
|---|---|---|---:|---:|---:|---:|
| L8 | E4B Q4_K | RTX 5090 | 241 | 不建議（173） | 0.04 秒 | 3.7 GB |
| L16 | E4B Q8_0 | RTX 5090 | 196 | 不建議（206） | 0.04 秒 | 5.6 GB |
| L24 | 26B-A4B Q4_K | RTX 5090 | 254 | **349**（+37%） | 0.13 秒 | 17.2 GB |
| L24 | 26B-A4B Q4_K | L40S 48 GB | 161 | — | 0.16 秒 | 17.1 GB |
| L48+ | 26B-A4B Q8_0 | L40S 48 GB | 121 | — | 0.16 秒 | 26.7 GB |
| L48+ | 26B-A4B Q8_0 | A40 48 GB | 85 | **119**（+40%） | 0.20 秒 | 26.5 GB |

E4B 層級的 VRAM 比檔案小，因為每層嵌入（per-layer embeddings）留在系統記憶體。

### MTP 加速（只建議 26B，L24 起）

MTP 用 Gemma 4 26B-A4B 的 assistant 頭當草稿模型，一次猜幾個 token 再驗證。
另外下載它的 GGUF（實測用的是 Hugging Face 上 `AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF` 的 Q8_0），然後：

```bash
llama-server -m helioslm-tw-26b-a4b-l24-Q4_K.gguf -c 32768 -ngl 99 --reasoning-budget 0 \
  -md <assistant 檔>.gguf -ngld 99 --spec-type draft-mtp --spec-draft-n-max 3 --port 8080
```

實測快 37–40%，多用約 0.5 GB VRAM，草稿接受率約 61–64%。
E4B 層級（L-Edge、L8、L16）開 MTP 沒有幫助（接受率 6–11%），L8 反而變慢 28%。
舊版 llama.cpp 的 `--draft-max` 已移除，請用 `--spec-draft-n-max`。

### CPU 執行緒

加 `-t <實體核心數>`。在核心很多或虛擬化的機器上，預設的執行緒數可能讓速度掉到個位數 token/秒。

## Mac（Apple Silicon）MLX 版

Apple M 系列 Mac（M1 起）建議用下載頁「Mac（Apple Silicon）· MLX」一節的 MLX 版：
Apple 自家的 MLX 執行環境在 Mac 上通常比 GGUF 快，LM Studio 的 MLX 引擎也能直接載入。
上面的 GGUF 套件在 Mac 上一樣能用（Metal）。

| Mac 統一記憶體 | MLX 版 | 類型 |
|---|---|---|
| 8 GB | E4B 4-bit | 基底模型（只適合短文字） |
| 16–24 GB | E4B 8-bit | 基底模型 |
| 32–48 GB | 26B-A4B 4-bit | 對話模型 |
| 64 GB 以上 | 26B-A4B 8-bit | 對話模型 |

```bash
python3 -m venv ~/mlx && source ~/mlx/bin/activate   # 需要 Python 3.10 以上（brew install python）
pip install -U mlx-lm
curl -L -O <下載連結> && shasum -a 256 <zip 檔> && unzip <zip 檔>
# 對話模型
mlx_lm.chat --model ./helioslm-tw-26b-a4b-mlx-4bit --max-tokens 1024
mlx_lm.server --model ./helioslm-tw-26b-a4b-mlx-4bit --port 8080   # OpenAI 相容 API
# 基底模型（接著寫）
mlx_lm.generate --model ./helioslm-tw-e4b-mlx-8bit --ignore-chat-template --max-tokens 256 --prompt "台灣的夜市文化"
```

LM Studio：把解壓出來的資料夾放進 `~/.lmstudio/models/helioslm/`，選它載入。
Intel Mac 沒有 MLX，請用 L-Edge GGUF（llama.cpp 只用 CPU）。
打包工具：`examples/package_mlx.py`（`mlx_lm.convert` 轉檔 + zip + 下載頁 Mac 區）。

## 7. 筆電當蒸餾老師（Qwen3.6-35B-A3B，8 GB 顯示卡）

`examples/distill_local_3070.ps1`（v5.253）在 RTX 3070 Laptop（8 GB VRAM、40 GB RAM、Windows）上，
用 llama-server 跑 Qwen3.6-35B-A3B 當老師，產生蒸餾資料。
Kimi K2.6 放不進這台：最小的實用 GGUF 約 340 GB。

- **模型檔**：`unsloth/Qwen3.6-35B-A3B-GGUF` 的 `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`，22.1 GB，Apache-2.0。
  加 `-Download` 由腳本用 `hf` CLI 下載（固定版本），或自己下載後放在 `models\qwen3.6-35b-a3b\`。
- **需要**：llama.cpp 的 CUDA 版（`llama-server` 在 PATH 上，或用 `-LlamaServer` 指定路徑）、Python 3。
- **記憶體配置**：專家權重放系統記憶體（`--cpu-moe`），其他放顯示卡；KV 快取用 q8_0。

```powershell
# 1. 冒煙測試：5 題，最後印出每秒 token 數
powershell -ExecutionPolicy Bypass -File examples\distill_local_3070.ps1 `
    -Prompts runs\distill\prompts.jsonl -Out runs\distill\qwen36_smoke.jsonl -Limit 5
# 2. 正式跑，8 小時後停；中斷後重跑會接續
powershell -ExecutionPolicy Bypass -File examples\distill_local_3070.ps1 `
    -Prompts runs\distill\prompts.jsonl -Out runs\distill\qwen36_answers.jsonl -MaxHours 8
```

調整：

- **`-NCpuMoe N`**：只把前 N 層的專家放系統記憶體，其餘放顯示卡。N 越小越快，但要放得進 8 GB。
  從預設（全部放系統記憶體）開始，冒煙測試成功後再試較小的 N，看工作管理員的「專用 GPU 記憶體」。
- **`-Parallel`**：同時處理幾個請求（預設 2）；每個請求有 `-CtxPerSlot`（預設 8192）個 token 的上下文。
- **`-NoThinking`**：關掉思考（例如讓老師出題時）。
- **醫療資料**：加 `-DataManifest <manifest> -Medical`。老師在 127.0.0.1，資料不會離開這台電腦。

速度還沒在 3070 Laptop 上實測。跑完冒煙測試後，請把印出的 token/秒補進上面第 6 節。
