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

- **把 `-c` 調小**（例如 `-c 8192`）：上下文越長，KV 快取越吃記憶體。
- **KV 快取量化**：加 `-fa on -ctk q8_0 -ctv q8_0`，幾乎不影響品質，KV 快取減半。
  Ollama 則在啟動 `ollama serve` 前設 `OLLAMA_FLASH_ATTENTION=1` 和 `OLLAMA_KV_CACHE_TYPE=q8_0`。
- **混合專家模型（L24 起）放不進顯示卡**：加 `--cpu-moe`，專家權重放在系統記憶體、其他放顯示卡；
  再試 `--n-cpu-moe N`（只放前 N 層的專家），在放得下的前提下把 N 調小會比較快。
  統一記憶體的電腦（Mac、DGX Spark、Ryzen AI Max+ 395）不需要。

  ```bash
  llama-server -m helioslm-tw-26b-a4b-l24-Q4_K.gguf -c 32768 -ngl 99 --cpu-moe --reasoning-budget 0 --port 8080
  ```

- **Windows 上變得很慢**：關掉其他吃顯示卡記憶體的程式（瀏覽器硬體加速、遊戲、Docker / WSL）。
