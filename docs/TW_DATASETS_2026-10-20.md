# 台灣與多語訓練資料集：HF 卡片查核（2026-10-20，TW1）

[`TW_DATASETS_2026Q4.md`](TW_DATASETS_2026Q4.md) 的更新。原文件保持不動：它是 TW4／TW5 tokenizer 量測的固定 held-out 文件，改動它會改變已提交的量測結果。

沙箱已可連 huggingface.co 的 metadata API，45 個資料集卡片已逐一讀取（`license` 欄位、gated 狀態、revision sha）。
結果見 `benchmarks/hf_license_audit_2026-10-20.json`，重跑：`python examples/hf_license_audit.py`（gated 卡片需 `HF_TOKEN`）。
分類只依卡片欄位，**不是法律意見**；卡片本身可能標錯（例如衍生自 CC BY-SA 卻標 MIT）。

| 分類 | 數量 | 說明 |
|---|---|---|
| commercial | 21 | CC BY / BY-SA / CC0 / ODC-By / ODbL / MIT / Apache / GFDL |
| noncommercial | 11 | 含 NC 條款 |
| no_derivatives | 1 | `HAERAE-HUB/KMMLU`（CC BY-ND） |
| unknown | 12 | 卡片沒有 license 欄位，使用前讀卡片 |

**與上方 2026-10-16 表格不同之處：**

- `yentinglin/zh_TW_c4` 不存在；改用 `botp/yentinglin-zh_TW_c4`（ODC-By）或 `liswei/c4-zhtw`（卡片無授權欄位）。
- FineWeb-zhtw 已找到：`lianghsun/fineweb-zhtw`（MIT，gated）、`lianghsun/fineweb-2-zhtw`（ODC-By，gated）、`twinkle-ai/fineweb-zhtw-filtered`（ODC-By）。
- `miulab/tmlu` 找不到，已從清單移除。
- `ikala/tmmluplus` 卡片目前標 **MIT**（不是先前摘錄的 CC BY-NC）。題目來源的版權另計，仍建議只做本地評測、不再散布。
- `lianghsun/tw-instruct-500k`：CC BY-NC-SA 4.0（不可商用）。
- `lianghsun/cosmopedia-zhtw`：卡片標 Apache-2.0；合成資料，生成模型的條款另計。
- `lianghsun/wikipedia-pretrain-zh-tw-chat` 卡片標 MIT，但上游 Wikipedia 是 CC BY-SA，**仍視為存疑**。
- `TCEval-v2` 卡片沒有授權欄位（unknown）。
- 新增簡體、西班牙文、日文、韓文候選資料集，同一份報告一起分類。

**限制：** 只讀 metadata。資料檔本身的主機（`cas-server.xethub.hf.co`、`*.cdn.hf.co`）仍被沙箱擋住，所以還不能量測西／日／韓文的 tokenizer 成本。

