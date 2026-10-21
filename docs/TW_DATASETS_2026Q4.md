# 台灣（zh-TW）訓練與評測資料集調查（2026-10-16）

**查證範圍：** 調查時沙箱連不到 huggingface.co，資料集卡片無法直接開啟，以下授權資訊來自搜尋引擎摘錄。
標 **[未驗證]** 的項目，使用前必須在 HF 卡片確認 `license:` 欄位與 gated 狀態：

```python
from huggingface_hub import dataset_info
print(dataset_info("ikala/tmmluplus").card_data.license)
```

## 2026-10-20 更新：HF 卡片實際查核（TW1）

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
- `MediaTek-Research/TCEval-v2` 卡片沒有授權欄位（unknown）。
- 新增簡體、西班牙文、日文、韓文候選資料集，同一份報告一起分類。

**限制：** 只讀 metadata。資料檔本身的主機（`cas-server.xethub.hf.co`、`*.cdn.hf.co`）仍被沙箱擋住，所以還不能量測西／日／韓文的 tokenizer 成本。

## 預訓練

| 名稱 / HF id | 維護者 | 規模 | 授權 | 商用／再散布 | 備註 |
|---|---|---|---|---|---|
| `wikimedia/wikipedia`（zh） | Wikimedia | 約 140 萬條 | CC BY-SA 4.0 + GFDL | 可，ShareAlike | 繁簡混合，需 OpenCC `s2twp` 轉換；config 名 [未驗證] |
| `HuggingFaceFW/fineweb-2`（中文子集） | Hugging Face | 數百 GB 級 | ODC-By 1.0 | 可（需標示；原文版權另計） | 沒有獨立繁體子集，需過濾 [config 名未驗證] |
| data.gov.tw 政府開放資料 | 數位發展部 | 依資料集 | 政府資料開放授權條款 v1（`OGDL-Taiwan-1.0`，與 CC BY 4.0 相容） | 可，須標示提供機關 | 法規、新聞稿等文字型資料 |
| `lianghsun/tw-news-551M` | Twinkle AI | 約 5.6 億 token | CC BY-NC-SA 4.0 | **否** | 新聞版權風險 |
| `lianghsun/tw-science-24M` | Twinkle AI | 2400 萬 token | CC BY-NC-SA 4.0 | **否** | |
| `lianghsun/Taiwan_c4` | Twinkle AI | ? | CC BY-NC-ND 4.0 | **否**（ND 能否訓練有疑義） | |
| `yentinglin/zh_TW_c4` | Yen-Ting Lin | ? | [未驗證] | ? | |
| `lianghsun/cosmopedia-zhtw` | Twinkle AI | 進行中 | 卡片兩處說法衝突 | 不明 | 合成資料 |
| FineWeb-zhtw（ROCLING 2024） | 聯發科研究院等 | ? | [未驗證] | ? | HF id 未找到 |
| 教育部《重編國語辭典修訂本》 | 教育部 | 約 16 萬詞條 | [未驗證] | 存疑 | |
| 中研院平衡語料庫 | 中研院 | 約 1000 萬詞 | 付費授權 | **否** | 不可再散布 |
| PTT／Dcard | — | — | 無合法散布的 dump | **否** | |
| TAIDE 資料 | 國科會 | — | 尚未釋出 | — | HF `taide` 只有模型 |

## 指令微調

| 名稱 / HF id | 規模 | 授權 | 商用 | 備註 |
|---|---|---|---|---|
| `yentinglin/TaiwanChat` | 約 48.6 萬筆 | CC BY-NC 4.0 | **否** | 部分由 GPT 產生，另受 OpenAI 條款約束 |
| `lianghsun/tw-instruct-500k` | 約 50 萬筆 | [未驗證] | ? | |
| `lianghsun/wikipedia-pretrain-zh-tw-chat` | ? | 卡片標 MIT，但上游是 CC BY-SA | 存疑 | |

**沒有找到** zh-TW 的程式碼、agentic coding 或 function-calling 資料集。可行做法：授權乾淨的英文 code 指令集，加 zh-TW 翻譯或合成。

## 評測

| 名稱 / HF id | 規模 | 授權 | 備註 |
|---|---|---|---|
| TMMLU+ `ikala/tmmluplus` | 66 科，約 2.27 萬題 | CC BY-NC（卡片 `other`） | 本地評測可，不再散布 |
| TMLU `miulab/tmlu` | 37 科 | [未驗證] | |
| TCEval-v2 `MediaTek-Research/TCEval-v2` | 含 TMMLU+ 子集、MT-Bench-TW | [未驗證] | |
| DRCD（GitHub `DRCKnowledgeTeam/DRCD`） | 3 萬多題 | CC BY-SA 3.0 或 BY-NC-SA 3.0（說法衝突） | 先確認 |
| `JerryWu/TW-JusticeBench` | ? | CC BY 4.0 | 可 |
| `lianghsun/tw-legal-benchmark-v1` | ? | [未驗證] | |
| `ericeric777777/medical-bench-tw` | ? | 限研究／評測 | 不可商用 |

## 建議起步組合（授權乾淨優先，CPU 小規模）

1. **預訓練：** zh Wikipedia（OpenCC 轉台灣繁體；衍生資料保留 CC BY-SA 並標示）加 FineWeb-2 中文子集，先取約 1 億 token（`streaming=True`），再加 data.gov.tw 文字型資料（依 OGDL 標示提供機關）。
2. **指令微調：** 目前沒有確認授權乾淨的 zh-TW 指令集。先用 Wikipedia／政府資料自製問答；TaiwanChat 只做內部研究，權重不商用、不再散布。
3. **評測：** TMMLU+、TMLU 本地跑分（不再散布）；加 TW-JusticeBench（CC BY 4.0）。

用到任何 NC 資料時，程式碼授權不受影響，但**訓練出的權重與衍生資料應標為非商用**。

## 法律注意事項

- NC／ND：TMMLU+、TaiwanChat、tw-news、Taiwan_c4 不可商用；ND 授權能否用於訓練不明確。
- ShareAlike：Wikipedia 衍生資料再散布須用 CC BY-SA。
- 合成資料：GPT 產生的資料另受模型供應商條款約束。
- 政府資料：OGDL 可商用，但未標示提供機關即視為未取得授權。
- PTT、Dcard、中研院語料庫不可用。

主要來源：[OGDL-Taiwan-1.0（SPDX）](https://spdx.org/licenses/OGDL-Taiwan-1.0.html)、
[FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)、
[TMMLU+ README](https://huggingface.co/datasets/ikala/tmmluplus)、
[TaiwanChat](https://huggingface.co/datasets/yentinglin/TaiwanChat)、
[TMLU 論文](https://arxiv.org/abs/2403.20180)、[DRCD 論文](https://arxiv.org/abs/1806.00920)、
[FineWeb-zhtw（ROCLING 2024）](https://aclanthology.org/2024.rocling-1.16)、
[TW-JusticeBench](https://huggingface.co/datasets/JerryWu/TW-JusticeBench)。
