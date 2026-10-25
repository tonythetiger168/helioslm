# HDL training data (H1)

H1 of the HDL track picks the Verilog / SystemVerilog / VHDL data for
continued pretraining (H2) and lists the instruction sets kept for the
instruct stage (H3). The H0 benchmarks are in [HDL_BENCH.md](HDL_BENCH.md).

## Continued-pretraining mixture

`data/mixtures/hdl_h1_cpt_v1.json`: about 75% HDL source code and 25%
English + maths replay, so the base keeps its general skills. Only raw
code columns are read. Card licences and revisions were read on
2026-10-25.

| Source | Column | Licence | Size | Weight | Status |
|---|---|---|---|---|---|
| `shailja/Verilog_GitHub` | `text` | MIT (card) | 1.9 GB CSV | 0.35 | usable |
| `bigcode/the-stack-dedup` `data/systemverilog` | `content` | other (terms, auto-gated) | 166 MB parquet | 0.15 | review |
| `bigcode/the-stack-dedup` `data/vhdl` | `content` | other (terms, auto-gated) | 445 MB parquet | 0.12 | review |
| `rtl-llm/vhdl_github_deduplicated` | `chosen` | MIT | 52 MB JSON | 0.08 | usable |
| `wilyub/VeriThoughtsTrainSet` | `ground_truth` | MIT | 399 MB JSONL | 0.05 | usable |
| `HuggingFaceFW/fineweb-edu` | `text` | ODC-BY-1.0 | | 0.20 | usable |
| `HuggingFaceTB/finemath` `finemath-3plus` | `text` | ODC-BY-1.0 | | 0.05 | usable |

The Stack sources stay `review` until a filter keeps only rows whose
`max_stars_repo_licenses` are all in the allow list; until then
`fetch_mixture.py` gives their share to the usable sources. The Stack's
`data/verilog` folder holds only 0.18 MB (most `.v` files are not
labelled Verilog there), so Verilog comes from `Verilog_GitHub`.

```
python examples/fetch_mixture.py --mixture data/mixtures/hdl_h1_cpt_v1.json \
    --tokenizer google/gemma-4-E2B --tokens 5e8 --out /workspace/hdl_data --check-only
```

Per-source options added for this mixture (v5.152): `field` (the text
column), `data_dir` / `data_files` (passed to `datasets.load_dataset`)
and `"heuristics": false`, because the prose heuristics drop most source
code (short modules, repeated `end` lines).

Before training, check the corpus for overlap with VerilogEval v2, RTLLM
v2.1 and CVDP: public GitHub Verilog contains HDLBits-style solutions.

## Left out of the pretraining mixture

| Dataset | Reason |
|---|---|
| `GaTech-EIC/MG-Verilog` (MIT) | its code is a subset of `Verilog_GitHub`; its descriptions came partly from GPT-3.5 / GPT-4 (repository README), which the teacher rules exclude |
| `bnadimi/PyraNet-Verilog` | CC-BY-NC-SA-4.0, non-commercial |
| `NOKHAB-Lab/LLM_4_VHDL` (MIT) | manually gated; request access first |
| `rtl-llm/github_vhdl_verilog`, `Nellyw888/RTL-Coder_small`, `hdl2v/vhdl-dataset` | no licence on the card |
| VeriThoughts `question` / `reasoning_trace` | written by a model the card does not name |

## Instruct-stage candidates (H3)

Model-written sets, kept for supervised fine-tuning once their generators
are checked against the teacher whitelist:

| Dataset | Licence | Generator (card) |
|---|---|---|
| `zhuyaoyu/CodeV-R1-dataset` (`codev_r1_sft.jsonl`, RL parquet) | CC-BY-SA-4.0 | DeepSeek-V3 instructions, DeepSeek-R1 responses (open weights) |
| `HPAI-BSC/SystemVerilog-SFT-86k` | none on the card (ask the authors) | Gemma-4-31B, testbenches checked by simulation |
