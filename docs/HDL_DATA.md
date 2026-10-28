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

Per-source options added for this mixture (v5.178): `field` (the text
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

## SystemVerilog and UVM mixture (H5, v5.183)

`data/mixtures/hdl_h5_sv_uvm_v1.json` continues from the H3 checkpoint
with SystemVerilog design and verification code. Three sources are git
repositories pinned to a commit (`git:<url>@<commit>`, `include` /
`exclude` globs); `fetch_mixture.py` checks out each one, reads its
top-level LICENSE file as the licence, and uses every matching file as a
document.

| Source | Licence (LICENSE file) | SystemVerilog | Of which UVM |
|---|---|---|---|
| `lowRISC/opentitan` @ `ecedaa5a` | Apache-2.0 | ~38 MB | ~12 MB |
| `lowRISC/ibex` @ `c0c07085` | Apache-2.0 | 1.7 MB (190 files) | 92 files |
| `openhwgroup/cv32e40p` @ `b353438f` (`rtl/`) | Solderpad Hardware 0.51 (based on Apache-2.0) | ~0.65 MB | |
| `openhwgroup/core-v-verif` @ `1dfa4836` | Solderpad Hardware 2.0 (may be taken as Apache-2.0) | ~4.3 MB | ~3.6 MB |

Directories matching `*vendor*` (third-party code under other licences)
are excluded; exact duplicate files and files over 1 MB are dropped.
Verilog GitHub code and English + maths are kept as replay.

### Projects that synthesise (v5.186)

Every HDL source of the H5 mixture is read as *projects*, and a project
is trained on only when it synthesises
(`helioslm_v5/src/training/hdl_projects.py`):

| Source | Project | Weight |
|---|---|---|
| OpenTitan | each IP directory (`hw/ip/*`, `hw/ip_templates/*`) | 0.28 |
| Ibex, CV32E40P | the whole core (`tops`: `ibex_top`, `cv32e40p_top`) | 0.05, 0.08 |
| core-v-verif | UVM environments; used only when CV32E40P passes (`requires`) | 0.05 |
| The Stack SystemVerilog | one GitHub repository, permissive licences only | 0.10 |
| `Verilog_GitHub` | one file (the rows carry no repository name) | 0.20 |

The gate finds the project's top module (the module named like the IP
directory, else the uninstantiated module with the largest design;
testbench paths never count), collects the files it needs from the whole
checkout (vendored dependencies included, though they are not trained
on), and runs `verilator --lint-only` and a Yosys `synth` of that top
with `SYNTHESIS` defined. The project passes when both succeed and the
netlist has cells. SystemVerilog goes through `sv2v` first when it is
installed: Yosys' own reader stops on most SystemVerilog packages.

A passing project is one document: a `// project:` line, then each file
as `// file: <path>` and its text, in dependency order (packages,
interfaces, modules, then the UVM interface, package, item, driver,
monitor, agent, scoreboard, environment, sequences and tests). Shared
libraries (OpenTitan's `hw/ip/prim`, `hw/ip/prim_generic` and `hw/dv`)
form one more document: their verification code once an IP passed, and
their design files only when a passing IP's synthesis read them.

Then, across the whole run:

- **near-duplicates**: a project whose MinHash (128 permutations, 16
  bands, about 0.8 Jaccard) matches an earlier one is dropped, so forks of
  the curated cores in GitHub data are not trained on twice (curated
  sources come first in the mixture);
- **decontamination**: `--decontam <bench files or directories>` drops
  every file sharing a 13-token window with VerilogEval, RTLLM or CVDP
  prompts and reference solutions;
- **licences**: a Stack row is kept only when every licence in
  `max_stars_repo_licenses` is permissive (`fetch_mixture.PERMISSIVE`:
  MIT, Apache-2.0, BSD, ISC, CC0, Unlicense, Solderpad, ...). The source
  still needs the account to accept The Stack's terms on the Hub.

`manifest.json` records every project: top module, lint, synth, cell
count, files, and why it failed.

```
python examples/fetch_mixture.py --mixture data/mixtures/hdl_h5_sv_uvm_v1.json \
    --tokenizer google/gemma-4-E2B --tokens 2e8 --out /workspace/h5_data \
    --decontam /workspace/benches --allow-blocked
```

Measured on 2026-10-28 (Yosys 0.33, sv2v v0.0.13, Verilator 5.052, 4 CPUs,
600 s per project):

| Source | Passed | Notes |
|---|---|---|
| OpenTitan IPs (`hw/ip/*` minus the prim libraries) | 24 of 38 | about 10 MB of passing project text; fails: 4 lint (missing cross-IP packages, vendor cell libraries), 3 synth timeouts (otbn, keymgr_dpe, prim_generic), 4 sv2v / Yosys front-end limits, 3 IPs with no top in `hw/ip` (generated from templates) |
| Ibex (`ibex_top`) | 1 of 1 | 22,104 cells |
| CV32E40P (`cv32e40p_top`) | 1 of 1 | 43,201 cells |
| CVA6 (`cva6`) | 0 of 1 | lint passes; sv2v cannot convert its parameterised structs, so CV32E40P replaced it |

Without sv2v, Yosys' own SystemVerilog reader stopped on the register
package of nearly every IP, and no OpenTitan IP passed. The pod installs
sv2v from its release (`zachjs/sv2v` v0.0.13, `sv2v-Linux.zip`).
