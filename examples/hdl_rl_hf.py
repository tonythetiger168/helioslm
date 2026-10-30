"""hdl_rl_hf.py — small GRPO-style RL for Verilog on one GPU (v5.185, HDL
track H4), reward from simulation equivalence with a golden design plus
a gate-count bonus (``helioslm_v5/eval/hdl_reward.py``).

  # 1. tasks (CPU, needs iverilog + yosys): golden designs that can be driven
  python examples/hdl_rl_hf.py prepare --parquet codev_r1_rl_train.parquet \\
      --out /workspace/rl/tasks.jsonl
  # 2. train (one GPU, vLLM for rollouts in a subprocess each round)
  python examples/hdl_rl_hf.py train --model /workspace/sft --tasks /workspace/rl/tasks.jsonl \\
      --out /workspace/runs/rl --max-hours 5

Each round samples ``--prompts`` tasks, generates ``--group`` answers per
task with vLLM from the current policy (saved in bf16 to
``<out>/policy``), scores them, and normalises rewards within each group
(advantage ``(r - mean) / (std + eps)``; groups where every answer got
the same reward are dropped, as in DAPO). The update is the on-policy
policy gradient: token-level ``-A * log p`` averaged over the response
tokens of the round, split into ``--minibatches`` optimizer steps. No
KL term and no reference model. The policy reuses ``cpt_hf.py``'s
loading (``--freeze``, ``--frozen-bf16``) and fp32 master weights.
Writes ``<out>/log.jsonl`` (per round: mean reward, pass rate, kept
groups, gate bonus), ``<out>/final`` and ``<out>/result.json``.

v5.207: ``--reward mcq`` trains on multiple-choice tasks instead
(``helioslm_v5/eval/mcq_reward.py``: reasoning, then 「答案：X」 checked
against the key), the DeepSeek-R1 recipe for exam-style questions:

  python examples/hdl_rl_hf.py prepare-mcq --mcq /data/tmmluplus --bench tmmluplus \
      --split train --eval /data/tmmluplus --eval-split test --out /workspace/rl/mcq.jsonl
  python examples/hdl_rl_hf.py prepare-mcq --distill answers.jsonl --out /workspace/rl/mcq.jsonl
  python examples/hdl_rl_hf.py train --reward mcq --model /workspace/sft \
      --tasks /workspace/rl/mcq.jsonl --max-new 1536 --out /workspace/runs/rl_mcq
"""
import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "examples"))

from helioslm_v5.eval import hdl_reward as hr  # noqa: E402
from helioslm_v5.eval import mcq_reward as mr  # noqa: E402
from helioslm_v5.eval.uvm_bench import cpu_limit  # noqa: E402

PROCESSOR_FILES = ("processor_config.json", "preprocessor_config.json", "chat_template.jinja",
                   "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


# ------------------------------------------------------------------ tasks

def _strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from _strings(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _strings(v)


def _maybe_json(s):
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return s


def row_prompt_golden(row):
    """(user prompt, golden code) of a CodeV-R1 RL row: the prompt is the
    user turn of ``prompt`` (messages or text); the golden design is the
    longest string with ``module`` ... ``endmodule`` under the other
    columns (``reward_model.ground_truth`` in the verl layout)."""
    p = row.get("prompt") or row.get("question") or row.get("instruction")
    if isinstance(p, str):
        p = _maybe_json(p)
    if isinstance(p, (list, tuple)):
        users = [m.get("content", "") for m in p if isinstance(m, dict) and m.get("role") == "user"]
        p = users[-1] if users else "\n".join(_strings(p))
    if not isinstance(p, str) or not p.strip():
        raise hr.RewardError("no prompt")
    def found(keys):
        out = []
        for k in keys:
            v = row.get(k)
            for s in _strings(_maybe_json(v) if isinstance(v, str) else v):
                for t in _strings(_maybe_json(s)):
                    if "module" in t and "endmodule" in t:
                        out.append(t)
        return out
    # the golden code lives in ground_truth; a model response column
    # (r1_response) is longer but is reasoning plus an answer, not the golden
    cands = found(("ground_truth", "reward_model")) or found(
        [k for k in row if k not in ("prompt", "question", "instruction")
         and "response" not in k])
    if not cands:
        raise hr.RewardError("no golden code")
    golden = max(cands, key=len)
    if "```" in golden or "<think>" in golden or "[BEGIN]" in golden:
        golden = hr.extract_answer(golden) or golden
    return p.strip(), golden            # raw source keeps its `timescale / `define lines


def cmd_prepare(a):
    import pyarrow.parquet as pq
    rows = pq.read_table(a.parquet).to_pylist()
    if a.limit:
        rows = rows[:a.limit]
    print("columns:", sorted(rows[0].keys()) if rows else [], flush=True)
    if rows:
        print("first row:", json.dumps({k: str(v)[:600] for k, v in rows[0].items()}), flush=True)

    def one(row):
        g = ""
        try:
            p, g = row_prompt_golden(row)
            return hr.make_task(p, g), None
        except hr.RewardError as e:
            return None, f"{e} | golden starts: {g[:160]!r}"
        except Exception as e:                       # a golden design we cannot parse
            return None, type(e).__name__

    with ThreadPoolExecutor(max_workers=a.workers or cpu_limit()) as ex:
        res = list(ex.map(one, rows))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    why, examples = {}, {}
    with open(out, "w", encoding="utf-8") as fh:
        for t, w in res:
            if t:
                fh.write(json.dumps(t) + "\n")
            else:
                key = w.split(":")[0][:60]
                why[key] = why.get(key, 0) + 1
                examples.setdefault(key, []).append(w[:400])
    kept = sum(1 for t, _ in res if t)
    st = {"rows": len(rows), "kept": kept,
          "with_ref_cells": sum(1 for t, _ in res if t and t.get("ref_cells")),
          "dropped": dict(sorted(why.items(), key=lambda kv: -kv[1])[:12])}
    for key in list(st["dropped"])[:4]:
        for w in examples[key][:2]:
            print("example", json.dumps(w), flush=True)
    print(json.dumps(st), flush=True)
    return st


def cmd_prepare_mcq(a):
    """MCQ tasks from a public set's train / dev split (``--mcq``) or from
    teacher answers (``--distill``), minus anything overlapping ``--eval``."""
    from helioslm_v5.eval import tw_bench as tb
    if bool(a.mcq) == bool(a.distill):
        raise SystemExit("give exactly one of --mcq / --distill")
    if a.mcq:
        tasks, st = mr.tasks_from_mcq(tb.load_mcq(a.mcq, a.bench, split=a.split), lang=a.lang), {}
    else:
        recs = [json.loads(x) for x in open(a.distill, encoding="utf-8") if x.strip()]
        tasks, st = mr.tasks_from_distill(recs, min_agree=a.min_agree, lang=a.lang)
    if a.eval:
        items = {}
        for path in a.eval:
            items.update(tb.eval_texts(tb.load_mcq(path, a.eval_bench or a.bench,
                                                   split=a.eval_split)))
        tasks, st["contaminated"] = mr.drop_contaminated(tasks, items)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    st["tasks"] = len(tasks)
    print(json.dumps(st), flush=True)
    return st


def score(text, task, a):
    if a.reward == "mcq":
        return mr.reward(text, task, lang_penalty=a.lang_penalty)
    return hr.reward(text, task, gate_weight=a.gate_weight)


# ------------------------------------------------------------------ rollouts

def render_prompt(tok, prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True)


def cmd_generate(a):
    """vLLM rollouts (run as a subprocess so its GPU memory is freed)."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(a.model)
    tasks = [json.loads(x) for x in open(a.tasks, encoding="utf-8") if x.strip()]
    prompts = [tok(render_prompt(tok, t["prompt"]), add_special_tokens=False)["input_ids"]
               for t in tasks]
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_mem, seed=a.seed)
    sp = SamplingParams(n=a.n, temperature=a.temperature, top_p=1.0,
                        max_tokens=a.max_new, seed=a.seed)
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts], sp)
    with open(a.out, "w", encoding="utf-8") as fh:
        for i, o in enumerate(outs):
            for c in o.outputs:
                fh.write(json.dumps({"task": i, "prompt_ids": prompts[i],
                                     "ids": list(c.token_ids), "text": c.text,
                                     "finish": c.finish_reason}) + "\n")


# ------------------------------------------------------------------ update

def group_advantages(rewards, group, eps=1e-4):
    """Per-group normalised advantages; groups with one reward value get
    None (dropped)."""
    out = []
    for i in range(0, len(rewards), group):
        g = np.asarray(rewards[i:i + group], dtype=np.float64)
        if g.std() < 1e-8:
            out += [None] * len(g)
        else:
            out += list((g - g.mean()) / (g.std() + eps))
    return out


def pg_loss(model, prompt_ids, resp_ids, adv, n_tokens, device):
    """``-adv * sum(log p(response))`` / ``n_tokens`` for one sample."""
    import torch
    import torch.nn.functional as F
    ids = torch.tensor([prompt_ids + resp_ids], device=device)
    k = len(resp_ids)
    out = model(input_ids=ids[:, :-1], logits_to_keep=k)
    logits = out.logits if hasattr(out, "logits") else out[0]
    lp = -F.cross_entropy(logits[0, -k:].float(), ids[0, -k:], reduction="sum")
    return -adv * lp / n_tokens


def save_policy(model, tok_dir, out):
    import torch
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    sd = {k: v.detach().to(torch.bfloat16) for k, v in model.state_dict().items()}
    model.save_pretrained(out, state_dict=sd, safe_serialization=True)
    for f in PROCESSOR_FILES:
        src = Path(tok_dir) / f
        if src.is_file():
            shutil.copy(src, out / f)


def cmd_train(a):
    import torch
    from transformers import AutoModelForCausalLM
    from cpt_hf import cast_frozen, freeze
    tasks = [json.loads(x) for x in open(a.tasks, encoding="utf-8") if x.strip()]
    rng = random.Random(a.seed)
    rng.shuffle(tasks)
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    trainable, frozen = freeze(model, a.freeze)
    if a.frozen_bf16:
        cast_frozen(model, torch.bfloat16)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(dict(vars(a), tasks_n=len(tasks),
                                                  trainable_params=trainable,
                                                  frozen_params=frozen), indent=2, default=str))
    logf = open(out / "log.jsonl", "a")
    policy = out / "policy"
    gen_model = a.model
    t0, step, cursor, rnd, stopped = time.time(), 0, 0, 0, None
    workers = a.reward_workers or cpu_limit()   # a pod's os.cpu_count() is the host's
    while True:
        if a.rounds and rnd >= a.rounds:
            break
        if (time.time() - t0) / 3600 > a.max_hours:
            stopped = "max_hours"
            break
        batch = [tasks[(cursor + i) % len(tasks)] for i in range(a.prompts)]
        cursor += a.prompts
        bfile, gfile = out / "batch.jsonl", out / "gen.jsonl"
        bfile.write_text("".join(json.dumps(t) + "\n" for t in batch))
        tg = time.time()
        torch.cuda.empty_cache()           # the vLLM subprocess needs the cached blocks
        r = subprocess.run([sys.executable, __file__, "generate", "--model", str(gen_model),
                            "--tasks", str(bfile), "--out", str(gfile), "--n", str(a.group),
                            "--max-new", str(a.max_new), "--gpu-mem", str(a.gpu_mem),
                            "--seed", str(a.seed + rnd)],
                           capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-2000:], r.stderr[-4000:], flush=True)
            stopped = "generate_failed"
            break
        gens = [json.loads(x) for x in open(gfile, encoding="utf-8") if x.strip()]
        t_gen = time.time() - tg
        with ThreadPoolExecutor(max_workers=workers) as ex:
            scored = list(ex.map(lambda g: score(g["text"], batch[g["task"]], a), gens))
        rewards = [s[0] for s in scored]
        advs = group_advantages(rewards, a.group)
        keep = [(g, ad) for g, ad in zip(gens, advs) if ad is not None and g["ids"]]
        n_tok = sum(len(g["ids"]) for g, _ in keep)
        tu = time.time()
        if keep:
            rng.shuffle(keep)
            parts = [keep[i::a.minibatches] for i in range(a.minibatches)]
            for part in parts:
                if not part:
                    continue
                pt = sum(len(g["ids"]) for g, _ in part)
                for g, ad in part:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = pg_loss(model, g["prompt_ids"], g["ids"], float(ad), pt, device)
                    loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
                if not bool(torch.isfinite(gn)):
                    opt.zero_grad(set_to_none=True)
                    stopped = "non_finite"
                    break
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            save_policy(model, a.model, policy)
            gen_model = policy
        if stopped:
            break
        correct = [s for s in scored if s[1].get("ok")]
        row = {"round": rnd, "step": step, "reward": round(float(np.mean(rewards)), 4),
               "pass": round(len(correct) / max(len(scored), 1), 4),
               "kept_samples": len(keep), "resp_tokens": n_tok,
               "truncated": sum(1 for g in gens if g["finish"] == "length"),
               "gate_bonus": round(float(np.mean([s[0] - 1 for s in correct])), 4)
               if correct and a.reward == "hdl" else None,
               "gen_s": round(t_gen, 1), "update_s": round(time.time() - tu, 1),
               "elapsed_s": round(time.time() - t0, 1)}
        logf.write(json.dumps(row) + "\n")
        logf.flush()
        print(json.dumps(row), flush=True)
        rnd += 1
    logf.close()
    save_policy(model, a.model, out / "final")
    res = {"rounds": rnd, "steps": step, "stopped": stopped}
    (out / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--parquet", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int)
    m = sub.add_parser("prepare-mcq")
    m.add_argument("--mcq", help="tw_bench.load_mcq file or directory")
    m.add_argument("--bench", default="tmmluplus")
    m.add_argument("--split", default="train")
    m.add_argument("--distill", help="distill_gen answers.jsonl of MCQ prompts")
    m.add_argument("--min-agree", type=float, default=1.0)
    m.add_argument("--eval", nargs="*", help="evaluation sets to decontaminate against")
    m.add_argument("--eval-bench")
    m.add_argument("--eval-split", default="test")
    m.add_argument("--lang", default="tw")
    m.add_argument("--out", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True)
    g.add_argument("--tasks", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--n", type=int, default=8)
    g.add_argument("--max-new", type=int, default=2048)
    g.add_argument("--max-model-len", type=int, default=4096)
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--gpu-mem", type=float, default=0.45)
    g.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("train")
    t.add_argument("--model", required=True)
    t.add_argument("--tasks", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--prompts", type=int, default=32)
    t.add_argument("--group", type=int, default=8)
    t.add_argument("--max-new", type=int, default=2048)
    t.add_argument("--lr", type=float, default=1e-6)
    t.add_argument("--minibatches", type=int, default=2)
    t.add_argument("--reward", choices=("hdl", "mcq"), default="hdl")
    t.add_argument("--gate-weight", type=float, default=hr.GATE_WEIGHT)
    t.add_argument("--lang-penalty", type=float, default=mr.LANG_PENALTY)
    t.add_argument("--gpu-mem", type=float, default=0.45)
    t.add_argument("--rounds", type=int, default=0, help="0: until --max-hours")
    t.add_argument("--max-hours", type=float, default=5.0)
    t.add_argument("--reward-workers", type=int)
    t.add_argument("--freeze", nargs="*", default=["per_layer", "vision", "audio"])
    t.add_argument("--frozen-bf16", action="store_true")
    t.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    return {"prepare": cmd_prepare, "prepare-mcq": cmd_prepare_mcq, "generate": cmd_generate,
            "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
