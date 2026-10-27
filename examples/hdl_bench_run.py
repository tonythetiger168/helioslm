"""hdl_bench_run.py — HDL track H0: score a Hugging Face model on
VerilogEval v2 spec-to-rtl (v5.147) or RTLLM v2.1 (v5.176).

  # on the pod (needs iverilog; yosys and pyslang optional)
  python -m helioslm_v5.eval.hdl_bench fetch
  python -m helioslm_v5.eval.hdl_bench self-check      # must be 156/156
  python examples/hdl_bench_run.py --model google/gemma-4-E2B \\
      --n 1 --out results/hdl/e2b_pass1.json
  # pass@1 with 20 samples at temperature 0.8 (upstream setting):
  python examples/hdl_bench_run.py --model ... --n 20 --temperature 0.8
  # RTLLM: fetch, self-check (prints any --exclude list), then score
  python -m helioslm_v5.eval.rtllm_bench fetch
  python -m helioslm_v5.eval.rtllm_bench self-check
  python examples/hdl_bench_run.py --bench rtllm --model ... --out results/hdl/rtllm.json

Greedy (``--n 1 --temperature 0``) is the cheap first number. Instruct
models get the system and user messages through their chat template;
base models get one plain prompt. Results follow helioslm_v5/eval/hdl_bench's
``hdl-bench/1`` schema. A greedy run of all 156 problems takes minutes on
one GPU, so results are written once at the end.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from helioslm_v5.eval import hdl_bench as hb  # noqa: E402
from helioslm_v5.eval import rtllm_bench as rb  # noqa: E402


def make_generate(model_id, n, temperature, top_p, max_new_tokens):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto")
    chat = tok.chat_template is not None

    def generate(system, user, i):
        if chat:
            ids = tok.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                add_generation_prompt=True, return_dict=True,
                return_tensors="pt")["input_ids"]
        else:
            ids = tok(system + "\n" + user, return_tensors="pt").input_ids
        ids = ids.to(model.device)
        kw = dict(max_new_tokens=max_new_tokens, num_return_sequences=n,
                  pad_token_id=tok.pad_token_id or tok.eos_token_id)
        if temperature > 0:
            kw.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            kw.update(do_sample=False)
        with torch.no_grad():
            out = model.generate(ids, **kw)
        return [tok.decode(o[ids.shape[1]:], skip_special_tokens=True)
                for o in out]
    return generate


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--bench", choices=("verilog-eval", "rtllm"),
                    default="verilog-eval")
    ap.add_argument("--cache", help="default: data/hdl_bench/<bench>")
    ap.add_argument("--exclude", default="",
                    help="rtllm: comma-separated designs to skip (from self-check)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--shots", type=int, default=0, choices=(0, 1))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--extra", action="store_true",
                    help="also run pyslang lint and yosys synth per sample")
    a = ap.parse_args(argv)
    if a.n > 1 and a.temperature == 0:
        ap.error("--n > 1 needs --temperature > 0")
    rtllm = a.bench == "rtllm"
    cache = a.cache or ("data/hdl_bench/rtllm_v2" if rtllm
                        else "data/hdl_bench/verilog_eval_v2")
    probs = (rb.load(cache) if rtllm else hb.load(cache))[:a.limit]
    gen = make_generate(a.model, a.n, a.temperature, a.top_p, a.max_new_tokens)
    ks = tuple(k for k in (1, 5, 10) if k <= a.n)
    if rtllm:
        res = rb.score(probs, gen, n=a.n, ks=ks,
                       exclude=tuple(x for x in a.exclude.split(",") if x))
    else:
        res = hb.score(probs, gen, n=a.n, ks=ks,
                       prefix=hb.shot_prefix(cache, a.shots), extra=a.extra)
    res.update(model=a.model, temperature=a.temperature, top_p=a.top_p,
               shots=a.shots)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("problems", "designs")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
