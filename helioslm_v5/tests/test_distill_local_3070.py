"""v5.253: examples/distill_local_3070.ps1, Qwen3.6-35B-A3B as a local
teacher behind llama-server on an 8 GB laptop GPU.

PowerShell and llama.cpp are not in CI, so the script is checked
statically and its request path is run against a stand-in server that
answers the way llama-server does. Pins:
1. the script's teacher alias, served name and GGUF repo match
   distill_gen.TEACHERS, and the licence review allows that teacher;
2. llama-server is started with the chat template (--jinja), the served
   name as --alias, reasoning split out (--reasoning-format deepseek) and
   bound to 127.0.0.1; every distill_generate.py flag it passes exists;
3. with llama-server's reply shape (reasoning_content, prompt_tokens /
   completion_tokens), generate records the reasoning, the token counts
   and the served name, and sends the Qwen thinking switch;
4. a medical manifest is accepted for that endpoint (local teacher).
"""
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from helioslm_v5.eval.teacher_license_audit import usable  # noqa: E402
from helioslm_v5.src.training import distill_gen as dg  # noqa: E402
from helioslm_v5.src.training.data_guard import check_teacher_endpoint  # noqa: E402

PS1 = (ROOT / "examples" / "distill_local_3070.ps1").read_text(encoding="utf-8")


def _var(name):
    m = re.search(rf'^\${name} = "([^"]+)"', PS1, re.M)
    assert m, name
    return m.group(1)


def test_pins_match_the_teacher_table():
    t = dg.TEACHERS[_var("Teacher")]
    assert t["served_model"] == _var("ServedModel")
    assert usable(t["repo"])
    assert _var("GgufRepo") == "unsloth/" + t["repo"].split("/")[1] + "-GGUF"
    assert re.fullmatch(r"[0-9a-f]{40}", _var("GgufRevision"))
    assert '[string]$Quant = "UD-Q4_K_M"' in PS1


def test_server_and_generate_arguments():
    for flag in ('"--jinja"', '"--alias", $ServedModel', '"--reasoning-format", "deepseek"',
                 '"--host", "127.0.0.1"', '"--cpu-moe"', '"--n-cpu-moe"'):
        assert flag in PS1, flag
    cli = (ROOT / "examples" / "distill_generate.py").read_text(encoding="utf-8")
    known = set(re.findall(r'add_argument\("(--[a-z-]+)"', cli))
    gen = PS1[PS1.index("$genArgs = @("):PS1.index("$t0 = Get-Date")]
    used = set(re.findall(r'"(--[a-z-]+)"', gen))
    assert used and used <= known, used - known
    assert '"--endpoint", "vllm"' in gen and '"--teacher", $Teacher' in gen


def test_generate_against_a_llama_server_stand_in(tmp_path, capsys):
    import distill_generate as cli
    bodies = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            bodies.append(body)
            out = {"model": body["model"],
                   "choices": [{"finish_reason": "stop",
                                "message": {"role": "assistant", "content": "答案是 4",
                                            "reasoning_content": "2 加 2 等於 4"}}],
                   "usage": {"prompt_tokens": 12, "completion_tokens": 30,
                             "total_tokens": 42}}
            data = json.dumps(out, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("".join(json.dumps({"prompt": f"題目 {i}：2 加 2 等於多少？"},
                                          ensure_ascii=False) + "\n" for i in range(3)),
                       encoding="utf-8")
    out = tmp_path / "answers.jsonl"
    try:
        cli.main(["generate", "--prompts", str(prompts), "--out", str(out),
                  "--teacher", _var("Teacher"), "--endpoint", "vllm",
                  "--base-url", f"http://127.0.0.1:{srv.server_port}/v1",
                  "--max-tokens", "6144", "--workers", "2"])
    finally:
        srv.shutdown()
    capsys.readouterr()
    recs = dg.read_jsonl(str(out))
    assert len(recs) == 3
    for r in recs:
        assert r["reasoning"] == "2 加 2 等於 4" and r["content"] == "答案是 4"
        assert r["usage"]["input_tokens"] == 12 and r["usage"]["output_tokens"] == 30
        assert r["teacher"]["repo"] == "Qwen/Qwen3.6-35B-A3B"
    for b in bodies:
        assert b["model"] == _var("ServedModel") and b["max_tokens"] == 6144
        assert b["chat_template_kwargs"]["enable_thinking"] is True


def test_medical_prompts_may_use_the_local_teacher():
    manifest = {"sources": [{"name": "reports", "domain": "medical"}]}
    check_teacher_endpoint(manifest, "http://127.0.0.1:8000/v1", "vllm")
