"""MidCandidate — BPE-tokenizer candidate for the mid_sft_v5.33 checkpoint.

Drop-in replacement for CharCandidate in agentic_candidate_run.py.
The mid model (360M params, BPE vocab 32768, chat-SFT format) is driven
through the same chat transcript format it was trained on:

    ##user## <task>
    ##assistant## <reply>
    ##tool## <observation>

For term_bench / swe_bench the tool protocol differs (bash commands in
```bash blocks), so we use a custom system prompt that matches the
mid model's SFT distribution while exposing the term_bench protocol.
"""

import hashlib
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


class MidCandidate:
    """One mid checkpoint -> every component's generate/policy function."""

    MAX_NEW = 128

    def __init__(self, checkpoint, tokenizer=None):
        import torch
        from helioslm_v5.configs.config_v5 import HeliosLMv5Config
        from helioslm_v5.src.model_v5 import HeliosLMv5
        from helioslm_v5.src.tokenizer.bpe import HeliosBPE, BOS

        self._torch = torch
        ckpt_dir = Path(checkpoint).parent
        tok_path = tokenizer or (ckpt_dir / "mid_sft_v5.33.tok.json")

        self.tok = HeliosBPE.load(str(tok_path))
        self.bos = self.tok.vocab[BOS]

        self.model = HeliosLMv5(HeliosLMv5Config(size="mid"))
        self.model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True))
        self.model.eval()

        self.checkpoint_sha256 = hashlib.sha256(
            Path(checkpoint).read_bytes()).hexdigest()
        self.model_name = "helioslm-mid-v5.33"

    def generate(self, prompt, max_new=MAX_NEW):
        """Greedy decode up to max_new tokens, return generated text."""
        torch = self._torch
        ids = [self.bos] + self.tok.encode(prompt)
        input_ids = torch.tensor([ids])
        with torch.no_grad():
            out = self.model.generate(input_ids, max_new_tokens=max_new,
                                      temperature=0)
        return self.tok.decode(out[0, len(ids):].tolist())

    # -- term_bench / swe_bench episode interface: (prompt, sample) -> text
    def episode(self, prompt, sample):
        return self.generate(prompt)

    # -- long_horizon policy interface: (scenario, observations) -> text
    def policy(self, scenario, observations):
        prompt = ("You are an agent that replies with exactly one tool call.\n"
                  f"Task: {scenario.task}\n"
                  + "".join(f"observation: {o}\n" for o in observations)
                  + "Your reply:")
        return self.generate(prompt)

    # -- hdl_bench interface: (system, user, i) -> [text] * n
    def hdl(self, system, user, i):
        return [self.generate(system + "\n" + user, max_new=384)]

    # -- chat-format wrapper for tasks that expect tool protocol
    def chat_episode(self, prompt, sample):
        """Wrap prompt in the chat transcript format the mid model knows."""
        full = (self._chat_system() + "\n"
                + "##user## " + prompt + "\n"
                + "##assistant##")
        return self.generate(full)

    def _chat_system(self):
        if not hasattr(self, "_cached_system"):
            from helioslm_v5.agent.chat import CHAT_SYSTEM
            from helioslm_v5.agent.tools import build_default_registry
            from helioslm_v5.agent.loop import render_tool_docs
            reg, _ = build_default_registry()
            self._cached_system = CHAT_SYSTEM.replace(
                "%%TOOLS%%", render_tool_docs(reg))
        return self._cached_system
