"""Teacher licence audit for distillation (v5.82): the ten open-weight
frontier models considered as teachers.

Before any output of an open-weight model is used to train HeliosLM, this
records which licence governs the weights, the exact licence text read
(sha256), and a reviewed reading of what it allows. The reading is a
human/assistant review checked into the repo (REVIEWS), not inferred by
code: licence text is legal language, and a keyword match is no
substitute. Card metadata and file hashes are fetched live so a changed
licence shows up as a hash mismatch.

Scope notes recorded with every row:
  - these are the WEIGHT licences. Outputs obtained through a vendor's own
    API are governed by that API's terms of service instead, which can
    differ (and for closed models - Claude, GPT, Gemini - forbid training
    competing models);
  - whether a student trained on a teacher's outputs is a "derivative
    work" of the teacher is unsettled; the conditions below are recorded
    as if it were (the conservative reading).
"""
import hashlib

VERDICTS = ("allowed", "allowed_with_conditions", "not_allowed", "unclear")

# Reviewed 2026-10-20 against the LICENSE text whose sha256 is recorded.
REVIEWS = {
    "deepseek-ai/DeepSeek-V4.1-Flash": {
        "reviewed_sha256": "f2c6c602815669d292889e5be8c802f2ed950653b77999b1584e8e6aed25d040",
        "license": "MIT",
        "verdict": "allowed",
        "conditions": ["keep the copyright and permission notice"],
    },
    "deepseek-ai/DeepSeek-V4-Pro": {
        "reviewed_sha256": "f2c6c602815669d292889e5be8c802f2ed950653b77999b1584e8e6aed25d040",
        "license": "MIT",
        "verdict": "allowed",
        "conditions": ["keep the copyright and permission notice"],
    },
    "moonshotai/Kimi-K3": {
        "reviewed_sha256": "20c797ce19af0c17de52c6afb144644768a591c521655f5ebf5712c9850f2887",
        "license": "Kimi K3 License",
        "verdict": "allowed_with_conditions",
        "conditions": [
            "keep the copyright and permission notice (s.1)",
            "commercial products over 100M MAU or US$20M monthly revenue "
            "must display 'Kimi K3' (s.3)",
            "a Model-as-a-Service business over US$20M revenue in 12 months "
            "needs a separate agreement before commercial use (s.2)",
            "internal use (nothing made available to third parties) is "
            "exempt from s.2 and s.3 (s.4)",
        ],
    },
    "Qwen/Qwen3.8-2.4T-A95B": {
        "reviewed_sha256": "a136bc4137ef3d7a7a54963164a47270f73e497917728ea0f4407a0115db1290",
        "license": "Qwen3.8-Max License",
        "verdict": "allowed_with_conditions",
        "conditions": [
            "keep the copyright and permission notice (s.1)",
            "commercial products over 100M MAU or US$20M monthly revenue "
            "must display the model name (s.1)",
            "a Model-as-a-Service or AI Work Assistant business over US$50M "
            "revenue in 12 months needs a separate licence (s.2)",
        ],
    },
    "zai-org/GLM-5.3": {
        "reviewed_sha256": "96e1622099fc9d6b70c9760f007d99e66d7497eec636b63c60fe208401e9170c",
        "license": "GLM-5.3 License",
        "verdict": "allowed_with_conditions",
        "conditions": [
            "keep the copyright and permission notice (s.1)",
            "a Model-as-a-Service business over US$10B revenue in 12 months "
            "must pass Z.AI's security review before commercial use (s.2)",
        ],
    },
    "XiaomiMiMo/MiMo-V2.6-Pro-MOPD": {
        "reviewed_sha256": None,   # no LICENSE file to hash
        "license": "MIT (model card; no LICENSE file in the repository)",
        "verdict": "allowed",
        "conditions": ["keep the copyright and permission notice",
                       "confirm the licence with the MiMo GitHub repository "
                       "before use: the Hub repo carries no LICENSE file"],
    },
    "MiniMaxAI/MiniMax-M3": {
        "reviewed_sha256": "b53f2fdda3049b0e9013207be51efc2d372cda1fcfdd8bb4bb8b22658ca5db9c",
        "license": "MiniMax Community License",
        "verdict": "allowed_with_conditions",
        "conditions": [
            "keep the copyright and permission notice (s.1)",
            "any commercial use of the Software or a derivative: display "
            "'Built with MiniMax M3' and send a one-time notice to MiniMax "
            "(s.2.1, s.2.2)",
            "commercial products over US$20M yearly revenue need prior "
            "written authorization from MiniMax (s.2.2)",
            "Appendix prohibited uses apply to derivatives (incl. military "
            "purposes)",
        ],
    },
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16": {
        "reviewed_sha256": None,   # no LICENSE file; text at openmdw.ai
        "license": "OpenMDW-1.1 (model card; text not readable here)",
        "verdict": "unclear",
        "conditions": [
            "read the OpenMDW-1.1 text before use: openmdw.ai is not "
            "reachable from this environment and no licence file is in the "
            "repository; secondary reports call it permissive (commercial "
            "use and derivatives, keep notices) but none states its terms "
            "on outputs",
        ],
    },
    "openai/gpt-oss-120b": {
        "reviewed_sha256": "58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd",
        "license": "Apache-2.0",
        "verdict": "allowed",
        "conditions": ["keep the licence and NOTICE (s.4)",
                       "OpenAI's separate gpt-oss usage policy (not in the "
                       "repository) should be read before use"],
    },
    "google/gemma-4-31B-it": {
        "reviewed_sha256": None,   # no LICENSE file in the repository
        "license": "Apache-2.0 (model card; no LICENSE file in the repository)",
        "verdict": "allowed",
        "conditions": ["keep the licence notice",
                       "confirm on Google's Gemma 4 licence page: Gemma 1-3n "
                       "used the Gemma Terms of Use, which constrained "
                       "derivatives; Gemma 4 is reported to be plain "
                       "Apache 2.0"],
    },
}
TEACHERS = tuple(REVIEWS)
SCOPE_NOTES = (
    "weight licences only: outputs from a vendor's own API follow that "
    "API's terms of service",
    "a distilled student is treated as a derivative work (conservative)",
)


def license_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hub_fetch(repo_id):
    """Live card licence, revision and LICENSE text (network)."""
    import urllib.request

    from huggingface_hub import HfApi
    info = HfApi().model_info(repo_id)
    card = info.card_data
    files = [s.rfilename for s in info.siblings
             if s.rfilename.upper().startswith("LICENSE")]
    text = None
    if files:
        url = f"https://huggingface.co/{repo_id}/raw/{info.sha}/{files[0]}"
        text = urllib.request.urlopen(url, timeout=30).read().decode("utf-8")
    return {"card_license": card.get("license") if card else None,
            "card_license_name": card.get("license_name") if card else None,
            "sha": info.sha, "license_file": files[0] if files else None,
            "license_text": text}


def audit(teachers=TEACHERS, fetch=hub_fetch):
    rows = []
    for rid in teachers:
        if rid not in REVIEWS:
            raise KeyError(f"{rid} has no reviewed licence reading; review "
                           "the licence text and add it to REVIEWS first")
        meta = fetch(rid)
        text = meta.pop("license_text")
        sha = license_sha(text) if text else None
        review = REVIEWS[rid]
        if sha != review["reviewed_sha256"]:
            raise ValueError(f"{rid}: licence text changed since it was "
                             f"reviewed ({review['reviewed_sha256']} -> "
                             f"{sha}); re-read it and update REVIEWS")
        rows.append({"id": rid, **meta, "license_sha256": sha, **review})
    return rows
