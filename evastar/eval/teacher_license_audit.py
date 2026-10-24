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
    # Re-read 2026-10-23 (v5.128): the repository still has no LICENSE
    # file; the card names OpenMDW-1.1 and links the canonical text,
    # 1.1/LICENSE.OpenMDW-1.1 in github.com/OpenMDW/OpenMDW, whose sha256
    # is recorded in license_text_sha256.
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16": {
        "reviewed_sha256": None,   # no LICENSE file in the repository
        "license": "OpenMDW-1.1 (model card; no LICENSE file in the repository)",
        "license_text_sha256": "2ab44b68365473c112f5092211a38f231cb23e50de68b75a13369adbd76a74df",
        "verdict": "allowed",
        "conditions": [
            "the agreement imposes no restriction or obligation on outputs",
            "distributing any of the Model Materials: keep a copy of the "
            "agreement and the copyright and origin notices",
            "a patent or copyright suit claiming the Model Materials "
            "infringe ends the grant",
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
# OS4 (v5.99): the five models of the 2026-10-22 open-model review. Read
# 2026-10-22 through a web reader because huggingface.co was not reachable
# from the build container, so no byte-exact hash could be taken; those
# rows stayed PENDING (not usable) until a live audit pinned them. Pinned
# 2026-10-23 (v5.131) from a live audit of the Hub, each fetched text
# compared with the recorded reading first (pin_pending). MiniMax M3 is
# already a pinned row above.
PENDING = "pending: hash not pinned"
REVIEWS.update({
    "tencent/Hy3": {
        # Tencent header, then the Apache-2.0 text word for word; no NOTICE
        "reviewed_sha256": "76a14d21e741dbbf1572e9c0e6bb809a9bd76f5b4da64a0b89e7331cc77ea84a",
        "license": "Apache-2.0 (with a Tencent header naming Hy3 and its "
                   "copyright)",
        "verdict": "allowed",
        "conditions": ["keep the licence, the Tencent copyright header and "
                       "any NOTICE (s.4); the repository has no NOTICE file",
                       "earlier Hunyuan releases used the Tencent Hunyuan "
                       "Community License (territory exclusions); the "
                       "pinned text is the plain Apache one"],
    },
    "XiaomiMiMo/MiMo-V2.5-Pro": {
        "reviewed_sha256": None,   # no LICENSE file in the repository
        "license": "MIT (model card; no LICENSE file in the repository)",
        "verdict": "allowed",
        "conditions": ["keep the copyright and permission notice",
                       "confirm the licence with the MiMo GitHub repository "
                       "before use: the Hub repo carries no LICENSE file"],
    },
    "deepseek-ai/DeepSeek-V4-Flash": {
        "reviewed_sha256": "f2c6c602815669d292889e5be8c802f2ed950653b77999b1584e8e6aed25d040",
        "license": "MIT",
        "verdict": "allowed",
        "conditions": ["keep the copyright and permission notice",
                       "same text as V4.1-Flash (same hash)"],
    },
    "zai-org/GLM-5.2": {
        "reviewed_sha256": "f4a18c6ae40b0a8e7d2b7667f52f6e1994e54a46430d2e172b73cb8c9b5eb0d7",
        "license": "MIT (Copyright (c) 2026 Zhipu AI)",
        "verdict": "allowed",
        "conditions": ["keep the copyright and permission notice",
                       "unlike GLM-5.3 (GLM-5.3 License, security review "
                       "over US$10B MaaS revenue), 5.2 is plain MIT"],
    },
})
# 2026-10-24 re-survey of the whitelist: three newer MIT releases, each
# LICENSE text read and hashed from a live audit of the Hub before pinning.
REVIEWS.update({
    "deepseek-ai/DeepSeek-V4.1-Flash-1017": {
        "reviewed_sha256": "f2c6c602815669d292889e5be8c802f2ed950653b77999b1584e8e6aed25d040",
        "license": "MIT",
        "verdict": "allowed",
        "conditions": ["keep the copyright and permission notice",
                       "same text as V4.1-Flash (same hash)",
                       "the release ships no Jinja chat template: prompts "
                       "go through the repository's encoding scripts"],
    },
    "deepseek-ai/DeepSeek-V4-Pro-0813": {
        "reviewed_sha256": "f2c6c602815669d292889e5be8c802f2ed950653b77999b1584e8e6aed25d040",
        "license": "MIT",
        "verdict": "allowed",
        "conditions": ["keep the copyright and permission notice",
                       "same text as V4.1-Flash (same hash); the official "
                       "release that supersedes the V4-Pro preview"],
    },
    "zai-org/GLM-5.3-Flash": {
        "reviewed_sha256": "30b85b6b9659f2e78aa259f8faf5d920a68dee7c9ced3fa6dba1f19f2bc4fca1",
        "license": "MIT (Copyright (c) 2026 Z.AI Co., Ltd)",
        "verdict": "allowed",
        "conditions": ["keep the copyright and permission notice",
                       "unlike GLM-5.3 (GLM-5.3 License), the Flash "
                       "release is plain MIT"],
    },
})
TEACHERS = tuple(r for r, v in REVIEWS.items()
                 if v["reviewed_sha256"] != PENDING)
PENDING_TEACHERS = tuple(r for r, v in REVIEWS.items()
                         if v["reviewed_sha256"] == PENDING)
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
        if review["reviewed_sha256"] == PENDING:
            rows.append({"id": rid, **meta, "license_sha256": sha, **review,
                         "pinned": False})
            continue
        if sha != review["reviewed_sha256"]:
            raise ValueError(f"{rid}: licence text changed since it was "
                             f"reviewed ({review['reviewed_sha256']} -> "
                             f"{sha}); re-read it and update REVIEWS")
        rows.append({"id": rid, **meta, "license_sha256": sha, **review,
                     "pinned": True})
    return rows


def usable(rid):
    """May ``rid``'s outputs be used for distillation today? Only a pinned
    row whose verdict allows it (conditions still apply)."""
    r = REVIEWS.get(rid)
    return bool(r) and r["reviewed_sha256"] != PENDING and \
        r["verdict"] in ("allowed", "allowed_with_conditions")


def pin_pending(rows):
    """REVIEWS entries to paste after a live audit of PENDING_TEACHERS:
    the reviewer compares each fetched text with the recorded reading
    first; this only fills in the hash it saw."""
    return {r["id"]: {**{k: REVIEWS[r["id"]][k] for k in
                         ("license", "verdict", "conditions")},
                      "reviewed_sha256": r["license_sha256"]}
            for r in rows if not r["pinned"]}
