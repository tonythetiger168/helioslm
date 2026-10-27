"""Dataset licence audit for the training data plan (roadmap TW1): Traditional
and Simplified Chinese, Spanish, Japanese, Korean, and the evaluation sets.

For every Hugging Face dataset id the audit reads the dataset card's
``license`` field, whether access is gated, and the revision (commit sha)
it read, and classifies what the licence allows for training a model that
may be used commercially:

  commercial         permissive or attribution / share-alike licences
                     (CC BY, CC BY-SA, CC0, ODC-By, ODbL, PDDL, MIT, Apache,
                     BSD, OGDL-Taiwan, GFDL);
  noncommercial      any licence with a NonCommercial term (CC BY-NC*);
  no_derivatives     CC BY-ND (whether training counts as a derivative work
                     is a legal question; flagged, not decided);
  unknown            missing, "other", "unknown", or anything unrecognised;
                     read the card before use.

Nothing here touches the network on import. ``audit()`` takes a
``fetch(repo_id) -> dict`` callable; the default one uses
``huggingface_hub.HfApi().dataset_info`` and needs network access to
huggingface.co (and ``HF_TOKEN`` for gated cards). The card is metadata
only: a card's licence can be wrong (e.g. a derivative relabelled MIT), so
the report is a screening list, not legal advice.
"""
import re

# The datasets the TW / Chinese plan names (docs/TW_DATASETS_2026Q4.md and
# docs/TW_DATASETS_2026-10-20.md). Ids marked in DATASETS_UNVERIFIED were found by
# search but not opened; the audit is how they get verified.
TRADITIONAL = (
    "wikimedia/wikipedia",
    "HuggingFaceFW/fineweb-2",
    "lianghsun/fineweb-zhtw",
    "lianghsun/fineweb-2-zhtw",
    "twinkle-ai/fineweb-zhtw-filtered",
    "liswei/Taiwan-Text-Excellence-2B",
    "lianghsun/tw-news-551M",
    "lianghsun/tw-science-24M",
    "lianghsun/Taiwan_c4",
    "botp/yentinglin-zh_TW_c4",
    "liswei/c4-zhtw",
    "lianghsun/cosmopedia-zhtw",
    "yentinglin/TaiwanChat",
    "lianghsun/tw-instruct-500k",
    "lianghsun/tw-instruct-pro",
    "lianghsun/tw-reasoning-instruct",
    "lianghsun/wikipedia-pretrain-zh-tw-chat",
    "raptorkwok/cantonese-traditional-chinese-parallel-corpus",
)
SIMPLIFIED = (
    "opencsg/chinese-fineweb-edu-v2",
    "Skywork/SkyPile-150B",
    "BAAI/CCI3-HQ",
    "m-a-p/MAP-CC",
)
# Spanish (2026-10-20). wikimedia/wikipedia (config es) and
# HuggingFaceFW/fineweb-2 (config spa_Latn) are already listed above.
SPANISH = (
    "uonlp/CulturaX",
    "josecannete/large_spanish_corpus",
    "spanish-ir/eswiki_20240401_corpus",
    "FreedomIntelligence/evol-instruct-spanish",
    "FreedomIntelligence/MMLU_Spanish",
    "amias-mx/mmlu-redux-2.0-spanish",
)
# Japanese and Korean (2026-10-20); pretraining, instruction and eval
# sets together. FineWeb-2 (jpn_Jpan, kor_Hang) and Wikipedia (ja, ko) are
# covered by the cards listed above.
JAPANESE = (
    "llm-book/japanese-wikipedia",
    "Podtech/llm-jp-corpus-v4-ja_wiki",
    "kanhatakeyama/japanese-corpus-categorized",
    "llm-jp/llm-jp-4.1-thinking-sft-data",
    "Kendamarron/cosmopedia-japanese-instruction-mistral-nemo",
    "nlp-waseda/JMMLU",
    "nlp-waseda/JMMLU_CC-BY-SA",
)
KOREAN = (
    "lcw99/wikipedia-korean-20240501",
    "korean-corpus/namu_wiki_512_char_seg",
    "heegyu/open-korean-instructions",
    "devngho/korean-instruction-mix",
    "HAERAE-HUB/KMMLU",
    "LGAI-EXAONE/KMMLU-Redux",
    "skt/kobest_v1",
)
EVAL = (
    "ikala/tmmluplus",
    "MediaTek-Research/TCEval-v2",
    "JerryWu/TW-JusticeBench",
)
# Ids that the audit has not yet resolved on the Hub. Empty after the
# 2026-10-20 run: voidful/fineweb-zhtw (404) was replaced by the lianghsun /
# twinkle-ai FineWeb-zhtw sets, yentinglin/zh_TW_c4 (404) by its
# botp mirror and liswei/c4-zhtw, and miulab/tmlu (not on the Hub under any
# name searched) was dropped.
DATASETS_UNVERIFIED = frozenset()

CLASSES = ("commercial", "noncommercial", "no_derivatives", "unknown")
_COMMERCIAL = re.compile(
    r"^(cc-by(-sa)?(-\d(\.\d)?)?|cc0(-1\.0)?|odc-by|odbl|pddl|mit|"
    r"apache(-2\.0)?|bsd(-\d-clause)?|ogdl-taiwan(-1\.0)?|unlicense|"
    r"gfdl(-1\.\d)?)$")


def classify(license_id):
    """Map a card licence id (str, list or None) to one of CLASSES. A list
    takes the most restrictive member."""
    if isinstance(license_id, (list, tuple)):
        if not license_id:
            return "unknown"
        order = {c: i for i, c in enumerate(
            ("unknown", "noncommercial", "no_derivatives", "commercial"))}
        return min((classify(x) for x in license_id), key=order.__getitem__)
    if not license_id or not isinstance(license_id, str):
        return "unknown"
    lic = license_id.strip().lower()
    if "-nc" in lic or "noncommercial" in lic:
        return "noncommercial"
    if "-nd" in lic:
        return "no_derivatives"
    if _COMMERCIAL.match(lic):
        return "commercial"
    return "unknown"


def hub_fetch(repo_id, token=None):
    """Card metadata from the Hub (network). Returns license, gated, sha."""
    from huggingface_hub import HfApi
    info = HfApi(token=token).dataset_info(repo_id)
    card = getattr(info, "card_data", None) or getattr(info, "cardData", None)
    lic = None
    if card is not None:
        lic = card.get("license") if hasattr(card, "get") else getattr(
            card, "license", None)
    return {"license": lic, "gated": getattr(info, "gated", False),
            "sha": getattr(info, "sha", None)}


ALL = TRADITIONAL + SIMPLIFIED + SPANISH + JAPANESE + KOREAN + EVAL


def audit(repo_ids, fetch=hub_fetch):
    """One row per id, in order. A fetch error becomes a row with
    ``error`` set and class "unknown" (the audit never aborts halfway)."""
    rows = []
    for rid in repo_ids:
        row = {"id": rid, "verified_id": rid not in DATASETS_UNVERIFIED}
        try:
            meta = fetch(rid)
        except Exception as exc:  # network, 404, gated without token
            row.update({"license": None, "gated": None, "sha": None,
                        "class": "unknown",
                        "error": f"{type(exc).__name__}: {exc}"[:200]})
        else:
            row.update({"license": meta.get("license"),
                        "gated": meta.get("gated"),
                        "sha": meta.get("sha"),
                        "class": classify(meta.get("license")),
                        "error": None})
        rows.append(row)
    return rows


def summary(rows):
    out = {c: [] for c in CLASSES}
    for r in rows:
        out[r["class"]].append(r["id"])
    out["errors"] = [r["id"] for r in rows if r["error"]]
    return out
