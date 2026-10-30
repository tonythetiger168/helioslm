"""Knowledge rephrasing: more CPT tokens from the same zh-TW documents
(v5.206).

The public-data pilots (docs/PILOT_B0.md) showed that a few hundred
million tokens of web and wiki text, seen once or repeated, do not move
TMMLU+. The Kimi K2 report's answer to a small high-quality corpus is to
rewrite it: an instruct model restates each document in several styles
and perspectives, each rewrite checked against its source, so a CPT run
sees the same knowledge many times without seeing the same text twice
(their 10 rewrites x 1 epoch beat 1 text x 10 epochs on SimpleQA). This
module is that pipeline on top of the D1 generator (``distill_gen``):

  chunks     a long document is split at paragraph, then sentence,
             boundaries into chunks of at most ``max_chars``; each chunk
             is rewritten with the tail of the preceding source chunk as
             context, so chunks can go to the teacher in parallel and
             still join into one coherent document;
  styles     ``STYLES``: textbook, Q&A, study notes, encyclopaedia,
             spoken explanation (and a step-by-step learning-note style
             for maths), all asking for Taiwan Traditional Chinese and
             forbidding new facts;
  fidelity   ``fidelity_check``: the teacher stopped normally, the output
             is not Simplified Chinese, its length is within the style's
             ratio band, the source's numbers survive, and its character
             bigram overlap with the source is high enough to be about the
             same thing yet low enough not to be a copy;
  assemble   a (document, style) rewrite is kept only when every chunk
             passed; the result keeps the source's licence and
             attribution plus the teacher's name, in the record layout of
             ``tw2_prepare_corpus.py`` (id, source, text, license,
             attribution), so ``pilot_prepare_data.py --text`` tokenises
             it unchanged.

Teachers go through ``distill_gen.check_teacher`` (licence review in
``eval/teacher_license_audit.py``). Run ``eval/decontam.py`` on the
output as on any training text: a rewrite of a page that quotes an exam
question is still contaminated. Nothing here needs torch.
"""
import re

from .data_quality import zh_profile

STYLES = {
    "textbook": {
        "instruction": ("請把下面這段文字改寫成高中教科書的寫法：先點出主題，"
                        "再有條理地說明重點與因果關係。"),
        "ratio": (0.6, 1.8)},
    "qa": {
        "instruction": ("請根據下面這段文字，寫出 3 到 6 組問答（「問：」「答：」），"
                        "問題涵蓋文中的重要事實，答案完整、可以單獨閱讀。"),
        "ratio": (0.5, 2.2)},
    "notes": {
        "instruction": ("請把下面這段文字整理成條列式的讀書筆記，"
                        "保留所有人名、地名、年份、數字與專有名詞。"),
        "ratio": (0.4, 1.4)},
    "encyclopedia": {
        "instruction": "請用百科全書條目的中立、客觀語氣重寫下面這段文字。",
        "ratio": (0.6, 1.6)},
    "spoken": {
        "instruction": ("請用老師上課講解給學生聽的口吻，重新說明下面這段文字的內容，"
                        "可以加上簡單的舉例幫助理解，但例子不能和原文事實矛盾。"),
        "ratio": (0.7, 2.2)},
    "math_steps": {
        "instruction": ("請把下面這段數學內容改寫成學習筆記：寫出定義、"
                        "逐步推導每一個步驟，最後總結重點。"),
        "ratio": (0.7, 2.5)},
}

SYSTEM = ("你是一位臺灣的專業編輯。只能使用臺灣正體中文與臺灣常用詞彙。"
          "改寫時不可新增原文沒有的事實、數字或引述，也不可遺漏重要資訊。"
          "直接輸出改寫後的內容，不要加任何前言或說明。")

DEFAULT_STYLES = ("textbook", "qa", "notes", "encyclopedia", "spoken")

_PARA = re.compile(r"\n\s*\n")
_SENT = re.compile(r"(?<=[。！？!?；;])")
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_PREAMBLE = re.compile(r"^\s*(?:好的|以下是|下面是|當然)[^\n]{0,40}[：:]\s*\n")


def _pieces(text, max_chars):
    """Paragraphs, with any paragraph over ``max_chars`` cut at sentence
    ends (and, for one huge sentence, hard-cut)."""
    for para in _PARA.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            yield para
            continue
        for sent in _SENT.split(para):
            while len(sent) > max_chars:
                yield sent[:max_chars]
                sent = sent[max_chars:]
            if sent.strip():
                yield sent


def split_chunks(text, max_chars=1500):
    """Greedy packing of paragraphs / sentences into chunks of at most
    ``max_chars`` characters; paragraph breaks inside a chunk are kept."""
    if max_chars < 50:
        raise ValueError("max_chars must be >= 50")
    chunks, cur = [], ""
    for p in _pieces(text, max_chars):
        sep = "\n\n" if cur else ""
        if len(cur) + len(sep) + len(p) <= max_chars:
            cur += sep + p
        else:
            chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks


def rephrase_prompts(doc, styles=DEFAULT_STYLES, max_chars=1500,
                     context_chars=300):
    """Prompt records (``distill_gen.normalize_prompt`` form) for every
    (style, chunk) of one document ``{"id", "text", ...}``."""
    for s in styles:
        if s not in STYLES:
            raise ValueError(f"unknown style {s!r}; known: {sorted(STYLES)}")
    chunks = split_chunks(doc["text"], max_chars)
    out = []
    for s in styles:
        for i, chunk in enumerate(chunks):
            ctx = chunks[i - 1][-context_chars:] if i and context_chars else ""
            user = STYLES[s]["instruction"]
            if ctx:
                user += ("\n\n【前文，只供銜接語氣，不要改寫】\n" + ctx)
            user += "\n\n【要改寫的文字】\n" + chunk
            out.append({"id": f"{doc['id']}~{s}~{i:04d}", "domain": "rephrase",
                        "messages": [{"role": "system", "content": SYSTEM},
                                     {"role": "user", "content": user}],
                        "meta": {"doc_id": doc["id"], "style": s, "chunk": i,
                                 "n_chunks": len(chunks), "source_text": chunk}})
    return out


def clean_output(text):
    """Drop a one-line preamble such as 「以下是改寫後的內容：」."""
    return _PREAMBLE.sub("", text or "", count=1).strip()


def _bigrams(text):
    cjk = "".join(_CJK.findall(text))
    return {cjk[i:i + 2] for i in range(len(cjk) - 1)}


def fidelity_check(source, output, style, *, finish_reason="stop",
                   max_simp_share=0.02, min_number_recall=0.8,
                   min_overlap=0.15, max_overlap=0.85):
    """(keep, reasons, stats) for one rewritten chunk.

    ``overlap`` is the share of the source's CJK character bigrams found
    in the output: below ``min_overlap`` the rewrite is about something
    else (or hallucinated), above ``max_overlap`` it is a near copy that
    adds no diversity. Numbers are compared as digit strings."""
    reasons = []
    if finish_reason != "stop":
        reasons.append(f"finish_{finish_reason}")
    lo, hi = STYLES[style]["ratio"]
    ratio = len(output) / max(1, len(source))
    if not lo <= ratio <= hi:
        reasons.append("length_ratio")
    prof = zh_profile(output)
    simp = prof["simp_only"] / prof["cjk"] if prof["cjk"] else 0.0
    if simp > max_simp_share:
        reasons.append("simplified")
    nums = set(_NUM.findall(source))
    kept = set(_NUM.findall(output))
    recall = len(nums & kept) / len(nums) if nums else 1.0
    if recall < min_number_recall:
        reasons.append("numbers_lost")
    src_bg = _bigrams(source)
    overlap = len(src_bg & _bigrams(output)) / len(src_bg) if src_bg else 0.0
    if src_bg and overlap < min_overlap:
        reasons.append("off_topic")
    if overlap > max_overlap:
        reasons.append("near_copy")
    stats = {"ratio": round(ratio, 3), "simp_share": round(simp, 4),
             "number_recall": round(recall, 3), "overlap": round(overlap, 3)}
    return not reasons, reasons, stats


def assemble(records, docs, teacher=None, **check_kw):
    """Join checked chunk rewrites into documents.

    ``records``: generator output (``distill_gen.Generator.run``) for
    prompts from ``rephrase_prompts``; ``docs``: ``{doc_id: source
    record}`` (for licence, attribution and source name). Returns
    ``(rewritten_docs, report)``; a (document, style) is dropped when a
    chunk is missing or fails ``fidelity_check``. With several samples
    per prompt the first passing one is used."""
    groups, report = {}, {"chunks": 0, "chunks_failed": 0, "reasons": {},
                          "docs": 0, "docs_dropped": 0}
    for rec in records:
        m = rec.get("meta") or {}
        if "doc_id" not in m:
            continue
        report["chunks"] += 1
        text = clean_output(rec.get("content"))
        ok, reasons, _ = fidelity_check(m["source_text"], text, m["style"],
                                        finish_reason=rec.get("finish_reason"),
                                        **check_kw)
        if not ok:
            report["chunks_failed"] += 1
            for r in reasons:
                report["reasons"][r] = report["reasons"].get(r, 0) + 1
            continue
        g = groups.setdefault((m["doc_id"], m["style"]),
                              {"n": m["n_chunks"], "parts": {}, "teacher": rec.get("teacher")})
        g["parts"].setdefault(m["chunk"], text)
    out = []
    for (doc_id, style), g in sorted(groups.items()):
        src = docs.get(doc_id)
        if src is None or len(g["parts"]) != g["n"]:
            report["docs_dropped"] += 1
            continue
        t = teacher or (g["teacher"] or {}).get("repo", "unknown teacher")
        attribution = src.get("attribution", "")
        out.append({"id": f"{doc_id}~{style}",
                    "source": f"{src.get('source', 'unknown')}+rephrase",
                    "text": "\n\n".join(g["parts"][i] for i in range(g["n"])),
                    "license": src.get("license"),
                    "attribution": (attribution + "；" if attribution else "") +
                    f"由 {t} 改寫（{style}）",
                    "rephrase": {"of": doc_id, "style": style, "teacher": t}})
    report["docs"] = len(out)
    return out, report


def plan_tokens(source_tokens, rewrites, epochs_raw=1):
    """Tokens a CPT run sees from one corpus: the raw text ``epochs_raw``
    times plus ``rewrites`` rephrased copies (each about the source's
    size). The K2 recipe is rewrites x 1 epoch instead of more epochs of
    the raw text."""
    return int(source_tokens * (epochs_raw + rewrites))
