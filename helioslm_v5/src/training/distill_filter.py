"""Filtering teacher answers before distillation (v5.112, D2).

Input: D1 records (``distill_gen``). Every record goes through the checks
below in order and is dropped at the first it fails; the report counts
drops per reason, so a filter that eats half the data shows up at once.

  finish        finish_reason must be "stop": "length" is a truncated
                answer (training on it teaches the student to stop
                mid-sentence), anything else is an error or a refusal;
  empty         no answer text, or a leaked ``<think>`` tag in it;
  length        reasoning / answer token caps. A small student cannot
                imitate a 30k-token chain, so the 0.5B config caps
                reasoning near 4k (reasoning tokens as ``token_stats``
                counts them);
  script        Traditional Chinese: the share of simplified-only
                characters among Han characters must stay under a
                threshold, separately for answer and reasoning. A fixed
                character list stands in for a converter (no opencc): it
                names simplified forms not used in Taiwanese writing
                (characters that are also valid or common variants in
                Taiwan, such as 台 and 着, are excluded);
  decontam      the prompt or the answer shares an n-gram with an
                evaluation item (``eval/decontam.EvalIndex``, 13-grams);
  verify        math: with several samples of a prompt, keep the ones
                whose final answer agrees with the majority (self-
                consistency; a prompt without a majority is dropped);
                code: every ```python block must parse;
  near_dup      a prompt whose word 8-grams are mostly (containment
                >= 0.8) already covered by kept prompts is dropped, so
                teacher-written prompts that repeat each other do not
                weight one question many times. Samples of the SAME
                prompt are not duplicates of each other.
"""
import ast
import re
from collections import Counter, defaultdict

from ...eval.decontam import EvalIndex, ngram_digests
from .distill_gen import reasoning_tokens

# Simplified forms that are not valid Traditional characters. Common ones
# only; the threshold is a rate, so recall does not need to be complete.
SIMPLIFIED_ONLY = set(
    "这们说时会对来个为学国发么过还没经应关问题开实现样当动进种东车长门见话让认"
    "气电书买卖写语读网页数据简体压历县乡兴华产务业丝两严丧举义乐习乱亚亲们仅从"
    "众优伟传伤体余佣侠侣侦俭债倾储儿兑兰关兴兹养兽冯冲决况冻净减凤凭击创则刚"
    "剂剑剧劝办务动劳势勋区医华协单卖卢卫却厂厅历压厌县参双变叙台叶号叹吓吗"
    "启员呜咏响哑唤啰团园围图圆圣场坏块坚坛垄垒执扩扫扬扰抚抢护报担拟拥择挡挤"
    "挥损换据掷摄摆摇摊撑敌数斋断无旧时显晒晓晕暂术机杀杂权条来杨极构枪柜标栏树"
    "样档桥梦检楼汇汉汤沟没沪泪泽洁洒浅测济浑浓涂润涨涩渊渐温湾湿满滚滞潜灭灯"
    "灵灾灿炉点炼烂烛烦烧热焕爱爷牵犹狭独猎献环现玛琐电画畅疗疮疯痒瘾皱盏盐监"
    "盘着睁矫矿码砖确礼祸禅离秃种积称稳穷窃窍竞笔笼筑筹签简粮紧纠红纪纤约级纯纲"
    "纳纵纷纸纹纺线练组细织终绍经结绕绘给络绝统继续维综绿缓编缘缩缴网罗罚罢职"
    "联聪肃肠肤肿胀胁脉脑脚脱脸腊舰艰节芜苍苏范茧荐药莱获蓝虑虚虫虽蚀蛮补衬袜"
    "袭装见观规视览觉触誉计订认讨让训议讯记讲讳许论设访证评识诈诉诊词译试诗诚"
    "话诞询该详语误说请诸读课谁调谅谈谊谋谓谢谣谦谨谱贝负贡财责贤败货质贩贪贫"
    "购贯贱贴贵贷贸费贺资赋赌赏赔赖赚赛赞赠赵赶趋跃践踪轨轩转轮软轰轻载较辅辆"
    "辈辉辑输辞边辽达迁过迈运还这进远违连迟适选递逻遗邮邻郑释针钉钓钟钢钥钱钻"
    "铁铃铜铭银铺链销锁锅锋错锦键锻镇镜长门闪闭问闯闲间闷闹闻阀阁阅阐队阳阴阵"
    "阶际陆陈险随隐难雾静韦韩页顶项顺须顾顿预领颇频颗题颜额风飞饭饮饰饱饼馆马"
    "驱驳驶驻驾验骂骑骗骤鱼鲁鲜鸟鸡鸣鸭鹅麦齐龄龙") - set(
    # rare-but-valid or common Taiwanese variants: never counted
    "台叶范么冲凭术余佣筑腊虫种着适体药")
_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_CODE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.S)
_BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
_NUMBER = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?")

DEFAULTS = {
    "max_reasoning_tokens": 4096,
    "max_answer_tokens": 4096,
    "max_simplified_answer": 0.02,
    "max_simplified_reasoning": 0.05,
    "near_dup_n": 8,
    "near_dup_containment": 0.8,
    "math_domains": ("math",),
    "code_domains": ("code",),
}
REASONS = ("finish", "empty", "length", "script", "decontam", "verify", "near_dup")


def simplified_rate(text):
    """Share of Han characters that are simplified-only (0 for no Han)."""
    han = _HAN.findall(text or "")
    if not han:
        return 0.0
    return sum(c in SIMPLIFIED_ONLY for c in han) / len(han)


def final_answer(text):
    """Last \\boxed{...}, else the last number; normalised for voting."""
    text = text or ""
    boxed = _BOXED.findall(text)
    if boxed:
        ans = boxed[-1]
    else:
        nums = _NUMBER.findall(text)
        if not nums:
            return None
        ans = nums[-1]
    ans = ans.replace(",", "").replace(" ", "").strip("$.")
    try:
        f = float(ans)
        return str(int(f)) if f == int(f) else repr(f)
    except ValueError:
        return ans or None


def code_parses(text):
    blocks = _CODE.findall(text or "")
    for b in blocks:
        try:
            ast.parse(b)
        except SyntaxError:
            return False
    return True


def prompt_text(rec):
    return "\n".join(m.get("content", "") for m in rec["messages"] if m["role"] == "user")


def _answer_tokens(rec):
    out = rec["usage"].get("output_tokens", 0)
    return out - reasoning_tokens(rec)[0]


class Filter:
    def __init__(self, eval_items=None, **cfg):
        unknown = set(cfg) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown filter options {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **cfg}
        self.index = EvalIndex(eval_items) if eval_items else None

    def _single(self, rec):
        c = self.cfg
        if rec.get("finish_reason") != "stop":
            return "finish"
        content = rec.get("content") or ""
        if not content.strip() or "<think>" in content or "</think>" in content:
            return "empty"
        if reasoning_tokens(rec)[0] > c["max_reasoning_tokens"] or \
                _answer_tokens(rec) > c["max_answer_tokens"]:
            return "length"
        if simplified_rate(content) > c["max_simplified_answer"] or \
                simplified_rate(rec.get("reasoning")) > c["max_simplified_reasoning"]:
            return "script"
        if self.index is not None and \
                (self.index.scan(prompt_text(rec)) or self.index.scan(content)):
            return "decontam"
        if rec.get("domain") in c["code_domains"] and not code_parses(content):
            return "verify"
        return None

    def run(self, records):
        """Returns (kept records, report)."""
        c = self.cfg
        drops = Counter()
        survivors = []
        for r in records:
            why = self._single(r)
            if why:
                drops[why] += 1
            else:
                survivors.append(r)

        # math self-consistency over the samples of one prompt
        by_prompt = defaultdict(list)
        for r in survivors:
            by_prompt[r["prompt_id"]].append(r)
        voted = []
        for pid, group in by_prompt.items():
            if group[0].get("domain") not in c["math_domains"] or len(group) < 2:
                voted.extend(group)
                continue
            answers = [final_answer(r["content"]) for r in group]
            top, cnt = Counter(a for a in answers if a is not None).most_common(1)[0] \
                if any(a is not None for a in answers) else (None, 0)
            if cnt * 2 <= len(group):
                drops["verify"] += len(group)
                continue
            for r, a in zip(group, answers):
                if a == top:
                    voted.append(r)
                else:
                    drops["verify"] += 1

        # near-duplicate prompts, first come first kept
        seen = set()
        kept_prompt = {}
        kept = []
        for r in voted:
            pid = r["prompt_id"]
            if pid not in kept_prompt:
                grams = ngram_digests(prompt_text(r), c["near_dup_n"], short_ok=True)
                cover = len(grams & seen) / len(grams) if grams else 0.0
                kept_prompt[pid] = cover < c["near_dup_containment"]
                if kept_prompt[pid]:
                    seen |= grams
            if kept_prompt[pid]:
                kept.append(r)
            else:
                drops["near_dup"] += 1

        n = len(records)
        report = {"input": n, "kept": len(kept),
                  "kept_rate": round(len(kept) / n, 4) if n else 0.0,
                  "dropped": {k: drops.get(k, 0) for k in REASONS},
                  "kept_prompts": len({r["prompt_id"] for r in kept}),
                  "config": {k: list(v) if isinstance(v, tuple) else v
                             for k, v in c.items()},
                  "eval_items": len(self.index.items) if self.index else 0}
        return kept, report


def load_eval_items(paths, fields=("question", "prompt", "text", "query")):
    """{id: text} from evaluation JSONL files, first matching field."""
    import json
    items = {}
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if not line.strip():
                    continue
                row = json.loads(line)
                text = next((row[f] for f in fields if isinstance(row.get(f), str)), None)
                if text:
                    items[f"{path}:{i}"] = text
    return items
