"""Mini SWE task set (roadmap Phase AC, AC1).

SWE-bench-style evaluation in miniature, on top of the terminal sandbox
protocol of ``term_bench`` (same reply format, same ``DONE confidence=``
verbalized confidence, same resource-limited ``bash -c`` per command).

Each task is a small Python repository with a bug and an issue text.
The agent reads files, edits code and may run the visible tests. The
verifier then checks the FINAL repository:

1. it copies the sandbox to a fresh directory, so the agent's state is
   never mutated by grading;
2. it RESTORES the visible test files from the task definition, so an
   agent cannot pass by weakening a test;
3. it writes the HIDDEN test (defined only in this module, never inside
   the sandbox — the M1 self-grading guard) and runs visible + hidden
   tests with plain ``python3``; the task passes iff all exit 0.

Every task carries three reference gates, pinned by T97:
- ``fix``: the reference patch passes;
- the untouched repository fails (the bug is real);
- ``trap``: a plausible patch that passes the VISIBLE tests but not the
  hidden ones fails (the hidden tests carry the issue's real intent).

Patches are lists of ``(path, old, new)`` exact replacements, applied
through ``patch_command`` so references and traps run through the same
shell path as an agent's commands.
"""
import json
import shutil
import tempfile
from pathlib import Path

from evastar.eval import term_bench as tb

HIDDEN_NAME = "_hidden_test.py"

_PRELUDE = "import sys\nsys.path.insert(0, '.')\n"


class SweBenchError(ValueError):
    """Raised for malformed tasks. Never swallowed."""


def patch_command(edits):
    """Shell command applying exact ``(path, old, new)`` replacements; it
    exits non-zero (and changes nothing further) if an ``old`` is absent.

    The payload travels base64-encoded: patches carry regex backslashes
    (swe/13-parse-duration's ``\\d+``), and on Windows the command line
    passes through multiple re-serialization layers (CreateProcess
    quoting, the Git Bash CRT parse, and python3 launcher shims that
    re-exec another interpreter); a raw JSON argv lost one backslash
    level there and died with ``JSONDecodeError: Invalid \\escape``.
    Base64's alphabet survives every one of those layers. Files are read
    and written with ``newline=''`` so LF task fixtures stay LF.
    """
    import base64
    payload = base64.b64encode(json.dumps(edits).encode("utf-8")).decode()
    script = ("import base64,json,sys\n"
              "for p, o, n in json.loads(base64.b64decode(sys.argv[1])):\n"
              "    s = open(p, newline='').read()\n"
              "    if o not in s: sys.exit('not found in ' + p)\n"
              "    open(p, 'w', newline='').write(s.replace(o, n, 1))\n")
    return "python3 -c " + _sq(script) + " " + _sq(payload)


def _sq(s):
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _test(body):
    return _PRELUDE + body.strip() + "\nprint('ok')\n"


# -------------------------------------------------------------------- tasks
# Each task: id, issue, files (repo incl. visible tests), visible (test
# paths), hidden (test source), fix / trap (edit lists).

_T = []


def _task(tid, issue, files, visible, hidden, fix, trap):
    _T.append({"id": f"swe/{len(_T):02d}-{tid}", "issue": issue,
               "files": files, "visible": visible, "hidden": _test(hidden),
               "fix": fix, "trap": trap})


_task("slugify",
      "slugify('  Hello,  World!  ') returns '-hello-world-' but should "
      "return 'hello-world': runs of separators must collapse to one dash "
      "and there must be no leading or trailing dash.",
      {"textutil/__init__.py": "",
       "textutil/slug.py":
           "import re\n\n\ndef slugify(text):\n"
           "    return re.sub(r'[^a-z0-9]', '-', text.lower()).replace('--', '-')\n",
       "tests/test_slug.py": _test(
           "from textutil.slug import slugify\n"
           "assert slugify('abc') == 'abc'\n"
           "assert slugify('A B') == 'a-b'\n")},
      ["tests/test_slug.py"],
      "from textutil.slug import slugify\n"
      "assert slugify('  Hello,  World!  ') == 'hello-world'\n"
      "assert slugify('a---b') == 'a-b'\n"
      "assert slugify('--x--') == 'x'\n",
      [("textutil/slug.py",
        "re.sub(r'[^a-z0-9]', '-', text.lower()).replace('--', '-')",
        "re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')")],
      [("textutil/slug.py", ".replace('--', '-')",
        ".replace('--', '-').strip('-')")])

_task("paginate",
      "pages(total_items, per_page) undercounts: 25 items at 10 per page "
      "should be 3 pages, it says 2. 0 items is 0 pages.",
      {"pager.py": "def pages(total_items, per_page):\n"
                   "    return total_items // per_page\n",
       "tests/test_pager.py": _test(
           "from pager import pages\nassert pages(20, 10) == 2\n")},
      ["tests/test_pager.py"],
      "from pager import pages\nassert pages(25, 10) == 3\n"
      "assert pages(0, 10) == 0\nassert pages(30, 10) == 3\n"
      "assert pages(1, 10) == 1\n",
      [("pager.py", "return total_items // per_page",
        "return -(-total_items // per_page)")],
      [("pager.py", "return total_items // per_page",
        "return round(total_items / per_page)")])

_task("config-zero",
      "get_int(cfg, 'retries', 3) returns 3 when the config says "
      "retries=0. An explicit 0 must be kept; the default applies only "
      "when the key is missing.",
      {"app/__init__.py": "",
       "app/config.py": "def get_int(cfg, key, default):\n"
                        "    return int(cfg.get(key) or default)\n",
       "app/client.py": "from app.config import get_int\n\n\n"
                        "def retries(cfg):\n"
                        "    return get_int(cfg, 'retries', 3)\n",
       "tests/test_client.py": _test(
           "from app.client import retries\n"
           "assert retries({}) == 3\nassert retries({'retries': '5'}) == 5\n")},
      ["tests/test_client.py"],
      "from app.client import retries\nfrom app.config import get_int\n"
      "assert retries({'retries': '0'}) == 0\n"
      "assert retries({'retries': 0}) == 0\n"
      "assert get_int({'t': 0}, 't', 9) == 0\n"
      "assert get_int({'t': '7'}, 't', 9) == 7\n"
      "assert retries({}) == 3\n",
      [("app/config.py", "return int(cfg.get(key) or default)",
        "value = cfg.get(key)\n    return default if value is None else int(value)")],
      [("app/client.py", "return get_int(cfg, 'retries', 3)",
        "return int(cfg['retries']) if 'retries' in cfg else 3")])

_task("lru-refresh",
      "LRUCache evicts the wrong key: after cache.get('a'), 'a' is the most "
      "recently used and must survive the next eviction.",
      {"lru.py": "from collections import OrderedDict\n\n\n"
                 "class LRUCache:\n"
                 "    def __init__(self, capacity):\n"
                 "        self.capacity = capacity\n"
                 "        self.data = OrderedDict()\n\n"
                 "    def get(self, key):\n"
                 "        return self.data.get(key)\n\n"
                 "    def put(self, key, value):\n"
                 "        self.data[key] = value\n"
                 "        self.data.move_to_end(key)\n"
                 "        if len(self.data) > self.capacity:\n"
                 "            self.data.popitem(last=False)\n",
       "tests/test_lru.py": _test(
           "from lru import LRUCache\nc = LRUCache(2)\nc.put('a', 1)\n"
           "assert c.get('a') == 1\nassert c.get('z') is None\n")},
      ["tests/test_lru.py"],
      "from lru import LRUCache\nc = LRUCache(2)\nc.put('a', 1); c.put('b', 2)\n"
      "c.get('a'); c.put('c', 3)\n"
      "assert c.get('a') == 1 and c.get('b') is None and c.get('c') == 3\n"
      "d = LRUCache(2)\nd.put('a', 1); d.put('b', 2)\n"
      "assert d.get('missing') is None\nd.put('c', 3)\n"
      "assert d.get('a') is None\n",
      [("lru.py", "        return self.data.get(key)\n",
        "        if key not in self.data:\n            return None\n"
        "        self.data.move_to_end(key)\n        return self.data[key]\n")],
      [("lru.py", "            self.data.popitem(last=False)",
        "            self.data.popitem(last=True)")])

_task("date-range",
      "days_between(start, end) should list every date from start to end "
      "INCLUSIVE; the end date is missing.",
      {"cal.py": "from datetime import timedelta\n\n\n"
                 "def days_between(start, end):\n"
                 "    out = []\n"
                 "    d = start\n"
                 "    while d < end:\n"
                 "        out.append(d)\n"
                 "        d += timedelta(days=1)\n"
                 "    return out\n",
       "tests/test_cal.py": _test(
           "from datetime import date\nfrom cal import days_between\n"
           "assert days_between(date(2026, 1, 2), date(2026, 1, 1)) == []\n")},
      ["tests/test_cal.py"],
      "from datetime import date\nfrom cal import days_between\n"
      "r = days_between(date(2026, 1, 30), date(2026, 2, 2))\n"
      "assert [d.day for d in r] == [30, 31, 1, 2]\n"
      "assert days_between(date(2026, 1, 1), date(2026, 1, 1)) == "
      "[date(2026, 1, 1)]\n"
      "assert days_between(date(2026, 1, 2), date(2026, 1, 1)) == []\n",
      [("cal.py", "while d < end:", "while d <= end:")],
      [("cal.py", "    return out\n",
        "    if out:\n        out.append(end)\n    return out\n")])

_task("csv-quotes",
      "load_rows breaks on quoted fields: the line '1,\"Smith, Ann\",42' "
      "must give ['1', 'Smith, Ann', '42'].",
      {"data/__init__.py": "",
       "data/loader.py": "def load_rows(text):\n"
                         "    return [line.split(',') for line in text.splitlines() if line]\n",
       "tests/test_loader.py": _test(
           "from data.loader import load_rows\n"
           "assert load_rows('a,b\\n1,2\\n') == [['a', 'b'], ['1', '2']]\n")},
      ["tests/test_loader.py"],
      "from data.loader import load_rows\n"
      "assert load_rows('1,\"Smith, Ann\",42\\n') == [['1', 'Smith, Ann', '42']]\n"
      "assert load_rows('x,\"a \"\"q\"\" b\"\\n') == [['x', 'a \"q\" b']]\n"
      "assert load_rows('a,b\\n\\n1,2\\n') == [['a', 'b'], ['1', '2']]\n",
      [("data/loader.py",
        "    return [line.split(',') for line in text.splitlines() if line]\n",
        "    import csv\n"
        "    return [r for r in csv.reader(text.splitlines()) if r]\n")],
      [("data/loader.py", "line.split(',')",
        "[f.strip('\"') for f in line.split(',')]")])

_task("median-even",
      "median([1, 2, 3, 4]) returns 3; for an even count it must be the "
      "mean of the two middle values (2.5). The input must not be mutated.",
      {"stats.py": "def median(xs):\n"
                   "    xs.sort()\n"
                   "    return xs[len(xs) // 2]\n",
       "tests/test_stats.py": _test(
           "from stats import median\nassert median([3, 1, 2]) == 2\n")},
      ["tests/test_stats.py"],
      "from stats import median\nassert median([1, 2, 3, 4]) == 2.5\n"
      "v = [4, 1, 3, 2]\nassert median(v) == 2.5 and v == [4, 1, 3, 2]\n"
      "assert median([5]) == 5\n",
      [("stats.py", "    xs.sort()\n    return xs[len(xs) // 2]\n",
        "    s = sorted(xs)\n    n = len(s)\n"
        "    if n % 2:\n        return s[n // 2]\n"
        "    return (s[n // 2 - 1] + s[n // 2]) / 2\n")],
      [("stats.py", "    return xs[len(xs) // 2]\n",
        "    n = len(xs)\n    if n % 2:\n        return xs[n // 2]\n"
        "    return (xs[n // 2 - 1] + xs[n // 2]) / 2\n")])

_task("retry-count",
      "@retry(times=3) should make up to 3 attempts in total and re-raise "
      "the last error; it currently makes 4 attempts.",
      {"net/__init__.py": "",
       "net/retry.py": "import functools\n\n\n"
                       "def retry(times):\n"
                       "    def deco(fn):\n"
                       "        @functools.wraps(fn)\n"
                       "        def wrapper(*a, **kw):\n"
                       "            for attempt in range(times + 1):\n"
                       "                try:\n"
                       "                    return fn(*a, **kw)\n"
                       "                except Exception:\n"
                       "                    if attempt == times:\n"
                       "                        raise\n"
                       "        return wrapper\n"
                       "    return deco\n",
       "tests/test_retry.py": _test(
           "from net.retry import retry\n"
           "@retry(times=3)\ndef ok():\n    return 7\nassert ok() == 7\n")},
      ["tests/test_retry.py"],
      "from net.retry import retry\ncalls = []\n"
      "@retry(times=3)\ndef bad():\n    calls.append(1)\n"
      "    raise KeyError('x')\n"
      "try:\n    bad()\n    raise AssertionError('no raise')\n"
      "except KeyError:\n    pass\nassert len(calls) == 3, calls\n"
      "n = []\n@retry(times=1)\ndef once():\n    n.append(1)\n"
      "    raise ValueError\n"
      "try:\n    once()\nexcept ValueError:\n    pass\nassert len(n) == 1\n",
      [("net/retry.py", "for attempt in range(times + 1):",
        "for attempt in range(times):"),
       ("net/retry.py", "if attempt == times:", "if attempt == times - 1:")],
      [("net/retry.py", "for attempt in range(times + 1):",
        "for attempt in range(times):")])

_task("deep-merge",
      "merge(base, override) must merge nested dicts recursively: "
      "merge({'db': {'host': 'a', 'port': 1}}, {'db': {'port': 2}}) should "
      "keep host. Neither input may be mutated.",
      {"settings.py": "def merge(base, override):\n"
                      "    out = dict(base)\n"
                      "    out.update(override)\n"
                      "    return out\n",
       "tests/test_settings.py": _test(
           "from settings import merge\n"
           "assert merge({'a': 1}, {'b': 2}) == {'a': 1, 'b': 2}\n")},
      ["tests/test_settings.py"],
      "from settings import merge\nb = {'db': {'host': 'a', 'port': 1}, 'x': 1}\n"
      "o = {'db': {'port': 2}}\nr = merge(b, o)\n"
      "assert r == {'db': {'host': 'a', 'port': 2}, 'x': 1}, r\n"
      "assert b == {'db': {'host': 'a', 'port': 1}, 'x': 1}\n"
      "assert merge({'a': {'b': {'c': 1}}}, {'a': {'b': {'d': 2}}}) == "
      "{'a': {'b': {'c': 1, 'd': 2}}}\n"
      "assert merge({'a': {'b': 1}}, {'a': 5}) == {'a': 5}\n",
      [("settings.py", "    out = dict(base)\n    out.update(override)\n",
        "    out = dict(base)\n    for k, v in override.items():\n"
        "        if isinstance(v, dict) and isinstance(out.get(k), dict):\n"
        "            out[k] = merge(out[k], v)\n        else:\n"
        "            out[k] = v\n")],
      [("settings.py", "    out = dict(base)\n    out.update(override)\n",
        "    out = base\n    for k, v in override.items():\n"
        "        if isinstance(v, dict) and isinstance(out.get(k), dict):\n"
        "            out[k].update(v)\n        else:\n"
        "            out[k] = v\n")])

_task("version-compare",
      "is_newer('1.10.0', '1.9.3') returns False; versions must be compared "
      "numerically per component, and '1.2' equals '1.2.0'.",
      {"release/__init__.py": "",
       "release/version.py": "def is_newer(a, b):\n    return a > b\n",
       "tests/test_version.py": _test(
           "from release.version import is_newer\n"
           "assert is_newer('2.0.0', '1.0.0')\n"
           "assert not is_newer('1.0.0', '1.0.0')\n")},
      ["tests/test_version.py"],
      "from release.version import is_newer\n"
      "assert is_newer('1.10.0', '1.9.3')\nassert not is_newer('1.9.3', '1.10.0')\n"
      "assert not is_newer('1.2', '1.2.0') and not is_newer('1.2.0', '1.2')\n"
      "assert is_newer('1.2.1', '1.2')\n",
      [("release/version.py", "    return a > b\n",
        "    def parts(v):\n        p = [int(x) for x in v.split('.')]\n"
        "        while p and p[-1] == 0:\n            p.pop()\n        return p\n"
        "    return parts(a) > parts(b)\n")],
      [("release/version.py", "    return a > b\n",
        "    return [int(x) for x in a.split('.')] > "
        "[int(x) for x in b.split('.')]\n")])

_task("word-count",
      "top_words counts 'The' and 'the.' as different words. Words must be "
      "case-insensitive with surrounding punctuation stripped, and ties "
      "broken alphabetically.",
      {"wc.py": "from collections import Counter\n\n\n"
                "def top_words(text, n):\n"
                "    return Counter(text.split()).most_common(n)\n",
       "tests/test_wc.py": _test(
           "from wc import top_words\n"
           "assert top_words('a a b', 1) == [('a', 2)]\n")},
      ["tests/test_wc.py"],
      "from wc import top_words\n"
      "assert top_words('The cat. the CAT, the dog!', 2) == "
      "[('the', 3), ('cat', 2)]\n"
      "assert top_words('b a', 2) == [('a', 1), ('b', 1)]\n"
      "assert top_words('... !!', 3) == []\n",
      [("wc.py", "    return Counter(text.split()).most_common(n)\n",
        "    import string\n"
        "    words = [w.strip(string.punctuation).lower() for w in text.split()]\n"
        "    c = Counter(w for w in words if w)\n"
        "    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:n]\n")],
      [("wc.py", "Counter(text.split())", "Counter(text.lower().split())")])

_task("mutable-default",
      "add_tag('x') leaks tags between calls: a second add_tag('y') "
      "returns ['x', 'y'] instead of ['y'].",
      {"tags.py": "def add_tag(tag, tags=[]):\n"
                  "    tags.append(tag)\n"
                  "    return tags\n",
       "tests/test_tags.py": _test(
           "from tags import add_tag\nassert add_tag('a', ['b']) == ['b', 'a']\n")},
      ["tests/test_tags.py"],
      "from tags import add_tag\nassert add_tag('x') == ['x']\n"
      "assert add_tag('y') == ['y']\nmine = ['k']\nadd_tag('z', mine)\n"
      "assert mine == ['k', 'z']\n",
      [("tags.py", "def add_tag(tag, tags=[]):\n",
        "def add_tag(tag, tags=None):\n    if tags is None:\n        tags = []\n")],
      [("tags.py", "    tags.append(tag)\n    return tags\n",
        "    return tags + [tag]\n")])

_task("binary-search",
      "find(xs, target) loops forever when the target is absent; it must "
      "return the index of target in the sorted list, or -1.",
      {"search.py": "def find(xs, target):\n"
                    "    lo, hi = 0, len(xs) - 1\n"
                    "    while lo <= hi:\n"
                    "        mid = (lo + hi) // 2\n"
                    "        if xs[mid] == target:\n"
                    "            return mid\n"
                    "        if xs[mid] < target:\n"
                    "            lo = mid\n"
                    "        else:\n"
                    "            hi = mid\n"
                    "    return -1\n",
       "tests/test_search.py": _test(
           "from search import find\nassert find([1, 3, 5], 3) == 1\n")},
      ["tests/test_search.py"],
      "from search import find\nxs = [1, 3, 5, 7, 9, 11]\n"
      "for i, x in enumerate(xs):\n    assert find(xs, x) == i\n"
      "for t in (0, 4, 12):\n    assert find(xs, t) == -1\n"
      "assert find([], 1) == -1\n",
      [("search.py", "            lo = mid\n", "            lo = mid + 1\n"),
       ("search.py", "            hi = mid\n", "            hi = mid - 1\n")],
      [("search.py", "            lo = mid\n", "            lo = mid + 1\n")])

_task("parse-duration",
      "parse_duration('1h30m') returns 3600; every unit in the string must "
      "count (h, m, s), e.g. '1h30m' -> 5400 and '45s' -> 45.",
      {"timefmt/__init__.py": "",
       "timefmt/duration.py": "import re\n\n_UNITS = {'h': 3600, 'm': 60, 's': 1}\n\n\n"
                              "def parse_duration(text):\n"
                              "    m = re.match(r'(\\d+)([hms])', text)\n"
                              "    return int(m.group(1)) * _UNITS[m.group(2)]\n",
       "tests/test_duration.py": _test(
           "from timefmt.duration import parse_duration\n"
           "assert parse_duration('2h') == 7200\n")},
      ["tests/test_duration.py"],
      "from timefmt.duration import parse_duration\n"
      "assert parse_duration('1h30m') == 5400\n"
      "assert parse_duration('45s') == 45\n"
      "assert parse_duration('1h2m3s') == 3723\n",
      [("timefmt/duration.py",
        "    m = re.match(r'(\\d+)([hms])', text)\n"
        "    return int(m.group(1)) * _UNITS[m.group(2)]\n",
        "    return sum(int(n) * _UNITS[u]\n"
        "               for n, u in re.findall(r'(\\d+)([hms])', text))\n")],
      [("timefmt/duration.py", "re.match(", "re.search(")])

_task("overdraft",
      "Account.withdraw lets the balance go negative; withdrawing more than "
      "the balance must raise ValueError and leave the balance unchanged. "
      "Non-positive amounts must also raise ValueError.",
      {"bank/__init__.py": "",
       "bank/account.py": "class Account:\n"
                          "    def __init__(self, balance=0):\n"
                          "        self.balance = balance\n\n"
                          "    def withdraw(self, amount):\n"
                          "        self.balance -= amount\n"
                          "        return self.balance\n",
       "tests/test_account.py": _test(
           "from bank.account import Account\n"
           "assert Account(10).withdraw(4) == 6\n")},
      ["tests/test_account.py"],
      "from bank.account import Account\na = Account(10)\n"
      "for bad in (11, 0, -5):\n    try:\n        a.withdraw(bad)\n"
      "        raise AssertionError(bad)\n    except ValueError:\n        pass\n"
      "assert a.balance == 10\nassert a.withdraw(10) == 0\n",
      [("bank/account.py", "        self.balance -= amount\n",
        "        if amount <= 0 or amount > self.balance:\n"
        "            raise ValueError('invalid amount')\n"
        "        self.balance -= amount\n")],
      [("bank/account.py", "        self.balance -= amount\n",
        "        self.balance -= amount\n        if self.balance < 0:\n"
        "            raise ValueError('overdraft')\n")])

_task("chunks",
      "chunks([1, 2, 3, 4, 5], 2) drops the trailing 5; the last partial "
      "chunk must be kept.",
      {"seq.py": "def chunks(xs, size):\n"
                 "    return [xs[i:i + size] for i in range(0, len(xs) - size + 1, size)]\n",
       "tests/test_seq.py": _test(
           "from seq import chunks\nassert chunks([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]\n")},
      ["tests/test_seq.py"],
      "from seq import chunks\n"
      "assert chunks([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]\n"
      "assert chunks([1, 2], 5) == [[1, 2]]\nassert chunks([], 3) == []\n",
      [("seq.py", "range(0, len(xs) - size + 1, size)", "range(0, len(xs), size)")],
      [("seq.py", "range(0, len(xs) - size + 1, size)",
        "range(0, len(xs) - size + 2, size)")])

_task("roman",
      "to_roman(4) returns 'IIII'; subtractive forms are required "
      "(4 -> 'IV', 9 -> 'IX', 40 -> 'XL', 90 -> 'XC', 400 -> 'CD', 900 -> 'CM').",
      {"numerals.py": "_TABLE = [(1000, 'M'), (500, 'D'), (100, 'C'), (50, 'L'),\n"
                      "          (10, 'X'), (5, 'V'), (1, 'I')]\n\n\n"
                      "def to_roman(n):\n"
                      "    out = ''\n"
                      "    for value, sym in _TABLE:\n"
                      "        while n >= value:\n"
                      "            out += sym\n"
                      "            n -= value\n"
                      "    return out\n",
       "tests/test_numerals.py": _test(
           "from numerals import to_roman\nassert to_roman(3) == 'III'\n")},
      ["tests/test_numerals.py"],
      "from numerals import to_roman\n"
      "for n, r in [(4, 'IV'), (9, 'IX'), (14, 'XIV'), (40, 'XL'), (90, 'XC'),\n"
      "             (400, 'CD'), (1994, 'MCMXCIV'), (2026, 'MMXXVI')]:\n"
      "    assert to_roman(n) == r, (n, to_roman(n))\n",
      [("numerals.py",
        "_TABLE = [(1000, 'M'), (500, 'D'), (100, 'C'), (50, 'L'),\n"
        "          (10, 'X'), (5, 'V'), (1, 'I')]\n",
        "_TABLE = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),\n"
        "          (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),\n"
        "          (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]\n")],
      [("numerals.py", "(5, 'V'), (1, 'I')]", "(5, 'V'), (4, 'IV'), (1, 'I')]")])

_task("flatten",
      "flatten([1, [2, [3, [4]]]]) returns [1, 2, [3, [4]]]; it must "
      "flatten arbitrarily deep lists (tuples and strings stay as items).",
      {"nested.py": "def flatten(xs):\n"
                    "    out = []\n"
                    "    for x in xs:\n"
                    "        if isinstance(x, list):\n"
                    "            out.extend(x)\n"
                    "        else:\n"
                    "            out.append(x)\n"
                    "    return out\n",
       "tests/test_nested.py": _test(
           "from nested import flatten\nassert flatten([1, [2, 3]]) == [1, 2, 3]\n")},
      ["tests/test_nested.py"],
      "from nested import flatten\n"
      "assert flatten([1, [2, [3, [4]]]]) == [1, 2, 3, 4]\n"
      "assert flatten([[[]], 'ab', (1, 2)]) == ['ab', (1, 2)]\n",
      [("nested.py", "            out.extend(x)\n", "            out.extend(flatten(x))\n")],
      [("nested.py", "            out.extend(x)\n",
        "            out.extend(y for z in x for y in "
        "(z if isinstance(z, list) else [z]))\n")])

_task("leap-year",
      "is_leap(1900) returns True; century years are leap years only when "
      "divisible by 400 (2000 is, 1900 is not). The report generator "
      "(report.days_in_year) uses it.",
      {"calendar_util/__init__.py": "",
       "calendar_util/leap.py": "def is_leap(year):\n    return year % 4 == 0\n",
       "calendar_util/report.py": "from calendar_util.leap import is_leap\n\n\n"
                                  "def days_in_year(year):\n"
                                  "    return 366 if is_leap(year) else 365\n",
       "tests/test_report.py": _test(
           "from calendar_util.report import days_in_year\n"
           "assert days_in_year(2024) == 366\nassert days_in_year(2023) == 365\n")},
      ["tests/test_report.py"],
      "from calendar_util.report import days_in_year\n"
      "from calendar_util.leap import is_leap\n"
      "assert days_in_year(1900) == 365 and days_in_year(2000) == 366\n"
      "assert not is_leap(2100) and is_leap(2400) and is_leap(2024)\n",
      [("calendar_util/leap.py", "    return year % 4 == 0\n",
        "    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)\n")],
      [("calendar_util/report.py", "    return 366 if is_leap(year) else 365\n",
        "    if year == 1900:\n        return 365\n"
        "    return 366 if is_leap(year) else 365\n")])

_task("env-bool",
      "env_flag('DEBUG') treats DEBUG=false and DEBUG=0 as enabled. Only "
      "'1', 'true', 'yes', 'on' (any case, surrounding spaces ignored) "
      "enable a flag; anything else, or an unset variable, is False.",
      {"envutil.py": "import os\n\n\n"
                     "def env_flag(name, env=None):\n"
                     "    env = os.environ if env is None else env\n"
                     "    return bool(env.get(name))\n",
       "tests/test_envutil.py": _test(
           "from envutil import env_flag\nassert env_flag('X', {'X': '1'})\n"
           "assert not env_flag('X', {})\n")},
      ["tests/test_envutil.py"],
      "from envutil import env_flag\n"
      "for v in ('false', '0', 'no', 'off', '', 'banana'):\n"
      "    assert not env_flag('D', {'D': v}), v\n"
      "for v in ('1', 'TRUE', ' yes ', 'On'):\n"
      "    assert env_flag('D', {'D': v}), v\n",
      [("envutil.py", "    return bool(env.get(name))\n",
        "    return str(env.get(name, '')).strip().lower() in "
        "{'1', 'true', 'yes', 'on'}\n")],
      [("envutil.py", "    return bool(env.get(name))\n",
        "    return env.get(name, '') not in ('false', '0', '')\n")])


TASKS = _T


# ---------------------------------------------------------------- verifier

def _verify(task, root):
    with tempfile.TemporaryDirectory(prefix="swe_verify_") as tmp:
        work = Path(tmp) / "repo"
        shutil.copytree(root, work, symlinks=True)
        for rel in task["visible"]:
            p = work / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            # newline="\n": keep the restored tests byte-identical to the
            # task definition on every platform (see term_bench setup).
            p.write_text(task["files"][rel], encoding="utf-8", newline="\n")
        (work / HIDDEN_NAME).write_text(task["hidden"], encoding="utf-8",
                                        newline="\n")
        for rel in list(task["visible"]) + [HIDDEN_NAME]:
            code, out = tb.run_command(f"python3 {rel}", work)
            if code != 0:
                last = out.strip().splitlines()[-1:] or [""]
                tag = "hidden" if rel == HIDDEN_NAME else rel
                return False, f"{tag} failed: {last[0][:80]}"
    return True, "visible + hidden tests pass"


def _as_term_task(task):
    """Adapter onto term_bench's task schema (instruction/files/verify)."""
    listing = ", ".join(sorted(task["files"]))
    tests = " ".join(f"python3 {v}" for v in task["visible"])
    instruction = (f"You are in a Python repository ({listing}). Issue: "
                   f"{task['issue']} Fix the code so the issue is resolved. "
                   f"Visible tests: `{tests}`. Hidden tests will also be run; "
                   f"do not edit the tests.")
    return {"id": task["id"], "instruction": instruction,
            "files": task["files"],
            "solution": [patch_command(task["fix"])],
            "verify": lambda root, _t=task: _verify(_t, root)}


def load_tasks():
    ids = [t["id"] for t in TASKS]
    if len(set(ids)) != len(ids):
        raise SweBenchError("duplicate task id")
    for t in TASKS:
        for key in ("issue", "files", "visible", "hidden", "fix", "trap"):
            if not t.get(key):
                raise SweBenchError(f"{t['id']}: missing {key!r}")
        for rel in t["visible"]:
            if rel not in t["files"]:
                raise SweBenchError(f"{t['id']}: visible test {rel} not in files")
    return [_as_term_task(t) for t in TASKS]


def trap_commands():
    """task id -> the trap patch as a shell command (for reference gates)."""
    return {t["id"]: patch_command(t["trap"]) for t in TASKS}


def run_swe_bench(tasks, generate_fn, n_samples=1, max_steps=tb.MAX_STEPS):
    """Run episodes through term_bench's loop; rows are probe-schema rows
    with ``family='swe'`` so they score on the calibration board."""
    rows = tb.run_term_bench(tasks, generate_fn, n_samples, max_steps)
    for r in rows:
        r["family"] = "swe"
    return rows
