import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from longctx import make_needle_corpus, probe_report


def test_longctx_probes():
    def model_fn(corpus, question):
        i = corpus.find("the magic number is ")
        needle = corpus[i + len("the magic number is "):].split()[0]
        return needle if i / max(1, len(corpus)) < 0.8 else "unknown"

    report = probe_report(model_fn, n_tokens=2000, trials=3)
    assert report["depth_0.1"] == 1.0
    assert report["depth_0.9"] == 0.0
    assert 0.0 < report["mean"] < 1.0
    c1, q1 = make_needle_corpus(random.Random(5), 500, "1234", 0.5)
    c2, q2 = make_needle_corpus(random.Random(5), 500, "1234", 0.5)
    assert (c1, q1) == (c2, q2)


if __name__ == "__main__":
    test_longctx_probes()
    print("PASS test_longctx_probes\n\n1/1 tests passed")
