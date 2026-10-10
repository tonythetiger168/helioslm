"""T62 - v1.20: DNA sequence analysis plugin."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.harness_core import Context
from harness.harness_plugins import DNAPlugin, PresetPlugin


SEQ = "ATGGCGTACGTTAGCCGGATCGATCGATGCTAGCTAGCATCGATCGTACGATCGATCGATCG"


def test_dna_ops():
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    ctx.use(DNAPlugin())
    assert ctx.dna.validate(SEQ)["ok"]
    assert ctx.dna.gc_content(SEQ) > 40
    rc = ctx.dna.reverse_complement("ATGC")
    assert rc == "GCAT"
    assert ctx.dna.transcribe("ATGC") == "AUGC"
    print("PASS test_dna_ops")


def test_dna_translate():
    ctx = Context()
    ctx.use(DNAPlugin())
    # ATG = M, GCG = A, TAA = stop
    prot = ctx.dna.translate("ATGGCGTAA")
    assert prot == "MA"
    print("PASS test_dna_translate")


def test_dna_align():
    ctx = Context()
    ctx.use(DNAPlugin())
    r = ctx.dna.align("GATTACA", "GCATGCU")
    assert "a" in r and "score" in r
    print("PASS test_dna_align")


def test_dna_pcr_and_restriction():
    ctx = Context()
    ctx.use(DNAPlugin())
    p = ctx.dna.pcr_primers(SEQ)
    assert "forward" in p and "reverse" in p
    r = ctx.dna.restriction_sites("AAA" + "GAATTC" + "BBB" + "GAATTC", "EcoRI")
    # 0-based positions: GAATTC starts at index 3 and 12 (the old
    # expectation [3, 13] mixed 0-based and 1-based conventions)
    assert r["positions"] == [3, 12]
    print("PASS test_dna_pcr_and_restriction")


if __name__ == "__main__":
    test_dna_ops()
    test_dna_translate()
    test_dna_align()
    test_dna_pcr_and_restriction()
    print("\nDNA plugin tests done")
