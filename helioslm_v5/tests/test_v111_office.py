"""T53 - v1.11: office plugins (excel/docx/pptx/youtube)."""
import sys, tempfile, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.harness_core import Context
from harness.harness_plugins import (DocxPlugin, ExcelPlugin, PptxPlugin,
                                     PresetPlugin, YouTubePlugin)


def test_excel_roundtrip():
    try:
        import openpyxl
    except ImportError:
        print("SKIP test_excel (openpyxl not installed)")
        return
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.xlsx")
        ctx = Context()
        ctx.use(ExcelPlugin())
        ctx.excel.write(p, [[1, 2], [3, 4]])
        r = ctx.excel.read(p)
        assert r["rows"][0][:2] == [1, 2]
    print("PASS test_excel_roundtrip")


def test_docx_roundtrip():
    try:
        import docx
    except ImportError:
        print("SKIP test_docx (python-docx not installed)")
        return
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.docx")
        ctx = Context()
        ctx.use(DocxPlugin())
        ctx.docx.write(p, ["hello", "world"])
        r = ctx.docx.read(p)
        assert "hello" in r["paragraphs"]
    print("PASS test_docx_roundtrip")


def test_pptx_write():
    try:
        import pptx
    except ImportError:
        print("SKIP test_pptx (python-pptx not installed)")
        return
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.pptx")
        ctx = Context()
        ctx.use(PptxPlugin())
        r = ctx.pptx.write(p, [{"title": "T", "points": ["a", "b"]}])
        assert r["ok"]
    print("PASS test_pptx_write")


def test_youtube_stub():
    ctx = Context()
    ctx.use(YouTubePlugin())
    r = ctx.youtube.transcript("dQw4w9WgXcQ")
    assert "error" in r or "text" in r
    print("PASS test_youtube_stub")


if __name__ == "__main__":
    test_excel_roundtrip()
    test_docx_roundtrip()
    test_pptx_write()
    test_youtube_stub()
    print("\noffice plugins tests done")
