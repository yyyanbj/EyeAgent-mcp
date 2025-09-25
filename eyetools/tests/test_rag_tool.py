from pathlib import Path

from tools.rag.tool_impl import RAGQueryTool


def _write_corpus(tmpdir: Path) -> list[Path]:
    docs = []
    d = tmpdir / "corpus"
    d.mkdir(parents=True, exist_ok=True)
    docs.append(d / "doc1.txt")
    docs.append(d / "doc2.md")
    docs.append(d / "notes.rst")
    (d / "doc1.txt").write_text(
        """
        Retinal detachment is a serious condition. Early signs may include flashes of light and floaters.
        Treatment options vary. Prompt diagnosis is important.
        """.strip()
    )
    (d / "doc2.md").write_text(
        """
        # Eye Health Notes

        The macula and retina are vital for vision. Detachment can cause vision loss.
        Prevention and screening are topics of ongoing research.
        """.strip()
    )
    (d / "notes.rst").write_text(
        """
        Symptoms
        ========

        Floaters, flashes, curtain-like shadow over the visual field.
        """.strip()
    )
    return docs


def test_rag_local_basic(tmp_path: Path):
    _write_corpus(tmp_path)
    meta = {"id": "rag:query", "entry": "tool_impl:RAGQueryTool", "root_dir": str((Path(__file__).resolve().parents[1] / "tools" / "rag").resolve()), "model": {"lazy": True}}
    params = {
        "mode": "local",
        "top_k": 3,
        "corpus_dirs": [str(tmp_path / "corpus")],
        "chunk_size": 120,
        "chunk_overlap": 20,
        "maxpages": 5,
    }
    tool = RAGQueryTool(meta, params)
    tool.prepare()
    out = tool.predict({"inputs": {"query": "retinal detachment floaters", "top_k": 2}})
    assert isinstance(out, dict)
    assert out.get("source") == "rag"
    items = out.get("items") or []
    # Expect at least one relevant chunk
    assert len(items) >= 1
    # top_k respected
    assert len(items) <= 2
    # Validate fields
    for it in items:
        assert {"title", "text", "source", "score"}.issubset(it.keys())
        assert isinstance(it["title"], str)
        assert isinstance(it["text"], str)
        assert isinstance(it["source"], str)


def test_rag_local_empty_query(tmp_path: Path):
    _write_corpus(tmp_path)
    meta = {"id": "rag:query", "entry": "tool_impl:RAGQueryTool", "root_dir": str((Path(__file__).resolve().parents[1] / "tools" / "rag").resolve()), "model": {"lazy": True}}
    params = {
        "mode": "local",
        "corpus_dirs": [str(tmp_path / "corpus")],
    }
    tool = RAGQueryTool(meta, params)
    tool.prepare()
    out = tool.predict({"inputs": {"query": ""}})
    assert out.get("items") == []
    assert out.get("source") == "rag"
    assert "warning" in out
