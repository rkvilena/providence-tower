"""Import-boundary tests: runtime modules must not load the local model stack.

Run from the project root:
    python -m pytest test/unit -q
"""

from __future__ import annotations

import subprocess
import sys

HEAVY_MODULES = ("torch", "sentence_transformers")


def test_vector_store_and_rag_imports_stay_model_free():
    code = """
import sys

import core.embedding
import core.vector_store
import core.rag.fetcher
from core.embedding.schema import ChunkDocument

heavy = [name for name in ("torch", "sentence_transformers") if name in sys.modules]
if heavy:
    raise SystemExit("unexpected heavy imports: " + ", ".join(heavy))

assert ChunkDocument(chunk_id="c", page_id=1, page_title="p", section="s",
                     subsection="ss", source_file="f", text="t").chunk_id == "c"
print("ok")
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_chunk_document_is_shared_with_embedding_service():
    from core.embedding.embedding_service import ChunkDocument as ServiceChunkDocument
    from core.embedding.schema import ChunkDocument

    assert ServiceChunkDocument is ChunkDocument
