from __future__ import annotations

"""
Algorithm (Seeding Colab-exported embeddings into Redis):
1) Read export_dir/manifest.json to determine vector_dim and the ordered list of part_*.jsonl files.
2) Choose Redis index parameters (index_name, key_prefix, distance_metric) from:
   - CLI overrides (highest priority), then
   - manifest fields if present, otherwise defaults used in the codebase.
3) Connect to Redis through the project's RedisVectorStore.
4) Ensure the vector index exists with the expected vector_dim.
5) For each part file:
   - Stream rows (one JSON per line).
   - Validate the embedding is a list of length vector_dim.
   - Write each row into Redis HASH at key "{key_prefix}{chunk_id}" with metadata + embedding bytes.
   - Flush writes in batches for memory/latency stability.

How to run:
python core/embedding/seed_embeddings.py
python core/embedding/seed_embeddings.py --export-dir e:\\MyProject\\providencetower-v2\\embed_chunk --index-name rag_chunks_idx --key-prefix rag:chunk: --distance-metric COSINE
python core/embedding/seed_embeddings.py --resume-from 41
"""

import argparse
import json
import sys
from pathlib import Path

# Allow direct script execution: `python core/embedding/seed_embeddings.py`
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import numpy as np

from core.embedding.embedding_service import ChunkDocument
from core.vector_store_factory import create_vector_store


def seed(
    export_dir: Path,
    *,
    index_name: str | None = None,
    write_batch_size: int = 1000,
    resume_from: int | None = None,
) -> None:
    manifest_path = export_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {export_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    resolved_index_name = str(
        index_name or manifest.get("index_name") or "rag_chunks_idx"
    )
    vector_dim = int(manifest.get("vector_dim") or 0)
    if vector_dim < 1:
        raise ValueError("Invalid vector_dim in manifest")

    parts = manifest.get("parts", [])
    if not isinstance(parts, list) or not parts:
        raise ValueError("No parts found in manifest")

    store = create_vector_store(index_name=resolved_index_name)

    store.ensure_index(vector_dim)

    total_skipped = 0
    total_written = 0
    for part in parts:
        part_number = part.get("part")
        part_file_name = part.get("file") or part.get("metadata_file")
        if not part_file_name:
            raise ValueError(f"Part entry missing file field: {part}")

        part_path = export_dir / str(part_file_name)
        if not part_path.exists():
            raise FileNotFoundError(f"Part file not found: {part_path}")

        # --resume-from: skip parts whose part number is less than resume_from
        if (
            resume_from is not None
            and part_number is not None
            and part_number < resume_from
        ):
            part_rows = part.get("rows", 0) or 0
            if part_rows == 0:
                # Count rows manually if manifest doesn't have row count
                with part_path.open("r", encoding="utf-8") as f:
                    part_rows = sum(1 for line in f if line.strip())
            total_skipped += part_rows
            print(
                f"Skipped {part_file_name}: {part_rows} rows (cumulative skipped={total_skipped})"
            )
            continue

        # Build items: convert raw JSONL rows into Upsertable format.
        # LocalRedisVectorStore uses flat HASH pipeline.
        # UpstashVectorStore uses nested metadata via its own upsert.
        # We use the store's upsert_documents but first convert rows to ChunkDocuments.

        rows_in_part = 0
        batch_docs: list[ChunkDocument] = []
        batch_vecs: list[np.ndarray] = []
        with part_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)

                embedding = row.get("embedding")
                if not isinstance(embedding, list):
                    raise ValueError(f"Row missing embedding list in file {part_path}")
                if len(embedding) != vector_dim:
                    raise ValueError(
                        f"Embedding dim mismatch for chunk_id={row.get('chunk_id')}: got={len(embedding)} expected={vector_dim}"
                    )

                batch_docs.append(
                    ChunkDocument(
                        chunk_id=str(row["chunk_id"]),
                        page_id=int(row.get("page_id", 0)),
                        page_title=str(row.get("page_title", "")),
                        section=str(row.get("section", "")),
                        subsection=str(row.get("subsection", "")),
                        source_file=str(row.get("source_file", "")),
                        text=str(row.get("text", "")),
                    )
                )
                batch_vecs.append(np.asarray(embedding, dtype=np.float32))
                rows_in_part += 1

                if len(batch_docs) >= write_batch_size:
                    store.upsert_documents(batch_docs, np.array(batch_vecs))
                    batch_docs.clear()
                    batch_vecs.clear()

        if batch_docs:
            store.upsert_documents(batch_docs, np.array(batch_vecs))

        total_written += rows_in_part
        print(
            f"Seeded {part_file_name}: {rows_in_part} rows (cumulative={total_written})"
        )

    print(f"Done. Seeded {total_written} rows into index={resolved_index_name}")
    if total_skipped > 0:
        print(f"  (skipped {total_skipped} rows via --resume-from={resume_from})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seed vector store from Colab JSON embedding export"
    )
    parser.add_argument(
        "--export-dir",
        default="e:\\MyProject\\providencetower-v2\\embed_chunk",
        help="Directory containing manifest.json and part_*.jsonl",
    )
    parser.add_argument(
        "--index-name",
        default="rag_chunks_idx",
        help="Override index name (otherwise manifest/default)",
    )
    parser.add_argument(
        "--write-batch-size",
        type=int,
        default=1000,
        help="Vector store write batch size",
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        default=None,
        help="Part number to resume from (inclusive). "
        "All parts with a lower part number are skipped. "
        "E.g. --resume-from 41 starts at part_00041.jsonl.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seed(
        export_dir=Path(args.export_dir),
        index_name=args.index_name,
        write_batch_size=args.write_batch_size,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
