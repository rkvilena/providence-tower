from __future__ import annotations

"""
Algorithm (Seeding Colab-exported embeddings into Redis):
1) Read export_dir/manifest.json to determine vector_dim and the ordered list of part_*.jsonl files.
2) Choose Redis index parameters (index_name, key_prefix, distance_metric) from:
   - CLI overrides (highest priority), then
   - manifest fields if present, otherwise defaults used in the codebase.
3) Connect to Redis through the project’s RedisVectorStore.
4) Ensure the vector index exists with the expected vector_dim.
5) For each part file:
   - Stream rows (one JSON per line).
   - Validate the embedding is a list of length vector_dim.
   - Write each row into Redis HASH at key "{key_prefix}{chunk_id}" with metadata + embedding bytes.
   - Flush writes in batches for memory/latency stability.

How to run:
python core/embedding/seed_embeddings.py
python core/embedding/seed_embeddings.py --export-dir e:\\MyProject\\providencetower-v2\\embed_chunk --index-name rag_chunks_idx --key-prefix rag:chunk: --distance-metric COSINE
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

from core.embedding.redis_store import RedisVectorStore


def seed(
    export_dir: Path,
    *,
    host: str,
    port: int,
    db: int,
    password: str | None,
    index_name: str | None,
    key_prefix: str | None,
    distance_metric: str | None,
    write_batch_size: int,
) -> None:
    manifest_path = export_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {export_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    resolved_index_name = str(index_name or manifest.get("index_name") or "rag_chunks_idx")
    resolved_key_prefix = str(key_prefix or manifest.get("key_prefix") or "rag:chunk:")
    resolved_distance_metric = str(distance_metric or manifest.get("distance_metric") or "COSINE")
    vector_dim = int(manifest.get("vector_dim") or 0)
    if vector_dim < 1:
        raise ValueError("Invalid vector_dim in manifest")

    parts = manifest.get("parts", [])
    if not isinstance(parts, list) or not parts:
        raise ValueError("No parts found in manifest")

    store = RedisVectorStore(
        host=host,
        port=port,
        db=db,
        password=password,
        index_name=resolved_index_name,
        key_prefix=resolved_key_prefix,
        distance_metric=resolved_distance_metric,
    )

    if not store.ping():
        raise RuntimeError(f"Cannot connect to Redis at {host}:{port}/{db}")

    store.ensure_index(vector_dim)

    total = 0
    for part in parts:
        part_file_name = part.get("file") or part.get("metadata_file")
        if not part_file_name:
            raise ValueError(f"Part entry missing file field: {part}")

        part_path = export_dir / str(part_file_name)
        if not part_path.exists():
            raise FileNotFoundError(f"Part file not found: {part_path}")

        pipe = store.client.pipeline(transaction=False)
        rows_in_part = 0
        buffered = 0
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

                key = f"{resolved_key_prefix}{row['chunk_id']}"
                mapping = {
                    "chunk_id": row["chunk_id"],
                    "page_id": row["page_id"],
                    "page_title": row["page_title"],
                    "section": row["section"],
                    "subsection": row["subsection"],
                    "source_file": row["source_file"],
                    "text": row["text"],
                    "embedding": np.asarray(embedding, dtype=np.float32).tobytes(),
                }
                pipe.hset(key, mapping=mapping)
                rows_in_part += 1
                buffered += 1
                if buffered >= write_batch_size:
                    pipe.execute()
                    pipe = store.client.pipeline(transaction=False)
                    buffered = 0

        if buffered:
            pipe.execute()
        total += rows_in_part
        print(f"Seeded {part_file_name}: {rows_in_part} rows (cumulative={total})")

    print(f"Done. Seeded {total} rows into Redis {host}:{port}/{db} index={resolved_index_name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seed Redis from Colab JSON embedding export")
    parser.add_argument("--export-dir", default="e:\\MyProject\\providencetower-v2\\embed_chunk", help="Directory containing manifest.json and part_*.jsonl")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--db", type=int, default=0)
    parser.add_argument("--password", default=None)
    parser.add_argument("--index-name", default="rag_chunks_idx", help="Override index name (otherwise manifest/default)")
    parser.add_argument("--key-prefix", default="rag:chunk:", help="Override key prefix (otherwise manifest/default)")
    parser.add_argument("--distance-metric", default=None, help="Override distance metric (otherwise manifest/default)")
    parser.add_argument("--write-batch-size", type=int, default=1000, help="Redis pipeline write batch size")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seed(
        export_dir=Path(args.export_dir),
        host=args.host,
        port=args.port,
        db=args.db,
        password=args.password,
        index_name=args.index_name,
        key_prefix=args.key_prefix,
        distance_metric=args.distance_metric,
        write_batch_size=args.write_batch_size,
    )

if __name__ == "__main__":
    main()
