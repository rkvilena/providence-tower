from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Allow direct script execution: `python core/embedding/embed.py`
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.embedding.embedding_service import EmbeddingService
from core.embedding.redis_store import RedisVectorStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embed chunked markdown files and store in Redis")
    parser.add_argument(
        "--input-dir",
        default=r"e:\MyProject\providencetower-v2\data\chunked_md",
        help="Directory containing *.chunked.md files",
    )
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5", help="Embedding model name")
    parser.add_argument("--device", default=None, help="Embedding device, e.g. cpu, cuda, cuda:0")
    parser.add_argument("--batch-size", type=int, default=256, help="Embedding batch size")
    parser.add_argument("--redis-host", default="127.0.0.1", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--redis-db", type=int, default=0, help="Redis DB index")
    parser.add_argument("--redis-password", default=None, help="Redis password if required")
    parser.add_argument("--index-name", default="rag_chunks_idx", help="Redis search index name")
    parser.add_argument("--key-prefix", default="rag:chunk:", help="Redis key prefix for chunk docs")
    parser.add_argument("--write-batch-size", type=int, default=1000, help="Redis pipeline write batch size")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    start = time.perf_counter()
    input_dir = Path(args.input_dir)

    embedding_service = EmbeddingService(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
    )
    store = RedisVectorStore(
        host=args.redis_host,
        port=args.redis_port,
        db=args.redis_db,
        password=args.redis_password,
        index_name=args.index_name,
        key_prefix=args.key_prefix,
    )

    if not store.ping():
        raise RuntimeError(f"Cannot connect to Redis at {args.redis_host}:{args.redis_port}")

    logger.info("Loading chunk documents from %s", input_dir)
    documents = embedding_service.load_documents_from_directory(input_dir)
    if not documents:
        raise RuntimeError(f"No chunk documents found in: {input_dir}")

    filtered_documents = embedding_service.filter_documents_for_embedding(documents)
    skipped_documents = len(documents) - len(filtered_documents)
    if skipped_documents:
        logger.info("Skipping %s chunks shorter than 50 characters", skipped_documents)
    if not filtered_documents:
        raise RuntimeError("No chunk documents met the minimum length requirement")

    logger.info("Embedding %s chunks using model '%s'", len(filtered_documents), args.model)
    vectors = embedding_service.embed_documents(filtered_documents)
    dim = int(vectors.shape[1])

    logger.info("Ensuring Redis vector index '%s' (dim=%s)", args.index_name, dim)
    store.ensure_index(dim)

    logger.info("Writing vectors to Redis")
    written = store.upsert_documents(filtered_documents, vectors, batch_size=args.write_batch_size)

    elapsed = time.perf_counter() - start
    print(
        json.dumps(
            {
                "status": "ok",
                "input_dir": str(input_dir),
                "documents": len(filtered_documents),
                "vectors_written": written,
                "vector_dim": dim,
                "index_name": args.index_name,
                "redis": f"{args.redis_host}:{args.redis_port}/{args.redis_db}",
                "elapsed_seconds": round(elapsed, 2),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
