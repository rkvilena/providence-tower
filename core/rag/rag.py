from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow direct script execution: `python core/rag/rag.py`
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.embedding.embedding_service import EmbeddingService
from core.embedding.redis_store import RedisVectorStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run semantic retrieval against Redis vector index")
    parser.add_argument("--query", required=True, help="User query text to search semantically")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top results to return")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--device", default=None, help="Embedding device, e.g. cpu, cuda, cuda:0")
    parser.add_argument("--redis-host", default="127.0.0.1", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--redis-db", type=int, default=0, help="Redis DB index")
    parser.add_argument("--redis-password", default=None, help="Redis password if required")
    parser.add_argument("--index-name", default="rag_chunks_idx", help="Redis vector index name")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")

    embedder = EmbeddingService(model_name=args.model, device=args.device)
    store = RedisVectorStore(
        host=args.redis_host,
        port=args.redis_port,
        db=args.redis_db,
        password=args.redis_password,
        index_name=args.index_name,
    )
    if not store.ping():
        raise RuntimeError(f"Cannot connect to Redis at {args.redis_host}:{args.redis_port}")

    query_vector = embedder.embed_query(args.query)
    search_results = store.search_similar(query_vector, top_k=args.top_k)

    output = {
        "query": args.query,
        "top_k": args.top_k,
        "results": [
            {
                "rank": idx + 1,
                "score": item.score,
                "key": item.key,
                "metadata": item.metadata,
                "text": item.text,
            }
            for idx, item in enumerate(search_results)
        ],
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
