# Embedding Module Documentation

## Overview

Converts chunked markdown into vector embeddings for Redis storage, supporting both local processing and Colab workflows. Corpus vectors are still produced with the local BGE model; only RAG-time query embedding and reranking run on Cloudflare Workers AI.

## Module Structure

- `embed.py` - CLI for local embedding operations
- `embedding_service.py` - Core local BGE embedding logic
- `schema.py` - Lightweight `ChunkDocument` model (no PyTorch imports)
- `redis_store.py` - Redis vector store interface
- `seed_embeddings.py` - Import Colab exports to Redis Stack or Upstash
- `ptv2_embed.ipynb` - Colab notebook for high-performance processing

Local model dependencies live in `core/requirements-embedding.txt`, which layers
on `core/requirements.txt`. The Cloud Run API image installs only the runtime
requirements and no longer downloads model weights at build time.

## Model Evolution

- **Original**: `all-MiniLM-L6-v2` - Poor retrieval for Ys-specific content
- **Current**: `BAAI/bge-small-en-v1.5` - Better for game terminology, slower locally

## Hybrid Processing Approach

1. **Local**: `embed.py` for development/testing
2. **Colab**: `ptv2_embed.ipynb` for production-scale processing
3. **Seeding**: `seed_embeddings.py` imports Colab results to local Redis

## Process Flow

### Local Processing

```bash
pip install -r core/requirements-embedding.txt
```

```python
embedding_service = EmbeddingService(
    model_name="BAAI/bge-small-en-v1.5",
    device="cpu",
    batch_size=256
)
```

1. **Load**: `load_documents_from_directory("./chunked_md")`
2. **Filter**: Skip chunks <60 characters
3. **Embed**: Batch processing with configurable sizes
4. **Store**: Redis vector storage with metadata

### Colab Export Format

```json
{
  "chunk_id": "11258-0003",
  "page_id": 11258,
  "text": "Along with Momiyama, he infused Falcom with a new style...",
  "embedding": [-0.0802999883890152, -0.01564083620905876, ...]
}
```

### Redis Seeding

The seeder writes through the existing store factory, so the same command
targets local Redis Stack or Upstash Vector depending on the environment:

```bash
python core/embedding/seed_embeddings.py --export-dir ./embed_chunk
```

- `VECTOR_STORE_PROVIDER=redis_stack`: local Redis Stack (`REDIS_*` settings).
- `VECTOR_STORE_PROVIDER=upstash`: Upstash (`UPSTASH_VECTOR_URL` and
  `UPSTASH_VECTOR_TOKEN`).

Seeding remains a manual, local step. Embeddings are not uploaded to Cloudflare
and the corpus is not automatically re-embedded.

## Key Features

- **Model**: BGE-small-en-v1.5 for niche content (384-dim)
- **Quality**: Filters short chunks, batch processing
- **Redis**: Automatic index management, metadata preservation
- **Hybrid**: Local dev + Colab production workflow
- **Runtime split**: Local BGE for corpus vectors; Cloudflare BGE for queries

## Usage Examples

### Local Embedding

```bash
python -m core.embedding.embed --input-dir ./data/chunked_md
```

### Redis Seeding

```bash
python -m core.embedding.seed_embeddings --export-dir ./embed_chunk
```

### Programmatic

```python
embedding_service = EmbeddingService()
store = RedisVectorStore()
for batch_docs, batch_vectors in embedding_service.iter_embeddings_by_batch(docs):
    store.upsert_documents(batch_docs, batch_vectors)
```

## RAG Integration

Corpus vectors are consumed by semantic search with metadata filtering using
384-dimensional vectors optimized for game content. At request time the Fetcher
embeds each query through Cloudflare Workers AI `@cf/baai/bge-small-en-v1.5`
(`mean` pooling) and reranks through `@cf/baai/bge-reranker-base`; see
[rag.md](rag.md).

Before relying on the mixed-provider path, run the manual compatibility smoke
check with Cloudflare credentials configured:

```bash
python -m test.manual_bge_compatibility
```

It compares local and Cloudflare vectors for one fixed sample and confirms both
are 384-dimensional and directionally compatible. It is a compatibility check,
not a model-quality benchmark.

## Performance

- Use local for development (which took a very long time since model upgrade), hence moving to Google Colab using jupyter notebook for free & fast processing
- Batch sizes: 256-512 for memory efficiency
- Redis: 1000-document write batches

