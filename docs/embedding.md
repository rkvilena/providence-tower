# Embedding Module Documentation

## Overview

Converts chunked markdown into vector embeddings for Redis storage, supporting both local processing and Colab workflows.

## Module Structure

- `embed.py` - CLI for local embedding operations
- `embedding_service.py` - Core embedding logic
- `redis_store.py` - Redis vector store interface
- `seed_embeddings.py` - Import Colab exports to Redis
- `ptv2_embed.ipynb` - Colab notebook for high-performance processing

## Model Evolution

- **Original**: `all-MiniLM-L6-v2` - Poor retrieval for Ys-specific content
- **Current**: `BAAI/bge-small-en-v1.5` - Better for game terminology, slower locally

## Hybrid Processing Approach

1. **Local**: `embed.py` for development/testing
2. **Colab**: `ptv2_embed.ipynb` for production-scale processing
3. **Seeding**: `seed_embeddings.py` imports Colab results to local Redis

## Process Flow

### Local Processing

```python
embedding_service = EmbeddingService(
    model_name="BAAI/bge-small-en-v1.5",
    device="cpu",
    batch_size=256
)
```

1. **Load**: `load_documents_from_directory("./chunked_md")`
2. **Filter**: Skip chunks <50 characters
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

```bash
python core/embedding/seed_embeddings.py --export-dir ./embed_chunk
```

## Key Features

- **Model**: BGE-small-en-v1.5 for niche content (384-dim)
- **Quality**: Filters short chunks, batch processing
- **Redis**: Automatic index management, metadata preservation
- **Hybrid**: Local dev + Colab production workflow

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

Enables semantic search with metadata filtering using 384-dimensional vectors optimized for game content.

## Performance

- Use local for development (which took a very long time since model upgrade), hence moving to Google Colab using jupyter notebook for free & fast processing
- Batch sizes: 256-512 for memory efficiency
- Redis: 1000-document write batches

