# Ingestion Module Documentation

## Overview
The ingestion module (`core/ingestion/`) is responsible for fetching and processing content from MediaWiki-based sources (like Fandom wikis) into structured data suitable for RAG (Retrieval-Augmented Generation) systems.

## Module Structure

### Files:
- `ingest.py` - Main CLI entry point for ingestion operations
- `mediawiki_ingestor.py` - Core MediaWiki API client and processing logic
- `__init__.py` - Package initialization

## Purpose

The ingestion module serves three primary purposes:

1. **Content Acquisition**: Fetch wiki pages from MediaWiki APIs
2. **Content Transformation**: Convert HTML wiki content to clean markdown
3. **Metadata Enrichment**: Extract and structure page metadata for RAG indexing

## Process Flow

### 1. API Client Initialization
```python
# MediaWikiAPIClient handles API communication
client = MediaWikiAPIClient(api_url="https://yswiki.com/api.php")
```

### 2. Page Discovery
```python
# Fetch all pages in namespace 0 (main articles)
pages = client.fetch_all_pages(namespace=0)
# Returns: [WikiPage(page_id=123, title="Adol Christin", is_redirect=False), ...]
```

### 3. Redirect Resolution
```python
# Build clean page index resolving redirects to targets
clean_pages = client.build_clean_page_index(namespace=0)
# Returns: [(123, "Adol Christin"), (456, "Dana Iclucia"), ...]
# Redirect pages like "Adol" -> "Adol Christin" are resolved to target pages
```

### 4. Content Fetching
```python
# Fetch page content and metadata
content = client.fetch_page_content(page_id=123)
# Returns structured content with HTML, categories, and metadata
```

### 5. Redirect Content Detection
```python
# Check if HTML contains redirect phrases
if _has_redirect_phrase(html_content):
    # Skip processing and mark as redirect in manifest
    status = "skipped_redirect_phrase"
```

### 6. HTML to Markdown Conversion
```python
# Convert wiki HTML to clean markdown
markdown_content = html_to_markdown(html_content)
# Removes wiki-specific markup, tables become simplified text
```

### 7. Output Generation
```python
# Create structured JSON output for RAG ingestion
output = {
    "page_id": 123,
    "title": "Adol Christin", 
    "content": "# Adol Christin\n\nAdol Christin is the protagonist...",
    "categories": ["Characters", "Protagonists"],
    "source_url": "https://yswiki.com/wiki/Adol_Christin",
    "metadata": {"is_redirect": False, "namespace": 0}
}
```

## Example Input → Output Transformation

### Input (API Response):
```json
{
  "parse": {
    "pageid": 123,
    "title": "Adol Christin",
    "text": {"*": "<h2>Character Overview</h2><p>Adol Christin is the protagonist...</p>"},
    "categories": [{"title": "Category:Characters"}]
  }
}
```

### Processing Steps:
1. Extract page metadata (ID, title, categories)
2. Convert HTML content to markdown
3. Build source URL from title
4. Structure for RAG consumption

### Output (RAG-ready JSON):
```json
{
  "page_id": 123,
  "title": "Adol Christin",
  "content": "## Character Overview\n\nAdol Christin is the protagonist...",
  "categories": ["Characters"],
  "source_url": "https://yswiki.com/wiki/Adol_Christin",
  "metadata": {
    "is_redirect": false,
    "namespace": 0,
    "ingestion_timestamp": "2024-01-15T10:30:00Z"
  }
}
```

## Redirect Resolution Process

The ingestion module implements a comprehensive redirect resolution system:

### Two-Phase Redirect Handling

1. **Pre-Processing Phase**: During page discovery, redirect pages are identified and their target destinations are resolved using MediaWiki's redirects API

2. **Content Validation Phase**: During content fetching, HTML content is checked for redirect phrases to catch any remaining redirects

### Redirect Resolution Flow

```python
# Phase 1: Pre-processing redirect resolution
redirect_titles = [page.title for page in pages if page.is_redirect]
redirect_map = client.resolve_redirect_targets(redirect_titles)

# Phase 2: Content validation  
if _has_redirect_phrase(html_content):
    # Skip processing and log in manifest
    status = "skipped_redirect_phrase"
```

### Example Redirect Scenarios

- **Simple Redirect**: "Adol" → "Adol Christin" (resolved during pre-processing)
- **Complex Redirect**: Multi-hop redirects (resolved by MediaWiki API)
- **Content Redirect**: Pages with redirect phrases in HTML (caught during content validation)

## Key Features

- **Rate Limiting**: Polite API pacing with configurable intervals
- **Error Handling**: Exponential backoff with retries for failed requests
- **Redirect Handling**: Two-phase redirect resolution ensuring no redirect content enters RAG system
- **Content Cleaning**: Removal of wiki-specific markup and navigation elements
- **Batch Processing**: Efficient handling of large wiki datasets
- **Metadata Preservation**: Maintains categories and structural information

## Usage Examples

### CLI Ingestion:
```bash
python -m core.ingestion.ingest \
  --api-url https://yswiki.com/api.php \
  --output-dir ./ingested_data \
  --namespace 0
```

### Programmatic Usage:
```python
from core.ingestion.mediawiki_ingestor import WikiIngestor

ingestor = WikiIngestor(api_url="https://yswiki.com/api.php")
ingestor.ingest_all_pages(output_dir=Path("./data"))
```

## Integration with RAG System

The ingested JSON files are designed for direct consumption by the RAG indexing pipeline:

1. **Chunking**: Markdown content is split into semantic chunks
2. **Embedding**: Each chunk is vectorized using embedding models
3. **Indexing**: Stored in Redis vector store with metadata filtering capabilities
4. **Retrieval**: Used by the fetcher node for hybrid search operations

The structured metadata (categories, page titles) enables advanced filtering during RAG retrieval, ensuring contextually relevant results for Ys-specific queries.