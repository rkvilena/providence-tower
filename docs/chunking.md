# Chunking Module Documentation

## Overview
The chunking module (`core/chunking/`) is responsible for splitting markdown content from the ingestion phase into semantically meaningful chunks suitable for RAG (Retrieval-Augmented Generation) systems. It transforms raw markdown files into structured chunks with proper context preservation.

## Module Structure

### Files:
- `chunk.py` - Main CLI entry point for chunking operations
- `markdown_chunker.py` - Core chunking logic and processing
- `__init__.py` - Package initialization

## Purpose
The chunking module serves three primary purposes:

1. **Semantic Segmentation**: Split markdown content at natural boundaries (headers, paragraphs)
2. **Context Preservation**: Maintain hierarchical structure and metadata in chunks
3. **Quality Filtering**: Remove low-value content and noise from chunks

## Process Flow

### 1. File Processing Initialization
```python
# MarkdownChunker handles chunking operations
chunker = MarkdownChunker(
    input_dir=Path("./raw_markdown"),
    output_dir=Path("./chunked_md"),
    min_chars=500,
    max_chars=1000,
    overlap_ratio=0.1
)
```

### 2. Metadata Extraction
```python
# Extract page ID and title from filename
page_id, page_title = chunker._extract_page_metadata("123__Adol_Christin.md")
# Returns: (123, "Adol Christin")
```

### 3. Content Block Parsing
```python
# Parse markdown into semantic blocks with context
blocks = chunker._parse_content_blocks(
    content, 
    page_id=123, 
    page_title="Adol Christin", 
    source_file="123__Adol_Christin.md"
)
# Returns: [ContentBlock(block_type="text", text="Character overview...", context=...)]
```

### 4. Block Type Processing
```python
# Different processing for text vs table blocks
if block.block_type == "table":
    pieces = chunker._split_table_block(block.text)
else:
    pieces = chunker._split_text_block(block.text)
```

### 5. Content Sanitization
```python
# Remove low-value content and noise
sanitized_piece = chunker._sanitize_piece(piece)
# Removes images, autolinks, citation markers, and stub notices
```

### 6. Breadcrumb Generation
```python
# Build contextual breadcrumbs for chunks
breadcrumb = chunker._build_breadcrumb(block.context)
# Returns: "[Adol Christin > Character Overview > Appearance]"
```

### 7. Chunk Assembly
```python
# Create final chunk with metadata
chunk_text = f"{breadcrumb} {sanitized_piece.strip()}"
chunks.append({
    "chunk_id": "123-0001",
    "page_id": 123,
    "page_title": "Adol Christin",
    "section": "Character Overview",
    "subsection": "Appearance",
    "text": chunk_text,
    "char_len": len(chunk_text)
})
```

### 8. Output Generation
```python
# Write chunked content to file
output_path.write_text(chunker._render_chunked_markdown(chunks))
```

## Example Input → Output Transformation

### Input (Raw Markdown):
```markdown
# Adol Christin

## Character Overview

Adol Christin is the protagonist of the Ys series.

### Appearance

Adol has red hair and wears adventurer's clothing.

## Abilities

### Sword Skills

Adol is a skilled swordsman.
```

### Processing Steps:
1. Extract metadata from filename
2. Parse hierarchical structure (H1 → H2 → H3)
3. Split into semantic blocks at header boundaries
4. Generate breadcrumbs for context
5. Sanitize and assemble chunks

### Output (Chunked Content):
```json
{
  "chunk_id": "123-0001",
  "page_id": 123,
  "page_title": "Adol Christin",
  "section": "Character Overview",
  "subsection": "General",
  "text": "Adol Christin is the protagonist of the Ys series.",
  "char_len": 87
}
{
  "chunk_id": "123-0002", 
  "page_id": 123,
  "page_title": "Adol Christin",
  "section": "Character Overview",
  "subsection": "Appearance",
  "text": "Adol has red hair and wears adventurer's clothing.",
  "char_len": 92
}
```
Here's how it looks in markdown document for embedding process
```markdown
## Chunk 2041-0020
[Page: Ys II Ancient Ys Vanished - The Final Chapter][Section: Gameplay][Subsection: General] The player controls Adol as he battles his way across the land of Ys. As in the first game, Adol's strength is measured in a typical RPG fashion: He has numerical statistics such as HP, Attack Power, and Defense Power that determine his strength. These stats are increased by raising his experience level through battling.
```

## Key Features

### Smart Chunking Strategy
- **Recursive Splitting**: Headers → paragraphs → lines for optimal granularity
- **Hierarchical Preservation**: Maintains section/subsection relationships
- **Atomic Table Handling**: Tables are kept intact as single chunks

### Content Quality Filtering
- **Image Removal**: Strips markdown image syntax `![]()`
- **Link Cleaning**: Preserves descriptive links, removes autolinks
- **Citation Filtering**: Removes Wikipedia-style citation markers `[1]`
- **Stub Detection**: Filters out "this article is a stub" notices
- **Low-Value Section Skipping**: Skips navigation, references, external links

### Context Preservation
- **Breadcrumb Injection**: Adds hierarchical context to each chunk
- **Metadata Embedding**: Includes page ID, title, section info in each chunk
- **Structural Integrity**: Maintains semantic boundaries between sections

### Configuration Flexibility
- **Size Control**: Configurable min/max chunk sizes (default: 500-1000 chars)
- **Overlap Support**: Configurable overlap between chunks (default: 10%)
- **Batch Processing**: Handles single files or entire directories

## Usage Examples

### CLI Chunking (All Files):
```bash
python -m core.chunking.chunk all \
  --input-dir ./data/raw_markdown \
  --output-dir ./data/chunked_md \
  --min-chars 500 \
  --max-chars 1000 \
  --overlap-ratio 0.1
```

### CLI Chunking (Single File):
```bash
python -m core.chunking.chunk one "123__Adol_Christin.md" \
  --input-dir ./data/raw_markdown \
  --output-dir ./data/chunked_md
```

### Programmatic Usage:
```python
from core.chunking.markdown_chunker import MarkdownChunker

chunker = MarkdownChunker(
    input_dir=Path("./raw_markdown"),
    output_dir=Path("./chunked_md"),
    min_chars=500,
    max_chars=1000,
    overlap_ratio=0.1
)

# Chunk all files
summary = chunker.chunk_all_files()

# Chunk single file  
output_path, chunks = chunker.chunk_single_file("123__Adol_Christin.md")
```

## Integration with RAG System

The chunked markdown files are optimized for direct consumption by the RAG indexing pipeline:

1. **Embedding Ready**: Clean, contextualized text perfect for embedding models
2. **Metadata Rich**: Structured metadata enables advanced filtering during retrieval
3. **Semantically Coherent**: Chunks maintain logical boundaries for better relevance
4. **Quality Controlled**: Noise-free content improves retrieval accuracy

The hierarchical context preservation enables the RAG system to:
- Filter chunks by specific sections (e.g., only "Abilities" sections)
- Understand the contextual relationship between chunks
- Provide more precise and relevant responses to Ys-specific queries

## Error Handling & Monitoring

- **Manifest Generation**: Creates `chunk_manifest.json` with processing status
- **Error Resilience**: Continues processing other files on individual failures
- **Progress Logging**: Detailed logging for monitoring large batch operations
- **Quality Metrics**: Tracks success rates, chunk counts, and processing statistics