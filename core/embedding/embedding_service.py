from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkDocument:
    chunk_id: str
    page_id: int
    page_title: str
    section: str
    subsection: str
    source_file: str
    text: str


class EmbeddingService:
    """
    Handles chunk parsing and high-throughput local embeddings.
    """

    def __init__(
        self,
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 256,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        
        # Limit PyTorch to use only 4 CPU threads to avoid maxing out system resources
        torch.set_num_threads(4)
        
        self.model = SentenceTransformer(model_name, device=device)
        self._chunk_block_pattern = re.compile(r"^##\s+Chunk\s+([^\n]+)\n(.*?)(?=^##\s+Chunk\s+|\Z)", re.MULTILINE | re.DOTALL)
        self._breadcrumb_pattern = re.compile(
            r"^\[Page:\s*(.*?)\]\[Section:\s*(.*?)\]\[Subsection:\s*(.*?)\]\s*(.*)$",
            re.DOTALL,
        )
        self._meta_from_filename_pattern = re.compile(r"^(?P<id>\d+)__(?P<title>.+)\.chunked\.md$", re.IGNORECASE)

    def load_documents_from_directory(self, input_dir: Path) -> list[ChunkDocument]:
        if not input_dir.exists():
            raise FileNotFoundError(f"Chunk input directory not found: {input_dir}")
        files = sorted(input_dir.glob("*.chunked.md"))
        documents: list[ChunkDocument] = []
        for file_path in files:
            documents.extend(self.load_documents_from_file(file_path))
        return documents

    def load_documents_from_file(self, file_path: Path) -> list[ChunkDocument]:
        text = file_path.read_text(encoding="utf-8")
        fallback_page_id, fallback_title = self._extract_page_metadata_from_file(file_path.name)
        docs: list[ChunkDocument] = []

        for match in self._chunk_block_pattern.finditer(text):
            chunk_id = match.group(1).strip()
            body = match.group(2).strip()
            if not body:
                continue

            page_id = fallback_page_id
            page_title = fallback_title
            section = "General"
            subsection = "General"
            chunk_text = body

            breadcrumb_match = self._breadcrumb_pattern.match(body)
            if breadcrumb_match:
                page_title = breadcrumb_match.group(1).strip() or fallback_title
                section = breadcrumb_match.group(2).strip() or "General"
                subsection = breadcrumb_match.group(3).strip() or "General"
                chunk_text = breadcrumb_match.group(4).strip()

            if not chunk_text:
                continue

            docs.append(
                ChunkDocument(
                    chunk_id=chunk_id,
                    page_id=page_id,
                    page_title=page_title,
                    section=section,
                    subsection=subsection,
                    source_file=file_path.name,
                    text=chunk_text,
                )
            )

        return docs

    def embed_documents(self, documents: list[ChunkDocument]) -> np.ndarray:
        if not documents:
            return np.empty((0, 0), dtype=np.float32)
        texts = [doc.text for doc in documents]
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        text = query.strip()
        if not text:
            raise ValueError("Query must not be empty")
        vector = self.model.encode(
            [text],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )[0]
        return np.asarray(vector, dtype=np.float32)

    def _extract_page_metadata_from_file(self, filename: str) -> tuple[int, str]:
        match = self._meta_from_filename_pattern.match(filename)
        if not match:
            return -1, filename
        page_id = int(match.group("id"))
        page_title = match.group("title").replace("_", " ").strip()
        return page_id, page_title
