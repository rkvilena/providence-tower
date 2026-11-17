"""Lightweight, model-independent shared data models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkDocument:
    """A parsed corpus chunk shared by embedding, seeding, and vector stores."""

    chunk_id: str
    page_id: int
    page_title: str
    section: str
    subsection: str
    source_file: str
    text: str
