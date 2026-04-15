from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkContext:
    page_id: int
    page_title: str
    section: str
    subsection: str
    source_file: str


@dataclass(frozen=True)
class ContentBlock:
    block_type: str  # "text" | "table"
    text: str
    context: ChunkContext


class MarkdownChunker:
    """
    Smart markdown chunker with:
    - recursive split strategy (headers -> paragraphs -> lines),
    - breadcrumb injection,
    - atomic table handling with header duplication.
    """

    def __init__(
        self,
        *,
        input_dir: Path,
        output_dir: Path,
        min_chars: int = 500,
        max_chars: int = 1000,
        overlap_ratio: float = 0.1,
    ) -> None:
        if min_chars < 1:
            raise ValueError("min_chars must be >= 1")
        if max_chars < min_chars:
            raise ValueError("max_chars must be >= min_chars")
        if not 0 <= overlap_ratio < 1:
            raise ValueError("overlap_ratio must be in range [0, 1)")

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.overlap_ratio = overlap_ratio
        self.overlap_chars = int(max_chars * overlap_ratio)
        self._image_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
        self._link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
        self._autolink_pattern = re.compile(r"<((?:https?://|data:)[^>\s]+)>", flags=re.IGNORECASE)
        self._http_pattern = re.compile(r"https?://[^\s)\]}]+", flags=re.IGNORECASE)
        self._data_uri_pattern = re.compile(r"data:[^\s)\]}]+", flags=re.IGNORECASE)
        self._orphan_label_pattern = re.compile(r"\[([^\[\]]+)\]")
        self._cite_link_pattern = re.compile(
            r"\[\s*\\?\d+(?:\.\d+)?\\?\s*\]\s*\\?\(#cite[^)\s]*\\?\)",
            flags=re.IGNORECASE,
        )
        self._cite_label_pattern = re.compile(r"\[\s*\\?\d+(?:\.\d+)?\\?\s*\]")
        self._stub_notice_pattern = re.compile(
            r"^:?\s*\*This article is a stub\b",
            flags=re.IGNORECASE,
        )
        self._see_list_of_pattern = re.compile(
            r"^\*See:\s*List of\b",
            flags=re.IGNORECASE,
        )
        self._useless_section_pattern = re.compile(
            r"^(Contents|Navigation|External links|References|Gallery|Official websites|Videos)$",
            flags=re.IGNORECASE,
        )

    def chunk_all_files(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        markdown_files = sorted(self.input_dir.glob("*.md"))
        run_manifest: list[dict[str, Any]] = []

        for markdown_file in markdown_files:
            try:
                output_path, chunks = self.chunk_single_file(markdown_file)
                run_manifest.append(
                    {
                        "source_file": str(markdown_file),
                        "output_file": str(output_path),
                        "chunk_count": len(chunks),
                        "status": "ok",
                    }
                )
            except Exception as exc:
                LOGGER.exception("Failed chunking file: %s", markdown_file)
                run_manifest.append(
                    {
                        "source_file": str(markdown_file),
                        "output_file": None,
                        "chunk_count": 0,
                        "status": "error",
                        "error": str(exc),
                    }
                )

        manifest_path = self.output_dir / "chunk_manifest.json"
        manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        success_count = sum(1 for item in run_manifest if item["status"] == "ok")
        error_count = len(run_manifest) - success_count
        total_chunks = sum(item.get("chunk_count", 0) for item in run_manifest)
        return {
            "total_files": len(markdown_files),
            "processed_files": success_count,
            "failed_files": error_count,
            "total_chunks": total_chunks,
            "manifest_path": str(manifest_path),
        }

    def chunk_single_file(self, markdown_file: Path) -> tuple[Path, list[dict[str, Any]]]:
        input_path = markdown_file if markdown_file.is_absolute() else (self.input_dir / markdown_file)
        input_path = input_path.resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {input_path}")
        if input_path.suffix.lower() != ".md":
            raise ValueError(f"Expected .md file, got: {input_path.name}")

        page_id, page_title = self._extract_page_metadata(input_path.name)
        content = input_path.read_text(encoding="utf-8")
        blocks = self._parse_content_blocks(content, page_id=page_id, page_title=page_title, source_file=input_path.name)

        chunks: list[dict[str, Any]] = []
        chunk_idx = 1
        for block in blocks:
            if block.block_type == "table":
                pieces = self._split_table_block(block.text)
            else:
                pieces = self._split_text_block(block.text)

            for piece in pieces:
                sanitized_piece = self._sanitize_piece(piece)
                if not sanitized_piece.strip():
                    continue
                if self._is_low_value_piece(sanitized_piece, block.block_type):
                    continue
                breadcrumb = self._build_breadcrumb(block.context)
                chunk_text = f"{breadcrumb} {sanitized_piece.strip()}"
                chunks.append(
                    {
                        "chunk_id": f"{page_id}-{chunk_idx:04d}",
                        "page_id": page_id,
                        "page_title": page_title,
                        "section": block.context.section,
                        "subsection": block.context.subsection,
                        "source_file": block.context.source_file,
                        "text": chunk_text,
                        "char_len": len(chunk_text),
                    }
                )
                chunk_idx += 1

        output_path = self.output_dir / f"{input_path.stem}.chunked.md"
        output_path.write_text(self._render_chunked_markdown(chunks), encoding="utf-8")
        return output_path, chunks

    def _parse_content_blocks(
        self,
        content: str,
        *,
        page_id: int,
        page_title: str,
        source_file: str,
    ) -> list[ContentBlock]:
        lines = content.splitlines()
        blocks: list[ContentBlock] = []
        section = "General"
        subsection = "General"
        idx = 0

        while idx < len(lines):
            line = lines[idx]
            stripped = line.strip()

            if stripped.startswith("## "):
                section = stripped[3:].strip().strip("[]#\\") or "General"
                subsection = "General"
                idx += 1
                continue
            if stripped.startswith("### "):
                subsection = stripped[4:].strip().strip("[]#\\") or "General"
                idx += 1
                continue

            if not stripped:
                idx += 1
                continue

            # Skip useless sections entirely
            if (self._useless_section_pattern.search(section) or 
                self._useless_section_pattern.search(subsection) or
                "navigation" in section.lower() or 
                "navigation" in subsection.lower()):
                idx += 1
                continue

            # Identify if it's a table
            if self._is_table_line(stripped):
                table_lines: list[str] = []
                while idx < len(lines) and self._is_table_line(lines[idx].strip()):
                    table_lines.append(lines[idx].rstrip())
                    idx += 1
                table_text = "\n".join(table_lines).strip()
                
                # Check if it's navigation or otherwise useless
                if not self._is_navigation_table(table_text):
                    context = ChunkContext(
                        page_id=page_id,
                        page_title=page_title,
                        section=section,
                        subsection=subsection,
                        source_file=source_file,
                    )
                    blocks.append(ContentBlock(block_type="table", text=table_text, context=context))
                continue

            # It's a text block (paragraph)
            paragraph_lines: list[str] = []
            while idx < len(lines):
                current = lines[idx]
                cur_strip = current.strip()
                if not cur_strip:
                    idx += 1
                    break
                if cur_strip.startswith("## ") or cur_strip.startswith("### "):
                    break
                if self._is_table_line(cur_strip):
                    break
                paragraph_lines.append(current.rstrip())
                idx += 1

            paragraph_text = "\n".join(paragraph_lines).strip()
            if paragraph_text:
                context = ChunkContext(
                    page_id=page_id,
                    page_title=page_title,
                    section=section,
                    subsection=subsection,
                    source_file=source_file,
                )
                blocks.append(ContentBlock(block_type="text", text=paragraph_text, context=context))

        return blocks

    def _split_text_block(self, text: str) -> list[str]:
        if len(text) <= self.max_chars:
            return [text]

        parts = self._split_by_paragraph(text)
        if any(len(p) > self.max_chars for p in parts):
            parts = self._split_long_parts_by_line(parts)
        if any(len(p) > self.max_chars for p in parts):
            parts = self._split_long_parts_by_char(parts)

        parts = [p.strip() for p in parts if p.strip()]
        return self._apply_overlap(parts)

    def _split_table_block(self, table_text: str) -> list[str]:
        lines = [line for line in table_text.splitlines() if line.strip()]
        if len(lines) <= 2 or len(table_text) <= self.max_chars:
            return [table_text]

        header = lines[0]
        separator = lines[1]
        rows = lines[2:]
        chunks: list[str] = []
        current_rows: list[str] = []

        for row in rows:
            candidate_rows = current_rows + [row]
            candidate_text = "\n".join([header, separator, *candidate_rows])
            if len(candidate_text) <= self.max_chars or not current_rows:
                current_rows = candidate_rows
                continue

            chunks.append("\n".join([header, separator, *current_rows]))
            current_rows = [row]

        if current_rows:
            chunks.append("\n".join([header, separator, *current_rows]))

        normalized_chunks: list[str] = []
        for chunk in chunks:
            if len(chunk) <= self.max_chars:
                normalized_chunks.append(chunk)
                continue
            normalized_chunks.extend(self._split_long_table_chunk(chunk, header, separator))

        return normalized_chunks

    def _split_long_table_chunk(self, chunk: str, header: str, separator: str) -> list[str]:
        body = "\n".join(chunk.splitlines()[2:])
        windows = self._char_windows(body, self.max_chars - len(header) - len(separator) - 2)
        return [f"{header}\n{separator}\n{window}".strip() for window in windows]

    def _split_by_paragraph(self, text: str) -> list[str]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) <= 1:
            return [text]
        return self._pack_units(paragraphs, joiner="\n\n")

    def _split_long_parts_by_line(self, parts: list[str]) -> list[str]:
        output: list[str] = []
        for part in parts:
            if len(part) <= self.max_chars:
                output.append(part)
                continue
            lines = [line for line in part.splitlines() if line.strip()]
            if len(lines) <= 1:
                output.append(part)
                continue
            output.extend(self._pack_units(lines, joiner="\n"))
        return output

    def _split_long_parts_by_char(self, parts: list[str]) -> list[str]:
        output: list[str] = []
        for part in parts:
            if len(part) <= self.max_chars:
                output.append(part)
                continue
            output.extend(self._char_windows(part, self.max_chars))
        return output

    def _pack_units(self, units: list[str], *, joiner: str) -> list[str]:
        chunks: list[str] = []
        current = ""
        for unit in units:
            candidate = f"{current}{joiner}{unit}" if current else unit
            if len(candidate) <= self.max_chars:
                current = candidate
                continue
            if current:
                chunks.append(current)
            if len(unit) <= self.max_chars:
                current = unit
            else:
                chunks.extend(self._char_windows(unit, self.max_chars))
                current = ""
        if current:
            chunks.append(current)

        return self._merge_small_chunks(chunks, joiner=joiner)

    def _merge_small_chunks(self, chunks: list[str], *, joiner: str) -> list[str]:
        if not chunks:
            return []
        merged: list[str] = []
        for chunk in chunks:
            if not merged:
                merged.append(chunk)
                continue
            if len(chunk) < self.min_chars and len(merged[-1]) + len(joiner) + len(chunk) <= self.max_chars:
                merged[-1] = f"{merged[-1]}{joiner}{chunk}"
                continue
            merged.append(chunk)
        return merged

    def _char_windows(self, text: str, width: int) -> list[str]:
        if width < 1:
            width = self.max_chars
        windows: list[str] = []
        step = max(1, width - self.overlap_chars)
        start = 0
        while start < len(text):
            end = min(len(text), start + width)
            piece = text[start:end].strip()
            if piece:
                windows.append(piece)
            if end >= len(text):
                break
            start += step
        return windows

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        if self.overlap_chars <= 0 or len(chunks) <= 1:
            return chunks
        overlapped: list[str] = []
        for idx, chunk in enumerate(chunks):
            if idx == 0:
                overlapped.append(chunk)
                continue
            prefix = chunks[idx - 1][-self.overlap_chars :].strip()
            combined = f"{prefix}\n{chunk}" if prefix else chunk
            overlapped.append(combined)
        return overlapped

    def _extract_page_metadata(self, filename: str) -> tuple[int, str]:
        match = re.match(r"^(?P<id>\d+)__(?P<title>.+)\.md$", filename, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"Filename must follow '<page_id>__<title>.md', got: {filename}")
        page_id = int(match.group("id"))
        page_title = match.group("title").replace("_", " ").strip()
        return page_id, page_title

    def _build_breadcrumb(self, context: ChunkContext) -> str:
        return (
            f"[Page: {context.page_title}]"
            f"[Section: {context.section}]"
            f"[Subsection: {context.subsection}]"
        )

    def _is_table_line(self, line: str) -> bool:
        return line.startswith("|") and line.endswith("|")

    def _is_navigation_table(self, table_text: str) -> bool:
        """
        Identify tables that are purely for navigation (e.g. wiki navbars).
        These often have many links and very little plain text.
        """
        lines = table_text.splitlines()
        if not lines:
            return False

        # Guardrail: large markdown tables with header separators are almost
        # always data tables, even if they contain words like "Equipment & Items"
        # inside one cell/link title.
        has_separator = any("---" in line for line in lines)
        if has_separator and len(lines) > 6:
            return False

        # Check for navigation-related keywords in the table text
        nav_keywords = [
            "Previous game", "Next game", "Timeline order", "Release order",
            "Chronological order", "Main series", "Spin-offs", "Related games"
        ]
        if any(keyword.lower() in table_text.lower() for keyword in nav_keywords):
            return True

        # Check for 3-line tables that are often nav links (at start of file)
        # In Ys I, these are 3-line tables like | | --- | [Link] |
        if len(lines) <= 3 and table_text.count("[") >= 1:
            return True
        if len(lines) <= 3:
            small_nav_keywords = [
                "characters", "areas", "walkthrough", "equipment & items",
                "enemies & bosses", "music", "trophies"
            ]
            lowered = table_text.lower()
            if any(keyword in lowered for keyword in small_nav_keywords):
                return True

        return False

    def _is_low_value_piece(self, text: str, block_type: str) -> bool:
        if block_type != "text":
            return False
        compact = text.strip()
        if not compact:
            return True
        if self._stub_notice_pattern.match(compact):
            return True
        if self._see_list_of_pattern.match(compact):
            return True
        return False

    def _sanitize_piece(self, text: str) -> str:
        """
        Sanitize link-heavy markdown while preserving semantic labels.
        This runs as post-processing inside chunking so output is retrieval-ready.
        """
        sanitized = text
        sanitized = self._remove_citation_artifacts(sanitized)
        sanitized = self._sanitize_markdown_images(sanitized)
        sanitized = self._sanitize_markdown_links(sanitized)
        sanitized = self._autolink_pattern.sub("", sanitized)
        sanitized = self._http_pattern.sub("", sanitized)
        sanitized = self._data_uri_pattern.sub("", sanitized)
        sanitized = self._orphan_label_pattern.sub(r"\1", sanitized)
        sanitized = self._remove_citation_artifacts(sanitized)
        return self._normalize_sanitized_lines(sanitized)

    def _remove_citation_artifacts(self, text: str) -> str:
        sanitized = self._cite_link_pattern.sub("", text)
        sanitized = self._cite_label_pattern.sub("", sanitized)
        return sanitized

    def _sanitize_markdown_images(self, text: str) -> str:
        def _replace(match: re.Match[str]) -> str:
            alt_text = match.group(1).strip()
            if not alt_text:
                return ""
            return alt_text

        previous = None
        current = text
        while previous != current:
            previous = current
            current = self._image_pattern.sub(_replace, current)
        return current

    def _sanitize_markdown_links(self, text: str) -> str:
        def _replace(match: re.Match[str]) -> str:
            label = match.group(1).strip()
            return label

        previous = None
        current = text
        while previous != current:
            previous = current
            current = self._link_pattern.sub(_replace, current)
        return current

    def _normalize_sanitized_lines(self, text: str) -> str:
        normalized_lines: list[str] = []
        for raw_line in text.splitlines():
            line = re.sub(r"[ \t]+", " ", raw_line).strip()
            if not line:
                normalized_lines.append("")
                continue
            line = re.sub(r"\(\s*\)", "", line).strip()
            line = re.sub(r"\s+([,.;:!?])", r"\1", line)
            normalized_lines.append(line)
        return "\n".join(normalized_lines).strip()

    def _render_chunked_markdown(self, chunks: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for chunk in chunks:
            lines.append(f"## Chunk {chunk['chunk_id']}")
            lines.append(chunk["text"])
            lines.append("")
        return "\n".join(lines).strip() + "\n"
