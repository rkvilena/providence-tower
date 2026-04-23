from __future__ import annotations

import argparse
import json
import logging
import sys
import os
from pathlib import Path

# Add root directory to sys.path to support direct script execution
root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from core.chunking.markdown_chunker import MarkdownChunker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Markdown chunking entrypoint")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    all_parser = subparsers.add_parser("all", help="Chunk all markdown files")
    all_parser.add_argument(
        "--input-dir",
        default=str(root_path / "data" / "raw_markdown"),
        help="Directory containing raw markdown files",
    )
    all_parser.add_argument(
        "--output-dir",
        default=str(root_path / "data" / "chunked_md"),
        help="Directory where chunked markdown files will be written",
    )
    all_parser.add_argument(
        "--min-chars", type=int, default=500, help="Minimum target chars per chunk"
    )
    all_parser.add_argument(
        "--max-chars", type=int, default=1000, help="Maximum target chars per chunk"
    )
    all_parser.add_argument(
        "--overlap-ratio", type=float, default=0.1, help="Overlap ratio between chunks"
    )
    all_parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )

    one_parser = subparsers.add_parser("one", help="Chunk one markdown file")
    one_parser.add_argument(
        "filename", help="Markdown filename to chunk (required), e.g. 2039__Games.md"
    )
    one_parser.add_argument(
        "--input-dir",
        default=str(root_path / "data" / "raw_markdown"),
        help="Directory containing raw markdown files",
    )
    one_parser.add_argument(
        "--output-dir",
        default=str(root_path / "data" / "chunked_md"),
        help="Directory where chunked markdown files will be written",
    )
    one_parser.add_argument(
        "--min-chars", type=int, default=500, help="Minimum target chars per chunk"
    )
    one_parser.add_argument(
        "--max-chars", type=int, default=1000, help="Maximum target chars per chunk"
    )
    one_parser.add_argument(
        "--overlap-ratio", type=float, default=0.1, help="Overlap ratio between chunks"
    )
    one_parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    chunker = MarkdownChunker(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        overlap_ratio=args.overlap_ratio,
    )

    if args.mode == "all":
        summary = chunker.chunk_all_files()
        print(json.dumps(summary, indent=2))
        return

    output_path, chunks = chunker.chunk_single_file(Path(args.filename))
    print(
        json.dumps(
            {
                "source_file": args.filename,
                "output_file": str(output_path),
                "chunk_count": len(chunks),
                "status": "ok",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
