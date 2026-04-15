from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

# Add root directory to sys.path to support direct script execution
root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from core.ingestion.mediawiki_ingestor import (
    MediaWikiAPIClient,
    WikiIngestor,
    html_to_markdown,
    _build_output_name,
    _build_source_url,
    _has_redirect_phrase,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


def _infer_base_wiki_url(api_url: str) -> str | None:
    normalized = api_url.strip()
    if not normalized:
        return None
    # Handle common MediaWiki API forms:
    # - https://host/wiki/api.php
    # - https://host/w/api.php
    base = re.sub(r"/(?:wiki|w)?/api\.php(?:\?.*)?$", "", normalized, flags=re.IGNORECASE)
    if base == normalized:
        return None
    return f"{base.rstrip('/')}/wiki"


def fetch_single_page(api_url: str, title: str, output_dir: Path, base_wiki_url: str | None = None):
    client = MediaWikiAPIClient(api_url=api_url)

    # Resolve title to ID
    LOGGER.info(f"Resolving title: {title}")
    id_map = client.resolve_titles_to_ids([title])

    if title not in id_map:
        # Try again with normalized title (Fandom uses underscores)
        normalized_title = title.replace(" ", "_")
        if normalized_title != title:
            id_map = client.resolve_titles_to_ids([normalized_title])
            if normalized_title in id_map:
                title = normalized_title

    if title not in id_map:
        LOGGER.error(f"Could not find page ID for title: {title}")
        return

    page_id = id_map[title]
    LOGGER.info(f"Found page ID: {page_id} for {title}")

    # Fetch HTML
    LOGGER.info(f"Fetching HTML for page ID {page_id}")
    fetched_title, html = client.fetch_page_html(page_id)

    if _has_redirect_phrase(html):
        LOGGER.warning(f"Page '{fetched_title}' is a redirect. Skipping.")
        return

    # Convert to Markdown
    LOGGER.info(f"Converting HTML to Markdown")
    markdown = html_to_markdown(
        html,
        heading_style="ATX",
        bullets="-",
        strip=["script", "style"],
    )

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = _build_output_name(page_id, fetched_title)
    file_path = output_dir / file_name
    file_path.write_text(markdown, encoding="utf-8")

    LOGGER.info(f"Successfully saved to: {file_path}")

    # Print summary
    print(
        json.dumps(
            {
                "page_id": page_id,
                "title": fetched_title,
                "output_file": str(file_path),
                "source_url": _build_source_url(base_wiki_url, page_id=page_id),
                "status": "ok",
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MediaWiki Ingestion Tool")
    parser.add_argument("--api-url", required=True, help="MediaWiki api.php endpoint")
    parser.add_argument(
        "--output-dir",
        default=r"e:\MyProject\providencetower-v2\data\raw_markdown",
        help="Directory where markdown files and manifest will be written",
    )
    parser.add_argument("--title", help="Specific page title to ingest (single fetch mode)")
    parser.add_argument("--namespace", type=int, default=0, help="MediaWiki namespace id (full ingest mode)")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for debugging (full ingest mode)")
    parser.add_argument("--request-interval", type=float, default=0.2, help="Delay between requests (seconds)")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout (seconds)")
    parser.add_argument("--max-retries", type=int, default=4, help="Retry count for transient failures")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for title/page queries")
    parser.add_argument("--user-agent", default="ProvidenceTowerIngestor/1.0 (+local ingestion script)")
    parser.add_argument(
        "--base-wiki-url",
        default=None,
        help="Optional base page URL for manifest source links, e.g. https://example.com/wiki",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    # Update logger level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    effective_base_wiki_url = args.base_wiki_url or _infer_base_wiki_url(args.api_url)

    if args.title:
        # Single page mode
        fetch_single_page(args.api_url, args.title, Path(args.output_dir), effective_base_wiki_url)
    else:
        # Full ingest mode
        client = MediaWikiAPIClient(
            api_url=args.api_url,
            request_timeout=args.timeout,
            request_interval_seconds=args.request_interval,
            max_retries=args.max_retries,
            batch_size=args.batch_size,
            user_agent=args.user_agent,
        )
        ingestor = WikiIngestor(
            client=client,
            output_dir=Path(args.output_dir),
            base_wiki_url=effective_base_wiki_url,
        )
        summary = ingestor.run(namespace=args.namespace, limit=args.limit)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
