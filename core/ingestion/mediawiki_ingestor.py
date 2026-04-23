from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from markdownify import markdownify as html_to_markdown


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class WikiPage:
    page_id: int
    title: str
    is_redirect: bool


class MediaWikiAPIClient:
    """
    Lightweight MediaWiki API client focused on ingestion use-cases.
    """

    def __init__(
        self,
        api_url: str,
        *,
        request_timeout: int = 30,
        request_interval_seconds: float = 0.2,
        max_retries: int = 4,
        batch_size: int = 50,
        user_agent: str = "ProvidenceTowerIngestor/1.0 (+local ingestion script)",
    ) -> None:
        self.api_url = api_url
        self.request_timeout = request_timeout
        self.request_interval_seconds = request_interval_seconds
        self.max_retries = max_retries
        self.batch_size = max(1, min(batch_size, 50))
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self._last_request_at = 0.0

    def _sleep_if_needed(self) -> None:
        elapsed = time.time() - self._last_request_at
        if elapsed < self.request_interval_seconds:
            time.sleep(self.request_interval_seconds - elapsed)

    def _request_json(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Request JSON with polite pacing and exponential backoff.
        """
        for attempt in range(self.max_retries + 1):
            self._sleep_if_needed()
            try:
                response = self.session.get(
                    self.api_url,
                    params={**params, "format": "json"},
                    timeout=self.request_timeout,
                )
                self._last_request_at = time.time()
                response.raise_for_status()
                payload: dict[str, Any] = response.json()
                if "error" in payload:
                    raise RuntimeError(f"MediaWiki API error: {payload['error']}")
                return payload
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"API request failed after retries: {params}"
                    ) from exc
                backoff_seconds = (2**attempt) * 0.5
                LOGGER.warning(
                    "Request failed (attempt %s), retrying in %.1fs: %s",
                    attempt + 1,
                    backoff_seconds,
                    exc,
                )
                time.sleep(backoff_seconds)
        raise RuntimeError("Unreachable retry flow in _request_json")

    def fetch_all_pages(self, namespace: int = 0) -> list[WikiPage]:
        """
        Return all pages in a namespace, including redirects.
        """
        pages: list[WikiPage] = []
        continuation: dict[str, Any] = {}
        while True:
            params: dict[str, Any] = {
                "action": "query",
                "list": "allpages",
                "apnamespace": namespace,
                "aplimit": "max",
                "apfilterredir": "all",
            }
            params.update(continuation)
            payload = self._request_json(params)
            items = payload.get("query", {}).get("allpages", [])
            for item in items:
                page_id = item.get("pageid")
                title = item.get("title")
                if page_id is None or not title:
                    continue
                pages.append(
                    WikiPage(
                        page_id=int(page_id),
                        title=str(title),
                        is_redirect=("redirect" in item),
                    )
                )
            if "continue" not in payload:
                break
            continuation = payload["continue"]
        return pages

    def resolve_redirect_targets(self, redirect_titles: list[str]) -> dict[str, str]:
        """
        Resolve redirect source title -> destination title mapping.
        """
        mapping: dict[str, str] = {}
        for batch in _chunked(redirect_titles, self.batch_size):
            payload = self._request_json(
                {
                    "action": "query",
                    "titles": "|".join(batch),
                    "redirects": 1,
                }
            )
            redirects = payload.get("query", {}).get("redirects", [])
            for item in redirects:
                source = item.get("from")
                target = item.get("to")
                if source and target:
                    mapping[str(source)] = str(target)
        return mapping

    def resolve_titles_to_ids(self, titles: list[str]) -> dict[str, int]:
        """
        Resolve title -> page id for existing pages.
        """
        mapping: dict[str, int] = {}
        for batch in _chunked(titles, self.batch_size):
            payload = self._request_json(
                {
                    "action": "query",
                    "titles": "|".join(batch),
                }
            )
            pages_obj: dict[str, Any] = payload.get("query", {}).get("pages", {})
            for page in pages_obj.values():
                if "missing" in page:
                    continue
                title = page.get("title")
                page_id = page.get("pageid")
                if title and page_id is not None:
                    mapping[str(title)] = int(page_id)
        return mapping

    def fetch_page_html(self, page_id: int) -> tuple[str, str]:
        """
        Fetch rendered HTML for a page id.
        Returns (title, html).
        """
        payload = self._request_json(
            {
                "action": "parse",
                "pageid": page_id,
                "prop": "text",
                "formatversion": 2,
            }
        )
        parse_obj = payload.get("parse", {})
        title = parse_obj.get("title")
        html = parse_obj.get("text")
        if not isinstance(title, str) or not isinstance(html, str):
            raise RuntimeError(f"Invalid parse response for page id {page_id}")
        return title, html

    def build_clean_page_index(self, namespace: int = 0) -> list[tuple[int, str]]:
        """
        Collect all pages, resolve redirects to targets, and return deduped (id, title).
        """
        pages = self.fetch_all_pages(namespace=namespace)
        non_redirect: dict[int, str] = {}
        redirect_titles: list[str] = []
        for page in pages:
            if page.is_redirect:
                redirect_titles.append(page.title)
                continue
            non_redirect[page.page_id] = page.title

        if not redirect_titles:
            return sorted(non_redirect.items(), key=lambda x: x[0])

        redirect_map = self.resolve_redirect_targets(redirect_titles)
        target_titles = sorted(set(redirect_map.values()))
        target_id_map = self.resolve_titles_to_ids(target_titles)

        for title, page_id in target_id_map.items():
            if page_id not in non_redirect:
                non_redirect[page_id] = title

        return sorted(non_redirect.items(), key=lambda x: x[0])


class WikiIngestor:
    def __init__(
        self,
        client: MediaWikiAPIClient,
        output_dir: Path,
        *,
        base_wiki_url: str | None = None,
    ) -> None:
        self.client = client
        self.output_dir = output_dir
        self.base_wiki_url = base_wiki_url.rstrip("/") if base_wiki_url else None

    def run(self, namespace: int = 0, limit: int | None = None) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        pages = self.client.build_clean_page_index(namespace=namespace)
        if limit is not None:
            pages = pages[: max(0, limit)]

        manifest: list[dict[str, Any]] = []
        total = len(pages)
        LOGGER.info("Starting ingestion for %s pages", total)

        for index, (page_id, discovered_title) in enumerate(pages, start=1):
            try:
                title, html = self.client.fetch_page_html(page_id)
                if _has_redirect_phrase(html):
                    manifest.append(
                        {
                            "page_id": page_id,
                            "title": title,
                            "source_title": discovered_title,
                            "output_file": None,
                            "source_url": _build_source_url(
                                self.base_wiki_url, page_id=page_id
                            ),
                            "status": "skipped_redirect_phrase",
                        }
                    )
                    continue
                markdown = html_to_markdown(
                    html,
                    heading_style="ATX",
                    bullets="-",
                    strip=["script", "style"],
                )
                file_name = _build_output_name(page_id, title)
                file_path = self.output_dir / file_name
                file_path.write_text(markdown, encoding="utf-8")
                manifest.append(
                    {
                        "page_id": page_id,
                        "title": title,
                        "source_title": discovered_title,
                        "output_file": str(file_path),
                        "source_url": _build_source_url(
                            self.base_wiki_url, page_id=page_id
                        ),
                        "status": "ok",
                    }
                )
                if index % 25 == 0 or index == total:
                    LOGGER.info("Processed %s/%s pages", index, total)
            except Exception as exc:  # Keep run going and capture failures.
                LOGGER.exception("Failed on page id %s (%s)", page_id, discovered_title)
                manifest.append(
                    {
                        "page_id": page_id,
                        "title": discovered_title,
                        "output_file": None,
                        "source_url": _build_source_url(
                            self.base_wiki_url, page_id=page_id
                        ),
                        "status": "error",
                        "error": str(exc),
                    }
                )

        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        success = sum(1 for item in manifest if item["status"] == "ok")
        skipped = sum(
            1 for item in manifest if item["status"] == "skipped_redirect_phrase"
        )
        failed = sum(1 for item in manifest if item["status"] == "error")
        summary = {
            "total": total,
            "success": success,
            "skipped": skipped,
            "failed": failed,
            "manifest_path": str(manifest_path),
        }
        LOGGER.info("Ingestion finished: %s", summary)
        return summary


def _chunked(items: list[str], chunk_size: int) -> list[list[str]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _has_redirect_phrase(html: str) -> bool:
    return "redirect to:" in html.lower()


def _slugify(value: str) -> str:
    value = value.strip().replace(" ", "_")
    value = re.sub(r"[^\w\-\.]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "untitled"


def _build_output_name(page_id: int, title: str) -> str:
    return f"{page_id}__{_slugify(title)}.md"


def _build_source_url(base_wiki_url: str | None, *, page_id: int) -> str | None:
    if not base_wiki_url:
        return None
    return urljoin(f"{base_wiki_url.rstrip('/')}/", f"?curid={page_id}")
