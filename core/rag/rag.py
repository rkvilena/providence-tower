from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow direct script execution: `python core/rag/rag.py`
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.rag.rag_graph import RagGraph
from core.rag.schema import Message, RagState


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RAG phase nodes (currently Planner only)")
    parser.add_argument("--phase", required=True, choices=["planner"], help="RAG phase to execute")
    parser.add_argument("--query", required=True, help="User query text")
    parser.add_argument("--session-id", default="local-session", help="Session id for traceable artifacts")
    parser.add_argument(
        "--chat-history-json",
        default="[]",
        help='JSON array for chat history, e.g. \'[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]\'',
    )
    parser.add_argument(
        "--output-dir",
        default=r"e:\MyProject\providencetower-v2\data\rag_result",
        help="Directory where JSON results will be written",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def _parse_history(raw_json: str) -> list[Message]:
    parsed = json.loads(raw_json)
    if not isinstance(parsed, list):
        raise ValueError("--chat-history-json must be a JSON array")
    history: list[Message] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        history.append(
            Message(
                role=str(item.get("role", "user")),
                content=str(item.get("content", "")),
            )
        )
    return history


def _persist_output(output_dir: Path, session_id: str, phase: str, payload: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_session = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in session_id)
    file_path = output_dir / f"{timestamp}__{safe_session}__{phase}.json"
    payload["result_file"] = str(file_path)
    file_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return file_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    output_dir = Path(args.output_dir)
    try:
        state = RagState(
            session_id=args.session_id,
            user_query=args.query,
            phase=args.phase,
            chat_history=_parse_history(args.chat_history_json),
        )
        graph = RagGraph(phase=args.phase)
        result_state = graph.run(state)
        output = {
            "status": "ok",
            "phase": args.phase,
            "state": result_state.model_dump(),
        }
    except Exception as exc:
        output = {
            "status": "error",
            "phase": args.phase,
            "error_type": exc.__class__.__name__,
            "error_message": str(exc),
        }

    _persist_output(output_dir, args.session_id, args.phase, output)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
