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
    parser = argparse.ArgumentParser(description="Run RAG planner/fetcher/thinker phases or full-flow")
    parser.add_argument("--phase", choices=["planner", "fetcher", "thinker"], help="RAG phase to execute")
    parser.add_argument("--full-flow", action="store_true", help="Run planner, fetcher, then thinker in one command")
    parser.add_argument("--query", help="User query text (required for planner phase)")
    parser.add_argument("--file", help="Path to a JSON file containing the RagState to load and update")
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
    input_file = Path(args.file) if args.file else None
    target_phase = "full-flow" if args.full_flow else args.phase

    if not args.full_flow and not args.phase:
        parser.error("--phase is required unless --full-flow is set")

    try:
        if input_file:
            if not input_file.exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")
            raw_data = json.loads(input_file.read_text(encoding="utf-8"))
            # If the file was created by _persist_output, the actual state is inside "state" key
            state_data = raw_data.get("state", raw_data)
            state = RagState(**state_data)
            # Update phase from args if provided
            if args.phase:
                state.phase = args.phase
            if args.query:
                state.user_query = args.query
        else:
            if (args.phase == "planner" or args.full_flow) and not args.query:
                raise ValueError("--query is required for planner phase when not using --file")
            
            state = RagState(
                session_id=args.session_id,
                user_query=args.query or "",
                phase=args.phase or "planner",
                chat_history=_parse_history(args.chat_history_json),
            )

        if args.full_flow:
            state.phase = "planner"
            planner_graph = RagGraph(phase="planner")
            state = planner_graph.run(state)
            state.phase = "fetcher"
            fetcher_graph = RagGraph(phase="fetcher")
            state = fetcher_graph.run(state)
            state.phase = "thinker"
            thinker_graph = RagGraph(phase="thinker")
            result_state = thinker_graph.run(state)
        else:
            graph = RagGraph(phase=args.phase)
            result_state = graph.run(state)

        output = {
            "status": "ok",
            "phase": target_phase,
            "state": result_state.model_dump(),
        }
        
        if input_file:
            # Update the original file in-place
            input_file.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Updated file: {input_file}")
        else:
            _persist_output(output_dir, args.session_id, target_phase or "unknown", output)

    except Exception as exc:
        output = {
            "status": "error",
            "phase": target_phase,
            "error_type": exc.__class__.__name__,
            "error_message": str(exc),
        }
        if input_file:
             # Even on error, update the file to reflect the failure if possible
             try:
                 input_file.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
             except:
                 pass

    print("Question:", state.user_query)
    print("Response:", state.thinker_state.response)


if __name__ == "__main__":
    main()
