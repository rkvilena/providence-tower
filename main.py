from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.env import settings
from core.rag.history import RedisHistoryStore, generate_session_id
from core.rag.rag_graph import RagGraph
from core.rag.schema import RagState


def _print_banner() -> None:
    title = "Providence Tower v2"
    subtitle = "Interactive CLI for chatting with the Ys RAG prototype."
    width = max(len(title), len(subtitle), 60)
    line = "=" * width
    print(line)
    print(title)
    print(subtitle)
    print(line)
    print(
        "Guide: include the Ys game title in your query when possible (e.g., 'Ys X: Nordics'); it helps retrieval a lot."
    )
    print("Commands: /clear (new session), /quit (exit)")
    print()


def _build_history_store() -> RedisHistoryStore | None:
    store = RedisHistoryStore(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        password=settings.REDIS_PASSWORD or None,
    )
    try:
        store.ping()
        return store
    except Exception:
        return None


def _run_full_flow(state: RagState) -> RagState:
    state.phase = "planner"
    state = RagGraph(phase="planner").run(state)
    state.phase = "fetcher"
    state = RagGraph(phase="fetcher").run(state)
    state.phase = "thinker"
    return RagGraph(phase="thinker").run(state)


def main() -> None:
    _print_banner()

    store = _build_history_store()
    if store is None:
        print("Session memory: OFF (Redis not reachable)")
    else:
        print("Session memory: ON (Redis)")
    if not (settings.OPENAI_API_KEY and settings.PLANNER_AGENT):
        print("Planner condensation: OFF (LLM disabled or missing OPENAI_API_KEY)")
    print()

    session_id = generate_session_id()
    while True:
        try:
            raw = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue
        if raw == "/quit":
            break
        if raw == "/clear":
            if store is not None:
                try:
                    store.clear_history(session_id)
                except Exception:
                    pass
            session_id = generate_session_id()
            print(f"Session cleared. New session: {session_id}")
            continue

        history = store.load_history(session_id) if store is not None else []
        state = RagState(session_id=session_id, user_query=raw, history=history)
        result = _run_full_flow(state)

        response_text = (result.thinker_state.response or "").strip()
        if store is not None and response_text:
            try:
                store.append_turn(
                    session_id,
                    question=raw,
                    answer=response_text,
                    history_window=settings.RAG_HISTORY_WINDOW,
                    ttl_seconds=settings.RAG_SESSION_TTL_SECONDS,
                )
            except Exception:
                pass

        print()
        print(response_text if response_text else "(No response)")
        print()
        print(f"Session: {session_id}")
        print()


if __name__ == "__main__":
    main()
