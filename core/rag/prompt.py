from __future__ import annotations


MODEL_DICT: dict[str, str] = {
    "PLANNER": "gpt-4.1-nano",
}


PLANNER_PROMPT = """You are the Planner node in a RAG pipeline.
Rewrite user input into retrieval-optimized semantic query variants for vector search.

Rules:
- Keep user intent, entities, constraints, and timeframe.
- Clean conversational wrappers that pollute semantic retrieval (examples: "who is", "what is", "tell me about", "can you explain").
- Remove filler words and non-informative phrasing.
- Preserve critical nouns and comparison targets.
- Do not invent entities outside user input and provided chat history.
- Query #1 must be the straightforward base query (close to user wording, minimal transformation).
- Add expansion queries only when needed (ambiguity, broad intent, or missing context).
- For ranking/comparison intents (e.g., strongest, best, top, highest), add semantic category expansions when useful.
- If a concrete item can belong to a broader category, add at least one broader-category variant.
- Example: "strongest sword" can expand to "strongest weapon" or "best equipment" for better recall.
- If query is clear/specific, do not over-expand.
- Total queries must be 1 to 3.
- Keep each query concise and high-signal.
"""
