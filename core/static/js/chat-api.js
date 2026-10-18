/*
 * Providence Tower v2 — SSE chat transport.
 *
 * Pure transport layer: performs the request, validates the HTTP response,
 * parses SSE frames, reports progress events to a callback, and resolves with
 * the `final_answer` event. It deliberately knows nothing about Alpine state.
 *
 * Usage:
 *   const final = await streamChat(query, sessionId, (eventName, event) => { ... });
 */

/**
 * POST a query to the RAG streaming endpoint and read it to completion.
 *
 * @param {string} query      The user's question.
 * @param {string} sessionId  Current chat session id.
 * @param {(eventName: string, event: object) => void} [onEvent]
 *        Called for every intermediate SSE event (lifecycle phases etc.).
 * @returns {Promise<object>} The `final_answer` event payload.
 */
async function streamChat(query, sessionId, onEvent = () => {}) {
  const resp = await fetch('/rag/chat/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'text/event-stream',
    },
    body: JSON.stringify({ query, session_id: sessionId }),
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    const msg = resp.status === 429
      ? err.detail || 'Rate limit exceeded. Usage is limited due to the application\'s nature as portfolio.'
      : err.detail || `HTTP ${resp.status}`;
    throw new Error(msg);
  }
  if (!resp.body) throw new Error('Streaming response body is unavailable');

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let finalAnswer = null;

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const frames = buffer.split('\n\n');
    buffer = frames.pop() || '';

    for (const frame of frames) {
      const lines = frame.split('\n');
      const eventName = (lines.find(line => line.startsWith('event: ')) || '').slice(7);
      const dataLine = lines.find(line => line.startsWith('data: '));
      if (!eventName || !dataLine) continue;

      const event = JSON.parse(dataLine.slice(6));
      if (eventName === 'pipeline_error') {
        throw new Error(event.detail || 'Pipeline failed');
      }
      if (eventName === 'final_answer') {
        finalAnswer = event;
      } else {
        onEvent(eventName, event);
      }
    }

    if (done) break;
  }

  if (!finalAnswer) throw new Error('Stream ended without a final answer');
  return finalAnswer;
}
