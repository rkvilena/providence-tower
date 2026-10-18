/*
 * Providence Tower v2 — shared frontend constants.
 *
 * PHASE_LABELS maps backend SSE phase names (from `*_started` events) to the
 * human-readable text shown in the progress bubble while the pipeline runs.
 * The progress bubble is hidden the moment `final_answer` arrives; it must
 * never be repurposed for the client-side typewriter ("Typing…") state.
 */
const PHASE_LABELS = {
  planner: 'Planning query…',
  retrieval: 'Searching knowledge…',
  generation: 'Thinking…',
  context: 'Finalizing…',
};

/*
 * Display-speed options for the client-side typewriter reveal, in
 * milliseconds per character. 0 = instant (no animation).
 */
const DISPLAY_SPEEDS = [
  { value: 0, label: 'Instant' },
  { value: 8, label: 'Very fast (8 ms/char)' },
  { value: 20, label: 'Fast (20 ms/char)' },
  { value: 45, label: 'Normal (45 ms/char)' },
  { value: 90, label: 'Slow (90 ms/char)' },
];
