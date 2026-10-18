/*
 * Providence Tower v2 — Alpine chat component.
 *
 * Owns all chat state and orchestration. Delegates to:
 *   - chat-api.js      (streamChat: SSE request + parsing)
 *   - chat-renderer.js (renderMessageText, revealAnswerText)
 *   - chat-config.js   (PHASE_LABELS)
 *
 * State model:
 *   loading    — backend/SSE request is active; drives the progress bubble.
 *   typing     — client-side typewriter is animating; NEVER rendered as a
 *                status bubble. It only keeps the input controls disabled.
 *   phaseLabel — backend progress text; must never become "Typing…".
 */
function chat() {
  return {
    messages: [],
    input: '',
    loading: false,
    typing: false,
    phaseLabel: 'Thinking…',
    sessionId: self.crypto.randomUUID(),
    autoScroll: true,
    // Response Display Speed: ms per character for the typewriter reveal
    // (0 = instant). Frontend-only; the backend SSE pipeline is unaffected.
    displaySpeedMs: 0,
    // Bumped on every reveal / reset so an in-flight typewriter can be
    // cancelled without killing the request itself.
    renderToken: 0,

    init() {
      this.$watch('messages', () => {
        if (this.autoScroll) this.$nextTick(() => this.scrollDown());
      });
      this.$watch('loading', () => {
        if (!this.loading) this.$nextTick(() => this.scrollDown());
      });
    },

    scrollDown() {
      const el = this.$refs.messages;
      if (el) el.scrollTop = el.scrollHeight;
    },

    checkScroll() {
      const el = this.$refs.messages;
      if (!el) return;
      const threshold = 60;
      this.autoScroll = el.scrollHeight - el.scrollTop - el.clientHeight < threshold;
    },

    render(text) {
      return renderMessageText(text);
    },

    async send() {
      const query = this.input.trim();
      if (!query || this.loading || this.typing) return;

      this.messages.push({ role: 'user', content: query });
      this.input = '';
      this.loading = true;
      this.phaseLabel = 'Planning query…';

      try {
        const data = await streamChat(query, this.sessionId, (eventName, event) => {
          this.handleStreamEvent(eventName, event);
        });
        // The answer is here: drop the progress spinner immediately and
        // reveal the answer (instantly or via the typewriter animation).
        this.loading = false;
        this.typing = true; // revealAnswer clears it when done (or instantly)
        await this.revealAnswer(data);
      } catch (err) {
        this.renderToken++; // cancel any in-flight typewriter
        this.typing = false;
        this.messages.push({
          role: 'assistant',
          content: `**Error:** ${err.message}`,
        });
      } finally {
        this.loading = false;
      }
    },

    /**
     * Handle intermediate SSE events (progress phases) while the pipeline
     * runs. Only called for non-terminal events; `final_answer` is returned
     * by streamChat() and `pipeline_error` is thrown as an Error.
     */
    handleStreamEvent(eventName, event) {
      if (eventName.endsWith('_started') && event.phase) {
        this.phaseLabel = PHASE_LABELS[event.phase] || 'Thinking…';
      } else if (eventName === 'retrieval_completed' && event.chunk_count != null) {
        this.phaseLabel = `Found ${event.chunk_count} chunks…`;
      }
    },

    /**
     * Append the completed assistant message and reveal its text
     * (instantly or progressively). Uses the full `final_answer.response` —
     * never raw answer_delta text, which is structured Thinker JSON, not
     * user-facing prose.
     */
    async revealAnswer(data) {
      const token = ++this.renderToken;
      this.messages.push({
        role: 'assistant',
        content: '',
        trace: data.debug_trace || [],
      });
      this.sessionId = data.session_id || this.sessionId;

      const text = data.response || '(No response generated)';
      const messageIndex = this.messages.length - 1;

      const completed = await revealAnswerText(text, this.displaySpeedMs, {
        onChar: (fullText) => {
          // Update through the reactive array so Alpine re-renders reliably;
          // mutating a detached local object is not always observed.
          this.messages[messageIndex].content = fullText;
          this.messages = [...this.messages];
          if (this.autoScroll) this.scrollDown();
        },
        isCancelled: () => token !== this.renderToken, // cancelled (reset chat)
      });

      // Always clear the typing flag: on success, when the reveal finishes;
      // on cancellation (reset), resetChat() has already cleared it, but a
      // defensive reset here can never leave the controls stuck disabled.
      this.typing = false;
    },

    resetChat() {
      this.renderToken++; // cancel any in-flight typewriter reveal
      this.messages = [];
      this.sessionId = self.crypto.randomUUID();
      this.loading = false;
      this.typing = false;
      this.input = '';
      this.autoScroll = true;
      // Keep displaySpeedMs — it is a user preference, not per-chat state.
    },
  };
}
