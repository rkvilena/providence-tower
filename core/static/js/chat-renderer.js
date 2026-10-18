/*
 * Providence Tower v2 — presentation helpers.
 *
 * Pure display logic: message formatting and the typewriter reveal. No Alpine
 * state or network access lives here, so it stays trivially testable.
 */

/**
 * Escape text as HTML, then apply the supported inline Markdown subset
 * (**bold**, *italic*, `code`) and auto-link URLs. Newlines become <br>.
 *
 * @param {string} text Raw message text.
 * @returns {string} HTML-safe string for use with x-html.
 */
function renderMessageText(text) {
  // Browser-native HTML escaping first, then inline formatting.
  const div = document.createElement('div');
  div.appendChild(document.createTextNode(text));
  return div.innerHTML
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`(.+?)`/g, '<code class="bg-gray-200 rounded px-1 text-xs">$1</code>')
    .replace(/(https?:\/\/[^\s<]+)/g, '<a href="$1" target="_blank" rel="noopener noreferrer" class="text-blue-600 underline hover:text-blue-800">$1</a>')
    .replace(/\n/g, '<br>');
}

/**
 * Reveal an answer, either instantly or with a typewriter effect.
 *
 * Characters are appended via the `onChar(fullText)` callback; the caller is
 * responsible for making each update reactive (e.g. reassigning the Alpine
 * messages array) and for scrolling.
 *
 * @param {string} text       Full answer text.
 * @param {number} delayMs    Milliseconds per character; 0 = instant.
 * @param {object} hooks
 * @param {(fullText: string) => void} hooks.onChar
 *        Called after each appended character (and once for instant mode).
 * @param {() => Promise<void>} [hooks.wait]
 *        Inter-frame pause; defaults to a plain setTimeout sleep.
 * @param {() => boolean} [hooks.isCancelled]
 *        Checked before every character; aborts and returns false when true.
 * @returns {Promise<boolean>} True if the full text was revealed, false if cancelled.
 */
async function revealAnswerText(text, delayMs, { onChar, wait = sleepMs, isCancelled = () => false } = {}) {
  const delay = Number(delayMs) || 0;

  if (delay <= 0) {
    onChar(text);
    return true;
  }

  for (let i = 1; i <= text.length; i++) {
    if (isCancelled()) return false;
    onChar(text.slice(0, i));
    // Yield to the browser so the text is actually painted each step.
    await wait(delay);
  }
  return !isCancelled();
}

/**
 * Small promise-based sleep used as the default typewriter pause.
 * @param {number} ms
 * @returns {Promise<void>}
 */
function sleepMs(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
