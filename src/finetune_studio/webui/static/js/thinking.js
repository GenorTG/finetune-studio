// Shared chat-render helpers used by inference.html + project chat pages.
// Renders thinking blocks in a collapsible <details>, response as normal text.

(function () {
  function escHtml(s) {
    const d = document.createElement('div');
    d.textContent = s == null ? '' : String(s);
    return d.innerHTML;
  }

  function renderThinkingBlock(thinking) {
    if (!thinking) return '';
    return `
<details class="mb-2 group">
  <summary class="cursor-pointer text-xs text-gray-400 hover:text-gray-300 select-none list-none flex items-center gap-1">
    <span class="inline-block transition-transform group-open:rotate-90 text-[10px]">▶</span>
    <span class="italic">thinking</span>
    <span class="text-gray-600">(${thinking.length} chars)</span>
  </summary>
  <div class="mt-1 px-3 py-2 bg-gray-900/60 border-l-2 border-purple-700 rounded-r text-xs text-gray-400 whitespace-pre-wrap font-mono">${escHtml(thinking)}</div>
</details>`;
  }

  // Render an assistant message bubble from a server payload {thinking, response}.
  function renderAssistantMessage(payload, extraClass = '') {
    const thinking = (payload && payload.thinking) || '';
    const response = (payload && payload.response) || '';
    const thinkingHtml = renderThinkingBlock(thinking);
    return `<div class="flex justify-start"><div class="bg-gray-700 rounded-lg px-3 py-2 text-sm max-w-[70%] ${extraClass}">${thinkingHtml}${escHtml(response)}</div></div>`;
  }

  window.FTSChat = { escHtml, renderThinkingBlock, renderAssistantMessage };
})();
