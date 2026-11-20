/* All recipe editing stays in this page. No credentials, storage or network probes. */
function connectRecipes(client, folder, keyfile, address, listenerPort = '') {
  const quote = value => "'" + value.replace(/'/g, "'\\''") + "'";
  if (!folder.startsWith('/') || !keyfile.startsWith('/') || /[\r\n\0]/.test(folder + keyfile)) {
    throw new Error('Use absolute package and passphrase-file paths without line breaks.');
  }
  const url = new URL(address);
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
    throw new Error('Use an HTTP(S) package address without credentials, query or fragment.');
  }
  // Allow reverse-proxy path prefixes; accept an already pasted /mcp endpoint.
  const base = url.href.replace(/\/$/, '').replace(/\/mcp$/, '');
  // Only a direct local HTTP URL supplies a listener port. Public/HTTPS/prefixed
  // addresses may terminate at a proxy: never turn its port into server config.
  const directLocal = url.protocol === 'http:' &&
    ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname) &&
    ['', '/', '/mcp', '/mcp/'].includes(url.pathname);
  const port = listenerPort || (directLocal ? url.port || '80' : '');
  if ((listenerPort !== '' && /[^0-9]/.test(listenerPort)) ||
      (port !== '' && (Number(port) < 1 || Number(port) > 65535))) {
    throw new Error('Use a package listener port from 1 to 65535.');
  }
  const args = [folder.replace(/\/$/, '') + '/run-mcp.sh', '--keyfile', keyfile];
  const stdio = {command: 'bash', args};
  const json = value => JSON.stringify(value, null, 2);
  const standard = json({mcpServers: {'smart-brain': stdio}});
  const rows = {
    'claude-code': ['Run this command, then use /mcp to inspect the server tools.',
      'https://code.claude.com/docs/en/mcp', 'claude mcp add smart-brain -- bash ' + args.map(quote).join(' ')],
    'claude-desktop': ['Manual stdio configuration: merge this entry into claude_desktop_config.json and restart the client. Claude Desktop is not certified on Linux. Exported packages offer a Linux-only MCPB bridge for compatible hosts, not a working one-click Claude Desktop installer.',
      'https://modelcontextprotocol.io/quickstart/user', standard],
    cursor: ['Merge into .cursor/mcp.json (project) or ~/.cursor/mcp.json (user), then enable the server in MCP settings.',
      'https://cursor.com/docs/mcp', standard],
    vscode: ['Merge into .vscode/mcp.json. Start the server through MCP: List Servers, then enable its tools in Copilot agent mode.',
      'https://code.visualstudio.com/docs/agent-customization/mcp-servers',
      json({servers: {'smart-brain': {type: 'stdio', ...stdio}}})],
    windsurf: ['Manual stdio recipe for MCP clients using mcpServers. Windsurf-specific current setup is not verified; consult your installed version before applying.',
      'https://docs.windsurf.com/windsurf/cascade/mcp', standard],
    gemini: ['Merge into ~/.gemini/settings.json, restart Gemini CLI, then inspect /mcp.',
      'https://geminicli.com/docs/tools/mcp-server/', standard],
    lmstudio: ['Open Program → Install → Edit mcp.json; merge this entry, enable the server, and use a tool-capable model.',
      'https://lmstudio.ai/docs/app/mcp', standard],
    openwebui: ['Open WebUI 0.6.31+: as admin, open Settings → Integrations → External Tool Servers → Add Connection. Choose MCP (Streamable HTTP), not OpenAPI. Use this URL and bearer auth if enabled. Start the package HTTP server first; its address must be reachable from Open WebUI, not just your browser.',
      'https://docs.openwebui.com/features/extensibility/mcp/', base + '/mcp'],
    'knowledge-api': ['Knowledge + your LLM: configure the package operator’s provider, then add the base URL below as an OpenAI-compatible connection in your client. Select smart-brain, not the upstream model. The client key is the package’s RAG_AUTH_TOKEN, never the upstream API key. Text chat only; SSE is buffered until retrieval and generation finish. Raw retrieval needs no LLM. Generic SDK support is tested separately from vendor UI certification.',
      'https://platform.openai.com/docs/api-reference/chat/create',
      'Base URL: ' + base + '/v1\nModel: smart-brain\nAPI key: package server access token (RAG_AUTH_TOKEN)\n\n' +
      '# On the package host, configure YOUR provider before starting:\n' +
      'export RAG_LLM_PROVIDER=openai\nexport RAG_LLM_BASE_URL=http://127.0.0.1:11434/v1\n' +
      'export RAG_LLM_MODEL=YOUR_INSTALLED_MODEL\n# Set RAG_LLM_API_KEY privately if your provider needs it.\n' +
      '# For Claude use RAG_LLM_PROVIDER=anthropic and its /v1 base URL.'],
    chatgpt: ['Use the optional Actions setup section below to generate an OpenAPI schema, a retrieval-only relay and opt-in manual tunnel commands. Nothing is published automatically. Cloud clients need public HTTPS, the separate relay bearer token and vendor-account setup. This is not OAuth or actual ChatGPT certification.',
      'https://developers.openai.com/api/docs/actions/introduction', 'Generate the schema in Actions setup below, then configure API Key → Bearer in the GPT editor.'],
    ollama: ['Ollama is a model runtime, not the MCP host in this setup. Use Open WebUI or another MCP host with an Ollama tool-capable model, then connect the package there.',
      'https://docs.ollama.com/capabilities/tool-calling', 'Choose Open WebUI above for the package connection.']
  };
  if (!rows[client]) throw new Error('Choose a supported client.');
  const [instructions, docs, config] = rows[client];
  const health = 'curl --fail-with-body --connect-timeout 3 --max-time 15 -H "Authorization: Bearer ${RAG_AUTH_TOKEN:-}" ';
  const handshake = JSON.stringify({jsonrpc:'2.0', id:1, method:'initialize', params:{
    protocolVersion:'2024-11-05', capabilities:{}, clientInfo:{name:'connect-check',version:'1.0'}}});
  const apiChecks = health + quote(base + '/v1/models') + '\n\n' + health +
    "-H 'Content-Type: application/json' --data " + quote(JSON.stringify({query:'What is in this package?', top_k:3, images:true})) +
    ' ' + quote(base + '/v1/retrieve') + '\n\n' + health.replace('--max-time 15', '--max-time 330') +
    "-H 'Content-Type: application/json' --data " + quote(JSON.stringify({model:'smart-brain',
      messages:[{role:'user',content:'What is in this package?'}], stream:false})) + ' ' + quote(base + '/v1/chat/completions');
  return {instructions, docs, config, availableSetup: !['chatgpt', 'ollama'].includes(client),
    start: port ? 'bash ' + quote(folder.replace(/\/$/, '') + '/run-http.sh') +
      ' --keyfile ' + quote(keyfile) + ' --port ' + Number(port) + (client === 'knowledge-api' ? ' --answer' : '') :
      'Enter the package listener port above to generate a startup command. A public or proxy URL does not reveal it.',
    test: client === 'knowledge-api' ? apiChecks : health + quote(base + '/health') + '\n\n' + health +
      '-H \'Content-Type: application/json\' -H \'Accept: application/json, text/event-stream\' --data ' +
      quote(handshake) + ' ' + quote(base + '/mcp')};
}

if (typeof module !== 'undefined') module.exports = {connectRecipes};
if (typeof document !== 'undefined') (() => {
  const byId = id => document.getElementById('connect-' + id);
  if (!byId('builder')) return;
  let recipe;
  const render = () => {
    try {
      recipe = connectRecipes(byId('client').value, byId('path').value, byId('keyfile').value,
        byId('url').value, byId('port').value);
      for (const key of ['instructions', 'config', 'start', 'test']) byId(key).textContent = recipe[key];
      byId('docs').href = recipe.docs;
      byId('copy').disabled = !recipe.availableSetup;
      byId('copy').textContent = recipe.availableSetup ? 'Copy setup' : 'No direct setup';
      byId('copy-test').disabled = false;
      byId('feedback').textContent = recipe.availableSetup ? 'Setup updated. Client connection not checked.' : 'No direct setup for this client. Read the connection options above.';
    } catch (error) {
      recipe = null;
      for (const key of ['config', 'start', 'test']) byId(key).textContent = '';
      byId('copy').disabled = byId('copy-test').disabled = true;
      byId('feedback').textContent = error.message;
    }
  };
  for (const field of ['client', 'path', 'keyfile', 'url', 'port']) byId(field).addEventListener('input', render);
  const copy = async key => {
    if (!recipe) return;
    try {
      await navigator.clipboard.writeText(recipe[key]);
      byId('feedback').textContent = key === 'test' ? 'Terminal checks copied. Run them on the client machine.' : 'Setup copied. Add it to your client; connection not checked.';
    } catch (_) {
      byId('feedback').textContent = 'Clipboard unavailable. Select and copy the displayed text manually.';
    }
  };
  byId('copy').addEventListener('click', () => copy('config'));
  byId('copy-test').addEventListener('click', () => copy('test'));
  render();
})();
