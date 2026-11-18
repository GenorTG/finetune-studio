'use strict';
(() => {
  const byId = id => document.getElementById(id);
  // Resolve relative to this page so reverse-proxy path prefixes are preserved.
  byId('connect-url').value = new URL('./', location.href).href.replace(/\/$/, '');
  const status = byId('package-check-status');
  for (const [id, answer] of [['package-retrieve', false], ['package-answer', true]]) {
    byId(id).addEventListener('click', async () => {
      const feedback = byId('package-api-status');
      const buttons = [byId('package-retrieve'), byId('package-answer')];
      buttons.forEach(button => { button.disabled = true; });
      feedback.textContent = answer ? 'Retrieving evidence and asking the operator’s LLM…' : 'Searching this package…';
      byId('package-api-result').textContent = '';
      try {
        const token = byId('package-token').value;
        if (token && location.protocol !== 'https:' &&
            !['localhost', '127.0.0.1', '[::1]'].includes(location.hostname)) {
          throw new Error('Use HTTPS before sending a server access token over a network.');
        }
        const query = byId('package-query').value.trim();
        if (!query) throw new Error('Enter a question about the package.');
        const headers = {'Content-Type': 'application/json'};
        if (token) headers.Authorization = 'Bearer ' + token;
        const response = await fetch(new URL(answer ? 'v1/chat/completions' : 'v1/retrieve', location.href), {
          method: 'POST', headers, body: JSON.stringify(answer ? {
            model: 'smart-brain', messages: [{role: 'user', content: query}], stream: false
          } : {query, top_k: 5, images: true}),
          signal: AbortSignal.timeout(330000), credentials: 'omit', cache: 'no-store', redirect: 'error'
        });
        const data = await response.json();
        if (!response.ok) throw new Error('HTTP ' + response.status + ': ' +
          (data.error?.message || data.error || 'Package request failed.'));
        byId('package-api-result').textContent = JSON.stringify(data, null, 2);
        feedback.textContent = answer ? (data.knowledge?.generated ?
          'Answer API passed: operator LLM returned an answer with retrieval evidence. Client app not checked.' :
          'No matching knowledge: package returned a retrieval miss, not an LLM-generated answer.') :
          'Retrieval API passed: ' + data.hits.length + ' passage(s). No LLM needed. Client app not checked.';
      } catch (error) {
        feedback.textContent = 'API check failed: ' + error.message;
      } finally {
        buttons.forEach(button => { button.disabled = false; });
      }
    });
  }
  byId('package-check').addEventListener('click', async () => {
    const button = byId('package-check');
    button.disabled = true;
    status.textContent = 'Checking this package’s health and MCP tools…';
    const token = byId('package-token').value;
    try {
      if (token && location.protocol !== 'https:' &&
          !['localhost', '127.0.0.1', '[::1]'].includes(location.hostname)) {
        throw new Error('Use HTTPS before sending a server access token over a network.');
      }
      const headers = {'Accept': 'application/json, text/event-stream'};
      if (token) headers.Authorization = 'Bearer ' + token;
      const request = async (endpoint, body) => {
        const response = await fetch(new URL(endpoint, location.href), {
          method: body ? 'POST' : 'GET', headers: body ? {...headers, 'Content-Type': 'application/json'} : headers,
          ...(body ? {body: JSON.stringify(body)} : {}), signal: AbortSignal.timeout(15000),
          credentials: 'omit', cache: 'no-store', redirect: 'error'
        });
        if (!response.ok) throw new Error(response.status === 401 ?
          'Server token missing or wrong (401).' : 'Package request failed (' + response.status + ').');
        if (response.status === 202) return null;
        const data = await response.json();
        if (data.error) throw new Error('Package returned an MCP error.');
        return data;
      };
      const health = await request('health');
      if (health.ok !== true) throw new Error('Package health response is invalid.');
      const init = await request('mcp', {jsonrpc: '2.0', id: 1, method: 'initialize', params: {
        protocolVersion: '2024-11-05', capabilities: {}, clientInfo: {name: 'package-connect', version: '1.0'}
      }});
      if (!init.result?.serverInfo) throw new Error('MCP initialization did not return serverInfo.');
      await request('mcp', {jsonrpc: '2.0', method: 'notifications/initialized'});
      const list = await request('mcp', {jsonrpc: '2.0', id: 2, method: 'tools/list'});
      if (!['rag_info', 'rag_search'].every(name => list.result?.tools?.some(tool => tool.name === name))) {
        throw new Error('MCP did not advertise the required Smart Brain tools.');
      }
      const info = await request('mcp', {jsonrpc: '2.0', id: 3, method: 'tools/call',
        params: {name: 'rag_info', arguments: {}}});
      if (info.result?.isError || !info.result?.content?.length) throw new Error('rag_info did not succeed.');
      status.textContent = 'Package check passed: health, MCP initialization, tools and rag_info. Your selected client is not checked; call rag_info and rag_search there too.';
    } catch (error) {
      status.textContent = 'Package check failed: ' + error.message;
    } finally {
      button.disabled = false;
    }
  });
})();
