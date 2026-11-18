'use strict';
(() => {
  const byId = id => document.getElementById(id);
  // Resolve relative to this page so reverse-proxy path prefixes are preserved.
  byId('connect-url').value = new URL('./', location.href).href.replace(/\/$/, '');
  const status = byId('package-check-status');
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
