"""Who may reach this instance: the listen address, the password, API tokens and browser sessions.

State lives under ``$FTS_ROOT`` so the web app, the supervisor and the CLI read the same thing from any cwd:

- ``server.json``        listen address ``{host, port}`` (default 127.0.0.1:7860, this computer only)
- ``auth/auth.json``     mode (``off`` | ``password``), the scrypt password hash, API token hashes (mode 0600)
- ``auth/sessions.sqlite`` browser sessions, stored as sha256 of the cookie value (mode 0600)

No plain password or token is ever written to disk or to a log. ``middleware.AccessMiddleware`` enforces it;
``fts auth`` / ``fts server`` and Settings -> Security change it; ``install.sh --config`` pre-configures it.
"""
from __future__ import annotations
