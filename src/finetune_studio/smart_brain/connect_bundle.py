"""Build the dependency-free MCPB bridge shipped with exported Linux packages."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path


def build_mcpb() -> bytes:
    """A generic bridge, never a copy of corpus data, keys, paths or package identity.

    The selected package must already be installed on the client's Linux machine.
    MCPB hosts collect the two paths; Node only relays the package's stdio.
    This is not a claim of Claude Desktop support on Linux.
    """
    manifest = {
        "manifest_version": "0.3",
        "name": "smart-brain",
        "display_name": "Smart Brain · installed Linux package",
        "version": "1.0.0",
        "description": "Connect an installed Linux knowledge package. Retrieval, not an LLM.",
        "author": {"name": "Finetune Studio"},
        "server": {
            "type": "node",
            "entry_point": "bridge.cjs",
            "mcp_config": {
                "command": "node",
                "args": ["${__dirname}/bridge.cjs", "${user_config.package_dir}", "${user_config.keyfile}"],
            },
        },
        "compatibility": {"platforms": ["linux"], "runtimes": {"node": ">=18"}},
        "tools_generated": True,
        "user_config": {
            "package_dir": {
                "type": "directory", "title": "Installed package folder",
                "description": "Extract the package and run bash install.sh on this Linux machine first.",
                "required": True,
            },
            "keyfile": {
                "type": "file", "title": "Passphrase file",
                "description": "Private passphrase file on this machine (chmod 600); never the passphrase itself.",
                "required": True,
            },
        },
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
        # .cjs isolates CommonJS from any parent directory's package.json type=module.
        archive.writestr("bridge.cjs", Path(__file__).with_name("mcpb_bridge.js").read_bytes())
        archive.writestr("README.md", (
            "# Smart Brain MCPB bridge\n\n"
            "Requires an already installed Linux package and Node >=18 on the client's machine.\n"
            "Install this bundle in an MCPB-capable host and select the package folder and private keyfile.\n"
            "The bridge starts that package's run-mcp.sh; no corpus, model, key or passphrase is bundled.\n"
            "It does not install Python dependencies or provide an LLM.\n"
            "Linux only. Claude Desktop's official MCPB hosts are macOS/Windows, outside the current\n"
            "product scope: this bundle is not certified for Claude Desktop and is unsigned.\n"
            "After installation, call rag_info and rag_search in the host to verify access.\n"
        ))
    return output.getvalue()
