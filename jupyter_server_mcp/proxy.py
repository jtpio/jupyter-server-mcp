"""Stdio MCP proxy bridging an MCP client to a running Jupyter MCP server.

The extension writes a runtime info file when it starts (see
``jupyter_server_mcp.runtime``). This module reads those files, picks the
server whose root directory contains the session folder of the MCP client,
and forwards MCP traffic from stdio to the server's HTTP endpoint.

Run it with ``python -m jupyter_server_mcp.proxy``. MCP clients only need
this stable command, regardless of which port the server is actually using.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import sys
from pathlib import Path
from typing import Any

import psutil
from fastmcp.server import create_proxy

from .runtime import list_running_mcp_servers

ENV_URL = "JUPYTER_SERVER_MCP_URL"

logger = logging.getLogger(__name__)


class ProxyError(RuntimeError):
    """Raised when the proxy cannot find or connect to a Jupyter MCP server."""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m jupyter_server_mcp.proxy",
        description=(
            "Bridge stdio MCP traffic to a running Jupyter MCP server. "
            "Auto-discovers the server by reading jpserver-mcp-*.json files "
            "in the Jupyter runtime directory."
        ),
    )
    parser.add_argument(
        "--url",
        default=None,
        help=(
            "Explicit MCP endpoint URL (e.g. http://localhost:3001/mcp). "
            "Overrides auto-discovery. Also settable via the "
            f"{ENV_URL} environment variable."
        ),
    )
    parser.add_argument(
        "--runtime-dir",
        default=None,
        help=(
            "Override the Jupyter runtime directory searched for info files. "
            "Defaults to jupyter_core.paths.jupyter_runtime_dir()."
        ),
    )
    parser.add_argument(
        "--cwd",
        default=None,
        help=(
            "Directory matched against the root directory of each running "
            "server. By default the proxy uses its working directory, or the "
            "working directory of the client that started it when a plugin host "
            "set the former to the plugin folder."
        ),
    )
    return parser.parse_args(argv)


def _is_ancestor(ancestor: Path, descendant: Path) -> bool:
    """Return True if ``ancestor`` equals or contains ``descendant``."""
    try:
        descendant.relative_to(ancestor)
    except ValueError:
        return False
    return True


def _match_score(root_dir: Any, cwd: Path) -> int | None:
    """Return a specificity score if ``cwd`` lives under ``root_dir``.

    Higher scores mean a more specific (deeper) match. ``None`` means
    ``root_dir`` is missing, invalid, or does not contain ``cwd``.
    """
    if not isinstance(root_dir, str) or not root_dir:
        return None
    try:
        root_path = Path(root_dir).resolve()
    except OSError:
        return None
    if not _is_ancestor(root_path, cwd):
        return None
    return len(root_path.parts)


def _describe(server: dict[str, Any]) -> str:
    """Return a short human-readable description of a discovered server."""
    return (
        f"url={server.get('url')!r} "
        f"root_dir={server.get('root_dir')!r} "
        f"pid={server.get('pid')}"
    )


LAUNCHERS = frozenset({"uv", "uvx", "uvw"})
SCRIPT_NAME = "jupyter-server-mcp-proxy"
MAX_PARENTS = 8
WINDOWS = sys.platform == "win32"
NO_SERVERS = (
    "No running Jupyter MCP servers were discovered. Start Jupyter Server with "
    f"the jupyter-server-mcp extension, or pass --url / set ${ENV_URL}."
)
PLUGIN_FOLDER_VARIABLES = (
    "PLUGIN_ROOT",
    "PLUGIN_DATA",
    "COPILOT_PLUGIN_ROOT",
    "COPILOT_PLUGIN_DATA",
)


def _resolve(path: Path) -> Path | None:
    """Return ``path`` with symlinks resolved, or None when that fails."""
    try:
        return path.resolve()
    except (OSError, RuntimeError):
        return None


def _is_launcher(process: psutil.Process) -> bool:
    """True for a process that only started the proxy.

    That is uv/uvx, the console script (a launcher .exe on Windows), or the
    venv ``python.exe`` on Windows, which runs the interpreter as a child.
    """
    stem = Path(process.name()).stem.lower()
    if stem in LAUNCHERS or stem == SCRIPT_NAME:
        return True
    if not WINDOWS:
        return False
    try:
        exe = process.exe()
    except (psutil.Error, OSError):
        return False
    return os.path.normcase(exe) == os.path.normcase(sys.executable)


def _client_cwd() -> Path | None:
    """Return the working directory of the process that started the proxy.

    Launchers between the client and the proxy are skipped. Plugin hosts start
    the proxy in the plugin folder, but the client itself runs in the session
    folder. The walk never goes past the client.
    """
    try:
        process = psutil.Process()
        for _ in range(MAX_PARENTS):
            process = process.parent()
            if process is None:
                return None
            if _is_launcher(process):
                continue
            cwd = process.cwd()
            # psutil returns "" when the directory cannot be determined.
            return Path(cwd) if cwd and Path(cwd).is_absolute() else None
    except (psutil.Error, OSError, NotImplementedError):
        return None
    return None


def _plugin_folders() -> list[Path]:
    """Return the plugin folders that plugin hosts announce in the environment.

    Codex CLI sets ``PLUGIN_ROOT`` and ``PLUGIN_DATA`` for a plugin in the
    Agent Plugins format, GitHub Copilot CLI the ``COPILOT_`` variants.
    """
    folders = (
        _resolve(Path(value))
        for name in PLUGIN_FOLDER_VARIABLES
        if (value := os.environ.get(name))
    )
    return [folder for folder in folders if folder is not None]


def session_dir(cwd: str | os.PathLike[str] | None = None) -> Path:
    """Return the directory to match against the servers.

    An explicit ``cwd`` wins. Otherwise it is the working directory, unless a
    plugin host set it inside the plugin folder or it cannot be read; then it
    is the working directory of the client process that started the proxy,
    unless that one is inside the plugin folder too.
    """
    if cwd is not None:
        return Path(cwd).resolve()
    plugin_folders = _plugin_folders()

    def in_plugin(directory: Path) -> bool:
        return any(_is_ancestor(folder, directory) for folder in plugin_folders)

    own = None
    with contextlib.suppress(OSError):
        own = Path.cwd().resolve()
    if own is not None and not in_plugin(own):
        return own
    client = _client_cwd()
    client = _resolve(client) if client is not None else None
    if client is not None and not in_plugin(client):
        return client
    if own is None:
        reason = "the working directory cannot be read"
    else:
        reason = f"the working directory {own} is inside the plugin folder"
    if client is None:
        client_note = "the client process folder is unknown"
    else:
        too = " too" if own is not None else ""
        client_note = (
            f"the client process folder {client} is inside the plugin folder{too}"
        )
    msg = (
        f"Cannot determine the session folder: {reason}, and {client_note}. "
        f"Start the agent in the project folder, or pass --url / set ${ENV_URL}."
    )
    raise ProxyError(msg)


def select_server(servers: list[dict[str, Any]], cwd: Path) -> dict[str, Any]:
    """Pick the server whose ``root_dir`` is the most specific ancestor of ``cwd``.

    A lone server is held to the same rule, so an agent session started in an
    unrelated folder never reaches a server that serves another project. No
    server, no containing server, or a tie for the deepest match raises
    :class:`ProxyError` with the candidates listed.
    """
    if not servers:
        raise ProxyError(NO_SERVERS)

    scored = [
        (score, server)
        for server in servers
        for score in [_match_score(server.get("root_dir"), cwd)]
        if score is not None
    ]

    if scored:
        scored.sort(key=lambda item: item[0], reverse=True)
        top_score = scored[0][0]
        top_matches = [server for score, server in scored if score == top_score]
        if len(top_matches) == 1:
            return top_matches[0]
        candidates = top_matches
        reason = f"Multiple Jupyter MCP servers match {cwd}"
        hint = f"Pick one explicitly with --url or ${ENV_URL}"
    else:
        candidates = servers
        reason = f"No running Jupyter MCP server serves {cwd}"
        hint = (
            "Start the agent inside the folder a server serves, start a "
            f"Jupyter server in this folder, or pass --url / set ${ENV_URL}"
        )

    listing = "\n".join(f"  - {_describe(server)}" for server in candidates)
    msg = f"{reason}. {hint}:\n{listing}"
    raise ProxyError(msg)


def _validate_explicit_url(url: str, source: str) -> str:
    """Ensure an externally-supplied URL has an http(s) scheme."""
    if "://" not in url:
        msg = (
            f"{source} must be an absolute http(s) URL "
            f"(e.g. http://localhost:3001/mcp), got: {url!r}"
        )
        raise ProxyError(msg)
    scheme = url.split("://", 1)[0].lower()
    if scheme not in {"http", "https"}:
        msg = f"{source} must use http or https, got scheme {scheme!r}: {url!r}"
        raise ProxyError(msg)
    return url


def resolve_url(
    url: str | None = None,
    runtime_dir: str | None = None,
    cwd: str | os.PathLike[str] | None = None,
) -> str:
    """Resolve the MCP URL from explicit input or runtime discovery."""
    if url:
        return _validate_explicit_url(url, source="--url")
    env_url = os.environ.get(ENV_URL)
    if env_url:
        return _validate_explicit_url(env_url, source=f"${ENV_URL}")

    servers = list(list_running_mcp_servers(runtime_dir))
    if not servers:
        raise ProxyError(NO_SERVERS)
    server = select_server(servers, session_dir(cwd))

    endpoint = server.get("url")
    if not isinstance(endpoint, str) or not endpoint:
        msg = f"Discovered MCP server info has no URL: {server!r}"
        raise ProxyError(msg)

    logger.info(
        "Discovered MCP server: %s (root_dir=%s, pid=%s)",
        endpoint,
        server.get("root_dir"),
        server.get("pid"),
    )
    return endpoint


async def run_proxy(url: str) -> None:
    """Run a FastMCP proxy to ``url`` over stdio."""
    proxy = create_proxy(url)
    await proxy.run_async(transport="stdio", show_banner=False)


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m jupyter_server_mcp.proxy``."""
    # MCP stdio traffic occupies stdout, so keep all logging on stderr.
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr, force=True)

    args = _parse_args(argv)

    try:
        url = resolve_url(args.url, args.runtime_dir, args.cwd)
    except ProxyError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    try:
        asyncio.run(run_proxy(url))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
