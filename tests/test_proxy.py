"""Tests for the stdio proxy discovery and CLI logic."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import psutil
import pytest

from jupyter_server_mcp import proxy, runtime

client_cwd = proxy._client_cwd


@pytest.fixture(autouse=True)
def no_client_process(monkeypatch):
    """Keep discovery deterministic: no client process and no plugin folder."""
    monkeypatch.setattr(proxy, "_client_cwd", lambda: None)
    for name in proxy.PLUGIN_FOLDER_VARIABLES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def isolated_runtime_dir(tmp_path, monkeypatch):
    """Point discovery at a temporary runtime directory with live pids."""
    monkeypatch.setattr(runtime, "check_pid", lambda _pid: True)
    monkeypatch.delenv(proxy.ENV_URL, raising=False)
    return tmp_path


def _publish_server(
    runtime_dir: Path,
    pid: int,
    *,
    root_dir: Path | str,
    port: int = 3001,
) -> None:
    """Write an MCP info file in ``runtime_dir`` for ``pid``."""
    path = runtime.info_file_path(runtime_dir, pid)
    runtime.write_info_file(
        path,
        {
            "pid": pid,
            "host": "localhost",
            "port": port,
            "url": f"http://localhost:{port}/mcp",
            "name": f"Jupyter MCP Server {pid}",
            "root_dir": str(root_dir),
        },
    )


class TestResolveUrl:
    """Tests for ``resolve_url``."""

    def test_explicit_url_wins(self, isolated_runtime_dir):
        """An explicit URL short-circuits discovery."""
        url = proxy.resolve_url(
            url="http://explicit:9999/mcp",
            runtime_dir=str(isolated_runtime_dir),
            cwd=str(isolated_runtime_dir),
        )
        assert url == "http://explicit:9999/mcp"

    def test_env_var_wins(self, isolated_runtime_dir, monkeypatch):
        """The env var is honored when no explicit URL is passed."""
        monkeypatch.setenv(proxy.ENV_URL, "http://env:1234/mcp")
        url = proxy.resolve_url(
            runtime_dir=str(isolated_runtime_dir),
            cwd=str(isolated_runtime_dir),
        )
        assert url == "http://env:1234/mcp"

    def test_single_server_discovery(self, isolated_runtime_dir):
        """A single running server is selected when it serves ``cwd``."""
        _publish_server(isolated_runtime_dir, 101, root_dir=isolated_runtime_dir)

        url = proxy.resolve_url(
            runtime_dir=str(isolated_runtime_dir),
            cwd=str(isolated_runtime_dir),
        )

        assert url == "http://localhost:3001/mcp"

    def test_single_server_outside_cwd_raises(self, isolated_runtime_dir, tmp_path):
        """A single running server is refused when ``cwd`` is not under its root."""
        root = tmp_path / "project"
        other = tmp_path / "other"
        root.mkdir()
        other.mkdir()
        _publish_server(isolated_runtime_dir, 101, root_dir=root)

        with pytest.raises(proxy.ProxyError, match="No running Jupyter MCP server"):
            proxy.resolve_url(
                runtime_dir=str(isolated_runtime_dir),
                cwd=str(other),
            )

    def test_no_servers_raises(self, isolated_runtime_dir):
        """An empty runtime directory produces a helpful error."""
        with pytest.raises(proxy.ProxyError, match="No running Jupyter MCP servers"):
            proxy.resolve_url(
                runtime_dir=str(isolated_runtime_dir),
                cwd=str(isolated_runtime_dir),
            )

    def test_discovered_url_missing_raises(self, isolated_runtime_dir):
        """A malformed info dict raises a clear error."""
        path = runtime.info_file_path(isolated_runtime_dir, 17)
        runtime.write_info_file(
            path,
            {
                "pid": 17,
                "host": "localhost",
                "port": 3001,
                "root_dir": str(isolated_runtime_dir),
            },
        )

        with pytest.raises(proxy.ProxyError, match="has no URL"):
            proxy.resolve_url(
                runtime_dir=str(isolated_runtime_dir),
                cwd=str(isolated_runtime_dir),
            )

    @pytest.mark.parametrize(
        ("source", "value", "match"),
        [
            ("arg", "localhost:3001/mcp", "absolute http"),
            ("arg", "file:///etc/passwd", "http or https"),
            ("env", "ftp://example.com/mcp", "http or https"),
        ],
    )
    def test_invalid_urls_are_rejected(
        self, isolated_runtime_dir, monkeypatch, source, value, match
    ):
        """Explicit and env-var URLs must be absolute http(s) URLs."""
        if source == "env":
            monkeypatch.setenv(proxy.ENV_URL, value)
            url = None
        else:
            url = value

        with pytest.raises(proxy.ProxyError, match=match):
            proxy.resolve_url(
                url=url,
                runtime_dir=str(isolated_runtime_dir),
                cwd=str(isolated_runtime_dir),
            )

    def test_unserved_cwd_does_not_fall_back_to_the_client(
        self, isolated_runtime_dir, tmp_path, monkeypatch
    ):
        """A readable cwd outside a plugin folder is the session folder, full stop."""
        root = tmp_path / "project"
        other = tmp_path / "other"
        root.mkdir()
        other.mkdir()
        _publish_server(isolated_runtime_dir, 101, root_dir=root)
        monkeypatch.chdir(other)
        monkeypatch.setattr(proxy, "_client_cwd", lambda: root)

        with pytest.raises(proxy.ProxyError, match="No running Jupyter MCP server"):
            proxy.resolve_url(runtime_dir=str(isolated_runtime_dir))

    def test_plugin_folder_cwd_is_ignored(
        self, isolated_runtime_dir, tmp_path, monkeypatch
    ):
        """A cwd inside the plugin folder must not pick a server that contains it."""
        home = tmp_path / "home"
        plugin = home / ".codex" / "plugins" / "xtralab"
        project = home / "project"
        plugin.mkdir(parents=True)
        project.mkdir()
        _publish_server(isolated_runtime_dir, 101, root_dir=home, port=3001)
        _publish_server(isolated_runtime_dir, 102, root_dir=project, port=3002)
        monkeypatch.chdir(plugin)
        monkeypatch.setenv("PLUGIN_ROOT", str(plugin))
        monkeypatch.setattr(proxy, "_client_cwd", lambda: project)

        url = proxy.resolve_url(runtime_dir=str(isolated_runtime_dir))

        assert url == "http://localhost:3002/mcp"

    def test_plugin_folder_cwd_never_reaches_an_unrelated_server(
        self, isolated_runtime_dir, tmp_path, monkeypatch
    ):
        """With the session unserved, a server that contains the plugin folder loses."""
        home = tmp_path / "home"
        plugin = home / ".copilot" / "plugins" / "xtralab"
        session = tmp_path / "work" / "project"
        plugin.mkdir(parents=True)
        session.mkdir(parents=True)
        _publish_server(isolated_runtime_dir, 101, root_dir=home)
        monkeypatch.chdir(plugin)
        monkeypatch.setenv("COPILOT_PLUGIN_ROOT", str(plugin))
        monkeypatch.setattr(proxy, "_client_cwd", lambda: session)

        with pytest.raises(proxy.ProxyError, match="No running Jupyter MCP server"):
            proxy.resolve_url(runtime_dir=str(isolated_runtime_dir))

    def test_explicit_cwd_has_no_fallback(
        self, isolated_runtime_dir, tmp_path, monkeypatch
    ):
        """An explicit ``cwd`` is the only directory that is matched."""
        root = tmp_path / "project"
        other = tmp_path / "other"
        root.mkdir()
        other.mkdir()
        _publish_server(isolated_runtime_dir, 101, root_dir=root)
        monkeypatch.setattr(proxy, "_client_cwd", lambda: root)

        with pytest.raises(proxy.ProxyError, match="No running Jupyter MCP server"):
            proxy.resolve_url(runtime_dir=str(isolated_runtime_dir), cwd=str(other))

    def test_client_in_plugin_folder_too_raises(
        self, isolated_runtime_dir, tmp_path, monkeypatch
    ):
        """A wrapper process left in the plugin folder is not taken as the client."""
        home = tmp_path / "home"
        plugin = home / ".codex" / "plugins" / "xtralab"
        plugin.mkdir(parents=True)
        _publish_server(isolated_runtime_dir, 101, root_dir=home)
        monkeypatch.chdir(plugin)
        monkeypatch.setenv("PLUGIN_ROOT", str(plugin))
        monkeypatch.setattr(proxy, "_client_cwd", lambda: plugin / "skills")

        with pytest.raises(proxy.ProxyError, match="Cannot determine the session"):
            proxy.resolve_url(runtime_dir=str(isolated_runtime_dir))

    def test_cwd_argument_steers_selection(self, isolated_runtime_dir, tmp_path):
        """Passing ``cwd`` must disambiguate between multiple running servers."""
        root_a = tmp_path / "alpha"
        root_b = tmp_path / "beta"
        root_a.mkdir()
        root_b.mkdir()
        _publish_server(isolated_runtime_dir, 201, root_dir=root_a, port=3101)
        _publish_server(isolated_runtime_dir, 202, root_dir=root_b, port=3102)

        url_a = proxy.resolve_url(
            runtime_dir=str(isolated_runtime_dir),
            cwd=str(root_a),
        )
        url_b = proxy.resolve_url(
            runtime_dir=str(isolated_runtime_dir),
            cwd=str(root_b),
        )

        assert url_a == "http://localhost:3101/mcp"
        assert url_b == "http://localhost:3102/mcp"


class TestSessionDir:
    """Tests for the directory the proxy matches against the servers."""

    def test_explicit_cwd_is_the_only_candidate(self, tmp_path, monkeypatch):
        """An explicit directory disables the client process fallback."""
        monkeypatch.setattr(proxy, "_client_cwd", lambda: tmp_path)

        assert proxy.session_dir(tmp_path / "given") == (tmp_path / "given").resolve()

    def test_readable_cwd_alone(self, tmp_path, monkeypatch):
        """Outside a plugin folder, the working directory is the session folder."""
        own = tmp_path / "cwd"
        own.mkdir()
        monkeypatch.chdir(own)
        monkeypatch.setattr(proxy, "_client_cwd", lambda: tmp_path / "client")

        assert proxy.session_dir() == own.resolve()

    @pytest.mark.parametrize("variable", proxy.PLUGIN_FOLDER_VARIABLES)
    def test_cwd_inside_plugin_folder_gives_the_client_cwd(
        self, tmp_path, monkeypatch, variable
    ):
        """A cwd that a plugin host set inside the plugin folder is not a session."""
        plugin = tmp_path / "plugin"
        inside = plugin / "skills"
        client = tmp_path / "project"
        inside.mkdir(parents=True)
        client.mkdir()
        monkeypatch.chdir(inside)
        monkeypatch.setenv(variable, str(plugin))
        monkeypatch.setattr(proxy, "_client_cwd", lambda: client)

        assert proxy.session_dir() == client.resolve()

    def test_cwd_outside_plugin_folder_is_kept(self, tmp_path, monkeypatch):
        """A plugin folder elsewhere does not change the rule."""
        plugin = tmp_path / "plugin"
        session = tmp_path / "project"
        plugin.mkdir()
        session.mkdir()
        monkeypatch.chdir(session)
        monkeypatch.setenv("PLUGIN_ROOT", str(plugin))
        monkeypatch.setattr(proxy, "_client_cwd", lambda: tmp_path / "client")

        assert proxy.session_dir() == session.resolve()

    def test_unreadable_cwd_gives_the_client_cwd(self, tmp_path, monkeypatch):
        """A deleted working directory falls back to the client process."""
        client = tmp_path / "project"
        client.mkdir()

        def raise_not_found():
            raise FileNotFoundError

        monkeypatch.setattr(os, "getcwd", raise_not_found)
        monkeypatch.setattr(proxy, "_client_cwd", lambda: client)

        assert proxy.session_dir() == client.resolve()

    def test_no_candidate_raises_with_the_reasons(self, tmp_path, monkeypatch):
        """Without a usable directory the error says what was tried."""
        plugin = tmp_path / "plugin"
        plugin.mkdir()
        monkeypatch.chdir(plugin)
        monkeypatch.setenv("COPILOT_PLUGIN_ROOT", str(plugin))
        monkeypatch.setattr(proxy, "_client_cwd", lambda: None)

        with pytest.raises(proxy.ProxyError) as excinfo:
            proxy.session_dir()

        message = str(excinfo.value)
        assert str(plugin.resolve()) in message
        assert "client process folder is unknown" in message
        assert proxy.ENV_URL in message

    def test_no_servers_is_reported_before_the_session_folder(
        self, isolated_runtime_dir, tmp_path, monkeypatch
    ):
        """With no server at all, the plugin folder problem is not the message."""
        plugin = tmp_path / "plugin"
        plugin.mkdir()
        monkeypatch.chdir(plugin)
        monkeypatch.setenv("PLUGIN_ROOT", str(plugin))

        with pytest.raises(proxy.ProxyError, match="No running Jupyter MCP servers"):
            proxy.resolve_url(runtime_dir=str(isolated_runtime_dir))


class FakeProcess:
    """A psutil-like process with a name, a cwd (or an error), a parent, an exe."""

    def __init__(self, name, cwd=None, parent=None, exe="/usr/bin/tool"):
        self._name = name
        self._cwd = cwd
        self._parent = parent
        self._exe = exe

    def name(self):
        return self._name

    def cwd(self):
        if isinstance(self._cwd, Exception):
            raise self._cwd
        return self._cwd

    def parent(self):
        if isinstance(self._parent, Exception):
            raise self._parent
        return self._parent

    def exe(self):
        if isinstance(self._exe, Exception):
            raise self._exe
        return self._exe


def _chain(monkeypatch, *parents):
    """Make ``psutil.Process()`` return a proxy process with the given parents.

    Each parent is ``(name, cwd)`` or ``(name, cwd, exe)``.
    """
    process = None
    for parent in reversed(parents):
        process = FakeProcess(*parent[:2], process, *parent[2:])
    monkeypatch.setattr(
        proxy.psutil, "Process", lambda: FakeProcess("python", "/plugin", process)
    )


class TestClientCwd:
    """Tests for the lookup of the client process directory."""

    def test_skips_uv_and_stops_at_the_client(self, monkeypatch, tmp_path):
        """uv and uvx are launchers; the first other ancestor is the client."""
        project = tmp_path / "project"
        _chain(
            monkeypatch,
            ("uv", "/plugin"),
            ("UVX.EXE", "/plugin"),
            ("codex", str(project)),
            ("zsh", str(tmp_path)),
        )

        assert client_cwd() == project

    def test_skips_the_windows_launchers(self, monkeypatch, tmp_path):
        """On Windows the script .exe and the venv python.exe sit between uv and us."""
        project = tmp_path / "project"
        monkeypatch.setattr(proxy, "WINDOWS", True)
        _chain(
            monkeypatch,
            ("python.exe", "/plugin", sys.executable),
            ("jupyter-server-mcp-proxy.exe", "/tmp/trampoline"),
            ("uv.exe", "/plugin"),
            ("copilot.exe", str(project)),
        )

        assert client_cwd() == project

    def test_python_client_is_not_a_launcher_elsewhere(self, monkeypatch, tmp_path):
        """Outside Windows a Python client in the same env is the client."""
        project = tmp_path / "project"
        monkeypatch.setattr(proxy, "WINDOWS", False)
        _chain(
            monkeypatch,
            ("python", str(project), sys.executable),
            ("zsh", str(tmp_path)),
        )

        assert client_cwd() == project

    def test_direct_parent_is_the_client(self, monkeypatch, tmp_path):
        """Without a launcher, the parent process is the client."""
        project = tmp_path / "project"
        _chain(monkeypatch, ("claude", str(project)), ("zsh", str(tmp_path)))

        assert client_cwd() == project

    def test_unreadable_exe_means_client(self, monkeypatch, tmp_path):
        """A parent whose executable cannot be read is still the client."""
        project = tmp_path / "project"
        monkeypatch.setattr(proxy, "WINDOWS", True)
        _chain(
            monkeypatch,
            ("agent", str(project), psutil.AccessDenied(4)),
            ("zsh", str(tmp_path)),
        )

        assert client_cwd() == project

    def test_unreadable_client_cwd_gives_none(self, monkeypatch):
        """An AccessDenied on the client is not a reason to look further."""
        _chain(
            monkeypatch,
            ("uv", "/plugin"),
            ("codex", psutil.AccessDenied(5)),
            ("zsh", "/home"),
        )

        assert client_cwd() is None

    @pytest.mark.parametrize("cwd", ["", "relative/dir"])
    def test_unusable_client_cwd_gives_none(self, monkeypatch, cwd):
        """An empty or relative cwd from psutil is not a directory to match."""
        _chain(monkeypatch, ("codex", cwd), ("zsh", "/home"))

        assert client_cwd() is None

    def test_gone_parent_gives_none(self, monkeypatch):
        """A parent that disappeared ends the lookup."""
        process = FakeProcess("uv", "/plugin", psutil.NoSuchProcess(3))
        monkeypatch.setattr(
            proxy.psutil, "Process", lambda: FakeProcess("python", "/plugin", process)
        )

        assert client_cwd() is None

    def test_end_of_chain_gives_none(self, monkeypatch):
        """Reaching the top of the process tree ends the lookup."""
        _chain(monkeypatch, ("uv", "/plugin"))

        assert client_cwd() is None

    def test_launcher_chain_is_bounded(self, monkeypatch):
        """A long chain of launchers is not followed forever."""
        _chain(monkeypatch, *[("uv", "/plugin")] * (proxy.MAX_PARENTS + 1))

        assert client_cwd() is None

    def test_own_process_lookup_failure_gives_none(self, monkeypatch):
        """A psutil error on the current process disables the lookup."""

        def raise_no_such_process():
            raise psutil.NoSuchProcess(0)

        monkeypatch.setattr(proxy.psutil, "Process", raise_no_such_process)

        assert client_cwd() is None


def _run_in_shell(session_dir: Path, code: str, env: dict) -> str:
    """Run ``code`` in a child interpreter whose parent is a shell in ``session_dir``.

    The shell must not replace itself with the interpreter: dash does so for
    a last command, hence the trailing ``exit``.
    """
    python = sys.executable
    if sys.platform == "win32":
        command: str | list[str] = f'cmd /s /c ""{python}" -c "{code}""'
    else:
        command = ["sh", "-c", '"$0" -c "$1"; exit $?', python, code]
    child_env = {k: v for k, v in os.environ.items() if k != proxy.ENV_URL}
    return subprocess.run(
        command,
        cwd=session_dir,
        capture_output=True,
        text=True,
        check=True,
        env={**child_env, **env},
    ).stdout.strip()


class TestRealProcessTree:
    """Tests that walk a real process tree."""

    def test_shell_in_the_session_folder_is_the_client(self, tmp_path):
        """A child started in the plugin folder sees the shell as its client."""
        session = tmp_path / "session"
        plugin = tmp_path / "plugin"
        session.mkdir()
        plugin.mkdir()
        code = (
            "import os; os.chdir(os.environ['PLUGIN_ROOT']); "
            "from jupyter_server_mcp.proxy import _client_cwd; print(_client_cwd())"
        )

        output = _run_in_shell(session, code, {"PLUGIN_ROOT": str(plugin)})

        assert Path(output).resolve() == session.resolve()

    def test_plugin_host_resolves_the_session_server(self, tmp_path):
        """In a plugin folder under a served home, the session server still wins."""
        home = tmp_path / "home"
        plugin = home / "plugins" / "xtralab"
        session = home / "project"
        runtime_dir = tmp_path / "runtime"
        for directory in (plugin, session, runtime_dir):
            directory.mkdir(parents=True)
        _publish_server(runtime_dir, os.getpid(), root_dir=home, port=3001)
        _publish_server(runtime_dir, os.getppid(), root_dir=session, port=3002)
        code = (
            "import os; os.chdir(os.environ['PLUGIN_ROOT']); "
            "from jupyter_server_mcp.proxy import resolve_url; "
            f"print(resolve_url(runtime_dir={str(runtime_dir)!r}))"
        )

        output = _run_in_shell(session, code, {"PLUGIN_ROOT": str(plugin)})

        assert output == "http://localhost:3002/mcp"


class TestSelectServer:
    """Tests for the server selection logic."""

    def test_picks_ancestor_with_highest_specificity(self, tmp_path):
        """Deeper ancestor root_dirs should win over shallower ones."""
        shallow = tmp_path / "projects"
        deep = shallow / "alpha"
        deep.mkdir(parents=True)
        cwd = deep / "src"
        cwd.mkdir()

        servers = [
            {"pid": 1, "url": "http://localhost:3001/mcp", "root_dir": str(shallow)},
            {"pid": 2, "url": "http://localhost:3002/mcp", "root_dir": str(deep)},
        ]

        chosen = proxy.select_server(servers, cwd.resolve())

        assert chosen["pid"] == 2

    def test_multiple_servers_none_containing_cwd_raises(self, tmp_path):
        """If no server contains the cwd, the user must disambiguate."""
        other_a = tmp_path / "a"
        other_b = tmp_path / "b"
        other_a.mkdir()
        other_b.mkdir()
        cwd = tmp_path / "c"
        cwd.mkdir()

        servers = [
            {"pid": 1, "url": "http://localhost:3001/mcp", "root_dir": str(other_a)},
            {"pid": 2, "url": "http://localhost:3002/mcp", "root_dir": str(other_b)},
        ]

        with pytest.raises(proxy.ProxyError, match="No running Jupyter MCP server"):
            proxy.select_server(servers, cwd.resolve())

    def test_ambiguous_same_root_dir(self, tmp_path):
        """Two servers with the same root_dir produce a disambiguation error."""
        root = tmp_path / "shared"
        root.mkdir()

        servers = [
            {"pid": 1, "url": "http://localhost:3001/mcp", "root_dir": str(root)},
            {"pid": 2, "url": "http://localhost:3002/mcp", "root_dir": str(root)},
        ]

        with pytest.raises(proxy.ProxyError, match="Multiple Jupyter MCP servers"):
            proxy.select_server(servers, root.resolve())

    @pytest.mark.parametrize("relative_cwd", ["", "src/pkg"])
    def test_single_server_containing_cwd_is_selected(self, tmp_path, relative_cwd):
        """A lone server is selected when cwd equals or sits below its root_dir."""
        root = tmp_path / "project"
        cwd = root / relative_cwd
        cwd.mkdir(parents=True)

        servers = [
            {"pid": 1, "url": "http://localhost:3001/mcp", "root_dir": str(root)},
        ]

        assert proxy.select_server(servers, cwd.resolve())["pid"] == 1

    def test_single_server_outside_cwd_raises(self, tmp_path):
        """A lone server is refused when cwd is not under its root_dir."""
        root = tmp_path / "project"
        unrelated = tmp_path / "unrelated"
        root.mkdir()
        unrelated.mkdir()

        server = {"pid": 1, "url": "http://localhost:3001/mcp", "root_dir": str(root)}

        with pytest.raises(proxy.ProxyError) as excinfo:
            proxy.select_server([server], unrelated.resolve())

        message = str(excinfo.value)
        assert message.startswith(f"No running Jupyter MCP server serves {unrelated}")
        assert proxy._describe(server) in message
        assert "--url" in message
        assert proxy.ENV_URL in message


class TestMainCLI:
    """Tests for the ``main`` CLI entry point."""

    def test_main_exits_cleanly_on_proxy_error(
        self, isolated_runtime_dir, capsys, monkeypatch
    ):
        """Discovery failures produce an error message and non-zero exit."""
        monkeypatch.chdir(isolated_runtime_dir)
        exit_code = proxy.main(["--runtime-dir", str(isolated_runtime_dir)])

        assert exit_code == 2
        captured = capsys.readouterr()
        assert "No running Jupyter MCP servers" in captured.err

    def test_main_refuses_single_server_outside_cwd(
        self, isolated_runtime_dir, capsys, tmp_path
    ):
        """A lone server that does not serve ``--cwd`` is an error, not a proxy."""
        root = tmp_path / "project"
        other = tmp_path / "other"
        root.mkdir()
        other.mkdir()
        _publish_server(isolated_runtime_dir, 101, root_dir=root)

        exit_code = proxy.main(
            ["--runtime-dir", str(isolated_runtime_dir), "--cwd", str(other)]
        )

        assert exit_code == 2
        assert f"root_dir={str(root)!r}" in capsys.readouterr().err

    def test_main_connects_with_explicit_url(self, monkeypatch):
        """An explicit URL should trigger ``run_proxy`` without discovery."""
        calls = []

        async def fake_run_proxy(url):
            calls.append(url)

        monkeypatch.setattr(proxy, "run_proxy", fake_run_proxy)

        assert proxy.main(["--url", "http://explicit:1/mcp"]) == 0
        assert calls == ["http://explicit:1/mcp"]

    def test_main_treats_keyboard_interrupt_as_success(self, monkeypatch):
        """KeyboardInterrupt while proxying should be a clean shutdown."""

        async def raise_kbi(_url):
            raise KeyboardInterrupt

        monkeypatch.setattr(proxy, "run_proxy", raise_kbi)

        assert proxy.main(["--url", "http://x/mcp"]) == 0


class TestRunProxy:
    """Test that ``run_proxy`` configures FastMCP correctly."""

    @pytest.mark.asyncio
    async def test_run_proxy_uses_stdio_transport(self):
        """``run_proxy`` should forward the URL and pick the stdio transport."""
        fake_proxy = AsyncMock()

        with patch(
            "jupyter_server_mcp.proxy.create_proxy", return_value=fake_proxy
        ) as create:
            await proxy.run_proxy("http://localhost:3001/mcp")

        create.assert_called_once_with("http://localhost:3001/mcp")
        fake_proxy.run_async.assert_awaited_once_with(
            transport="stdio", show_banner=False
        )
