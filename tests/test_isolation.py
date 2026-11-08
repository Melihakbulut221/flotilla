"""Properties the bwrap backend must guarantee for untrusted model code."""

import os
from pathlib import Path


def one(fl, cmd, **kw):
    return fl.swarms.run(cmd, count=1, timeout_seconds=30, **kw)["sandboxes"][0]


def test_home_dir_is_hidden(iso):
    home = str(Path.home())
    sb = one(iso, f"ls -a {home} 2>&1; cat {home}/.bashrc 2>&1; ls {home}/.ssh 2>&1; true")
    # only the exposed python prefix may exist under $HOME, never dotfiles or keys
    assert ".bashrc" not in sb["stdoutTail"].split()
    assert "No such file" in sb["stdoutTail"]


def test_no_network_by_default(iso):
    sb = one(iso, "python3 -c \"import socket; socket.create_connection(('1.1.1.1', 53), timeout=2)\"")
    assert sb["status"] == "failed"
    assert "Network is unreachable" in sb["stderrTail"] or "OSError" in sb["stderrTail"]


def test_host_env_does_not_leak(iso, monkeypatch):
    monkeypatch.setenv("SUPER_SECRET_TOKEN", "abc123")
    sb = one(iso, "env")
    assert "SUPER_SECRET_TOKEN" not in sb["stdoutTail"]


def test_system_dirs_read_only(iso):
    sb = one(iso, "touch /usr/x 2>&1; touch /etc/x 2>&1; true")
    assert sb["stdoutTail"].count("Read-only") == 2


def test_symlinked_result_cannot_read_host_files(iso, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("host-secret")
    sb = one(iso, f"ln -s {secret} /work/result.json", result_path="/work/result.json")
    assert sb["result"] is None


def _procs_with_cmdline(needle: str) -> list[int]:
    found = []
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                cmd = Path(f"/proc/{d}/cmdline").read_bytes().replace(b"\0", b" ").decode()
            except OSError:
                continue
            if needle in cmd and "pytest" not in cmd:
                found.append(int(d))
    return found


def test_background_processes_die_with_sandbox(iso):
    sb = one(iso, "sleep 47.123 & echo started")
    assert sb["status"] == "succeeded"
    assert _procs_with_cmdline("sleep 47.123") == []


def test_pid_namespace(iso):
    sb = one(iso, "ls /proc | grep -c '^[0-9]'")
    assert int(sb["stdoutTail"].strip()) < 10
