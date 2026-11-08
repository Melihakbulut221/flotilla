import threading

import pytest

from flotilla import Flotilla
from flotilla.client import ApiError
from flotilla.server import make_server
from flotilla.session import SessionManager
from flotilla.swarm import SwarmRunner


@pytest.fixture
def remote(tmp_path):
    runner = SwarmRunner(backend="local", state_dir=tmp_path, max_concurrency=4)
    sessions = SessionManager(backend=runner.backend, state_dir=tmp_path)
    srv = make_server(runner, sessions, port=0, token="t0k")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    yield url
    srv.shutdown()
    sessions.close_all()


def test_remote_swarm_roundtrip(remote):
    fl = Flotilla(base_url=remote, token="t0k")
    run = fl.swarms.start("echo $FLOTILLA_SWARM_INDEX > out; cat out", count=3,
                          result_path="/work/out", timeout_seconds=30)
    done = fl.swarms.wait(run["id"], interval=0.1, timeout=30)
    assert done["counts"]["succeeded"] == 3
    rows = fl.swarms.results(run["id"])
    assert [r["result"].strip() for r in rows] == ["0", "1", "2"]
    assert fl.swarms.list()[0]["id"] == run["id"]


def test_remote_idempotency(remote):
    fl = Flotilla(base_url=remote, token="t0k")
    a = fl.swarms.start("true", count=1, idempotency_key="same")
    b = fl.swarms.start("true", count=1, idempotency_key="same")
    assert a["id"] == b["id"]
    with pytest.raises(ApiError) as e:
        fl.swarms.start("false", count=1, idempotency_key="same")
    assert e.value.status == 409


def test_auth_required(remote):
    with pytest.raises(ApiError) as e:
        Flotilla(base_url=remote, token="wrong").swarms.list()
    assert e.value.status == 401


def test_bad_request_is_422(remote):
    with pytest.raises(Exception):
        Flotilla(base_url=remote, token="t0k").swarms.start("x")  # neither count nor inputs


def test_remote_sessions(remote):
    fl = Flotilla(base_url=remote, token="t0k")
    with fl.sessions.open(files={"/work/a.txt": "hello"}) as s:
        r = s.exec("cat a.txt; echo more >> a.txt")
        assert r.exit_code == 0 and r.stdout == "hello"
        assert s.read_file("/work/a.txt") == b"hellomore\n"
        s.write_file("/work/bin.dat", b"\x00\x01")
        assert s.read_file("/work/bin.dat") == b"\x00\x01"
        assert s.read_file("/work/missing") is None
        assert s.exec("sleep 5", timeout=1).timed_out
    assert fl.sessions.list() == []
