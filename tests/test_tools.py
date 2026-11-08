import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from flotilla import Flotilla
from flotilla.rl import (
    TOOLS,
    OpenAIChatPolicy,
    OpenAIToolPolicy,
    ToolTaskEnv,
    parse_tool_calls,
    run_episode,
    run_episodes,
)
from flotilla.rl.tools import glob_to_regex, middle_truncate

SORT_TASK = {
    "id": "sort",
    "prompt": "Write the numbers from /data/in.txt sorted ascending to /work/out.txt, one per line.",
    "files": {"/data/in.txt": "3\n1\n2\n", "/data/expected.txt": "1\n2\n3\n",
              "/work/notes.md": "# notes\nTODO: sort\nTODO: check\n"},
    "verify": "cmp -s /work/out.txt /data/expected.txt",
    "max_turns": 6,
}


def call(name, _id=None, **args):
    return {"id": _id or f"c_{name}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def msg(*calls, content=""):
    m = {"role": "assistant", "content": content}
    if calls:
        m["tool_calls"] = list(calls)
    return m


def scripted(replies):
    it = iter(replies)
    return lambda messages: next(it)


@pytest.fixture
def env(iso):
    e = ToolTaskEnv(iso, SORT_TASK)
    e.reset()
    yield e
    e.close()


# ------------------------------------------------------------------ parsing


def test_parse_native_hermes_and_qwen_xml():
    calls, native = parse_tool_calls(msg(call("bash", command="ls")))
    assert native and calls[0]["name"] == "bash" and json.loads(calls[0]["arguments"]) == {"command": "ls"}

    text = 'thinking\n<tool_call>{"name": "read", "arguments": {"path": "a.py"}}</tool_call>'
    calls, native = parse_tool_calls(text)
    assert not native and calls == [{"id": "call_0", "name": "read", "arguments": {"path": "a.py"}}]

    xml = ("<tool_call>\n<function=read>\n<parameter=path>\na.py\n</parameter>\n"
           "<parameter=limit>\n20\n</parameter>\n</function>\n</tool_call>")
    calls, _ = parse_tool_calls(xml)
    assert calls[0]["arguments"] == {"path": "a.py", "limit": 20}

    assert parse_tool_calls("just text") == ([], False)


def test_tool_schemas_are_valid_openai_functions():
    for t in TOOLS:
        f = t["function"]
        assert t["type"] == "function" and f["parameters"]["type"] == "object"
        assert set(f["parameters"]["required"]) <= set(f["parameters"]["properties"])


def test_helpers():
    assert glob_to_regex("**/*.py").match("a/b/c.py") and glob_to_regex("**/*.py").match("c.py")
    assert not glob_to_regex("*.py").match("a/c.py")
    assert glob_to_regex("src/?.[ch]").match("src/x.h")
    t = middle_truncate("a" * 1000 + "b" * 1000, 200)
    assert t.startswith("a") and t.endswith("b") and "truncated" in t and len(t) < 300


# ------------------------------------------------------------------ tools


def test_read_write_paths(env):
    out = env.run_tool("read", {"path": "notes.md"})
    assert out.splitlines()[0] == "     1\t# notes"
    assert "     3\tTODO: check" in out
    assert env.run_tool("read", {"path": "/work/notes.md", "offset": 2, "limit": 1}).startswith("     2\tTODO: sort")
    assert "1 more lines" in env.run_tool("read", {"path": "notes.md", "limit": 2})
    env.run_tool("write", {"path": "deep/dir/x.txt", "content": "hi\n"})
    assert env.run_tool("bash", {"command": "cat /work/deep/dir/x.txt"}).startswith("hi")
    with pytest.raises(Exception, match="not found"):
        env.run_tool("read", {"path": "missing.txt"})
    with pytest.raises(Exception, match="read-only"):
        env.run_tool("write", {"path": "/usr/evil", "content": "x"})


def test_edit(env):
    env.run_tool("edit", {"path": "notes.md", "old_string": "TODO: sort", "new_string": "DONE: sort"})
    assert "DONE: sort" in env.run_tool("read", {"path": "notes.md"})
    env.run_tool("write", {"path": "t.txt", "content": "x x\n"})
    with pytest.raises(Exception, match="occurs 2 times"):
        env.run_tool("edit", {"path": "t.txt", "old_string": "x", "new_string": "y"})
    assert "2 replacements" in env.run_tool("edit", {"path": "t.txt", "old_string": "x", "new_string": "y",
                                                     "replace_all": True})
    with pytest.raises(Exception, match="not found"):
        env.run_tool("edit", {"path": "t.txt", "old_string": "zzz", "new_string": "q"})


def test_grep_and_glob(env):
    env.run_tool("write", {"path": "src/a.py", "content": "def Foo():\n    return 1\n"})
    env.run_tool("write", {"path": "src/sub/b.py", "content": "foo = 2\n"})
    env.run_tool("write", {"path": "src/c.txt", "content": "foo\n"})
    out = env.run_tool("grep", {"pattern": "foo", "path": "src", "glob": "*.py", "ignore_case": True})
    assert "/work/src/a.py:1:def Foo():" in out and "/work/src/sub/b.py:1:foo = 2" in out and "c.txt" not in out
    assert env.run_tool("grep", {"pattern": "nomatch_xyz"}) == "No matches."
    files = env.run_tool("glob", {"pattern": "**/*.py"}).splitlines()
    assert files == ["/work/src/a.py", "/work/src/sub/b.py"]
    assert env.run_tool("glob", {"pattern": "*.py", "path": "src"}).splitlines() == ["/work/src/a.py"]


def test_bash_exit_code_and_timeout(env):
    assert env.run_tool("bash", {"command": "echo hi; exit 3"}).endswith("[exit code 3]")
    out = env.run_tool("bash", {"command": "sleep 5", "timeout": 1})
    assert "timed out after 1s" in out


# ------------------------------------------------------------------ episodes


def test_native_episode_solves_task(iso):
    policy = scripted([
        msg(call("read", path="/data/in.txt"), call("glob", pattern="**/*")),
        msg(call("bash", command="sort -n /data/in.txt > out.txt")),
        msg(call("submit", summary="sorted")),
    ])
    t = run_episode(ToolTaskEnv(iso, SORT_TASK), policy)
    assert t["reward"] == 1.0 and t["info"]["end"] == "submitted", t["info"]
    assert t["info"]["tool_calls"] == 4 and t["info"]["tool_errors"] == 0
    tool_msgs = [m for m in t["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["c_read", "c_glob", "c_bash", "c_submit"]
    assert "     1\t3" in tool_msgs[0]["content"]


def test_bad_calls_are_reported_and_counted(iso):
    bad_json = {"id": "x", "type": "function", "function": {"name": "bash", "arguments": "{not json"}}
    policy = scripted([
        msg(call("rm_rf"), bad_json, call("read"), call("bash", command="sort -n /data/in.txt > out.txt")),
        msg(call("submit")),
    ])
    t = run_episode(ToolTaskEnv(iso, SORT_TASK), policy)
    contents = [m["content"] for m in t["messages"] if m["role"] == "tool"]
    assert contents[0].startswith("Error: unknown tool 'rm_rf'")
    assert "not valid JSON" in contents[1]
    assert "missing required argument(s): path" in contents[2]
    assert t["info"]["tool_errors"] == 3 and t["reward"] == 1.0


def test_no_tool_call_ends_or_reminds(iso):
    t = run_episode(ToolTaskEnv(iso, SORT_TASK), scripted([msg(content="I think I am done.")]))
    assert t["info"]["end"] == "no_tool_call" and t["reward"] == 0.0 and t["turns"] == 1

    policy = scripted([msg(content="hmm"), msg(call("bash", command="sort -n /data/in.txt > out.txt")),
                       msg(call("submit"))])
    t = run_episode(ToolTaskEnv(iso, SORT_TASK, on_no_tool_call="remind"), policy)
    assert t["reward"] == 1.0 and t["turns"] == 3


def test_max_turns_and_calls_per_turn(iso):
    many = msg(*[call("bash", _id=f"b{i}", command="true") for i in range(5)])
    env = ToolTaskEnv(iso, dict(SORT_TASK, max_turns=2), max_tool_calls_per_turn=3)
    t = run_episode(env, scripted([many, many]))
    assert t["info"]["end"] == "max_turns" and t["turns"] == 2
    skipped = [m for m in t["messages"] if m["role"] == "tool" and "skipped" in m["content"]]
    assert len(skipped) == 4  # 2 per turn


def test_isolated_verify_applies_to_tool_env(iso):
    hack = scripted([msg(call("write", path="out.txt", content="x\n"),
                         call("write", path="/data/expected.txt", content="x\n")), msg(call("submit"))])
    assert run_episode(ToolTaskEnv(iso, SORT_TASK), hack)["reward"] == 0.0


def test_text_protocol_without_native_tool_calls(iso):
    seen = {}

    def policy(messages):
        seen.setdefault("system", messages[0]["content"])
        n = sum(m["role"] == "assistant" for m in messages)
        if n == 0:
            return ('<tool_call>{"name": "bash", "arguments": {"command": "sort -n /data/in.txt > out.txt"}}'
                    '</tool_call>')
        return "<tool_call>\n<function=submit>\n</function>\n</tool_call>"

    t = run_episode(ToolTaskEnv(iso, SORT_TASK), policy, native_tools=False)
    assert t["reward"] == 1.0
    assert '"name": "bash"' in seen["system"] and "<tool_call>" in seen["system"]
    responses = [m for m in t["messages"] if m["role"] == "user" and "<tool_response>" in m["content"]]
    assert responses and "[exit code 0]" in responses[0]["content"]


def test_run_episodes_with_tool_env(iso):
    policy = lambda messages: msg(call("bash", command="sort -n /data/in.txt > out.txt"),  # noqa: E731
                                  call("submit"))
    trajs = run_episodes(iso, policy, [SORT_TASK], samples_per_task=3, parallel=3, env_cls=ToolTaskEnv)
    assert [t["reward"] for t in trajs] == [1.0, 1.0, 1.0]


# ------------------------------------------------------------------ OpenAI-compatible server


class _FakeServer:
    """Minimal /v1/chat/completions that replays canned assistant messages."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                data = json.dumps({"choices": [{"message": outer.replies.pop(0)}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/v1"


def test_openai_tool_policy_roundtrip(iso):
    fake = _FakeServer([
        {"role": "assistant", "content": None,
         "tool_calls": [call("bash", _id="srv1", command="sort -n /data/in.txt > out.txt")]},
        {"role": "assistant", "content": None, "tool_calls": [call("submit", _id="srv2")]},
    ])
    t = run_episode(ToolTaskEnv(iso, SORT_TASK), OpenAIToolPolicy(fake.url, "m"))
    fake.srv.shutdown()
    assert t["reward"] == 1.0
    first, second = fake.requests
    assert first["tool_choice"] == "auto" and {f["function"]["name"] for f in first["tools"]} >= {"bash", "submit"}
    # the follow-up request carries the assistant tool call and the tool result in OpenAI format
    assert second["messages"][2]["tool_calls"][0]["id"] == "srv1"
    assert second["messages"][3]["role"] == "tool" and second["messages"][3]["tool_call_id"] == "srv1"


def test_text_policy_against_server(iso):
    fake = _FakeServer([{"role": "assistant", "content": "<tool_call>{\"name\": \"submit\", \"arguments\": {}}"
                                                         "</tool_call>"}])
    t = run_episode(ToolTaskEnv(iso, SORT_TASK), OpenAIChatPolicy(fake.url, "m"), native_tools=False)
    fake.srv.shutdown()
    assert t["info"]["end"] == "submitted" and "tools" not in fake.requests[0]


# ------------------------------------------------------------------ remote sandboxes


def test_tool_env_over_http(tmp_path):
    from flotilla.backends import BwrapBackend
    from flotilla.server import make_server
    from flotilla.session import SessionManager
    from flotilla.swarm import SwarmRunner

    if not BwrapBackend.available()[0]:
        pytest.skip("bwrap not available")
    runner = SwarmRunner(backend="bwrap", state_dir=tmp_path)
    sessions = SessionManager(backend=runner.backend, state_dir=tmp_path)
    srv = make_server(runner, sessions, port=0, token="t")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        fl = Flotilla(base_url=f"http://127.0.0.1:{srv.server_address[1]}", token="t")
        policy = scripted([
            msg(call("write", path="/usr/x", content="nope"),         # 422 from server -> tool error
                call("read", path="/data/in.txt"),
                call("bash", command="sort -n /data/in.txt > out.txt")),
            msg(call("submit")),
        ])
        t = run_episode(ToolTaskEnv(fl, SORT_TASK), policy)
        assert t["reward"] == 1.0 and t["info"]["tool_errors"] == 1 and not t["infra_error"]
    finally:
        srv.shutdown()
        sessions.close_all()
