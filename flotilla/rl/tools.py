"""Function-calling environment: the policy acts through structured tool calls.

Tools: ``bash``, ``read``, ``write``, ``edit``, ``grep``, ``glob`` and
``submit``, declared as OpenAI ``tools`` JSON schemas (``TOOLS``). Relative
paths resolve against ``/work``.

Each assistant message is one turn. Its tool calls run in order and every
call gets a ``role: tool`` reply. The episode ends, and the task is verified
exactly as in ``ShellTaskEnv`` (isolated by default), when one of these
happens:

- ``submit`` is called;
- an assistant message carries no tool call (``on_no_tool_call="submit"``,
  the default); with ``"remind"`` the policy is nudged instead;
- ``max_turns`` is reached.

Tool calls are accepted in three encodings, so the same environment works
with native function calling and with plain text completions:

- native ``message["tool_calls"]``, e.g. from ``vllm serve ...
  --enable-auto-tool-choice --tool-call-parser hermes``;
- Hermes JSON in the text:
  ``<tool_call>{"name": "bash", "arguments": {...}}</tool_call>``;
- Qwen3-Coder XML in the text:
  ``<tool_call><function=bash><parameter=command>ls</parameter></function></tool_call>``.

A malformed call (unknown tool, bad JSON, missing argument) does not end the
episode. It returns an error message as the tool result and is counted in
``info["tool_errors"]``.
"""

from __future__ import annotations

import json
import posixpath
import re
import shlex
from typing import Any

from ..spec import SpecError
from .env import ShellTaskEnv, StepResult

WORKDIR = "/work"
READ_MAX_BYTES = 2 * 1024 * 1024


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }}


TOOLS = [
    _fn("bash", "Run a shell command in the sandbox (cwd /work, no network). Returns stdout, stderr and the exit code.",
        {"command": {"type": "string", "description": "The command to run."},
         "timeout": {"type": "integer", "description": "Seconds before the command is killed (default 30, max 300)."}},
        ["command"]),
    _fn("read", "Read a text file. Lines are numbered from 1.",
        {"path": {"type": "string", "description": "File path, absolute or relative to /work."},
         "offset": {"type": "integer", "description": "First line to show (1-based, default 1)."},
         "limit": {"type": "integer", "description": "Maximum number of lines (default 500)."}},
        ["path"]),
    _fn("write", "Create or overwrite a file with the given content. Parent directories are created.",
        {"path": {"type": "string"}, "content": {"type": "string"}},
        ["path", "content"]),
    _fn("edit", "Replace an exact string in a file. old_string must occur exactly once unless replace_all is true.",
        {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"},
         "replace_all": {"type": "boolean", "description": "Replace every occurrence (default false)."}},
        ["path", "old_string", "new_string"]),
    _fn("grep", "Search file contents with an extended regular expression. Returns path:line:text matches.",
        {"pattern": {"type": "string"},
         "path": {"type": "string", "description": "File or directory to search (default /work)."},
         "glob": {"type": "string", "description": "Only search files whose name matches, e.g. *.py."},
         "ignore_case": {"type": "boolean"},
         "max_results": {"type": "integer", "description": "Default 250."}},
        ["pattern"]),
    _fn("glob", "List files matching a glob pattern such as **/*.py, relative to path.",
        {"pattern": {"type": "string"},
         "path": {"type": "string", "description": "Directory to search (default /work)."}},
        ["pattern"]),
    _fn("submit", "Finish the task. Call this once the work in /work is complete.",
        {"summary": {"type": "string", "description": "One or two sentences on what was done."}},
        []),
]
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}
_SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}

TOOL_SYSTEM_PROMPT = (
    "You are working in a Linux sandbox; your working directory is /work and there is no network access. "
    "Use the provided tools to inspect and change files and to run commands. "
    "Call submit when the task is complete."
)


class ToolError(Exception):
    """A tool call the policy got wrong; reported back to it, not raised."""


# ------------------------------------------------------------------ parsing

_HERMES = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_XML_CALL = re.compile(r"<tool_call>\s*<function=([\w.-]+)>(.*?)</function>\s*</tool_call>", re.S)
_XML_PARAM = re.compile(r"<parameter=([\w.-]+)>\n?(.*?)\n?</parameter>", re.S)


def _coerce(name: str, key: str, value: str) -> Any:
    """XML parameters arrive as strings; convert using the tool schema."""
    typ = _SCHEMAS.get(name, {}).get("properties", {}).get(key, {}).get("type")
    if typ == "integer":
        return int(value.strip())
    if typ == "number":
        return float(value.strip())
    if typ == "boolean":
        return value.strip().lower() in ("true", "1", "yes")
    return value


def parse_tool_calls(message: dict | str) -> tuple[list[dict], bool]:
    """Return (calls, native). Each call is {"id", "name", "arguments" (raw)}.

    ``arguments`` stays raw (dict or JSON string) so bad JSON can be reported
    to the policy as a tool error rather than dropped silently.
    """
    if isinstance(message, str):
        message = {"role": "assistant", "content": message}
    native = message.get("tool_calls") or []
    if native:
        calls = []
        for i, c in enumerate(native):
            fn = c.get("function") or {}
            calls.append({"id": c.get("id") or f"call_{i}", "name": fn.get("name"),
                          "arguments": fn.get("arguments", {})})
        return calls, True
    text = message.get("content") or ""
    calls = []
    for m in _XML_CALL.finditer(text):
        name = m.group(1)
        try:
            args = {k: _coerce(name, k, v) for k, v in _XML_PARAM.findall(m.group(2))}
        except ValueError as e:
            args = f"<unparseable parameters: {e}>"
        calls.append({"id": f"call_{len(calls)}", "name": name, "arguments": args})
    if not calls:
        for m in _HERMES.finditer(text):
            try:
                obj = json.loads(m.group(1))
                name = obj.get("name")
                args = obj.get("arguments", obj.get("parameters", {}))
            except ValueError:
                name, args = None, m.group(1)
            calls.append({"id": f"call_{len(calls)}", "name": name, "arguments": args})
    return calls, False


def render_tools_prompt(tools: list[dict] = TOOLS) -> str:
    """Tool schemas plus calling instructions, for servers without a tool parser."""
    lines = ["You can call these tools. To call one, write exactly:",
             '<tool_call>{"name": "<tool>", "arguments": {...}}</tool_call>',
             "You may make several calls in one reply. Results come back in <tool_response> blocks.",
             "", "Tools:"]
    for t in tools:
        lines.append(json.dumps(t["function"]))
    return "\n".join(lines)


# ---------------------------------------------------------------- execution


def middle_truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    keep = max(0, limit - 60)
    head, tail = text[: keep // 2], text[len(text) - keep // 2:]
    return f"{head}\n... [{len(text) - len(head) - len(tail)} characters truncated] ...\n{tail}"


def _resolve(path: Any) -> str:
    if not isinstance(path, str) or not path.strip():
        raise ToolError("path must be a non-empty string")
    p = path.strip()
    p = posixpath.normpath(p if p.startswith("/") else posixpath.join(WORKDIR, p))
    if p == "/":
        raise ToolError("path must not be /")
    return p


def glob_to_regex(pattern: str) -> re.Pattern:
    i, out = 0, []
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        elif pattern[i] == "[":
            j = pattern.find("]", i + 1)
            if j == -1:
                out.append(re.escape(pattern[i]))
                i += 1
            else:
                out.append("[" + pattern[i + 1:j].replace("\\", "\\\\") + "]")
                i = j + 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def _is_tool_error(exc: Exception) -> bool:
    """File-operation failures the policy caused (bad path, read-only dir)
    versus infrastructure failures, which must propagate."""
    if isinstance(exc, (SpecError, ToolError)):
        return True
    status = getattr(exc, "status", None)  # client.ApiError
    return status is not None and status < 500


class ToolTaskEnv(ShellTaskEnv):
    def __init__(self, client, task: dict, *, max_turns: int = 30, max_tool_calls_per_turn: int = 8,
                 max_observation_chars: int = 16000, bash_timeout: float = 30, bash_max_timeout: float = 300,
                 on_no_tool_call: str = "submit", **kwargs):
        super().__init__(client, task, **kwargs)
        self.max_turns = int(task.get("max_turns", max_turns))
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self.max_observation_chars = max_observation_chars
        self.bash_timeout = bash_timeout
        self.bash_max_timeout = bash_max_timeout
        if on_no_tool_call not in ("submit", "remind"):
            raise ValueError("on_no_tool_call must be 'submit' or 'remind'")
        self.on_no_tool_call = on_no_tool_call
        self.tool_calls = 0
        self.tool_errors = 0

    tools = TOOLS

    def reset(self) -> str:
        self.tool_calls = self.tool_errors = 0
        return super().reset()

    # one assistant message -> tool results (observation is a list of messages)
    def step(self, message: dict | str) -> StepResult:
        if self.done:
            raise RuntimeError("episode is over; call reset()")
        if self.session is None:
            raise RuntimeError("call reset() first")
        self.turn += 1
        calls, native = parse_tool_calls(message)
        results: list[dict] = []
        submitted = False
        if not calls:
            if self.on_no_tool_call == "submit":
                return self._done("no_tool_call", results)
            results.append(self._reply(None, "No tool call found. Use a tool, or call submit when done.", native))
        for n, call in enumerate(calls):
            if n >= self.max_tool_calls_per_turn:
                self.tool_errors += 1
                results.append(self._reply(call, f"Error: at most {self.max_tool_calls_per_turn} tool calls "
                                                  "per turn; this call was skipped.", native))
                continue
            self.tool_calls += 1
            if call["name"] == "submit":
                submitted = True
                results.append(self._reply(call, "Submitted.", native))
                break
            try:
                out = self.run_tool(call["name"], call["arguments"])
            except Exception as e:
                if not _is_tool_error(e):
                    raise
                self.tool_errors += 1
                out = f"Error: {e}"
            results.append(self._reply(call, middle_truncate(out, self.max_observation_chars), native))
        if submitted:
            return self._done("submitted", results)
        if self.turn >= self.max_turns:
            return self._done("max_turns", results)
        return StepResult(results, 0.0, False, {"turn": self.turn, "tool_calls": self.tool_calls,
                                                "tool_errors": self.tool_errors})

    def _done(self, why: str, results: list[dict]) -> StepResult:
        step = self._finish(why)
        step.observation = results
        step.info.update(tool_calls=self.tool_calls, tool_errors=self.tool_errors)
        return step

    @staticmethod
    def _reply(call: dict | None, content: str, native: bool) -> dict:
        if native and call is not None:
            return {"role": "tool", "tool_call_id": call["id"], "content": content}
        # text protocol: results go back as a user turn the model was told to expect
        return {"role": "user", "content": f"<tool_response>\n{content}\n</tool_response>"}

    # ------------------------------------------------------------ the tools

    def run_tool(self, name: Any, arguments: Any) -> str:
        if name not in TOOL_NAMES:
            raise ToolError(f"unknown tool {name!r}; available: {', '.join(sorted(TOOL_NAMES))}")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except ValueError as e:
                raise ToolError(f"arguments are not valid JSON: {e}") from None
        if not isinstance(arguments, dict):
            raise ToolError("arguments must be a JSON object")
        missing = [k for k in _SCHEMAS[name]["required"] if k not in arguments]
        if missing:
            raise ToolError(f"{name} is missing required argument(s): {', '.join(missing)}")
        return getattr(self, f"tool_{name}")(**{k: v for k, v in arguments.items()
                                               if k in _SCHEMAS[name]["properties"]})

    def tool_bash(self, command: str, timeout: Any = None) -> str:
        if not isinstance(command, str) or not command.strip():
            raise ToolError("command must be a non-empty string")
        try:
            t = float(timeout) if timeout is not None else self.bash_timeout
        except (TypeError, ValueError):
            raise ToolError("timeout must be a number") from None
        t = max(1.0, min(t, self.bash_max_timeout))
        r = self.session.exec(command, timeout=t, output_limit=self.max_observation_chars * 2)
        body = r.stdout
        if r.stderr:
            body += ("\n" if body and not body.endswith("\n") else "") + r.stderr
        status = f"[command timed out after {t:g}s]" if r.timed_out else f"[exit code {r.exit_code}]"
        return (body.rstrip("\n") + "\n" + status).lstrip("\n")

    def _read_text(self, path: str) -> str:
        data = self.session.read_file(path, READ_MAX_BYTES + 1)
        if data is None:
            raise ToolError(f"file not found: {path}")
        if len(data) > READ_MAX_BYTES:
            raise ToolError(f"{path} is larger than {READ_MAX_BYTES} bytes; use bash (head, sed -n) instead")
        if b"\0" in data[:8192]:
            raise ToolError(f"{path} looks binary")
        return data.decode("utf-8", errors="replace")

    def tool_read(self, path: str, offset: Any = 1, limit: Any = 500) -> str:
        p = _resolve(path)
        lines = self._read_text(p).split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        try:
            start, count = max(1, int(offset or 1)), max(1, int(limit or 500))
        except (TypeError, ValueError):
            raise ToolError("offset and limit must be integers") from None
        chunk = lines[start - 1:start - 1 + count]
        if not chunk:
            return f"(file has {len(lines)} lines; nothing at offset {start})"
        body = "\n".join(f"{start + i:6d}\t{ln[:2000]}" for i, ln in enumerate(chunk))
        rest = len(lines) - (start - 1 + len(chunk))
        return body + (f"\n... {rest} more lines (use offset {start + len(chunk)})" if rest > 0 else "")

    def tool_write(self, path: str, content: Any) -> str:
        p = _resolve(path)
        if not isinstance(content, str):
            raise ToolError("content must be a string")
        self.session.write_file(p, content)
        return f"Wrote {len(content.encode())} bytes to {p}"

    def tool_edit(self, path: str, old_string: Any, new_string: Any, replace_all: Any = False) -> str:
        p = _resolve(path)
        if not isinstance(old_string, str) or not isinstance(new_string, str):
            raise ToolError("old_string and new_string must be strings")
        if old_string == "":
            raise ToolError("old_string must not be empty; use write to create a file")
        if old_string == new_string:
            raise ToolError("old_string and new_string are identical")
        text = self._read_text(p)
        n = text.count(old_string)
        if n == 0:
            raise ToolError(f"old_string not found in {p}")
        if n > 1 and not replace_all:
            raise ToolError(f"old_string occurs {n} times in {p}; add surrounding context or set replace_all")
        self.session.write_file(p, text.replace(old_string, new_string) if replace_all
                                else text.replace(old_string, new_string, 1))
        return f"Edited {p} ({n if replace_all else 1} replacement{'s' if replace_all and n > 1 else ''})"

    def tool_grep(self, pattern: str, path: Any = None, glob: Any = None, ignore_case: Any = False,
                  max_results: Any = 250) -> str:
        if not isinstance(pattern, str) or not pattern:
            raise ToolError("pattern must be a non-empty string")
        target = _resolve(path) if path else WORKDIR
        try:
            limit = max(1, min(int(max_results or 250), 2000))
        except (TypeError, ValueError):
            raise ToolError("max_results must be an integer") from None
        cmd = ["grep", "-rnE", "--binary-files=without-match", "--exclude-dir=.git"]
        if ignore_case:
            cmd.append("-i")
        if glob:
            cmd.append(f"--include={glob}")
        cmd += ["-e", pattern, "--", target]
        r = self.session.exec(f"{shlex.join(cmd)} 2>&1 | head -n {limit + 1}", timeout=60,
                              output_limit=self.max_observation_chars * 2)
        lines = [ln for ln in r.stdout.splitlines() if ln]
        if not lines:
            return "No matches."
        more = len(lines) > limit
        return "\n".join(lines[:limit]) + (f"\n... more than {limit} matches; narrow the search" if more else "")

    def tool_glob(self, pattern: str, path: Any = None) -> str:
        if not isinstance(pattern, str) or not pattern:
            raise ToolError("pattern must be a non-empty string")
        base = _resolve(path) if path else WORKDIR
        r = self.session.exec(
            f"find {shlex.quote(base)} -type f -not -path '*/.git/*' -not -path '*/node_modules/*' "
            f"2>/dev/null | head -n 20000", timeout=60, output_limit=4 * 1024 * 1024)
        rx = glob_to_regex(pattern.lstrip("/") if not pattern.startswith(base) else pattern[len(base):].lstrip("/"))
        prefix = base.rstrip("/") + "/"
        hits = sorted(f for f in r.stdout.splitlines()
                      if f.startswith(prefix) and rx.match(f[len(prefix):]))
        if not hits:
            return "No files matched."
        shown = hits[:250]
        return "\n".join(shown) + (f"\n... {len(hits) - 250} more" if len(hits) > 250 else "")
