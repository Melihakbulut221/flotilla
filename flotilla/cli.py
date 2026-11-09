"""Command line interface.

    flotilla doctor
    flotilla run --count 50 --command 'python eval.py --case $FLOTILLA_SWARM_INDEX' --wait
    flotilla run --inputs inputs.jsonl --command 'python rollout.py' --result-path /work/result.json --wait
    flotilla serve --port 8642 --token "$FLOTILLA_TOKEN"
    flotilla list | get ID | results ID -o results.jsonl | cancel ID
    flotilla verl export --kind code tasks.jsonl -o train.parquet
    flotilla verl import mimo_rows.parquet -o tasks/     # writes tasks/<kind>.jsonl

``run`` without ``--url`` executes in this process (and therefore must wait).
With ``--url`` (or ``$FLOTILLA_URL``) it submits to a server.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path

from . import __version__
from .backends import doctor
from .spec import SpecError, parse_duration


def _print(obj) -> None:
    print(json.dumps(obj, indent=2))


def _client(args):
    from .client import Flotilla

    return Flotilla(
        base_url=args.url, token=args.token, backend=args.backend,
        state_dir=args.home, max_concurrency=getattr(args, "max_concurrency", None),
    )


def _progress(run: dict) -> str:
    c = run["counts"]
    return (f"\r{run['id']}  done {run['done']}/{run['count']}  ok {c['succeeded']}  "
            f"failed {c['failed']}  running {c['running'] + c['booting']}  queued {c['queued']}   ")


def cmd_run(args) -> int:
    inputs = None
    if args.inputs:
        with open(args.inputs) as f:
            inputs = [json.loads(line) for line in f if line.strip()]
    if (args.count is None) == (inputs is None):
        raise SpecError("give exactly one of --count or --inputs")
    fl = _client(args)
    env = dict(kv.split("=", 1) for kv in args.env or [])
    run = fl.swarms.start(
        args.command, count=args.count, inputs=inputs, name=args.name or "",
        image=args.image, size=args.size, timeout_seconds=parse_duration(args.timeout),
        concurrency=args.concurrency, result_path=args.result_path,
        network=args.network, env=env,
    )
    if fl.remote and not args.wait:
        _print(run)
        return 0
    if not fl.remote:
        # an in-process run dies with this process: cancel cleanly on Ctrl-C
        signal.signal(signal.SIGINT, lambda *a: fl.swarms.cancel(run["id"]))
    import time

    while True:
        cur = fl.swarms.get(run["id"])
        if sys.stderr.isatty():
            sys.stderr.write(_progress(cur))
        if cur["status"] != "running":
            break
        time.sleep(0.5)
    if sys.stderr.isatty():
        sys.stderr.write("\n")
    if args.output:
        with open(args.output, "w") as f:
            for r in fl.swarms.results(run["id"]):
                f.write(json.dumps(r) + "\n")
    summary = {k: v for k, v in cur.items() if k != "sandboxes"}
    _print(summary)
    return 0 if cur["counts"]["succeeded"] == cur["count"] else 1


def cmd_serve(args) -> int:
    from .server import make_server
    from .session import SessionManager
    from .swarm import SwarmRunner

    runner = SwarmRunner(backend=args.backend, state_dir=args.home, max_concurrency=args.max_concurrency)
    sessions = SessionManager(backend=runner.backend, state_dir=runner.home,
                              idle_seconds=parse_duration(args.session_idle))
    token = args.token or os.environ.get("FLOTILLA_TOKEN")
    if args.host not in ("127.0.0.1", "localhost", "::1") and not token:
        print("refusing to listen on a non-loopback address without --token", file=sys.stderr)
        return 2
    srv = make_server(runner, sessions, args.host, args.port, token, verbose=args.verbose)
    iso = "isolated" if runner.backend.isolated else "NOT isolated"
    print(f"flotilla {__version__} serving on http://{args.host}:{args.port}  "
          f"backend={runner.backend.name} ({iso})  max_concurrency={runner.max_concurrency}",
          file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        sessions.close_all()
    return 0


def cmd_verl(args) -> int:
    from . import verl

    if args.verl_cmd == "export":
        tasks = [json.loads(line) for line in open(args.input) if line.strip()]
        rows = verl.to_verl_rows(tasks, args.kind, split=args.split)
        verl.write_rows(rows, args.output)
        print(f"wrote {len(rows)} verl rows to {args.output}", file=sys.stderr)
        return 0
    rows = verl.read_rows(args.input)
    by_kind, skipped = verl.import_rows(rows, image_repo=args.image_repo, apptainer=args.apptainer)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for kind, tasks in sorted(by_kind.items()):
        with open(out / f"{kind}.jsonl", "w") as f:
            for t in tasks:
                f.write(json.dumps(t) + "\n")
        print(f"{kind:6s} {len(tasks):6d} tasks -> {out / (kind + '.jsonl')}", file=sys.stderr)
    for reason, n in skipped.most_common():
        print(f"skipped {n:6d}  {reason}", file=sys.stderr)
    return 0 if by_kind else 1


def cmd_doctor(args) -> int:
    report = doctor()
    for name, (ok, why) in report.items():
        print(f"{name:10s} {'ok' if ok else '--'}  {why}")
    return 0 if any(ok for n, (ok, _) in report.items() if n != "local") else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="flotilla", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--url", default=os.environ.get("FLOTILLA_URL"), help="server URL (default: in-process)")
    p.add_argument("--token", default=os.environ.get("FLOTILLA_TOKEN"))
    p.add_argument("--backend", default=os.environ.get("FLOTILLA_BACKEND", "auto"),
                   help="auto | bwrap | podman | docker | apptainer | local")
    p.add_argument("--home", default=None, help="state dir (default $FLOTILLA_HOME or ~/.flotilla)")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="start a swarm")
    r.add_argument("--command", "-c", required=True)
    r.add_argument("--count", "-n", type=int)
    r.add_argument("--inputs", help="JSON Lines file, one {args, env, files} object per sandbox")
    r.add_argument("--name")
    r.add_argument("--image", help="container image (docker/podman/apptainer backends)")
    r.add_argument("--size", default="2x4", help="VCPUSxGB, e.g. 1x2, 2x4, 4x8")
    r.add_argument("--timeout", default="10m", help="per sandbox: 90, 90s, 10m, 1h")
    r.add_argument("--concurrency", type=int)
    r.add_argument("--max-concurrency", type=int, help="global cap for in-process runs")
    r.add_argument("--result-path")
    r.add_argument("--network", action="store_true", help="allow network (default: none)")
    r.add_argument("--env", "-e", action="append", help="KEY=VALUE for every sandbox")
    r.add_argument("--wait", action="store_true", help="wait for a remote run to finish")
    r.add_argument("--output", "-o", help="write results JSON Lines here")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8642)
    s.add_argument("--max-concurrency", type=int)
    s.add_argument("--session-idle", default="30m")
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(fn=cmd_serve)

    vp = sub.add_parser("verl", help="convert between flotilla tasks and verl RL rows")
    vsub = vp.add_subparsers(dest="verl_cmd", required=True)
    ve = vsub.add_parser("export", help="flotilla tasks (JSON Lines) -> verl rows (.parquet or .jsonl)")
    ve.add_argument("input")
    ve.add_argument("--kind", required=True, choices=["code", "music", "shell", "tool"])
    ve.add_argument("--split", default="train")
    ve.add_argument("-o", "--output", required=True)
    vi = vsub.add_parser("import", help="verl or MiMo rows -> flotilla tasks, one JSON Lines file per kind")
    vi.add_argument("input", help=".parquet, .jsonl, or .json (list or datasets-server dump)")
    vi.add_argument("-o", "--output", required=True, help="output directory")
    vi.add_argument("--image-repo", default="docker.io/xiaomimimo/mimo-v2.6-rl-oss")
    vi.add_argument("--apptainer", action="store_true", help="emit docker:// image references")
    vp.set_defaults(fn=cmd_verl)

    d = sub.add_parser("doctor", help="show which backends work on this machine")
    d.set_defaults(fn=cmd_doctor)

    for name in ("get", "cancel", "results"):
        c = sub.add_parser(name)
        c.add_argument("id")
        if name == "results":
            c.add_argument("--output", "-o")
        c.set_defaults(fn=lambda a, n=name: _simple(a, n))
    ls = sub.add_parser("list")
    ls.add_argument("--limit", type=int, default=50)
    ls.set_defaults(fn=lambda a: _simple(a, "list"))

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except (SpecError, RuntimeError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def _simple(args, name: str) -> int:
    fl = _client(args)
    if name == "list":
        _print(fl.swarms.list(args.limit))
    elif name == "get":
        _print(fl.swarms.get(args.id))
    elif name == "cancel":
        _print(fl.swarms.cancel(args.id))
    elif name == "results":
        rows = fl.swarms.results(args.id)
        out = open(args.output, "w") if args.output else sys.stdout
        for row in rows:
            out.write(json.dumps(row) + "\n")
        if args.output:
            out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
