#!/bin/sh
# Swarm basics from the command line. Run from the repo root after `pip install -e .`
set -e

# Which backends work here?
flotilla doctor

# 1) count: same command on 8 fresh sandboxes, each sees its own index
flotilla run --count 8 --timeout 30s \
  --command 'echo "sandbox $FLOTILLA_SWARM_INDEX of $FLOTILLA_SWARM_COUNT"; test $FLOTILLA_SWARM_INDEX -ne 5' \
  -o /tmp/flotilla-count.jsonl || echo "(one sandbox fails on purpose)"

# 2) inputs: one JSON line per sandbox with args / env / files, plus a result file
cat > /tmp/flotilla-inputs.jsonl <<'EOF'
{"args": ["--seed", "1"]}
{"args": ["--seed", "2"], "env": {"TEMPERATURE": "0.7"}}
{"files": {"/work/task.json": "{\"id\": 3}"}}
EOF
flotilla run --inputs /tmp/flotilla-inputs.jsonl --result-path /work/result.json --timeout 30s \
  --command 'python3 -c "import json,os,sys; json.dump({\"argv\": sys.argv[1:], \"t\": os.environ.get(\"TEMPERATURE\")}, open(\"result.json\",\"w\"))"' \
  -o /tmp/flotilla-inputs-results.jsonl
cat /tmp/flotilla-inputs-results.jsonl | python3 -c 'import json,sys; [print(r["index"], r["status"], r["result"]) for r in map(json.loads, sys.stdin)]'

# 3) server mode: sandboxes on this machine, trainer anywhere
#   FLOTILLA_TOKEN=secret flotilla serve --host 0.0.0.0 --port 8642
#   FLOTILLA_URL=http://this-host:8642 FLOTILLA_TOKEN=secret flotilla run --count 50 -c '...' --wait
