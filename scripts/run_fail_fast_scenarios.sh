#!/usr/bin/env bash
# Validates that a lost participant fails the run quickly with a clear error (default fail-fast ON).
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src:.
V=${1:-reports/results/native_flower/v014/fail_fast}
ENVD=${ENV_DIR:-$HOME/runs}
mkdir -p "$V" "$ENVD"
run() { # name target when
  local name=$1 target=$2 when=$3
  [ -f "$V/$name.exit" ] && return
  mkdir -p "$V/$name"
  local t0=$(date +%s)
  python scripts/run_flower_docker.py --config configs/training/fedavg_digits_iid.yaml \
    --data-dir data/training/digits-docker-iid-v012 --run-dir "$V/$name" --env-file "$ENVD/ff-$name.env" \
    --app-dir experiments/flower_secagg --no-require-evaluation --timeout 400 \
    --fault-stop-client 2 --fault-target $target --fault-when $when \
    > "$V/$name.stdout.log" 2> "$V/$name.stderr.log"
  echo "exit=$? wall_seconds=$(( $(date +%s) - t0 ))" > "$V/$name.exit"
  docker system prune -f >/dev/null 2>&1
}
run clientexec-before-run clientexec before-run
run clientexec-after-round1 clientexec after-first-train
run supernode-after-round1 supernode after-first-train
echo done > "$V/fail_fast.done"
