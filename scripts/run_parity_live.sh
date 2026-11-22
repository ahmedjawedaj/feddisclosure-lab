#!/usr/bin/env bash
# Live confirmation of the opt-in DeterministicFedAvg on real Flower in Docker (hosted runner).
# Each scenario runs twice on the same data, once with the upstream strategy and once with
# aggregation="deterministic", then the unchanged 1e-5 float64 parity gate is applied to both.
# Failures are expected for some upstream arms and are recorded, not hidden.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src:.
V=${1:-reports/results/native_flower/v0150/live_parity}
ENVD=${ENV_DIR:-/tmp/parity-env}
mkdir -p "$ENVD" "$V"
scenario() { # name config datadir arm extra-run-config
  local name=$1 cfg=$2 data=$3 arm=$4 extra=$5
  local dir="$V/$name-$arm"
  mkdir -p "$dir"
  python scripts/run_flower_docker.py --config "$cfg" --data-dir "$data" --run-dir "$dir" \
    --env-file "$ENVD/$name-$arm.env" ${extra:+--extra-run-config "$extra"} --timeout 1500 \
    > "$dir.stdout.log" 2> "$dir.stderr.log"
  echo "run_exit=$?" > "$dir.exit"
  python scripts/run_torch_reference.py --config "$cfg" --data-dir "$data" \
    --output "$dir/torch_reference.json" > "$dir/torch_reference.stdout.log" 2>&1
  local id; id=$(python -c "import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))['id'])" "$cfg")
  python scripts/verify_flower_parity.py --flower-dir "$dir/server" \
    --reference-json "$dir/torch_reference.json" \
    --reference-checkpoint "$dir/checkpoints/${id}_final.pt" --manifest "$data/manifest.json" \
    --output "$dir/flower_parity.json" --require-deployment \
    > "$dir/flower_parity.stdout.log" 2>&1
  echo "parity_exit=$?" >> "$dir.exit"
}
for arm in upstream deterministic; do
  extra=""; [ "$arm" = deterministic ] && extra='aggregation="deterministic"'
  scenario iid3        configs/training/fedavg_digits_iid.yaml                 data/training/digits-live-iid3        "$arm" "$extra"
  scenario noniid3     configs/training/fedavg_digits_non_iid.yaml             data/training/digits-live-noniid3     "$arm" "$extra"
  scenario iid4-10r    configs/training/fedavg_digits_iid_4clients_10round.yaml data/training/digits-live-iid4-10r   "$arm" "$extra"
done
echo done > "$V/live_parity.done"
