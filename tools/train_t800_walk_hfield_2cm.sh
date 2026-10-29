#!/usr/bin/env bash
# Upstream TrainCfg defaults; choose only the robot, clip, floor and run length.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec uv run python -m morphmimic.scripts.train \
  --robot t800 \
  --subset-yaml data/t800/subsets/walk_straight_slow.yaml \
  --terrain smooth_random_2cm \
  --iterations 20000 \
  --run-name t800_walk_hfield_2cm_20k \
  "$@"
