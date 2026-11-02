#!/usr/bin/env bash
# Download released models into models/<name>.
#   bash scripts/download_models.sh                 # Stage2
#   bash scripts/download_models.sh Stage1 Stage2
set -euo pipefail
cd "$(dirname "$0")/.."
for name in "${@:-Stage2}"; do
  case "$name" in
    Stage1|Stage2) hf download "niejunnan25/Event2Reward-Qwen3.5-4B-${name}" --local-dir "models/${name}" ;;
    *) echo "usage: $0 [Stage1] [Stage2]   (the LIBERO models are not yet released)" >&2; exit 2 ;;
  esac
done
