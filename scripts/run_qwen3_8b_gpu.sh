#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
exec bash scripts/run_language_three_stage.sh
