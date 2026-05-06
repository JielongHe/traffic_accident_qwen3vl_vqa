#!/usr/bin/env bash
set -euo pipefail

: "${ADAPTER:?Please set ADAPTER to your LoRA checkpoint directory, e.g. output/.../checkpoint-xxx}"

swift export \
  --adapters "$ADAPTER" \
  --merge_lora true
