#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ -x "$PROJECT_DIR/.venv/Scripts/python.exe" ]]; then
    PYTHON="$PROJECT_DIR/.venv/Scripts/python.exe"
elif [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    PYTHON=python3
fi

export PYTHONUTF8=1
# Use a reachable Hugging Face endpoint by default; set HF_ENDPOINT to override.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
OIE_LLM="${OIE_LLM:-Qwen/Qwen3-1.7B}"
SD_LLM="${SD_LLM:-Qwen/Qwen3-1.7B}"
SC_LLM="${SC_LLM:-Qwen/Qwen3-1.7B}"
SC_EMBEDDER="${SC_EMBEDDER:-intfloat/multilingual-e5-small}"
DATASET="${DATASET:-example}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/${DATASET}_target_alignment_$(date +%Y%m%d_%H%M%S)}"

"$PYTHON" run.py \
    --oie_llm "$OIE_LLM" \
    --oie_few_shot_example_file_path "./few_shot_examples/${DATASET}/oie_few_shot_examples.txt" \
    --sd_llm "$SD_LLM" \
    --sd_few_shot_example_file_path "./few_shot_examples/${DATASET}/sd_few_shot_examples.txt" \
    --sc_llm "$SC_LLM" \
    --sc_embedder "$SC_EMBEDDER" \
    --input_text_file_path "./datasets/${DATASET}.txt" \
    --target_schema_path "./schemas/${DATASET}_schema.csv" \
    --output_dir "$OUTPUT_DIR" \
    --logging_verbose
