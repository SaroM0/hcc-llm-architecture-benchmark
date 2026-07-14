#!/bin/bash
# Run A2 and A3 with best models only

set -a
source .env
set +a

echo "OPENROUTER_BASE_URL: ${OPENROUTER_BASE_URL:0:30}..."
echo "Starting experiments..."

echo "Starting A2 with claude_opus_4_8..."
PYTHONPATH=src python3 -m oncology_rag.cli.eval matrix \
  --dataset data/eval/sct_validated_ground_truth.csv \
  --provider-config configs/providers/openrouter.yaml \
  --embeddings-config configs/rag/embeddings.yaml \
  --chroma-config configs/rag/chroma.yaml \
  --runs-dir runs \
  --arms A2 \
  --model-groups best_large &
A2_PID=$!

echo "Starting A3 with qwen3_6_27b..."
PYTHONPATH=src python3 -m oncology_rag.cli.eval matrix \
  --dataset data/eval/sct_validated_ground_truth.csv \
  --provider-config configs/providers/openrouter.yaml \
  --embeddings-config configs/rag/embeddings.yaml \
  --chroma-config configs/rag/chroma.yaml \
  --runs-dir runs \
  --arms A3 \
  --model-groups best_small &
A3_PID=$!

echo "A2 PID: $A2_PID, A3 PID: $A3_PID"
echo "Waiting for experiments to complete..."

while true; do
  a2_count=$(find runs/matrix -name "*_A2_claude_opus_4_8" -type d 2>/dev/null | head -1 | xargs -I{} sh -c 'wc -l < "{}/predictions.jsonl" 2>/dev/null || echo 0')
  a3_count=$(find runs/matrix -name "*_A3_qwen3_6_27b" -type d 2>/dev/null | head -1 | xargs -I{} sh -c 'wc -l < "{}/predictions.jsonl" 2>/dev/null || echo 0')
  echo "Progress - A2: $a2_count/88, A3: $a3_count/88"

  if ! kill -0 $A2_PID 2>/dev/null && ! kill -0 $A3_PID 2>/dev/null; then
    echo "Both experiments finished!"
    break
  fi

  sleep 10
done

wait $A2_PID
A2_EXIT=$?
wait $A3_PID
A3_EXIT=$?

echo "A2 exit code: $A2_EXIT"
echo "A3 exit code: $A3_EXIT"
echo "Done!"
