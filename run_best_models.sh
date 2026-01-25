#!/bin/bash
# Run A3 and A4 with best models only

set -a
source .env
set +a

echo "OPENROUTER_BASE_URL: ${OPENROUTER_BASE_URL:0:30}..."
echo "Starting experiments..."

# Run A3 with best large model (gpt52)
echo "Starting A3 with gpt52..."
PYTHONPATH=src python3 -m oncology_rag.cli.eval matrix \
  --dataset data/eval/sct_validated_ground_truth.csv \
  --provider-config configs/providers/openrouter.yaml \
  --embeddings-config configs/rag/embeddings.yaml \
  --chroma-config configs/rag/chroma.yaml \
  --runs-dir runs \
  --arms A3 \
  --model-groups best_large &
A3_PID=$!

# Run A4 with best small model (qwen3_vl_30b)
echo "Starting A4 with qwen3_vl_30b..."
PYTHONPATH=src python3 -m oncology_rag.cli.eval matrix \
  --dataset data/eval/sct_validated_ground_truth.csv \
  --provider-config configs/providers/openrouter.yaml \
  --embeddings-config configs/rag/embeddings.yaml \
  --chroma-config configs/rag/chroma.yaml \
  --runs-dir runs \
  --arms A4 \
  --model-groups best_small &
A4_PID=$!

echo "A3 PID: $A3_PID, A4 PID: $A4_PID"
echo "Waiting for experiments to complete..."

# Monitor progress
while true; do
  a3_count=$(find runs/matrix -name "*_A3_gpt52" -type d 2>/dev/null | head -1 | xargs -I{} sh -c 'wc -l < "{}/predictions.jsonl" 2>/dev/null || echo 0')
  a4_count=$(find runs/matrix -name "*_A4_qwen3_vl_30b" -type d 2>/dev/null | head -1 | xargs -I{} sh -c 'wc -l < "{}/predictions.jsonl" 2>/dev/null || echo 0')
  echo "Progress - A3: $a3_count/88, A4: $a4_count/88"

  # Check if processes finished
  if ! kill -0 $A3_PID 2>/dev/null && ! kill -0 $A4_PID 2>/dev/null; then
    echo "Both experiments finished!"
    break
  fi

  sleep 10
done

wait $A3_PID
A3_EXIT=$?
wait $A4_PID
A4_EXIT=$?

echo "A3 exit code: $A3_EXIT"
echo "A4 exit code: $A4_EXIT"
echo "Done!"
