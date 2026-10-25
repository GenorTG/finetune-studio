#!/bin/bash
# KL divergence of plain vs imatrix Q4_K_M against the f16 reference, on two texts (neither was calibration text).
set -u
: "${CUDA_VISIBLE_DEVICES:?set it to the training GPU (its UUID from nvidia-smi -L) so llama.cpp sees only that card}"
cd "$(git rev-parse --show-toplevel)"   # run from inside the checkout
B=.llama.cpp/build/bin/llama-perplexity
RUN=${RUN:?set RUN to the run folder, e.g. output/projects/<pid>/runs/<rid>}
F16=$RUN/gguf/model-f16.gguf
PLAIN=$RUN/gguf/model-q4_k_m.gguf
IMAT=$RUN/gguf-imatrix/model-q4_k_m.gguf
OUT=${OUT:-.tmp/imat-eval}   # holds korvane_docs.txt and general.txt
for t in korvane_docs general; do
  "$B" -m "$F16" -f "$OUT/$t.txt" --chunks 40 -c 512 --fit on --kl-divergence-base "$OUT/$t.f16logits" > "$OUT/$t.f16.log" 2>&1
  for q in plain imat; do
    M=$PLAIN; [ "$q" = imat ] && M=$IMAT
    "$B" -m "$M" -f "$OUT/$t.txt" --chunks 40 -c 512 --fit on --kl-divergence-base "$OUT/$t.f16logits" --kl-divergence > "$OUT/$t.$q.log" 2>&1
  done
  rm -f "$OUT/$t.f16logits"
done
echo DONE
