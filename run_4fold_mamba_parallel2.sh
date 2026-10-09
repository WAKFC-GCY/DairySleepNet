#!/usr/bin/env bash
# Usage: GPUS=0,1,2,3 bash run_4fold_mamba_parallel2.sh PSG [training options]
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
MODALITY="${1:-PSG}"
if (( $# > 0 )); then shift; fi
MODALITY="$($PYTHON -c 'import sys; sys.path.insert(0, sys.argv[1]); from args import canonical_modality; print(canonical_modality(sys.argv[2]))' "$SCRIPT_DIR" "$MODALITY")"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/outputs/$MODALITY}"
IFS=',' read -r -a GPU_IDS <<< "${GPUS:-0}"
for gpu in "${GPU_IDS[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "GPUS must be a comma-separated list of GPU indices" >&2; exit 2; }
done
mkdir -p "$OUT_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
pids=()
trap 'for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; exit 130' INT TERM
wait_batch() {
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then failed=1; fi
    done
    pids=()
    if (( failed )); then
        echo "A fold failed. Inspect $OUT_DIR/fold*.log" >&2
        exit 1
    fi
}
for fold in 1 2 3 4; do
    gpu="${GPU_IDS[$(( (fold - 1) % ${#GPU_IDS[@]} ))]}"
    if [[ -e "$OUT_DIR/fold${fold}.log" ]]; then
        echo "Existing fold log: $OUT_DIR/fold${fold}.log. Choose a fresh OUT_DIR." >&2
        wait_batch
        exit 1
    fi
    "$PYTHON" -u "$SCRIPT_DIR/kfold4_subject_level_mamba1.py" "$@" \
        --modality "$MODALITY" --fold "$fold" --gpu "$gpu" --out-dir "$OUT_DIR" \
        > "$OUT_DIR/fold${fold}.log" 2>&1 &
    pids+=("$!")
    echo "Started fold $fold on GPU $gpu; log: $OUT_DIR/fold${fold}.log"
    if (( ${#pids[@]} == ${#GPU_IDS[@]} )); then wait_batch; fi
done
wait_batch
"$PYTHON" "$SCRIPT_DIR/kfold4_subject_level_mamba1.py" --summary --out-dir "$OUT_DIR"
