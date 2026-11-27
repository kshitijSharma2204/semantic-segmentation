#!/usr/bin/env bash
# Runs the long parts of the v2 work in order. Each stage shows tqdm progress
# and also logs to outputs/logs/<stage>.log.
#
#   bash run_v2.sh                 tests, rescore, train, eval, media
#   bash run_v2.sh train           only the v2 training run (resumes if last.pt exists)
#   bash run_v2.sh bench resume-check   the optional checks
#
# Stages: tests rescore bench resume-check train eval media
# Run it inside tmux (tmux new -s v2) so the training survives a closed terminal.
# EPOCHS=600 and EXTRA="--schedule cosine" etc. can be set from the environment.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-semantic_segmentation_venv/bin/python}
EPOCHS=${EPOCHS:-600}
EXTRA=${EXTRA:-}
mkdir -p outputs/logs

log() { tee "outputs/logs/$1.log"; }
banner() { echo; echo "=== $* ($(date '+%H:%M:%S')) ==="; }

stage_tests() {
    banner "unit tests"
    $PY -m unittest discover -s tests -v 2>&1 | log tests
}

stage_rescore() {
    banner "re-scoring the 50 v1 checkpoints on splits/val.txt"
    $PY evaluate.py outputs/v1/checkpoints/unet_epoch*.pt --out-dir outputs/v1_rescore 2>&1 | log rescore
    $PY plot_rescore.py --eval-dir outputs/v1_rescore --out-dir results/v1_rescore 2>&1 | tee -a outputs/logs/rescore.log
}

stage_bench() {
    banner "speed options, 50 timed steps each"
    rm -f outputs/bench.md
    $PY bench.py train --out outputs/bench.md 2>&1 | log bench
    $PY bench.py confusion --out outputs/bench.md 2>&1 | tee -a outputs/logs/bench.log
    $PY bench.py synthetic --classes 34 --batch 8 --no-cudnn-benchmark --out outputs/bench.md 2>&1 | tee -a outputs/logs/bench.log
    $PY bench.py synthetic --classes 19 --batch 4 --height 352 --width 592 --out outputs/bench.md 2>&1 | tee -a outputs/logs/bench.log
    $PY bench.py synthetic --classes 19 --batch 4 --height 352 --width 592 --bf16 --out outputs/bench.md 2>&1 | tee -a outputs/logs/bench.log
    $PY bench.py synthetic --classes 19 --batch 8 --height 352 --width 592 --bf16 --out outputs/bench.md 2>&1 | tee -a outputs/logs/bench.log
    echo "table written to outputs/bench.md"
}

stage_resume_check() {
    # 1 x 20 epochs against 10 + resume + 10, same schedule. About 25 minutes in total.
    banner "resume check: 1 x 20 epochs"
    $PY train.py --run-name resume_check/full --epochs 20 --val-every 5 $EXTRA 2>&1 | log resume_full
    banner "resume check: 10 epochs, then resume for 10 more"
    rm -rf outputs/resume_check/split
    $PY train.py --run-name resume_check/split --epochs 20 --val-every 5 --stop-after 10 $EXTRA 2>&1 | log resume_split
    $PY train.py --resume outputs/resume_check/split/checkpoints/last.pt 2>&1 | tee -a outputs/logs/resume_split.log
    $PY - <<'EOF'
import csv
a = list(csv.DictReader(open("outputs/resume_check/full/log.csv")))
b = list(csv.DictReader(open("outputs/resume_check/split/log.csv")))
print("epoch  train loss (1x20 / 10+10)   val mIoU (1x20 / 10+10)")
for x, y in zip(a, b):
    print(f"{x['epoch']:>5}  {x['train_loss']:>10} / {y['train_loss']:<10}  {x['val_miou']:>8} / {y['val_miou']}")
EOF
}

stage_train() {
    if [ -f outputs/v2/checkpoints/last.pt ]; then
        banner "resuming the v2 run from outputs/v2/checkpoints/last.pt"
        $PY train.py --resume outputs/v2/checkpoints/last.pt 2>&1 | tee -a outputs/logs/train.log
    else
        banner "v2 training run, $EPOCHS epochs"
        $PY train.py --run-name v2 --epochs "$EPOCHS" $EXTRA 2>&1 | log train
    fi
    mkdir -p results/v2
    cp outputs/v2/curves.png results/v2/training_curves.png
}

stage_eval() {
    banner "scoring the v2 best.pt and last.pt with the Phase 1 protocol"
    $PY evaluate.py outputs/v2/checkpoints/best.pt outputs/v2/checkpoints/last.pt --out-dir outputs/v2/eval 2>&1 | log eval
    mkdir -p results/v2
    cp outputs/v2/eval/best_classes.csv results/v2/best_classes.csv
}

stage_media() {
    banner "README images and GIF for v2"
    $PY make_media.py 2>&1 | log media
    ls -la results/v2
}

stages=("$@")
[ ${#stages[@]} -eq 0 ] && stages=(tests rescore train eval media)
for s in "${stages[@]}"; do
    "stage_${s//-/_}"
done
banner "finished: ${stages[*]}"
