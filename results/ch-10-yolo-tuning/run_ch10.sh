#!/usr/bin/env bash
# CH-10 full runs, one after another on one GPU. Every step is resumable: re-running this script skips
# finished work, resumes an interrupted training from outputs/<run>/train/weights/last.pt, and the curve
# step skips epochs already in results/<run>/curve.json.
#
#   bash results/ch-10-yolo-tuning/run_ch10.sh [python] [run ...]     (from the repository root)
#
# Default runs: yolo_imgsz1600 yolo_p2 yolo_baseline_5070. Logs: outputs/ch10_logs/<run>_{train,curve,final,failures}.log
# yolo_p2 needs outputs/weights/yolo11s-p2-init.pt from scripts/make_p2_init.py.
set -euo pipefail
PY=${1:-python}
shift || true
RUNS=("$@")
[ ${#RUNS[@]} -eq 0 ] && RUNS=(yolo_imgsz1600 yolo_p2 yolo_baseline_5070)
mkdir -p outputs/ch10_logs

for run in "${RUNS[@]}"; do
  log=outputs/ch10_logs/$run
  weights=outputs/$run/train/weights
  if [ -f "results/$run/metrics.json" ] && [ -f "results/$run/speed.json" ] && [ -f "results/$run/failure_cases.csv" ]; then
    echo "$(date '+%F %T') $run: finished, skipped"
    continue
  fi
  if [ -f "$weights/epoch5.pt" ] && grep -q '"status": "completed"' "results/$run/runtime.json" 2>/dev/null; then
    echo "$(date '+%F %T') $run: training already complete"
  elif [ -f "$weights/last.pt" ]; then
    echo "$(date '+%F %T') $run: resuming training"
    "$PY" scripts/train_ultra.py --config "configs/runs/$run.json" --device 0 --resume --observe >> "${log}_train.log" 2>&1
  else
    echo "$(date '+%F %T') $run: training"
    "$PY" scripts/train_ultra.py --config "configs/runs/$run.json" --device 0 --observe > "${log}_train.log" 2>&1
  fi
  echo "$(date '+%F %T') $run: val_fast curve"
  "$PY" scripts/predict_ultra.py curve --run-id "$run" --device 0 >> "${log}_curve.log" 2>&1
  if [ ! -f "results/$run/speed.json" ]; then
    echo "$(date '+%F %T') $run: val_full + speed"
    "$PY" scripts/predict_ultra.py final --run-id "$run" --device 0 > "${log}_final.log" 2>&1
  fi
  echo "$(date '+%F %T') $run: failure cases"
  "$PY" scripts/failures_ultra.py --run-id "$run" > "${log}_failures.log" 2>&1
  echo "$(date '+%F %T') $run: done"
done
