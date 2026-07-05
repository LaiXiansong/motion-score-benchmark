# Motion Magnitude Detection Benchmark for Human Videos

This project compares three ways to estimate motion magnitude in human videos:

1. **Optical flow**: SEA-RAFT with the VBench-style top-5% flow magnitude score.
2. **Human keypoints**: RTMO/RTMPose via `rtmlib`, using scale-normalized joint velocity.
3. **Learned scorer**: VideoMAE backbone with a dual head — tier classification plus motion regression.

The intended dataset is a curated Kinetics-700 subset spanning low, medium, and high motion:
speaking-like clips, exercise/dance clips, and fighting or combat-sport clips.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### WSL + Clash for Windows proxy (required in China / restricted networks)

WSL does **not** inherit Windows proxy automatically. Use the gateway IP (not `127.0.0.1`).

On Windows (PowerShell), ensure Clash has **Allow LAN** enabled. Optional helper:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/windows_enable_clash_lan.ps1
```

In WSL, enable and persist proxy:

```bash
source scripts/setup_wsl_proxy.sh --persist --test
```

This writes `~/.bashrc.d/clash-proxy.sh` and auto-detects the Windows host IP on each shell start.
Test anytime:

```bash
bash scripts/test_wsl_proxy.sh
```

If HuggingFace is still slow after proxy works, you can additionally set:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

System tools are also required:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

### ONNX Runtime (pose scoring)

For GPU pose inference, install **only** `onnxruntime-gpu`. Do not install the CPU `onnxruntime` package alongside it — they conflict and will silently fall back to CPU.

### SEA-RAFT optical flow

Clone SEA-RAFT and download a general-purpose checkpoint (recommended for this benchmark):

```bash
git clone https://github.com/princeton-vl/SEA-RAFT external/SEA-RAFT
export SEA_RAFT_ROOT=$PWD/external/SEA-RAFT
export SEA_RAFT_CHECKPOINT=$PWD/external/SEA-RAFT/models/Tartan-C-T-TSKH432x960-M.pth
```

Use with `config/eval/sintel-M.json` (handled automatically by `src/flow_score.py`). This Tartan-trained checkpoint generalizes better across datasets than the Spring-tuned demo model.

Without SEA-RAFT, `src/flow_score.py --backend auto` falls back to OpenCV Farneback for smoke tests only.

## 1. Download Kinetics Clips

Prepare or download an official Kinetics annotation CSV with columns equivalent to:
`youtube_id,time_start,time_end,label`.

```bash
python scripts/download_kinetics.py \
  --annotations data/train.csv \
  --config configs/classes.yaml \
  --output-dir data/clips \
  --labels data/labels.csv
```

The class-to-tier selection is in `configs/classes.yaml`.

## 2. Create Human Rating Sheet

```bash
python scripts/create_rating_sheet.py \
  --labels data/labels.csv \
  --output data/human_rating_sheet.csv \
  --clips 60
```

Fill `human_rating` with integers 1-5, where 1 means nearly static and 5 means very intense motion.
Merge the completed ratings back into `data/labels.csv` before final evaluation.

```bash
python scripts/merge_human_ratings.py \
  --labels data/labels.csv \
  --ratings data/human_rating_sheet.csv \
  --output data/labels_with_ratings.csv
```

When human ratings are present, training can distill from them with `--distill-target human`.

## 3. Pose Scores

```bash
python -m src.pose_score \
  --labels data/labels.csv \
  --output results/pose_scores.csv \
  --boxes-json results/pose_boxes.json \
  --device cuda
```

The `pose_boxes.json` file can be reused by optical flow to compute foreground-masked flow.

## 4. Optical-Flow Scores

```bash
export SEA_RAFT_ROOT=$PWD/external/SEA-RAFT
export SEA_RAFT_CHECKPOINT=$PWD/external/SEA-RAFT/models/Tartan-C-T-TSKH432x960-M.pth

python -m src.flow_score \
  --labels data/labels.csv \
  --output results/flow_scores.csv \
  --boxes-json results/pose_boxes.json \
  --backend sea-raft \
  --sea-raft-root "$SEA_RAFT_ROOT" \
  --sea-raft-checkpoint "$SEA_RAFT_CHECKPOINT" \
  --device cuda
```

The main score is `flow_foreground` when pose boxes are available, otherwise `flow_global`.

## 5. Prepare Training CSV

Merge flow and pose scores into one CSV for the learned method:

```bash
python -m src.evaluate \
  --labels data/labels.csv \
  --score-csv results/flow_scores.csv \
  --score-csv results/pose_scores.csv \
  --output results/fused_training_scores.csv \
  --metrics results/pretrain_metrics.json
```

## 6. Train Learned Scorer

Recommended defaults (frozen backbone, pose distillation, combined checkpoint selection):

```bash
python -m src.train_learned \
  --labels results/fused_training_scores.csv \
  --output checkpoints/learned_motion.pt \
  --epochs 10 \
  --batch-size 4 \
  --freeze-backbone \
  --distill-target pose \
  --lambda-reg 2.0 \
  --selection-metric combined \
  --device cuda
```

Training options:

| Flag | Default | Purpose |
|------|---------|---------|
| `--freeze-backbone` | on | Train only the motion head; much faster and more stable on ~200 clips |
| `--no-freeze-backbone` | — | Fine-tune the full VideoMAE backbone (needs more VRAM) |
| `--distill-target` | `pose` | Regression target: `pose`, `flow`, `fused`, `tier`, or `human` |
| `--lambda-reg` | `2.0` | Weight of regression loss vs tier classification |
| `--selection-metric` | `combined` | Save best checkpoint by `tier_acc`, `spearman_tier_prob`, `spearman_reg`, or `combined` |

The regression head outputs a linear score (no sigmoid); values are clamped to `[0, 1]` during training and inference.

## 7. Learned Scores

```bash
python -m src.learned_score \
  --labels data/labels.csv \
  --checkpoint checkpoints/learned_motion.pt \
  --output results/learned_scores.csv \
  --device cuda
```

Each video gets several learned columns:

| Column | Meaning |
|--------|---------|
| `learned_score` | **Primary benchmark score** — tier probability expectation: `0·P(low) + 0.5·P(medium) + 1.0·P(high)` |
| `learned_tier_prob` | Same as `learned_score` (alias for clarity) |
| `learned_motion` | Regression head output, clamped to `[0, 1]` |
| `learned_motion_raw` | Unclamped regression output |
| `learned_tier_pred` | Argmax tier prediction (1/2/3) |
| `learned_prob_*` | Per-tier softmax probabilities |

**Why two continuous scores?** Tier classification often separates low/medium/high well even when the regression head collapses. Evaluation uses `learned_score` (tier-probability) as the main learned method; `learned_motion` is reported separately for diagnosis.

## 8. Final Evaluation

```bash
python -m src.evaluate \
  --labels data/labels.csv \
  --score-csv results/flow_scores.csv \
  --score-csv results/pose_scores.csv \
  --score-csv results/learned_scores.csv \
  --output results/scores.csv \
  --metrics results/metrics.json \
  --plots-dir results/plots
```

Outputs:

- `results/scores.csv`: per-video raw and normalized scores.
- `results/metrics.json`: Spearman/Kendall vs tiers and human ratings, tier accuracy, confusion matrix, inter-method correlation, `learned_tier_accuracy`, and auxiliary metrics for `learned_motion` vs `learned_tier_prob`.
- `results/plots/`:
  - `tier_boxplot_{flow,pose,learned,fused}.png` — per-method normalized scores
  - `tier_boxplot_learned_motion.png` — regression head only
  - `tier_boxplot_learned_tier_prob.png` — tier-probability score
  - `tier_boxplot_comparison.png` — side-by-side comparison
  - `method_correlation.png` — inter-method Spearman heatmap
  - `human_scatter.png` — human rating agreement (when ratings are filled)

## One-command Runner

After data is ready, the non-training path can be run with:

```bash
python scripts/run_benchmark.py \
  --labels data/labels_with_ratings.csv \
  --results-dir results \
  --device cuda \
  --flow-backend sea-raft \
  --sea-raft-root "$SEA_RAFT_ROOT" \
  --sea-raft-checkpoint "$SEA_RAFT_CHECKPOINT" \
  --learned-checkpoint checkpoints/learned_motion.pt
```

For a CPU-only smoke test that skips pose and learned models:

```bash
python scripts/make_synthetic_dataset.py \
  --output-dir data/synthetic \
  --labels data/synthetic_labels.csv

python scripts/run_benchmark.py \
  --labels data/synthetic_labels.csv \
  --results-dir results/synthetic \
  --flow-backend farneback \
  --device cpu \
  --skip-pose \
  --skip-learned
```

## Interpreting Results

On the current ~230-clip Kinetics subset (without human ratings), typical Spearman correlation vs tier labels is:

- **pose** (~0.49): best single geometric signal for human motion magnitude.
- **flow** (~0.40): useful but mixes camera motion with body motion.
- **learned_score** (tier probability): should track tier labels much better than `learned_motion` alone.
- **fused**: average of normalized flow + pose + learned scores.

If `learned_motion` stays flat across tiers while `learned_tier_pred` accuracy is high, the classifier is working but regression distillation needs tuning — try `--distill-target pose`, higher `--lambda-reg`, or fill human ratings.

## Notes

- Kinetics videos can have camera motion. Use the pose-box foreground mask for optical flow whenever possible.
- Pose scores are normalized by person scale, which makes close-up talking and full-body dance clips more comparable.
- The learned method is intentionally a small head on a pretrained backbone; the curated dataset is too small to train a video model from scratch.
- Retrain after changing distillation or backbone settings, then re-run `src.learned_score` before final evaluation.
