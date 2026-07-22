# RSGNet Experiment Commands

Run commands from the project directory:

```bash
cd /home/pengfei/HTCnet/RSGNet
```

## Final Model

```bash
GPUS=2,3 \
PROTOCOL=sunrgbd37 \
SEED=3407 \
RUN_ID=sunrgbd37_seed3407 \
EPOCHS=100 \
BATCH_SIZE=4 \
TARGET_GLOBAL_BATCH=16 \
bash scripts/experiments.sh train A5_rsgnet
```

Resume the same run:

```bash
GPUS=2,3 RUN_ID=sunrgbd37_seed3407 \
bash scripts/experiments.sh resume A5_rsgnet
```

Evaluate the best checkpoint:

```bash
GPUS=2,3 RUN_ID=sunrgbd37_seed3407 \
bash scripts/experiments.sh eval A5_rsgnet
```

## Controlled Ablations

```bash
bash scripts/experiments.sh list
GPUS=2,3 RUN_ID=sunrgbd37_seed3407 \
bash scripts/experiments.sh train-suite
```

## Robustness

```bash
GPUS=2,3 RUN_ID=sunrgbd37_seed3407 \
bash scripts/experiments.sh robust A5_rsgnet
```

Each run contains checkpoints, logs, cached class weights, and evaluation JSON
files in one directory under `/home/pengfei/HTCnet/runs_rsgnet/`.

## Refined Development Ablation

The refined path is intentionally separate from the legacy A0-A5 checkpoints.
It uses reliability-conditioned pixel fusion, linear semantic mixing,
identity-preserving cross-scale residuals, and single-application reliability
routing. By default it disables intermediate test evaluation and early stopping;
the official test split is reserved for post-training reporting.

```bash
GPUS=2,3 \
PROTOCOL=sunrgbd37 \
RUN_ID=refined_seed3407 \
EPOCHS=100 \
BATCH_SIZE=8 \
TARGET_GLOBAL_BATCH=16 \
bash scripts/experiments.sh train-refined-suite
```

List or run one refined row:

```bash
bash scripts/experiments.sh list
bash scripts/experiments.sh config B5_refined_rsgnet
GPUS=2,3 RUN_ID=refined_seed3407 \
  bash scripts/experiments.sh train B5_refined_rsgnet
```

Use `eval-refined-suite` after training for the same multi-scale and flip TTA
protocol across B0-B5. Do not load an A0-A5 checkpoint into a B-row model.

## Topology-Aware Candidate

`D0_topology_reliability` adds boundary-aware local interaction at C1/C2 and a
token-bounded topology-biased attention block at C4. It uses the factorized
HHA path with final soft reliability gating, so invalid geometry is masked
without attenuating every intermediate residual. This is an opt-in candidate;
it does not modify the published C5 baseline.

```bash
GPUS=2,3 RUN_ID=topology_seed3407 EPOCHS=100 BATCH_SIZE=8 \
TARGET_GLOBAL_BATCH=16 FACTORIZED_HHA_DIR_NAME=HHA_PFHR \
FACTORIZED_HHA_CHANNEL_ORDER=dha \
  bash scripts/experiments.sh train D0_topology_reliability
```

Run `eval D0_topology_reliability` after training and compare against C5 under the same
split, TTA, and checkpoint-selection protocol.

## Physics-Factorized Paper Runs

Generate canonical geometry first:

```bash
/home/pengfei/miniconda3/envs/PFseg/bin/python \
  /home/pengfei/HTCnet/getDATA/prepare_sunrgbd.py/SUN2HHA.py \
  --raw-root /home/pengfei/datasets_Original/SUNRGBD/SUNRGBD_Raw \
  --out-root /home/pengfei/HTCnet/DataSets/SUNRGBD \
  --hha-dir-name HHA_PFHR --depth-dir-name Depth --workers 8
```

Then train C0-C6:

```bash
GPUS=2,3 RUN_ID=pfhr_seed3407 EPOCHS=100 BATCH_SIZE=8 \
TARGET_GLOBAL_BATCH=16 FACTORIZED_HHA_DIR_NAME=HHA_PFHR \
FACTORIZED_HHA_CHANNEL_ORDER=dha \
  bash scripts/experiments.sh train-factorized-suite
```

Run the raw-depth and channel controls independently:

```bash
GPUS=2,3 RUN_ID=depth_seed3407 \
  bash scripts/experiments.sh train R0_raw_depth_control

GPUS=2,3 RUN_ID=channels_seed3407 FACTORIZED_HHA_DIR_NAME=HHA_PFHR \
FACTORIZED_HHA_CHANNEL_ORDER=dha \
  bash scripts/experiments.sh train-channel-suite
```

Evaluate only after each fixed 100-epoch run:

```bash
GPUS=2,3 RUN_ID=pfhr_seed3407 FACTORIZED_HHA_DIR_NAME=HHA_PFHR \
FACTORIZED_HHA_CHANNEL_ORDER=dha \
  bash scripts/experiments.sh eval-factorized-suite
```

The HHA channel-relation candidate is isolated as `C9_hha_channel_relation`.
It keeps separate disparity, height, and angle stems, then learns spatial
channel weights and pairwise D-H/D-A/H-A residual relations. Both additions
are identity-initialized. For a fair academic ablation, train it from the
common MiT-B2 initialization with no segmentation checkpoint resume:

```bash
GPUS=2,3 RUN_ID=hha_relation_scratch_seed3407 EPOCHS=100 BATCH_SIZE=8 \
  VAL_FRACTION=0.1 VAL_SEED=2105 \
  FACTORIZED_HHA_DIR_NAME=HHA_PFHR FACTORIZED_HHA_CHANNEL_ORDER=dha \
  bash scripts/experiments.sh train C9_hha_channel_relation
```

For an engineering fine-tuning result, it can instead warm-start from the
best C5 weights; report that result separately:

```bash
GPUS=2,3 RUN_ID=hha_relation_seed3407 EPOCHS=100 BATCH_SIZE=8 \
  FACTORIZED_HHA_DIR_NAME=HHA_PFHR FACTORIZED_HHA_CHANNEL_ORDER=dha \
  RESUME_CKPT=/home/pengfei/HTCnet/runs_rsgnet/C5_factorized_routed/pfhr_c5ft60_seed3407/best_miou_0.4897_epoch_060.pth \
  bash scripts/experiments.sh finetune C9_hha_channel_relation
```

`C9_hha_channel_relation` now sets `VAL_INTERVAL=1`, so `train.log` records
one mIoU value after every epoch. For a clean, non-leaking development curve,
use the held-out 10% split below; the official test split should be evaluated
only once after selecting the checkpoint:

SUNRGBD's official partition contains 5,285 training images and 5,050 test
images. `VAL_FRACTION=0.1` changes only the training side to 4,757 train plus
528 held-out validation images; it does not remove data from SUNRGBD. To train
on all 5,285 official training images and still log one official-test mIoU per
epoch, use `VAL_FRACTION=0.0` explicitly:

```bash
GPUS=2,3 RUN_ID=hha_relation_full_seed3407 EPOCHS=100 BATCH_SIZE=8 \
  VAL_FRACTION=0.0 HHA_DEGRADE_PROB=0.0 \
  FACTORIZED_HHA_DIR_NAME=HHA_PFHR FACTORIZED_HHA_CHANNEL_ORDER=dha \
  RESUME_CKPT=/home/pengfei/HTCnet/runs_rsgnet/C5_factorized_routed/pfhr_c5ft60_seed3407/best_miou_0.4897_epoch_060.pth \
  bash scripts/experiments.sh finetune C9_hha_channel_relation
```

This full-data mode evaluates the official test split every epoch, so its
curve is diagnostic and selecting the best epoch by that curve is test-set
leakage. Use the held-out mode below for model selection and report official
test mIoU only once afterward.

After choosing `BEST_EPOCH` from the held-out curve, the paper-style final
run can suppress intermediate test evaluation while using all 5,285 training
images:

```bash
BEST_EPOCH=60
GPUS=2,3 RUN_ID=hha_relation_final_seed3407 \
  EPOCHS="$BEST_EPOCH" VAL_FRACTION=0.0 VAL_INTERVAL=100000 \
  HHA_DEGRADE_PROB=0.0 FACTORIZED_HHA_DIR_NAME=HHA_PFHR \
  FACTORIZED_HHA_CHANNEL_ORDER=dha \
  bash scripts/experiments.sh train C9_hha_channel_relation
```

Then run `eval C9_hha_channel_relation` once on the resulting final
checkpoint with the fixed SUNRGBD-37 TTA settings.

```bash
GPUS=2,3 RUN_ID=hha_relation_val_seed3407 EPOCHS=100 BATCH_SIZE=8 \
  VAL_FRACTION=0.1 VAL_SEED=2105 HHA_DEGRADE_PROB=0.0 \
  FACTORIZED_HHA_DIR_NAME=HHA_PFHR FACTORIZED_HHA_CHANNEL_ORDER=dha \
  RESUME_CKPT=/home/pengfei/HTCnet/runs_rsgnet/C5_factorized_routed/pfhr_c5ft60_seed3407/best_miou_0.4897_epoch_060.pth \
  bash scripts/experiments.sh finetune C9_hha_channel_relation
```

Compare C9 against C5 using the same split, seed, epoch budget, EMA, TTA,
and checkpoint-selection rule. Do not infer an mIoU gain from the CPU smoke
test; a gain is established only after the fixed validation/test protocol.

Generate HHA from physically degraded raw geometry for the official 5,050-image
test split. The default `core` profile creates one primary severity for each
failure type and writes a separate manifest inside every HHA directory:

```bash
PFHR_ROBUST_PROFILE=core \
  bash scripts/experiments.sh prepare-pfhr-robustness

PFHR_ROBUST_PROFILE=core \
  bash scripts/experiments.sh verify-pfhr-robustness

PFHR_ROBUST_PROFILE=core \
  bash scripts/experiments.sh analyze-pfhr-robustness
```

Evaluate the clean C0 control and clean/degraded C5-C6 checkpoints:

```bash
GPUS=2,3 RUN_ID=pfhr_seed3407 PFHR_ROBUST_PROFILE=core \
  bash scripts/experiments.sh robust-physical-factorized
```

The core settings are raw-depth dropout area 0.30, Gaussian depth noise with
sigma 0.05 m, raw-depth shift 8 pixels, and tilt-rotation error 5 degrees.
Use `PFHR_ROBUST_PROFILE=full` for three severity levels per corruption. The
existing `robust-factorized` action instead modifies encoded HHA directly and
is reported only as a representation-space diagnostic.
