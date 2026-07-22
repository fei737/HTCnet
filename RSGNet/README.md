# RSGNet

RSGNet is a Reliability-aware Stage-wise Geometry-guided Network for RGB-D
semantic segmentation. Its paper-facing PFHR path keeps disparity, height, and
angle separate until it constructs layout and boundary relations. RGB remains
the only semantic backbone.

## Model Line

- Geometry encoding: channel-specific D/H/A stems, a D-H low-frequency layout
  relation, and an adaptive D/H/A high-frequency boundary relation.
- Routing: separate layout and boundary prompts with static, swapped, and
  content-adaptive stage controls.
- Stage guidance: pixel interaction at C1/C2, a prompt adapter at C3, and
  linear geometry-conditioned semantic mixing at C4.
- Reliability head: separate layout and boundary reliability with explicit
  invalid-pixel masking and controlled degradation supervision.
- Decoder: fixed lightweight multi-scale segmentation decoder.

## Quick Start

```bash
bash scripts/experiments.sh list
```

Generate canonical HHA and the aligned raw-depth control once:

```bash
/home/pengfei/miniconda3/envs/PFseg/bin/python \
  /home/pengfei/HTCnet/getDATA/prepare_sunrgbd.py/SUN2HHA.py \
  --raw-root /home/pengfei/datasets_Original/SUNRGBD/SUNRGBD_Raw \
  --out-root /home/pengfei/HTCnet/DataSets/SUNRGBD \
  --hha-dir-name HHA_PFHR \
  --depth-dir-name Depth \
  --workers 8
```

Run the paper-facing physics-factorized suite:

```bash
GPUS=2,3 \
RUN_ID=pfhr_seed3407 \
FACTORIZED_HHA_DIR_NAME=HHA_PFHR \
FACTORIZED_HHA_CHANNEL_ORDER=dha \
bash scripts/experiments.sh train-factorized-suite
```

Outputs are isolated under `../runs_rsgnet/<experiment>/<run_id>/`.
See `docs/SUNRGBD37_PROTOCOL.md` for the reporting protocol.

## Refined Model Line

The legacy A0-A5 sequence is kept unchanged for reproducibility. The refined
implementation is opt-in and adds four controlled changes: reliability-
conditioned pixel interaction at C1/C2, linear geometry-conditioned semantic
mixing at C4, identity-preserving cross-scale residuals, and one final
reliability application. It is implemented without a second geometry backbone.

Run the complete refined development ablation with a fixed 100-epoch schedule
and test evaluation only at the end:

```bash
GPUS=2,3 \
PROTOCOL=sunrgbd37 \
RUN_ID=refined_seed3407 \
BATCH_SIZE=8 \
TARGET_GLOBAL_BATCH=16 \
EPOCHS=100 \
bash scripts/experiments.sh train-refined-suite
```

The refined rows are `B0_refined_rgb` through `B5_refined_rsgnet`. Their
checkpoints are stored separately under `../runs_rsgnet/` and cannot load the
legacy A0-A5 architecture checkpoints.

## Physics-Factorized Model Line

The C line preserves disparity, height, and angle embeddings until it builds a
disparity-height layout relation and a D/H/A high-frequency boundary relation.
`C3` uses correct fixed stage priors, `C4` swaps them, `C5` learns spatial
routes, and `C6` adds separate layout and boundary reliability. C checkpoints
cannot load A/B weights.

The processed `HHA_PFHR/` directory is canonical `dha` and is the C/H default.
The historical `HHA/` directory and loader behavior are retained only for A/B
checkpoint reproducibility. It must not be used for paper-facing C results; a
diagnostic C run on those files must explicitly set the directory to `HHA` and
the decoded order to `ahd`.

Raw-depth and channel-role controls are separate from the default C suite:

```bash
GPUS=2,3 RUN_ID=depth_seed3407 \
  bash scripts/experiments.sh train R0_raw_depth_control

GPUS=2,3 RUN_ID=channels_seed3407 \
FACTORIZED_HHA_DIR_NAME=HHA_PFHR FACTORIZED_HHA_CHANNEL_ORDER=dha \
  bash scripts/experiments.sh train-channel-suite
```

Generate the primary physical robustness set from raw depth for the official
test split. The core profile uses 30% dropout, 0.05 m depth noise, an 8-pixel
shift, and a 5-degree tilt calibration error:

```bash
PFHR_ROBUST_PROFILE=core \
  bash scripts/experiments.sh prepare-pfhr-robustness

PFHR_ROBUST_PROFILE=core \
  bash scripts/experiments.sh verify-pfhr-robustness

PFHR_ROBUST_PROFILE=core \
  bash scripts/experiments.sh analyze-pfhr-robustness
```

After C0, C5, and C6 have been trained with the same `RUN_ID`, evaluate the
clean and physically degraded inputs without applying a second image-space
corruption:

```bash
GPUS=2,3 RUN_ID=pfhr_seed3407 PFHR_ROBUST_PROFILE=core \
  bash scripts/experiments.sh robust-physical-factorized
```

Set `PFHR_ROBUST_PROFILE=full` to generate and evaluate three severity levels
for each physical corruption. `robust-factorized` remains the cheaper
representation-space HHA diagnostic. The analysis action writes
`SUNRGBD/pfhr_robustness_core.json` with channel-wise corruption statistics.
