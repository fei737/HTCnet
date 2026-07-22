# SUNRGBD-37 Protocol

`PROTOCOL=sunrgbd37` is the paper-facing SUN RGB-D setting:

- official 5,285 training and 5,050 testing split;
- 37 evaluated classes;
- void and excluded NYU40 labels mapped to 255;
- 480 x 480 training crops;
- native-resolution sliding-window evaluation;
- fixed MiT-B2 initialization for every ablation;
- fixed lightweight decoder for every ablation;
- dense CE with Lovasz and edge supervision;
- OHEM disabled by default;
- target global batch size fixed across GPU counts;
- legacy development runs may early-stop only after epoch 30;
- paper-facing B/C runs train for a fixed 100 epochs and use the official
  test split only for post-training reporting, never for checkpoint selection.

## DFormerV2-Compatible Evaluation

Use `PROTOCOL=sunrgbd37_dformerv2` for a checkpoint evaluation that follows
the current DFormerV2 `eval.sh -> utils/eval.py -> utils/val_mm.py` path:

- pad each native SUNRGBD sample at the bottom/right to `531 x 730` before
  normalization; label padding is `255`;
- evaluate all 37 classes in the published order;
- use scales `0.5, 0.75, 1.0, 1.25, 1.5` plus horizontal flip;
- truncate each scaled side and round upward to a multiple of 32;
- use bilinear resizing with `align_corners=True`;
- use `480 x 480` sliding crops at stride `320`;
- average overlapping crop logits, apply softmax per scale/flip prediction,
  then average probabilities before `argmax`;
- reproduce DFormerV2's small-scale behavior by resizing the whole scaled
  input to `480 x 480` whenever either side is shorter than the crop.

This protocol aligns the test mechanics, split, label mapping, and class order.
It intentionally does not replace RSGNet's HHA input with DFormerV2's repeated
single-channel depth input; modality and architecture remain method-specific.
Report old `sunrgbd37` numbers and `sunrgbd37_dformerv2` numbers in separate
columns because they are different inference protocols.

In the checked DFormerV2 repository, the depth loader reads an 8-bit grayscale
PNG, repeats it to three channels, and normalizes it with mean `0.48` and
standard deviation `0.28`. The repository README says the source depth arrays
were exported through a grayscale color map. This is a depth appearance input,
not a preserved metric-depth tensor in meters. Do not reuse that preprocessing
for the planned metric-depth axial bias; decode the local 16-bit depth in
millimetres and keep an explicit invalid-depth mask for that module.

Example (the architecture flags must match the checkpoint):

```bash
PYTHONPATH=. /home/pengfei/miniconda3/envs/PFseg/bin/python evaluate.py \
  --protocol sunrgbd37_dformerv2 \
  --data-root /home/pengfei/HTCnet/DataSets \
  --ckpt /path/to/checkpoint.pth \
  --hha-dir-name HHA_PFHR --hha-channel-order dha \
  --architecture-variant factorized \
  --geometry-encoding factorized_final \
  --fusion-mode stagewise --use-tta \
  --metrics-json /path/to/dformerv2_protocol.json
```

## Academic Training and Reporting

The standard SUNRGBD semantic-segmentation setting is **SUNRGBD-37**:

- 10,335 RGB-D images in total;
- 5,285 official training images (`05051.png`-`10335.png`);
- 5,050 official test images (`00001.png`-`05050.png`);
- 37 evaluated classes, with remapped IDs `0..36`;
- raw void/background and excluded NYU40 labels mapped to `255` and ignored.

The local DFormer-style `train.txt` contains 5,282 unique IDs. The omitted
images (`05407`, `08010`, and `08104`) exist on disk but their masks are all
void. DFormerV2 still sets `num_train_imgs=5285` to determine its iteration
budget. Therefore the official split has 5,285 IDs but only 5,282 samples with
at least one supervised pixel. State this distinction explicitly in logs and
the paper instead of presenting 5,282 as a truncated dataset.

Use a two-stage protocol for a fair paper-style result:

1. Development run: hold out 10% of the official training split (`4757 train +
   528 validation`) and validate once per epoch (`VAL_INTERVAL=1`). Select the
   epoch and hyperparameters using only those 528 images.
2. Final run: retrain for the selected fixed epoch count on all 5,285 official
   training images (`VAL_FRACTION=0.0`), then evaluate the resulting checkpoint
   once on the 5,050 official test images with the fixed TTA protocol.

For a fair architecture ablation, initialize every row from the same
`mit_b2.pth` ImageNet checkpoint and leave `RESUME_CKPT` empty. Loading the C5
segmentation checkpoint into C9 is a valid engineering fine-tuning experiment,
but it must be reported separately because it gives C9 a trained geometry and
decoder initialization.

Do not select a checkpoint by the official-test mIoU. Running an official-test
evaluation every epoch is useful for a diagnostic curve, but it is not a fair
benchmark protocol.

## Ablation Sequence

1. `A0_rgb_baseline`: RGB backbone and fixed decoder.
2. `A1_geometry_prompt`: compact HHA prompt adapters.
3. `A2_shallow_geometry`: shallow C1/C2 geometry-detail fusion.
4. `A3_deep_semantics`: deepest-stage geometry-guided RGB attention.
5. `A4_stagewise_fusion`: C1/C2 shallow fusion, C3 adapter, C4 attention.
6. `A5_rsgnet`: stage-wise fusion plus learned reliability routing.

`A0` through `A4` disable degradation supervision and the reliability
head. `A5` uses HHA degradation probability 0.35 and reliability loss
weight 0.05 by default. This isolates reliability from the stage policy.

The refined B0-B5 sequence is separate from A0-A5. It uses an
identity-preserving cross-scale residual, reliability-conditioned pixel
fusion, a linear geometry-conditioned semantic mixer, and a single final
reliability application. B rows use the same fixed schedule and cannot reuse
A-row checkpoints.

The paper-facing C sequence is:

1. `C0_factorized_rgb`: RGB control with the same cross-scale aligner.
2. `C1_unified_hha`: one early D-H-A stem.
3. `C2_independent_hha`: independent channel stems without physical relations.
4. `C3_factorized_static`: D-H layout and D/H/A boundary relations with correct fixed priors.
5. `C4_factorized_swapped`: reversed shallow/deep route priors.
6. `C5_factorized_routed`: spatial content-adaptive routing without reliability.
7. `C6_pfhr_rsgnet`: adaptive routing with dual reliability.

`R0_raw_depth_control` uses decoded metric depth with the same unified geometry
width. H0-H3 retain `d`, `h`, `a`, or `dh` while leaving model capacity fixed.
Paper-facing C/R/H runs use `HHA_PFHR` in canonical `dha` order.

## Repeated Runs

```bash
for seed in 3407 1234 5678; do
  SEED="$seed" RUN_ID="sunrgbd37_seed_$seed" \
  FACTORIZED_HHA_DIR_NAME=HHA_PFHR FACTORIZED_HHA_CHANNEL_ORDER=dha \
    bash scripts/experiments.sh train C6_pfhr_rsgnet
done
```

Report mean and standard deviation over seeds. Do not compare single-scale and
TTA numbers in the same table column. Retain a module only when its gain is
repeatable and its degraded-HHA behavior does not regress materially.

## Robustness Grid

The representation-space diagnostic includes clean HHA, dropout at 10/30/50
percent, encoded-value Gaussian noise at sigma 8/16/24, and shifts of 4/8/12
pixels:

```bash
SEED=3407 RUN_ID=sunrgbd37_seed3407 \
FACTORIZED_HHA_DIR_NAME=HHA_PFHR FACTORIZED_HHA_CHANNEL_ORDER=dha \
  bash scripts/experiments.sh robust-factorized
```

The primary physical robustness table instead corrupts metric depth before HHA
generation and perturbs the provided tilt rotation. Generate only the official
test split, verify it, and then evaluate C0/C5/C6:

```bash
PFHR_ROBUST_PROFILE=core \
  bash scripts/experiments.sh prepare-pfhr-robustness

PFHR_ROBUST_PROFILE=core \
  bash scripts/experiments.sh verify-pfhr-robustness

PFHR_ROBUST_PROFILE=core \
  bash scripts/experiments.sh analyze-pfhr-robustness

GPUS=2,3 SEED=3407 RUN_ID=pfhr_seed3407 PFHR_ROBUST_PROFILE=core \
  bash scripts/experiments.sh robust-physical-factorized
```

The core physical settings are 30% rectangular raw-depth dropout, Gaussian
metric-depth noise with sigma 0.05 m, an 8-pixel raw-depth shift, and a 5-degree
tilt calibration perturbation. `PFHR_ROBUST_PROFILE=full` expands each failure
type to three severities. Every generated directory has its own manifest and
uses deterministic per-sample seed `3407` by default.

Never load an earlier architecture checkpoint into RSGNet. A/B/C variants,
geometry encodings, input sources, and channel masks are checked separately.
Only the ImageNet MiT-B2 backbone weight is shared.
