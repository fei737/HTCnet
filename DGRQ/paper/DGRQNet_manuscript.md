# DGRQ-Net: Code-Accurate Architecture Note

This document describes the current implementation in
`/home/pengfei/HTCnet/SUNpy_manba/models.py`. The model is an RGB-D semantic
segmentation network for SUN RGB-D. RGB is encoded by a deep semantic encoder,
while HHA is converted into reliability-aware geometric prompts rather than a
second full semantic backbone.

The training script currently uses a V9 capacity recipe in `run.sh`. The Python
model constructor keeps V7-compatible defaults so old checkpoints can still be
loaded. Therefore this note separates **model-code defaults** from **run-script
experiment defaults** where the values differ.

## 1. High-Level Data Flow

Inputs:

- `rgb`: RGB image, shape `(B, 3, H, W)`.
- `hha`: HHA image, shape `(B, 3, H, W)`, with channels interpreted as
  horizontal disparity, height, and angle.

Forward path:

```text
RGB
  -> rgb_encoder
  -> multi-scale RGB features
  -> DGLSR-M fusion
  -> SP-ASPP on deepest scale
  -> CS-DGAM top-down alignment
  -> QDHS decoder
  -> MSDE-BDN boundary branch
  -> SE-CBRL semantic-edge refinement
  -> CERRH upsampling
  -> seg_logits

HHA
  -> MUEB confidence
  -> optional GeometryPromptRecoveryBlock
  -> MUEB confidence recomputation
  -> BSDE on HD channels
  -> AGFD angular gradient
  -> FD-GPF geometric prompts
  -> GSTF on shallow prompt scales
  -> DGLSR-M geometry input and side guidance
```

When `return_aux=True`, the model returns:

```text
(seg_logits, edge_logits, aux_seg_logits)
```

Otherwise, it returns only:

```text
seg_logits
```

## 2. RGB Semantic Encoder

The RGB branch uses `segmentation_models_pytorch.encoders.get_encoder`.
The current run script uses:

- `encoder_name = mit_b2`
- ImageNet/MiT checkpoint path: `/home/pengfei/HTCnet/Checkpoint/mit_b2.pth`
- optional encoder stochastic depth through `set_encoder_drop_path`

There is no parallel deep HHA encoder. HHA is only used to generate confidence,
geometry prompts, side guides, and boundary cues.

## 3. HHA Reliability and Prompt Preparation

### 3.1 Modal Uncertainty Estimation Block (MUEB)

`ModalUncertaintyEstimationBlock` estimates a one-channel HHA confidence map.
It concatenates:

- raw HHA,
- local noise level from local variance,
- an invalid hint where HHA magnitude is near zero.

The network output is wrapped as:

```text
hha_confidence = 0.5 + 0.5 * MUEB(hha)
```

So the confidence lies in `[0.5, 1.0]`.

### 3.2 Geometry Prompt Recovery Block (GPRB)

`GeometryPromptRecoveryBlock` is enabled by default unless
`--no-prompt-recovery` is passed. It predicts a bounded residual correction in
normalized HHA space:

```text
recovered = hha + max_residual * (1 - confidence) * residual
```

Important implementation details:

- `max_residual = 0.25`.
- the final convolution is zero-initialized, so the block starts as identity.
- output is `nan_to_num` sanitized and clamped to `[-3, 3]`.
- after recovery, MUEB is run again to refresh `hha_confidence`.

### 3.3 BSDE

`BimodalStructuralDifferentialEnhancer` processes the first two HHA channels
(`hd = hha[:, 0:2]`). It applies a fixed Laplacian-like differential response,
normalizes it, gates it, and adds a learnable residual. This produces enhanced
height/disparity structure for FD-GPF.

### 3.4 AGFD and MAGP

`AngularGeometryFieldDescriptor` uses fixed directional kernels on the HHA angle
channel to produce an angular gradient map. It is gated by the HHA confidence.

`MultiScaleAngularGuidancePyramid` projects this one-channel angular map into
one side-guide tensor per valid RGB encoder scale.

## 4. FD-GPF: Frequency-Disentangled Geometric Prompt Field

`FrequencyDisentangledGeometricPromptField` converts enhanced HD geometry into
multi-scale prompt features.

Inputs:

- `hd`: enhanced two-channel HD geometry from BSDE.
- `angle_grad`: confidence-gated angular gradient.
- `confidence`: refreshed HHA confidence.
- `target_sizes`: spatial sizes of valid RGB encoder stages.

### 4.1 Layout Stream

The layout stream uses low-pass HD:

```text
hd_low = AvgPool9x9(hd)
layout_input = concat(hd_low, confidence)  # 3 channels
```

The layout stream first projects to `prompt_channels`, then applies one of:

- `ssm`: `GeometryStateSpaceScan`, the default proposed layout mode.
- `conv`: two local convolutional blocks.
- `avgpool`: 9x9 average pooling plus convolution.

For `ssm`, `GeometryStateSpaceScan` scans a coarse layout grid in four
directions: left-to-right, right-to-left, top-to-bottom, and bottom-to-top.
The scan uses `mamba_ssm` CUDA when available, otherwise a PyTorch reference
implementation. The scan grid is capped by `scan_size=32` inside
`mamba_layout.py`.

### 4.2 Boundary Stream

The boundary stream uses high-pass HD:

```text
hd_high = hd - hd_low
boundary_input = concat(hd_high, angle_grad, confidence)  # 4 channels
```

It remains convolutional and local, because the boundary signal is high
frequency.

### 4.3 Scale Routing

A router combines layout and boundary prompts per scale:

```text
learned_layout_weight = sigmoid(router_logits_s)
layout_weight = (0.25 + 0.50 * depth_prior) + 0.25 * learned_layout_weight
boundary_weight = 1 - layout_weight
prompt_s = layout_weight * layout_prompt + boundary_weight * boundary_prompt
```

The prior increases layout contribution in deeper stages and preserves more
boundary contribution in shallow stages. Each prompt is projected to the
corresponding encoder channel count and scaled by a learnable `prompt_scale`
initialized to `0.1`.

### 4.4 GSTF on Shallow Prompt Scales

`GuidanceDrivenStructuralTextureFilter` is applied to the first two valid prompt
scales. It smooths the prompt, predicts a guide map, gates texture-like prompt
components, and adds a residual. This is an existing prompt cleanup block, not a
separate semantic encoder.

## 5. DGLSR-M: Dynamic Gated Local-Semantic Routing

Each valid RGB scale is fused with:

- RGB feature from the encoder,
- the corresponding FD-GPF prompt feature,
- the corresponding angular side guide.

Before fusion, DGLSR-M predicts an internal HD confidence from RGB, prompt, and
side-guide features. If confidence routing is enabled, this internal confidence
is further multiplied by an interpolated global HHA reliability map:

```text
hd_confidence *= 0.5 + 0.5 * geometry_reliability
```

### 5.1 Local Branch

The local branch:

1. predicts offsets from prompt and side-guide features;
2. warps RGB with bilinear grid sampling;
3. applies spatial feature transform with bounded `gamma` and `beta`;
4. decomposes RGB into geometry-parallel and geometry-orthogonal components;
5. filters texture-like residuals with a learned gate;
6. adds a residual controlled by `local_scale`.

### 5.2 Semantic Branch

The semantic branch:

1. adaptively downsamples features if tokens exceed `max_attention_tokens`;
2. applies `MultiHeadTopologicalAttention`;
3. applies `AngleGuidedGSA` only on the deepest two valid scales, unless
   `safe_mode=True`;
4. upsamples back to the original scale;
5. adds a residual controlled by `semantic_scale`.

### 5.3 Route Gate

The local and semantic residual deltas are mixed by a learned two-way route
gate. The gate bias is initialized by scale depth:

- shallow scales prefer local updates,
- deep scales prefer semantic updates.

If confidence routing is enabled, the semantic delta is also reduced when the
global HHA reliability is low.

## 6. Cross-Scale Fusion

After per-scale DGLSR-M:

1. `SP_ASPP` is applied to the deepest fused feature.
2. `CrossScaleDualGatedAlignmentModule` performs top-down refinement.

`CS-DGAM` upsamples the deeper feature, projects it to the shallow channel
count, and uses channel and spatial gates to align shallow and deep features.

## 7. QDHS Decoder

`QueryDrivenHierarchicalSpatialFusionDecoder` receives the refined feature
pyramid.

It performs:

1. lateral projection of all scales to `decoder_channels`;
2. pyramid pooling on the deepest lateral feature;
3. FPN-style top-down addition and refinement;
4. multi-scale concatenation at the shallowest resolution;
5. detail refinement;
6. parallel semantic and geometry streams;
7. class-query construction and sparse multi-scale query attention.

Queries are built from:

- static class embeddings,
- scene-conditioned dynamic queries,
- a global scene token.

`MultiScaleSparseDeformableObjectQueryAttention` samples `query_points` positions
per level for each class query. The decoder combines convolutional semantic
logits and query-mask similarity:

```text
semantic_logits = classifier(semantic_feat) + query_scale * mask_logits
geometry_logits = geometry_classifier(geometry_feat)
seg_base = semantic_logits * (1 + geometry_scale * geometry_gate)
           + geometry_scale * geometry_logits
```

The decoder returns:

```text
seg_base, detail_feat, aux_logits
```

## 8. Boundary Branch and Final Heads

### 8.1 MSDE-BDN

`MultiSourceDetailEnhancedBoundaryDiscriminationNetwork` receives:

- decoder `detail_feat`,
- the shallowest fused feature,
- the shallowest angular side guide,
- detached boundary evidence from `seg_base`.

It outputs:

- enhanced `detail_feat`,
- `edge_feat`,
- `detail_logits`,
- `boundary_feat`.

The main semantic logits are updated before SE-CBRL:

```text
seg_base += detail_logit_scale * detail_logits * (1 + sigmoid(edge_feat))
```

### 8.2 RGB High-Frequency Boundary Guide

`FixedBoundaryExtractor` extracts Sobel/Laplacian boundary cues from the
shallowest RGB encoder feature. This is added to `boundary_feat`:

```text
boundary_feat += rgb_boundary_scale * rgb_high_freq
```

### 8.3 SE-CBRL and CERRH

`SemanticEdgeCoGatedBoundaryRefinementLayer` refines `seg_base` using edge
probabilities and boundary features. Then:

- `cerrh_seg` projects/refines and upsamples segmentation logits by 4.
- `cerrh_edge` projects/refines and upsamples edge logits by 4.

## 9. Important Configuration Values

### 9.1 Model Constructor Defaults

The Python constructor defaults are kept conservative for checkpoint
compatibility:

```text
prompt_channels = 32
layout_state_dim = 16
decoder_channels = 256
attention_tokens = 1024
local_scale_init = 1e-4
semantic_scale_init = 1e-4
query_points = 4
query_scale_init = 0.05
geometry_scale_init = 0.1
detail_logit_scale = 0.3
rgb_boundary_scale = 0.1
```

### 9.2 Current V9 Run-Script Defaults

`run.sh` currently uses a stronger capacity recipe:

```text
SAVE_DIR = ../Checkpoint_SUN_V9
GPUS = 1,2
encoder_name = mit_b2
layout_mode = ssm
input = 480 x 640
batch_size = 4 per GPU
target_global_batch = 16
lr = 8e-5
amp_dtype = bf16
drop_path_rate = 0.05

prompt_channels = 48
layout_state_dim = 24
decoder_channels = 320
attention_tokens = 1600
local_scale_init = 0.001
semantic_scale_init = 0.001
query_points = 6
query_scale_init = 0.08
geometry_scale_init = 0.15
detail_logit_scale = 0.35
rgb_boundary_scale = 0.15
```

Training recipe defaults in `run.sh`:

```text
train_cutout_prob = 0.10
aug_level = base
use_class_weights = 1
class_weight_mode = inverse_log
class_weight_clamp = 4.0
ce_only_epochs = 10
ohem_start_epoch = 40
ohem_min_kept = 80000
lambda_lovasz = 0.5
lambda_dice = 0.4
lambda_edge = 0.03
lambda_boundary = 0.01
lambda_feature_precision = 0.01
aux_weight = 0.1
ema_decay = 0.9996
ema_warmup_epochs = 10
auto_lr = enabled
auto_lr_start_epoch = 75
auto_lr_patience = 10
auto_lr_factor = 0.7
```

Standalone `run.sh val` enables multi-scale and flip TTA through `--use-tta`.

## 10. Loss Function

`DGRQCombinedLoss` combines:

- OHEM cross entropy,
- Lovasz loss,
- Dice loss,
- balanced edge BCE,
- boundary-weighted cross entropy,
- feature precision loss,
- auxiliary segmentation CE.

During CE-only warm-up, only main CE, auxiliary CE, and a zero edge anchor are
used. After warm-up, edge, boundary, and feature precision losses are ramped in
using `rampup_factor`.

When fixed dataset-level class weights are enabled, OHEM does hard-example
mining only and disables its batch-adaptive class reweighting to avoid double
weighting rare classes.

## 11. Checkpoint Compatibility

`train.py` filters shape-mismatched checkpoint keys before loading. This allows
fine-tuning from a V7-sized checkpoint into the wider V9 configuration: matched
weights are reused, and widened layers are trained from initialization.

The state-dict remapper also supports several older module-name prefixes, such
as `hha_prompt_encoder -> fdgpf`, `fusion_layers -> dglsr_layers`, and
`decoder -> qdhs_decoder`.

## 12. What the Current Model Is and Is Not

The current model **is**:

- an RGB-primary semantic encoder with HHA-derived geometry prompts;
- reliability-aware through MUEB, GPRB, and confidence routing;
- frequency-disentangled through FD-GPF;
- locally and semantically routed through DGLSR-M;
- query-decoded through QDHS;
- boundary-refined through MSDE-BDN, RGB fixed-boundary cues, and SE-CBRL.

The current model **is not**:

- a two-backbone RGB/HHA semantic encoder;
- a pure Mamba segmentation network;
- a pure Transformer decoder;
- a simple RGB-HHA concatenation baseline.

