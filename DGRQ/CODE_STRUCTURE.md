# UGF-Lite Code Structure

This project keeps a compact layout so training scripts, checkpoints, and paper
drafts stay easy to trace.

## Public Model Entry

- `models.py`
  - `UGFLiteNet`: paper-facing network name.
  - `DGRQNet`: backward-compatible alias for older scripts and checkpoints.
  - `remap_dgrq_state_dict`: checkpoint key migration helper.

## Encoder

- `encoder/backbone.py`
  - `UncertaintyGuidedFusionEncoder`: RGB-D encoder trunk with reliability-guided
    geometry prompt fusion.
  - `DGRQEncoder`: backward-compatible alias.

- `encoder/geometry.py`
  - `GeometryReliabilityEstimator`: HHA reliability estimation.
  - `GeometryPromptRecovery`: bounded geometry recovery.
  - `RGBGuidedGeometryRecovery`: RGB-guided bounded HHA recovery.
  - `CrossModalReliabilityEstimator`: RGB-HHA structural consistency estimator.
  - `StructuralContrastEnhancer`: local structural contrast enhancement.
  - `AngularBoundaryDescriptor`: angular boundary descriptor from HHA.
  - `AngularGuidancePyramid`: multi-scale angular side guidance.
  - `FrequencyAwareGeometryPrompt`: layout/boundary geometry prompt field.
  - `StructuralPromptFilter`: shallow prompt cleanup.

- `encoder/fusion.py`
  - `LocalSemanticFusionBlock`: local alignment plus semantic aggregation.
  - `GeometryAwareContextAttention`: geometry-aware context attention.
  - `AngularGuidedGlobalAttention`: optional deep angular global attention.
  - `CrossScaleFeatureAligner`: top-down feature alignment.
  - `StripPyramidASPP`: deepest-stage context block.

- `encoder/common.py`
  - Shared convolution blocks, boundary extractor, and encoder utilities.

## Decoder

- `decoder/simple.py`
  - `LightweightMLPDecoder`: main paper decoder.

- `decoder/qdhs.py`
  - `QueryGuidedFusionDecoder`: diagnostic or upper-bound decoder.

- `decoder/refinement.py`
  - `BoundaryDetailHead`: diagnostic boundary detail head.
  - `SemanticBoundaryRefinement`: semantic-boundary refinement.
  - `EfficientUpsamplingHead`: lightweight upsampling head.

## Training and Evaluation

- `train.py`: training entry.
- `val.py`: validation entry.
- `inference.py`: single-image inference and visualization.
- `losses.py`: segmentation, edge, boundary, and auxiliary losses.
- `dataset.py`: SUN RGB-D / NYU-style RGB-HHA dataset loading.
- `utils.py`: validation, logging, EMA, distributed training helpers.

## Experiment Scripts

- `run.sh`: low-level train/validation wrapper.
- `ugf_lite.sh`: paper-facing ablation automation.
- `ugf_lite_gpus_1_3.sh`: two-GPU wrapper using devices 1 and 3.
- `ablation.sh`: older ablation wrapper kept for compatibility.
- `diagnose_residual_scales.py`: branch activity diagnostics.
- `verify_mamba_layout.py`: geometry prompt layout verification.

## Paper and Notes

- `paper/`: Pattern Recognition draft, references, figures, and source notes.
- `UGF_LITE_PR_STEPS.md`: staged model improvement and experiment plan.
- `论文调整内容.md`: historical writing notes.

## Naming Policy

The clean academic names above are the preferred names for new code and paper
writing. Older names such as `DGRQNet`, `DGRQEncoder`, `FDGPF`, `DGLSR`, and
`GACR` are retained only as compatibility aliases or configuration values, so
existing checkpoints and scripts continue to load.
