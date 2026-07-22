# RSGNet Code Structure

```text
RSGNet/
+-- rsgnet/
|   +-- network.py
|   +-- losses.py
|   +-- decoder.py
|   +-- encoder/
|       +-- stage_encoder.py
|       +-- geometry.py
|       +-- fusion.py
|       +-- layers.py
+-- data/
|   +-- sunrgbd.py
|   +-- protocols.py
+-- scripts/
|   +-- run_rsgnet.sh
|   +-- experiments.sh
|   +-- prepare_pfhr_robustness.sh
+-- tools/
|   +-- analyze_pfhr_corruptions.py
|   +-- analyze_pfhr_routes.py
+-- tests/
+-- configs/
+-- docs/
+-- paper/
+-- train.py
+-- evaluate.py
+-- inference.py
+-- utils.py
```

## Public API

- `RSGNet`: complete segmentation network.
- `StageAwareGeometryEncoder`: RGB backbone and stage policy.
- `GeometryReliabilityHead`: dense HHA reliability prediction.
- `FactorizedGeometryReliabilityHead`: separate layout and boundary reliability.
- `GeometryPromptEncoder`: layout and boundary prompt construction.
- `PhysicsFactorizedHHAEncoder`: D-H layout and D/H/A boundary relations.
- `FactorizedPromptRouter`: fixed, swapped, or spatial stage routing.
- `ShallowGeometryFusion`: local C1/C2 detail fusion.
- `GeometryPromptAdapter`: economical middle-stage prompt residual.
- `GeometryGuidedSemanticAttention`: C4 RGB self-attention with geometry
  affinity bias.
- `ReliabilityConditionedPixelFusion`: refined C1/C2 pixel interaction with
  local RGB--geometry agreement.
- `ReliabilityConditionedLinearAttention`: refined C4 linear semantic mixer.
- `ResidualCrossScaleAligner`: refined identity-preserving top-down alignment.
- `LightweightSegmentationDecoder`: fixed decoder for every ablation.
- `RSGNetLoss`: segmentation, auxiliary, edge, and robustness losses.

There are no compatibility aliases for earlier model names. New checkpoints
expose one unambiguous RSGNet state-dict namespace.

`architecture_variant=legacy` is the default and preserves A0-A5 checkpoints.
`architecture_variant=refined` is selected by the B0-B5 scripts and changes
the fusion operators intentionally; the two checkpoint families are not
interchangeable.

`architecture_variant=factorized` is selected by C0-C6. Its
`geometry_encoding` records unified, independent, static, swapped, or adaptive
routing, and checkpoint loading also verifies `geometry_source` and
`geometry_channels`. Canonical HHA ordering is handled in the dataset before
normalization.
