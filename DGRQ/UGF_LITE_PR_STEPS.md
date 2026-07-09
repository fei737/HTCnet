# UGF-Lite PR Improvement Plan

This plan is for turning the current DGRQ prototype into a Pattern Recognition
style encoder/fusion paper line:

```text
UGF-Lite = strong RGB encoder + reliability-aware geometry prompt/fusion + simple decoder
```

The goal is to improve mIoU while keeping the contribution easy to defend:
encoding and fusion innovation first, simple decoder fixed.

## 0. Main Rules

- Do not add a new module before the clean ablation is stable.
- Keep `decoder_mode=simple_mlp` for the main paper line.
- Treat QDHS, RCFR, GSA, and GACR as diagnostics until they earn a stable gain.
- Change one factor at a time.
- Report single-scale validation as the primary result; report TTA separately.
- If a module gives less than 0.2 mIoU or is not stable across seeds, do not
  present it as a contribution.

## 1. Phase 1: Clean Main Ablation

Run the clean single-mainline ablation:

```bash
cd /home/pengfei/HTCnet/DGRQ
bash ugf_lite.sh list
bash ugf_lite.sh all-phase1
bash ugf_lite.sh report
```

If you want to run one experiment at a time:

```bash
bash ugf_lite.sh all E0_rgb_only
bash ugf_lite.sh all E1_hha_prompt
bash ugf_lite.sh all E2_rgpr
bash ugf_lite.sh all E3_local_only
bash ugf_lite.sh all E4_semantic_only
bash ugf_lite.sh all E5_full_lite
```

The six experiments are:

| ID | Meaning | Paper Use |
| --- | --- | --- |
| `E0_rgb_only` | RGB encoder + simple decoder | baseline |
| `E1_hha_prompt` | add HHA prompt without recovery | shows whether geometry helps |
| `E2_rgpr` | add RGB-guided geometry recovery | tests reliability/recovery |
| `E3_local_only` | recovered geometry + local branch | tests alignment |
| `E4_semantic_only` | recovered geometry + semantic branch | tests global reasoning |
| `E5_full_lite` | recovered geometry + both branches | full UGF-Lite |

Decision rules:

- `E1 <= E0`: HHA prompt quality is weak. Fix geometry prompt before adding modules.
- `E2 > E1`: RGB-guided recovery is useful and can be a core component.
- `E3 > E2`: local alignment is useful.
- `E4 > E2`: semantic aggregation is useful.
- `E5 < max(E3, E4)`: the branch mixing is hurting; simplify or bias the gate.
- `E5 > E0` and robust tests are strong: enough for the UGF-Lite mainline.

## 2. Phase 2: Residual Scale Sweep

Only after Phase 1 has a sensible trend, sweep residual strengths:

```bash
bash ugf_lite.sh all R_003_003
bash ugf_lite.sh all R_005_003
bash ugf_lite.sh all R_003_005
bash ugf_lite.sh all R_005_005
bash ugf_lite.sh report
```

Interpretation:

- If all larger scales hurt, geometry residuals are too noisy; keep `0.001`.
- If stronger local helps, emphasize misalignment/local structure in the paper.
- If stronger semantic helps, emphasize global structural context in the paper.
- If both stronger branches help, rerun the best setting with a second seed.

## 3. Phase 3: Prompt Width and Layout

Prompt width:

```bash
bash ugf_lite.sh all P32
bash ugf_lite.sh all P48
bash ugf_lite.sh all P64
```

Layout stream:

```bash
bash ugf_lite.sh all L_conv
bash ugf_lite.sh all L_ssm
bash ugf_lite.sh all L_avgpool
```

Decision rules:

- If `P64` improves train but not validation, use `P48`.
- If `L_conv >= L_ssm`, do not claim SSM as a main innovation.
- If `L_avgpool` is close, the geometry field is useful but layout modeling is
  not the main contribution.

## 4. Phase 4: Training Recipe Sweep

After the best structure is chosen, tune training:

```bash
bash ugf_lite.sh all T_lr02_dp10
bash ugf_lite.sh all T_lr03_dp10
bash ugf_lite.sh all T_lr05_dp10
```

Recommended final recipe to try:

```bash
EPOCHS=200 ENCODER_LR_MULT=0.3 DROP_PATH_RATE=0.1 bash ugf_lite.sh all E5_full_lite
```

If `encoder_lr_mult=0.5` is worse, the pretrained RGB encoder is being disturbed
too much.

## 5. Diagnostics Only

These should not enter the main contribution unless they clearly improve:

```bash
bash ugf_lite.sh all Q_confidence_check
bash ugf_lite.sh all Q_gacr_check
bash ugf_lite.sh all Q_ssm_safeoff
```

Rules:

- If confidence routing is within 0.2 mIoU, keep it out of the mainline.
- If GACR is not clearly better, do not include it in the core model.
- If SSM+GSA does not clearly improve, keep `SAFE_MODE=1` and a simple layout.

## 6. Validation Policy

Primary single-scale validation:

```bash
bash ugf_lite.sh val E5_full_lite
```

TTA validation:

```bash
USE_TTA_VAL=1 bash ugf_lite.sh val E5_full_lite
```

Never mix single-scale and TTA values in the same comparison table.

## 7. Robustness Experiments for PR

After the clean model is stable, add robustness validation. These experiments
are more important for Pattern Recognition than another tiny clean mIoU gain:

- Depth/HHA Gaussian noise.
- Random rectangular HHA dropout.
- RGB-D misalignment by shifting HHA 4, 8, 12, and 16 pixels.

Compare:

| Model | Purpose |
| --- | --- |
| RGB baseline | lower bound, not affected by HHA degradation |
| HHA prompt without uncertainty/recovery | naive geometry fusion |
| UGF-Lite without recovery | isolates reliability/recovery |
| UGF-Lite full | final robust model |

The paper claim should be:

```text
UGF-Lite improves clean RGB-D parsing and degrades more gracefully when geometry
is noisy, missing, or misaligned.
```

## 8. Paper Contribution Line

Use three contributions only:

1. A reliability-first RGB-D fusion formulation that treats HHA as calibrated
   structural evidence, not a second semantic stream.
2. A simple local/semantic geometry reasoning block under the same reliability
   principle.
3. A controlled evaluation showing clean mIoU gains and stronger robustness
   under unreliable geometry.

Do not present RCFR, GSA, QDHS, SSM, and boundary heads as equal contributions.

## 9. When to Change Code

Only change code after the script-level ablation shows a clear issue.

Likely code changes:

- Replace the two-way `route_gate` with a simpler stage-wise scalar if
  `E5_full_lite < max(E3_local_only, E4_semantic_only)`.
- Keep `SAFE_MODE=1` by default if `Q_ssm_safeoff` is weak.
- Remove or keep disabled `reliability_router` if RCFR remains weak.
- Keep the simple decoder as the main model. Use QDHS only as an optional
  appendix or future extension.

## 10. Minimum PR-Ready Evidence

Before writing the final PR manuscript, collect:

- SUN-RGBD main comparison.
- NYUDv2 full model and key ablation.
- Clean main ablation table from Phase 1.
- Complexity table: Params, FLOPs, FPS.
- Robustness table: noise, dropout, shift.
- Qualitative reliability maps and failure cases.
- At least two seeds for the final model and one key baseline; three seeds if
  time permits.
