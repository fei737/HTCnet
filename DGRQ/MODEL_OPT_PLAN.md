# 模型优化方案 — 编码器 + 融合（解码器不动）

目标：提升 mIoU；多用 transformer 等新技术；去除写死的固定线性系数；重设计置信度。
约束：只改 `encoder/` 与 `encoder/fusion.py`，解码器保持通用简单结构。当前基线 E5_full ≈ 0.4969 (single-scale)。

## 核心设计原则（贯穿所有改动）
> 训练方式：**从新开始训练**（不需要严格复现旧 checkpoint 行为）。因此初始化约束放宽，但仍采用稳定训练的标准做法。
1. **稳定初始化**：新增 transformer/残差分支输出投影**零初始化** + LayerScale 门控，起点接近恒等，避免从头训早期发散；可学习门控初值设在合理中点（如原固定系数附近）。
2. **不破坏 backbone 预训练**：mit_b2 仍加载 imagenet 预训练权重，新增模块不影响其加载。
3. **逐单元可消融**：每个改动加一个开关（构造参数），默认开，可单独关掉做消融，符合 PR 论文要求。
4. **每阶段做 smoke test**：PFseg 环境跑前向验证，确认输出形状不变、无 NaN/崩溃、参数量与显存增幅可控。

## 新增复用组件（放 `encoder/common.py`）
- `LearnableBlend(n_inputs, ctx_channels)`：小 1×1 卷积从上下文预测逐像素 softmax 权重，替换 `0.65*a+0.35*b` 这类固定凸组合；bias 初始化成复现原比例。
- `LearnableGate(init_lo, init_span)`：可学习 `lo + span*x` 仿射，替换 `0.5+0.5*x`、`0.25+0.75*x` 这类固定 remap；`lo/span` 经 sigmoid 约束在 [0,1]，初值复现原系数。

---

## 阶段 1：置信度可学习化 + 去除固定线性系数（低风险，直接命中三诉求中的两个）

替换点（全部改成 `LearnableGate`/`LearnableBlend`，初值复现原值）：

**geometry.py**
- `GeometryReliabilityEstimator` 输出、`CrossModalReliabilityEstimator` L241 `0.65*conf+0.35*consistency` → `LearnableBlend`，输入 [conf, consistency, rgb_edge, hha_edge, edge_gap]
- `.clamp(0.05,1.0)` / `.clamp(0.10,0.90)` 硬边界 → sigmoid + 可学习范围
- `DirectionalEdgePromptRefiner` L348、`ReliabilityAwareAutocorrelationPromptMixer` L430 `residual_gate=0.25+0.75*rel` → `LearnableGate`
- `FrequencyAwareGeometryPrompt` L557 layout_weight 固定系数 → 可学习（保留 depth_prior 结构）

**fusion.py**
- `LocalSemanticFusionBlock` L291 `0.5+0.5*hd_conf`、L301 `0.65/0.35`、L303/305、L349-351 `reliability_support` 固定系数 → `LearnableGate`/`LearnableBlend`

**backbone.py**
- L153/159 `0.5+0.5*mueb`、L161 `0.25+0.75*conf` → `LearnableGate`

预期：置信度融合从人工先验变为数据自适应，去掉全部写死凸组合。风险低，通常小幅稳定涨点。

---

## 阶段 2：跨模态可靠性 Transformer（新技术核心，最有论文价值）

`CrossModalReliabilityEstimator` 现在只用 3×3 局部边缘一致性判断"深度可不可信"。但不可信区域（远处/反光/薄结构/边界）本质需要全局上下文。

改法：加一个**窗口化多头 cross-attention** 块——RGB 特征作 query 去 attend HHA 特征（在低分辨率可靠性图尺度上，token 数可控），产出上下文感知的可靠性修正。输出投影**零初始化** → 起点等于当前可靠性，增益逐步学出。

这是同时服务"transformer 新技术" + "置信度重设计"两个诉求的关键贡献点。

---

## 阶段 3（可选）：浅层融合 Transformer 化

`StructuralPromptFilter`(GSTF) 目前是纯卷积纹理门。可把纹理门换成轻量**窗口注意力**（高分辨率，必须窗口化控算力，参考即插即用里的 MCA/AFFN）。零初始化残差。

此阶段收益不确定、算力更敏感，建议放在 1、2 验证有效后再做。

---

## 训练侧（正交、几乎免费的涨点，需你点头才碰 train.py）
- 当前 **class weighting 默认关**，SUN RGB-D 长尾严重 → 开 inverse-log 类别权重，通常先白捡零点几个点。
- 这一条不属于"编码器+融合"，默认**不动**，除非你要我一起改。

## 落地顺序
1. 先做阶段 1 → smoke test（前向 + 载旧 ckpt）→ 报告参数量/形状。
2. 再做阶段 2 → smoke test。
3. 阶段 3 视情况。
每阶段一个 git commit，你可单独训练验证增益，不达标可单独回退。
