# DGRQ-Net 当前代码版模型说明

本文档对应当前实现：

```text
/home/pengfei/HTCnet/SUNpy_manba/models.py
```

当前 DGRQ-Net 是一个用于 SUN RGB-D 语义分割的 RGB-D 模型。它的核心设计是：

- RGB 走深层语义编码器；
- HHA 不再作为第二个完整语义编码器；
- HHA 被转换成置信度、几何提示、角度引导和边界信息，再逐层辅助 RGB 特征。

需要注意：`models.py` 里的构造函数默认值保持 V7 兼容，方便加载旧权重；`run.sh`
里的默认值是当前 V9 实验配置，会把一些现有模块加宽、加大残差强度。

## 1. 总体数据流

输入：

- `rgb`: RGB 图像，形状为 `(B, 3, H, W)`。
- `hha`: HHA 图像，形状为 `(B, 3, H, W)`，通常表示水平视差、高度和表面角度。

主干流程：

```text
RGB
  -> rgb_encoder
  -> 多尺度 RGB 特征
  -> DGLSR-M 动态融合
  -> 最深层 SP-ASPP
  -> CS-DGAM 自顶向下跨尺度对齐
  -> QDHS 解码器
  -> MSDE-BDN 边界分支
  -> SE-CBRL 语义-边界协同细化
  -> CERRH 上采样头
  -> seg_logits

HHA
  -> MUEB 估计几何置信度
  -> 可选 GeometryPromptRecoveryBlock 修复 HHA
  -> 再次 MUEB 更新置信度
  -> BSDE 增强 HD 结构
  -> AGFD 提取角度梯度
  -> FD-GPF 生成几何提示
  -> 浅层 GSTF 过滤几何提示
  -> 输入 DGLSR-M 参与融合
```

当 `return_aux=True` 时，模型返回：

```text
(seg_logits, edge_logits, aux_seg_logits)
```

否则只返回：

```text
seg_logits
```

## 2. RGB 语义编码器

RGB 分支使用 `segmentation_models_pytorch.encoders.get_encoder` 构建编码器。
当前 `run.sh` 默认使用：

```text
encoder_name = mit_b2
pretrained_encoder = /home/pengfei/HTCnet/Checkpoint/mit_b2.pth
```

如果编码器里存在 DropPath 层，`set_encoder_drop_path` 会按深度线性设置随机深度。

关键点：当前模型没有 HHA 的深层并行语义编码器。HHA 只负责产生几何提示和边界辅助信息。

## 3. HHA 可靠性估计与几何预处理

### 3.1 MUEB: 模态不确定性估计

`ModalUncertaintyEstimationBlock` 输出一张一通道 HHA 置信度图。输入包括：

- 原始 HHA；
- 局部方差得到的噪声水平；
- HHA 近零区域形成的 invalid hint。

代码中置信度写法为：

```text
hha_confidence = 0.5 + 0.5 * MUEB(hha)
```

所以最终置信度范围是 `[0.5, 1.0]`。

### 3.2 GPRB: HHA 几何提示恢复块

`GeometryPromptRecoveryBlock` 默认开启，除非传入 `--no-prompt-recovery`。

它不会重建真实深度，而是在归一化 HHA 空间里预测一个小残差：

```text
recovered = hha + max_residual * (1 - confidence) * residual
```

实现细节：

- `max_residual = 0.25`；
- 最后一层卷积零初始化，所以刚开始等价于 identity；
- 输出经过 `nan_to_num`，并 clamp 到 `[-3, 3]`；
- 修复后会再次运行 MUEB，重新得到 `hha_confidence`。

### 3.3 BSDE: 双通道结构差分增强

`BimodalStructuralDifferentialEnhancer` 只处理 HHA 前两个通道：

```text
hd = hha[:, 0:2]
```

它用固定 Laplacian 响应提取局部结构差异，再通过门控和残差增强 HD 结构。

### 3.4 AGFD 与 MAGP

`AngularGeometryFieldDescriptor` 使用四个固定方向卷积核处理 HHA 的角度通道，得到角度梯度图。

`MultiScaleAngularGuidancePyramid` 把这张一通道角度图投影到每个 RGB 有效尺度，形成 side guides。

## 4. FD-GPF: 频率解耦几何提示场

`FrequencyDisentangledGeometricPromptField` 把增强后的 HD 几何转换成多尺度 prompt。

输入包括：

- `hd`: BSDE 后的两通道 HD；
- `angle_grad`: 置信度调制后的角度梯度；
- `confidence`: 更新后的 HHA 置信度；
- `target_sizes`: RGB 编码器有效尺度的空间大小。

### 4.1 低频布局流

布局流先对 HD 做 9x9 平均池化：

```text
hd_low = AvgPool9x9(hd)
layout_input = concat(hd_low, confidence)
```

然后投影到 `prompt_channels`，再根据 `layout_mode` 选择布局聚合方式：

- `ssm`: 默认方式，使用 `GeometryStateSpaceScan`；
- `conv`: 两层局部卷积；
- `avgpool`: 9x9 平均池化加卷积。

`GeometryStateSpaceScan` 会在低频布局图上做四方向扫描：

- 从左到右；
- 从右到左；
- 从上到下；
- 从下到上。

如果 `mamba_ssm` CUDA 可用，会走 CUDA selective scan；否则走 PyTorch 参考实现。
为了控制计算量，实际扫描网格在 `mamba_layout.py` 中限制为 `scan_size=32`。

### 4.2 高频边界流

边界流使用高频 HD：

```text
hd_high = hd - hd_low
boundary_input = concat(hd_high, angle_grad, confidence)
```

边界流保持局部卷积，因为边界是高频信息，不适合强行用全局扫描替代。

### 4.3 尺度路由

FD-GPF 对每个尺度融合布局流和边界流：

```text
learned_layout_weight = sigmoid(router_logits_s)
layout_weight = (0.25 + 0.50 * depth_prior) + 0.25 * learned_layout_weight
boundary_weight = 1 - layout_weight
prompt_s = layout_weight * layout_prompt + boundary_weight * boundary_prompt
```

浅层更偏边界，深层更偏布局。每个尺度的 prompt 会被投影到对应 RGB 编码器通道数，并乘以可学习的 `prompt_scale`。

### 4.4 GSTF: 浅层几何提示过滤

`GuidanceDrivenStructuralTextureFilter` 作用在前两个有效尺度上。它会平滑 prompt、生成 guide map、过滤纹理化残差，再输出更干净的浅层几何提示。

## 5. DGLSR-M: 动态局部-语义路由融合

每个有效尺度的 DGLSR-M 接收：

- RGB 编码器特征；
- FD-GPF 生成的几何 prompt；
- MAGP 生成的角度 side guide。

DGLSR-M 会先根据 RGB、prompt 和 side guide 估计内部 HD 置信度。如果开启 confidence routing，还会用全局 HHA 置信度进一步压制不可靠几何：

```text
hd_confidence *= 0.5 + 0.5 * geometry_reliability
```

### 5.1 局部分支

局部分支做的事情：

1. 根据 prompt 和 side guide 预测 offset；
2. 用 `grid_sample` 对 RGB 特征做双线性 warp；
3. 用 SFT 的 `gamma` 和 `beta` 调制 RGB；
4. 把 RGB 分解为几何平行分量和几何正交分量；
5. 对纹理分量做门控过滤；
6. 通过 `local_scale` 控制残差强度。

### 5.2 语义分支

语义分支做的事情：

1. 如果 token 数超过 `max_attention_tokens`，先自适应降采样；
2. 使用 `MultiHeadTopologicalAttention`；
3. 在最深两个有效尺度上额外使用 `AngleGuidedGSA`，除非 `safe_mode=True`；
4. 上采样回原尺度；
5. 通过 `semantic_scale` 控制残差强度。

### 5.3 路由门控

DGLSR-M 用两路 softmax gate 混合局部分支和语义分支：

```text
fused = rgb + route_local * local_delta + route_semantic * semantic_delta
```

gate 的 bias 按尺度深度初始化：

- 浅层更偏局部更新；
- 深层更偏语义更新。

如果开启 confidence routing，当全局 HHA 置信度较低时，语义几何残差也会被压小。

## 6. 跨尺度融合

DGLSR-M 之后：

1. 最深层特征先进入 `SP_ASPP`；
2. 然后通过 `CrossScaleDualGatedAlignmentModule` 自顶向下融合。

`CS-DGAM` 会把深层特征上采样到浅层大小，通道对齐后通过通道门控和空间门控融合。

## 7. QDHS 解码器

`QueryDrivenHierarchicalSpatialFusionDecoder` 接收跨尺度融合后的特征列表。

内部流程：

1. 所有尺度 lateral projection 到 `decoder_channels`；
2. 最深层做 pyramid pooling；
3. FPN 式自顶向下相加和卷积细化；
4. 在最高分辨率尺度拼接所有尺度；
5. detail refinement；
6. 分成 semantic stream 和 geometry stream；
7. 构造类别 queries；
8. 使用稀疏多尺度 query attention。

queries 由三部分组成：

- 静态类别 embedding；
- scene vector 生成的动态 queries；
- 全局 scene token。

`MultiScaleSparseDeformableObjectQueryAttention` 对每个类别 query 在每个特征层采样 `query_points` 个位置。

最终 decoder logits 形式为：

```text
semantic_logits = classifier(semantic_feat) + query_scale * mask_logits
geometry_logits = geometry_classifier(geometry_feat)
seg_base = semantic_logits * (1 + geometry_scale * geometry_gate)
           + geometry_scale * geometry_logits
```

decoder 返回：

```text
seg_base, detail_feat, aux_logits
```

## 8. 边界分支与最终输出

### 8.1 MSDE-BDN

`MultiSourceDetailEnhancedBoundaryDiscriminationNetwork` 输入：

- decoder 的 `detail_feat`；
- 最浅层 fused feature；
- 最浅层 side guide；
- `seg_base` 生成的语义边界图。

输出：

- 增强后的 `detail_feat`；
- `edge_feat`；
- `detail_logits`；
- `boundary_feat`。

随后主语义 logits 会被 detail logits 更新：

```text
seg_base += detail_logit_scale * detail_logits * (1 + sigmoid(edge_feat))
```

### 8.2 RGB 高频边界提示

模型还会从最浅层 RGB 编码器特征中提取固定 Sobel/Laplacian 边界提示：

```text
rgb_high_freq = FixedBoundaryExtractor(feats_rgb[first_valid_scale])
boundary_feat += rgb_boundary_scale * rgb_high_freq
```

### 8.3 SE-CBRL 与 CERRH

`SemanticEdgeCoGatedBoundaryRefinementLayer` 使用 edge probability 和 boundary feature 细化 `seg_base`。

最后：

- `cerrh_seg` 输出并上采样语义 logits；
- `cerrh_edge` 输出并上采样边缘 logits。

## 9. 当前关键配置

### 9.1 `models.py` 构造函数默认值

这些默认值偏保守，主要为了兼容旧 checkpoint：

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

### 9.2 `run.sh` 当前 V9 默认值

当前脚本默认保存到：

```text
../Checkpoint_SUN_V9
```

V9 结构参数：

```text
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

V9 训练参数：

```text
encoder_name = mit_b2
input = 480 x 640
batch_size = 4 per GPU
target_global_batch = 16
lr = 8e-5
amp_dtype = bf16
drop_path_rate = 0.05
train_cutout_prob = 0.10
aug_level = base
class_weight_mode = inverse_log
class_weight_clamp = 4.0
ce_only_epochs = 10
ohem_start_epoch = 40
ohem_min_kept = 80000
```

验证时 `run.sh val` 会默认启用 `--use-tta`。

## 10. 损失函数

`DGRQCombinedLoss` 包含：

- OHEM Cross Entropy；
- Lovasz loss；
- Dice loss；
- balanced edge BCE；
- boundary-weighted cross entropy；
- feature precision loss；
- auxiliary segmentation CE。

CE-only warm-up 阶段只使用主 CE、辅助 CE，以及一个用于保持图连接的零 edge anchor。

当启用数据集级 class weights 时，OHEM 内部的 batch-adaptive class reweighting 会关闭，避免稀有类别被重复加权。

## 11. checkpoint 兼容性

`train.py` 现在会过滤 shape 不匹配的 checkpoint 权重。这样可以从 V7 尺寸的 checkpoint 微调到 V9 加宽配置：

- 尺寸匹配的层正常加载；
- 尺寸不匹配的层跳过，重新初始化训练。

`remap_dgrq_state_dict` 还兼容一些旧模块命名，例如：

```text
hha_prompt_encoder -> fdgpf
fusion_layers -> dglsr_layers
decoder -> qdhs_decoder
```

## 12. 当前模型一句话总结

当前 DGRQ-Net 不是双编码器 RGB-D 网络，而是：

```text
RGB 深层语义编码器
+ HHA 可靠性估计
+ 频率解耦几何提示
+ 动态局部/语义融合
+ query 解码
+ 边界协同细化
```

它也不是纯 Mamba 或纯 Transformer 模型。Mamba/SSM 只用于低频布局提示流；注意力用于语义路由和 query 解码；边界和细节仍主要由卷积模块处理。

