# 2023--2026 相关工作核验记录

本文档记录 RSGNet 主稿中新增文献及其对代码设计的实际影响。题名、作者、
年份和录用信息均来自论文 arXiv 页面或官方代码仓库；没有把未验证的排行榜数字
写入主稿。

| 年份 | 工作 | 出处 | 与 RSGNet 的关系 |
|---|---|---|---|
| 2023 | CMX | IEEE T-ITS | 跨模态校正与 RGB-X 特征交换 |
| 2023 | CMNeXt / DeLiVER | CVPR | 非对称多模态分支、传感器失效评估 |
| 2024 | DFormer | ICLR | RGB-D 专用预训练与统一表示 |
| 2024 | GeminiFusion | ICML | 线性复杂度像素级多模态融合 |
| 2025 | DFormerv2 | CVPR | 深度作为几何先验参与注意力 |
| 2025 | HDBFormer | IEEE SPL | RGB/深度异构分支和局部--全局交互 |
| 2025 | OmniSegmentor | NeurIPS | 灵活多模态预训练与任意模态组合 |
| 2025 | UniMRSeg | NeurIPS | 缺失/损坏模态的分层补偿 |
| 2026 | GeomPrompt | CVPR URVIS Workshop | 缺失/退化深度下的任务驱动几何提示 |

## 已落实到 refined 代码的原则

1. `ReliabilityConditionedPixelFusion` 使用 RGB--几何逐像素交互和局部一致性，
   计算复杂度随像素数线性增长；它吸收 GeminiFusion 的效率原则，但不是其实现复刻。
2. `ReliabilityConditionedLinearAttention` 用正核线性注意力完成深层语义混合，
   几何只调制 RGB key，避免建立第二条语义 value 流。
3. `ResidualCrossScaleAligner` 保留浅层恒等路径，防止深层语义无缩放覆盖边界细节。
4. refined 路径只在每个阶段的最终几何残差上应用一次可靠性；已知无效像素在
   HHA 入口硬屏蔽，避免中间特征被连续多次衰减。

## 没有直接引入的技术

- 没有替换为 DFormer/DFormerv2 或 OmniSegmentor 大骨干，因为这会改变预训练、
  参数量和消融基线，无法证明 RSGNet 自身模块有效。
- 没有加入扩散模型、Mamba 或查询式重型解码器；当前数据和消融尚不能支持额外
  复杂度，且这些模块不直接解决 HHA 可靠性问题。
- GeomPrompt 负责从 RGB 合成/恢复几何输入；RSGNet 负责控制实测 HHA 的注入，
  两者问题设定不同，主稿中已作为并行方向说明。

## 可核验链接

- DFormer: <https://arxiv.org/abs/2309.09668>
- GeminiFusion: <https://arxiv.org/abs/2406.01210>
- DFormerv2: <https://arxiv.org/abs/2504.04701>
- HDBFormer: <https://arxiv.org/abs/2504.13579>
- OmniSegmentor: <https://arxiv.org/abs/2509.15096>
- UniMRSeg: <https://arxiv.org/abs/2509.16170>
- GeomPrompt: <https://arxiv.org/abs/2604.11585>
- CMNeXt / DeLiVER: <https://arxiv.org/abs/2303.01480>
- CMX: <https://arxiv.org/abs/2203.04838>
