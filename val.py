import torch
import numpy as np
from scipy.ndimage import convolve


def calculate_metrics(pred, gt):
    # pred: 预测显著图（B×1×H×W）, gt: 真实标签（B×1×H×W）
    pred = pred.squeeze().cpu().numpy()
    gt = gt.squeeze().cpu().numpy()
    B, H, W = pred.shape if len(pred.shape) == 3 else (1, pred.shape[0], pred.shape[1])

    metrics = {'Sm': 0, 'Em': 0, 'Fm': 0, 'MAE': 0, 'IoU': 0}
    for i in range(B):
        p = pred[i] if B > 1 else pred
        g = gt[i] if B > 1 else gt

        # 1. MAE
        metrics['MAE'] += np.mean(np.abs(p - g))

        # 2. IoU
        intersection = (p * g).sum()
        union = p.sum() + g.sum() - intersection
        metrics['IoU'] += (intersection + 1e-6) / (union + 1e-6)

        # 3. Fm（β²=0.2）
        thresh = np.linspace(0, 1, 255)
        precision = np.zeros_like(thresh)
        recall = np.zeros_like(thresh)
        for t_idx, t in enumerate(thresh):
            p_bin = (p >= t).astype(np.float32)
            tp = (p_bin * g).sum()
            precision[t_idx] = tp / (p_bin.sum() + 1e-6)
            recall[t_idx] = tp / (g.sum() + 1e-6)
        fm = (1 + 0.2) * (precision * recall) / (0.2 * precision + recall + 1e-6)
        metrics['Fm'] += np.max(fm)

        # 4. Sm（权重α=0.5）
        # 区域相似度S_r
        p_mean = np.mean(p)
        g_mean = np.mean(g)
        cov = np.cov(p.flatten(), g.flatten())[0, 1]
        var_p = np.var(p.flatten())
        var_g = np.var(g.flatten())
        s_r = (2 * p_mean * g_mean + 1e-6) / (p_mean ** 2 + g_mean ** 2 + 1e-6)
        s_o = (2 * cov + 1e-6) / (var_p + var_g + 1e-6)
        metrics['Sm'] += 0.5 * s_r + 0.5 * s_o

        # 5. Em（简化实现，参考论文[32]）
        g_bin = (g > 0).astype(np.float32)
        p_bin = (p > 0.5).astype(np.float32)
        align = (p_bin == g_bin).astype(np.float32)
        metrics['Em'] += np.mean(align)

    # 平均指标
    for k in metrics:
        metrics[k] /= B
    return metrics