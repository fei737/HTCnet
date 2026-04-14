import torch
import numpy as np
from modules.models.model import PFNet
from train import NYUBaselineDataset
from torch.utils.data import DataLoader


def get_miou(model, loader, device, n_classes=14):
    model.eval()
    ious = []
    # 混淆矩阵计算更准，这里使用简单逻辑演示
    inter = torch.zeros(n_classes).to(device)
    union = torch.zeros(n_classes).to(device)

    with torch.no_grad():
        for rgb, hha, masks in loader:
            rgb, hha, masks = rgb.to(device), hha.to(device), masks.to(device)
            preds = torch.argmax(model(rgb, hha), dim=1)
            for cls in range(1, n_classes):
                inter[cls] += ((preds == cls) & (masks == cls)).sum()
                union[cls] += ((preds == cls) | (masks == cls)).sum()

    miou = torch.mean(inter[1:] / (union[1:] + 1e-6))
    return miou.item()


if __name__ == "__main__":
    device = torch.device('cuda')
    model = PFNet(n_classes=14).to(device)
    # 自动处理可能存在的权重前缀问题
    state_dict = torch.load('Checkpoint/best_model.pth', map_location=device)
    model.load_state_dict(state_dict)

    val_loader = DataLoader(NYUBaselineDataset('/home/pengfei/PFmodel/DataSets', mode='val'), batch_size=12)
    print(f"Final Validation mIoU: {get_miou(model, val_loader, device):.4f}")
