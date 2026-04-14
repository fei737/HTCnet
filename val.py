import argparse
import torch
from torch.utils.data import DataLoader

from models import PFNet
from train import NYUBaselineDataset


def get_miou(model, loader, device, n_classes=14):
    model.eval()
    inter = torch.zeros(n_classes).to(device)
    union = torch.zeros(n_classes).to(device)

    with torch.no_grad():
        for rgb, hha, masks in loader:
            rgb, hha, masks = rgb.to(device), hha.to(device), masks.to(device)
            out = model(rgb, hha)
            if isinstance(out, tuple):
                out = out[0]
            preds = torch.argmax(out, dim=1)
            for cls in range(1, n_classes):
                inter[cls] += ((preds == cls) & (masks == cls)).sum()
                union[cls] += ((preds == cls) | (masks == cls)).sum()

    miou = torch.mean(inter[1:] / (union[1:] + 1e-6))
    return miou.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n-classes', type=int, default=14)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PFNet(n_classes=args.n_classes).to(device)
    state_dict = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    val_loader = DataLoader(NYUBaselineDataset(args.data_root, mode='val'), batch_size=args.batch_size)
    print(f"Final Validation mIoU: {get_miou(model, val_loader, device, n_classes=args.n_classes):.4f}")
