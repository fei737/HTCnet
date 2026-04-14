import os
import cv2
import argparse
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models import PFNet


class NYUBaselineDataset(Dataset):
    def __init__(self, root_dir, mode='train'):
        self.root_dir = root_dir
        self.mode = mode
        self.rgb_dir = os.path.join(self.root_dir, 'NYURGB')
        self.hha_dir = os.path.join(self.root_dir, 'HHA_datasets')
        self.label_dir = os.path.join(self.root_dir, 'labels')
        self.samples = self._build_samples()

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _extract_id(name):
        stem = os.path.splitext(name)[0]
        m = re.search(r'(\d+)$', stem)
        if m:
            return int(m.group(1))
        return None

    def _scan_with_ids(self, folder):
        id2path = {}
        if not os.path.isdir(folder):
            return id2path
        for fn in os.listdir(folder):
            fpath = os.path.join(folder, fn)
            if not os.path.isfile(fpath):
                continue
            fid = self._extract_id(fn)
            if fid is not None:
                id2path[fid] = fpath
        return id2path

    def _build_samples(self):
        rgb_map = self._scan_with_ids(self.rgb_dir)
        hha_map = self._scan_with_ids(self.hha_dir)
        label_map = self._scan_with_ids(self.label_dir)

        valid_ids = sorted(set(rgb_map.keys()) & set(hha_map.keys()) & set(label_map.keys()))
        if not valid_ids:
            raise RuntimeError(
                f"未找到可用样本，请检查目录: {self.rgb_dir}, {self.hha_dir}, {self.label_dir}"
            )

        split_idx = int(len(valid_ids) * 0.55)  # 约等于原始 795/1449
        ids = valid_ids[:split_idx] if self.mode == 'train' else valid_ids[split_idx:]
        samples = [(rgb_map[i], hha_map[i], label_map[i]) for i in ids]
        print(f"[{self.mode}] Loaded {len(samples)} samples (total valid ids: {len(valid_ids)})")
        return samples

    def __getitem__(self, index):
        rgb_path, hha_path, label_path = self.samples[index]
        rgb_raw = cv2.imread(rgb_path)
        hha_raw = cv2.imread(hha_path)
        label = cv2.imread(label_path, 0)

        if rgb_raw is None or hha_raw is None or label is None:
            raise RuntimeError(
                f"读取失败: rgb={rgb_path}, hha={hha_path}, label={label_path}"
            )
        rgb = cv2.cvtColor(rgb_raw, cv2.COLOR_BGR2RGB)
        hha = cv2.cvtColor(hha_raw, cv2.COLOR_BGR2RGB)

        label_13 = (label % 13) + 1
        label_13[label == 0] = 0

        rgb = cv2.resize(rgb, (320, 320))
        hha = cv2.resize(hha, (320, 320))
        label_13 = cv2.resize(label_13, (320, 320), interpolation=cv2.INTER_NEAREST)

        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        hha_t = torch.from_numpy(hha).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(label_13).long()
        return rgb_t, hha_t, mask_t


def dice_loss(logits, target, ignore_index=0, eps=1e-6):
    n_classes = logits.shape[1]
    probs = torch.softmax(logits, dim=1)
    valid = (target != ignore_index).float().unsqueeze(1)
    onehot = F.one_hot(torch.clamp(target, min=0), num_classes=n_classes).permute(0, 3, 1, 2).float()
    onehot = onehot * valid
    probs = probs * valid

    inter = (probs * onehot).sum(dim=(0, 2, 3))
    den = probs.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
    dice = (2 * inter + eps) / (den + eps)
    return 1 - dice[1:].mean()


def edge_target_from_mask(masks):
    """从语义 mask 生成边界监督 (B,1,H,W)."""
    x = masks.float().unsqueeze(1)
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(x, kernel_x, padding=1)
    gy = F.conv2d(x, kernel_y, padding=1)
    g = torch.abs(gx) + torch.abs(gy)
    return (g > 0).float()


def validate(model, loader, device, n_classes=14):
    model.eval()
    inter = torch.zeros(n_classes, device=device)
    union = torch.zeros(n_classes, device=device)
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
    return torch.mean(inter[1:] / (union[1:] + 1e-6)).item()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PFNet(n_classes=args.n_classes, pretrained_path=args.pretrained_encoder, return_aux=True)
    if args.resume and os.path.exists(args.resume):
        print(f"🔥 恢复训练权重: {args.resume}")
        state_dict = torch.load(args.resume, map_location=device)
        model.load_state_dict(state_dict, strict=False)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    train_loader = DataLoader(NYUBaselineDataset(args.data_root, 'train'), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(NYUBaselineDataset(args.data_root, 'val'), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    ce_loss = nn.CrossEntropyLoss(ignore_index=0)
    bce_loss = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    os.makedirs(args.save_dir, exist_ok=True)
    best_miou = 0.0

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for rgb, hha, masks in pbar:
            rgb, hha, masks = rgb.to(device), hha.to(device), masks.to(device)
            optimizer.zero_grad()

            seg_logits, edge_logits = model(rgb, hha)
            loss_seg = ce_loss(seg_logits, masks)
            loss_dice = dice_loss(seg_logits, masks, ignore_index=0)
            edge_tgt = edge_target_from_mask(masks)
            loss_edge = bce_loss(edge_logits, edge_tgt)

            loss = loss_seg + args.lambda_dice * loss_dice + args.lambda_edge * loss_edge
            loss.backward()
            optimizer.step()

            pbar.set_postfix(total=f"{loss.item():.4f}", seg=f"{loss_seg.item():.4f}", edge=f"{loss_edge.item():.4f}")

        scheduler.step()
        miou = validate(model, val_loader, device=device, n_classes=args.n_classes)
        print(f"Epoch {epoch + 1}: val mIoU = {miou:.4f}")

        raw_model = model.module if hasattr(model, 'module') else model
        torch.save(raw_model.state_dict(), os.path.join(args.save_dir, 'latest_model.pth'))
        if miou > best_miou:
            best_miou = miou
            ckpt = os.path.join(args.save_dir, f'best_model_{best_miou:.3f}.pth')
            torch.save(raw_model.state_dict(), ckpt)
            print(f"✅ New best checkpoint: {ckpt}")

    print(f"Training done. Best mIoU: {best_miou:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--save-dir', type=str, default='Checkpoint')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--pretrained-encoder', type=str, default='')
    parser.add_argument('--n-classes', type=int, default=14)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda-dice', type=float, default=0.5)
    parser.add_argument('--lambda-edge', type=float, default=0.2)
    args = parser.parse_args()

    main(args)
