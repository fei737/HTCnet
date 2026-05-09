import os
import random
import cv2
import math
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import segmentation_models_pytorch as smp 
from models import PFNet

def get_hha_grad_target(hha):
    x = hha[:, 2:3, :, :] 
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(x, kernel_x, padding=1)
    gy = F.conv2d(x, kernel_y, padding=1)
    g = torch.abs(gx) + torch.abs(gy)
    return (g > 0.1).float()

class SUNRGBDDataset(Dataset):
    def __init__(self, root_dir, mode='train'):
        self.root_dir = os.path.join(root_dir, 'SUNRGBD_Processed')
        self.mode = mode
        list_path = os.path.join(self.root_dir, f'{mode}.txt')
        with open(list_path, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        img_name = self.image_ids[index]
        rgb = cv2.cvtColor(cv2.imread(os.path.join(self.root_dir, 'RGB', img_name)), cv2.COLOR_BGR2RGB)
        hha = cv2.cvtColor(cv2.imread(os.path.join(self.root_dir, 'HHA', img_name)), cv2.COLOR_BGR2RGB)
        label = cv2.imread(os.path.join(self.root_dir, 'Labels', img_name), 0)

        if self.mode == 'train':
            h, w = rgb.shape[:2]
            rgb = rgb[5:h-5, 5:w-5]
            hha = hha[5:h-5, 5:w-5]
            label = label[5:h-5, 5:w-5]

            if random.random() > 0.5:
                rgb = cv2.flip(rgb, 1)
                hha = cv2.flip(hha, 1)
                label = cv2.flip(label, 1)

            if random.random() > 0.5:
                hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
                hsv = np.array(hsv, dtype=np.float64)
                hsv[:, :, 1] = hsv[:, :, 1] * random.uniform(0.7, 1.3)
                hsv[:, :, 2] = hsv[:, :, 2] * random.uniform(0.7, 1.3)
                hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
                hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
                hsv = np.array(hsv, dtype=np.uint8)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

            scale = random.uniform(1.0, 1.4) 
            h, w = rgb.shape[:2]
            new_h, new_w = max(int(h * scale), 480), max(int(w * scale), 480)

            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            hha = cv2.resize(hha, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            y1 = random.randint(0, new_h - 480)
            x1 = random.randint(0, new_w - 480)
            
            rgb = rgb[y1:y1 + 480, x1:x1 + 480]
            hha = hha[y1:y1 + 480, x1:x1 + 480]
            label = label[y1:y1 + 480, x1:x1 + 480]

            # ==============================================================
            # 🚀 进阶优化 1：RGB 专属 Modality Cutout (强迫模型看 HHA)
            # ==============================================================
            if random.random() > 0.5:
                # 在 RGB 图像上随机挖掉一块 60~120 像素的区域，填为全黑 (0)
                # HHA 和 Label 保持原样不动！
                cut_h = random.randint(60, 120)
                cut_w = random.randint(60, 120)
                cut_y = random.randint(0, 480 - cut_h)
                cut_x = random.randint(0, 480 - cut_w)
                rgb[cut_y:cut_y+cut_h, cut_x:cut_x+cut_w, :] = 0

        else:
            rgb = cv2.resize(rgb, (480, 480))
            hha = cv2.resize(hha, (480, 480))
            label = cv2.resize(label, (480, 480), interpolation=cv2.INTER_NEAREST)

        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        hha_t = torch.from_numpy(hha).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(label).long()
        
        return rgb_t, hha_t, mask_t


class OHEMCrossEntropyLoss(nn.Module):
    def __init__(self, thresh=0.7, ignore_index=0):
        super(OHEMCrossEntropyLoss, self).__init__()
        self.thresh = -math.log(thresh) 
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, logits, labels):
        loss = self.criterion(logits, labels) 
        valid_mask = labels != self.ignore_index
        hard_mask = loss > self.thresh
        keep_mask = valid_mask & hard_mask
        
        if keep_mask.sum() > 0:
            return loss[keep_mask].mean()
        else:
            return loss[valid_mask].mean()

class PFNetCombinedLoss(nn.Module):
    def __init__(self, lambda_lovasz=0.5, lambda_edge=0.3): 
        super().__init__()
        self.ohem_ce = OHEMCrossEntropyLoss(thresh=0.7, ignore_index=0)
        self.lovasz = smp.losses.LovaszLoss(mode='multiclass', ignore_index=0)
        self.bce = nn.BCEWithLogitsLoss()
        
        self.lambda_lovasz = lambda_lovasz
        self.lambda_edge = lambda_edge
        
    def forward(self, seg_logits, seg_labels, edge_logits, edge_labels, hha_grad_target=None):
        loss_ce = self.ohem_ce(seg_logits, seg_labels)
        loss_lov = self.lovasz(seg_logits, seg_labels)
        loss_edge = self.bce(edge_logits.squeeze(1), edge_labels.squeeze(1).float())
        
        if hha_grad_target is not None:
            loss_edge += 0.5 * self.bce(edge_logits.squeeze(1), hha_grad_target.squeeze(1).float())
        
        total_loss = loss_ce + (self.lambda_lovasz * loss_lov) + (self.lambda_edge * loss_edge)
        return total_loss, loss_ce, loss_lov, loss_edge

def edge_target_from_mask(masks):
    x = masks.float().unsqueeze(1)
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(x, kernel_x, padding=1)
    gy = F.conv2d(x, kernel_y, padding=1)
    g = torch.abs(gx) + torch.abs(gy)

    raw_edge = (g > 0).float()
    dilated_edge = F.max_pool2d(raw_edge, kernel_size=3, stride=1, padding=1)
    return dilated_edge

def validate(model, loader, device, n_classes=38): 
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

    model = PFNet(n_classes=args.n_classes, pretrained_path=args.pretrained_encoder, return_aux=True, encoder_name=args.encoder_name)

    start_epoch = 0
    best_miou = 0.0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

    if torch.cuda.device_count() > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.DataParallel(model)
    model.to(device)

    train_loader = DataLoader(SUNRGBDDataset(args.data_root, 'train'), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(SUNRGBDDataset(args.data_root, 'test'), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    criterion = PFNetCombinedLoss(lambda_lovasz=args.lambda_lovasz, lambda_edge=args.lambda_edge).to(device)

    rgb_params = model.module.rgb_encoder.parameters() if hasattr(model, 'module') else model.rgb_encoder.parameters()
    hd_params = model.module.hd_encoder.parameters() if hasattr(model, 'module') else model.hd_encoder.parameters()
    be_params = model.module.be_encoder.parameters() if hasattr(model, 'module') else model.be_encoder.parameters()
    head_params = [p for n, p in model.named_parameters() if 'encoder' not in n]

    optimizer = torch.optim.AdamW([
        {'params': rgb_params, 'lr': args.lr * 0.1}, 
        {'params': hd_params, 'lr': args.lr * 0.8},  
        {'params': be_params, 'lr': args.lr * 0.8},  
        {'params': head_params, 'lr': args.lr}
    ], weight_decay=1e-4)

    accumulation_steps = max(1, 16 // args.batch_size)
    actual_steps_per_epoch = len(train_loader) // accumulation_steps + (1 if len(train_loader) % accumulation_steps != 0 else 0)
    total_steps = args.epochs * actual_steps_per_epoch
    
    # ==============================================================
    # 🚀 进阶优化 2：带 Warmup 的多项式衰减 (防止起步阶段梯度爆炸)
    # ==============================================================
    warmup_steps = 5 * actual_steps_per_epoch
    def warmup_poly_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps)) # 线性预热
        return (1.0 - (step - warmup_steps) / (total_steps - warmup_steps)) ** 0.9
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_poly_lambda)

    if args.resume and os.path.exists(args.resume):
        if 'model_state_dict' in checkpoint:
            start_epoch = checkpoint.get('epoch', 0)
            best_miou = checkpoint.get('best_miou', 0.0)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    os.makedirs(args.save_dir, exist_ok=True)

    # ==============================================================
    # 🚀 进阶优化 3：引入 AMP 混合精度训练引擎
    # ==============================================================
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        optimizer.zero_grad()

        for i, (rgb, hha, masks) in enumerate(pbar):
            rgb, hha, masks = rgb.to(device), hha.to(device), masks.to(device)

            # ==============================================================
            # 🚀 绝对防御机制：强行把所有异常/越界标签抹除为 0 (ignore_index)
            # 防止数据集中极其个别的脏像素导致 CUDA 崩溃
            # ==============================================================
            masks[masks < 0] = 0
            masks[masks >= args.n_classes] = 0
            # 使用 autocast 上下文自动加速 FP16 前向传播
            with torch.amp.autocast('cuda'):
                seg_logits, edge_logits = model(rgb, hha)
                edge_tgt = edge_target_from_mask(masks)
                hha_grad_tgt = get_hha_grad_target(hha)
                
                total_loss, loss_ce, loss_lov, loss_edge = criterion(
                    seg_logits, masks, edge_logits, edge_tgt, hha_grad_target=hha_grad_tgt
                )
                loss = total_loss / accumulation_steps

            # AMP 梯度缩放器接管反向传播
            scaler.scale(loss).backward()

            # 修复边界条件，确保最后一批梯度一定被更新
            # ... 前面的 loss backward 保持不变 ...

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer) 
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 🚀 修复点：记录 scaler 更新前的比例
                scale_before = scaler.get_scale()
                
                scaler.step(optimizer)
                scaler.update()
                
                # 🚀 只有当 scale 没有因为异常缩小（即 optimizer 成功执行）时，才更新学习率
                scale_after = scaler.get_scale()
                if scale_before <= scale_after:
                    scheduler.step()
                    
                optimizer.zero_grad()

            pbar.set_postfix(
                tot=f"{total_loss.item():.3f}",
                ce=f"{loss_ce.item():.3f}",
                lov=f"{loss_lov.item():.3f}",
                edg=f"{loss_edge.item():.3f}"
            )

        raw_model = model.module if hasattr(model, 'module') else model

        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            miou = validate(model, val_loader, device=device, n_classes=args.n_classes)
            print(f"Epoch {epoch + 1}: val mIoU = {miou:.4f}")

            if miou > best_miou:
                best_miou = miou
                ckpt = os.path.join(args.save_dir, 'best_model.pth')
                torch.save(raw_model.state_dict(), ckpt)
                print(f"✅ New best checkpoint reached: {best_miou:.4f} (Saved to {ckpt})")
        else:
            print(f"Epoch {epoch + 1}: Training loss logged. Validation skipped (runs every 10 epochs).")

        checkpoint_dict = {
            'epoch': epoch + 1,
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_miou': best_miou
        }
        torch.save(checkpoint_dict, os.path.join(args.save_dir, 'latest_model.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='/home/pengfei/HTCnet/DataSets')
    # 建议换个保存名字，标记这是极致优化版
    parser.add_argument('--save-dir', type=str, default='Checkpoint_SUNRGBD_V10_Extreme') 
    parser.add_argument('--pretrained-encoder', type=str, default='/home/pengfei/HTCnet/Checkpoint/mit_b2.pth')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--n-classes', type=int, default=38) 
    
    # 🚀 现在有了 AMP 加持，如果卡 2、卡 3 的显存是 24G，你完全可以把 batch-size 调到 8 或 12！
    parser.add_argument('--batch-size', type=int, default=4) 
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=200) 
    parser.add_argument('--lr', type=float, default=6e-5) 
    
    parser.add_argument('--lambda-lovasz', type=float, default=0.5) 
    parser.add_argument('--lambda-edge', type=float, default=0.3)
    parser.add_argument('--encoder-name', type=str, default='mit_b2')
    args = parser.parse_args()

    main(args)
