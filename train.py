import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from modules.models.model import PFNet, get_spatial_gradient
import torch.nn.functional as F

weight_path = 'Checkpoint/best_model_0.403.pth'


class NYUBaselineDataset(Dataset):
    def __init__(self, root_dir, mode='train'):
        self.root_dir = root_dir
        self.indices = range(0, 795) if mode == 'train' else range(795, 1449)

    def __len__(self): return len(self.indices)

    def __getitem__(self, index):
        real_idx = self.indices[index]
        rgb = cv2.cvtColor(cv2.imread(os.path.join(self.root_dir, 'NYURGB', f'{real_idx:04d}.png')), cv2.COLOR_BGR2RGB)
        hha = cv2.cvtColor(cv2.imread(os.path.join(self.root_dir, 'HHA_datasets', f'hha_{real_idx:04d}.png')),
                           cv2.COLOR_BGR2RGB)
        label = cv2.imread(os.path.join(self.root_dir, 'labels', f'{real_idx:04d}.png'), 0)

        # 标签处理
        label_13 = (label % 13) + 1
        label_13[label == 0] = 0

        # Resize & Normalize
        rgb = cv2.resize(rgb, (320, 320))
        hha = cv2.resize(hha, (320, 320))
        label_13 = cv2.resize(label_13, (320, 320), interpolation=cv2.INTER_NEAREST)

        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        hha_t = torch.from_numpy(hha).permute(2, 0, 1).float() / 255.0
        return rgb_t, hha_t, torch.from_numpy(label_13).long()


def validate(model, loader, device):
    model.eval()
    inter, union = 0, 0
    with torch.no_grad():
        for rgb, hha, masks in loader:
            rgb, hha, masks = rgb.to(device), hha.to(device), masks.to(device)
            preds = torch.argmax(model(rgb, hha), dim=1)
            for cls in range(1, 14):  # 排除背景类 0
                inter += ((preds == cls) & (masks == cls)).sum().item()
                union += ((preds == cls) | (masks == cls)).sum().item()
    return inter / (union + 1e-6)


# 配置
device = torch.device("cuda")
root_path = '/home/pengfei/PFmodel/DataSets'

model = PFNet(n_classes=14)  # 这里的初始化不再传入全模型权重路径
# 在这里进行全模型权重加载
if os.path.exists(weight_path):
    print(f"🔥 正在加载全模型权重进行微调: {weight_path}")
    state_dict = torch.load(weight_path, map_location=device)

    # 如果你是用 DataParallel 训练的，可能需要处理 'module.' 前缀
    # new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)  # 这样就能完美对齐了

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.to(device)

train_loader = DataLoader(NYUBaselineDataset(root_path, 'train'), batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(NYUBaselineDataset(root_path, 'val'), batch_size=8, shuffle=False)

criterion = nn.CrossEntropyLoss(ignore_index=0)
# 必须使用极小的学习率进行微调
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

best_miou = 0.0
os.makedirs('Checkpoint', exist_ok=True)

for epoch in range(50):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for rgb, hha, masks in pbar:
        # 将数据移至 GPU
        rgb, hha, masks = rgb.to(device), hha.to(device), masks.to(device)

        # 1. 梯度清零
        optimizer.zero_grad()

        # 2. 前向传播：统一获取 outputs
        outputs = model(rgb, hha)

        # 3. 计算基础交叉熵损失 (保持维度，不求均值)
        loss_ce = F.cross_entropy(outputs, masks, reduction='none')

        # 4. 计算并处理边界权重
        # 假设 get_spatial_gradient 内部没有做去梯度操作，如果在其中不需要求导，
        # 建议将其包裹在 torch.no_grad() 中以节省显存
        with torch.no_grad():
            edge_weight = get_spatial_gradient(hha)
            # 限制范围，并直接 squeeze 到与 masks 相同的维度 (B, H, W)
            edge_weight = torch.clamp(edge_weight, 0, 1).squeeze(1)

            # 5. 边界加权融合损失并求均值
        # 核心逻辑：边缘区域的 Loss 会被放大 (1 + edge_weight) 倍
        loss = (loss_ce * (1.0 + edge_weight)).mean()

        # 6. 反向传播与参数更新
        loss.backward()
        optimizer.step()

        # 7. 更新进度条 (保留 4 位小数，看起来更清爽)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    # 每个 Epoch 结束后更新学习率
    scheduler.step()


# ===================== 放在所有 epoch 结束之后 =====================
# 最后统一计算一次 mIoU
current_miou = validate(model, val_loader, device)
print(f"Final mIoU after 10 epochs: {current_miou:.4f}")

# 最后保存最终模型
raw_model = model.module if hasattr(model, "module") else model
torch.save(raw_model.state_dict(), f"Checkpoint/best_model_{current_miou:.3f}.pth")
