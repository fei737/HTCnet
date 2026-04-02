import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class NEURSDDSDataset(Dataset):
    def __init__(self, data_root, split='train', img_size=352):
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        self.rgb_paths = self._load_paths('rgb')  # 自定义路径加载逻辑
        self.depth_paths = self._load_paths('depth')
        self.gt_paths = self._load_paths('gt')

        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        self.rgb_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _load_paths(self, modal):
        # 实现：读取数据集文件夹下的文件路径（按split划分）
        import os
        paths = sorted(
            [os.path.join(self.data_root, modal, f) for f in os.listdir(os.path.join(self.data_root, modal))])
        split_idx = int(len(paths) * 0.8)
        return paths[:split_idx] if self.split == 'train' else paths[split_idx:]

    def _generate_rgbd(self, rgb, depth):
        # Mixup生成RGB-D：I_rgbd = 0.5*rgb + 0.5*depth（论文α=0.5）
        rgb_np = rgb.permute(1, 2, 0).numpy()
        depth_np = depth.squeeze().numpy()
        depth_np = cv2.resize(depth_np, (self.img_size, self.img_size))
        depth_np = np.expand_dims(depth_np, axis=2)
        rgbd_np = 0.5 * rgb_np + 0.5 * depth_np
        rgbd = torch.from_numpy(rgbd_np).permute(2, 0, 1).float()
        return rgbd

    def _generate_edge(self, gt):
        # Canny边缘检测生成边界图E
        gt_np = gt.squeeze().numpy() * 255
        edge = cv2.Canny(gt_np.astype(np.uint8), 50, 150)
        edge = edge / 255.0
        return torch.from_numpy(edge).unsqueeze(0).float()

    def __getitem__(self, idx):
        # 加载RGB
        rgb = Image.open(self.rgb_paths[idx]).convert('RGB')
        rgb = self.transform(rgb)
        rgb = self.rgb_norm(rgb)

        # 加载深度图（单通道）
        depth = Image.open(self.depth_paths[idx]).convert('L')
        depth = self.transform(depth)  # 归一化到[0,1]

        # 生成RGB-D
        rgbd = self._generate_rgbd(rgb, depth)

        # 加载GT并生成边界图
        gt = Image.open(self.gt_paths[idx]).convert('L')
        gt = self.transform(gt)
        edge = self._generate_edge(gt)

        return {
            'rgb': rgb, 'depth': depth, 'rgbd': rgbd,
            'gt': gt, 'edge': edge
        }

    def __len__(self):
        return len(self.rgb_paths)


# 数据加载器
def get_dataloader(data_root, split='train', batch_size=16):
    dataset = NEURSDDSDataset(data_root, split)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=(split == 'train'),
        num_workers=4, pin_memory=True
    )