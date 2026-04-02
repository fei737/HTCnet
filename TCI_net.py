import torch
import torch.nn as nn
import torchvision.models as models
from modules import DGM, FPM, CFRM, SGM


class TCI_Net(nn.Module):
    def __init__(self, num_classes=1):
        super(TCI_Net, self).__init__()
        self.num_classes = num_classes
        # 1. 三分支骨干网络（适配不同通道数）
        self.rgb_backbone = self._build_backbone(models.resnet34, in_channels=3)  # RGB: 3通道
        self.depth_backbone = self._build_backbone(models.resnet18, in_channels=1)  # 深度: 1通道
        self.rgbd_backbone = self._build_backbone(models.resnet50, in_channels=4)  # RGB-D:4通道

        # 2. Encoder模块
        self.dgm3, self.dgm4, self.dgm5 = DGM(128), DGM(256), DGM(512)  # RGB分支各层DGM
        self.fpm3, self.fpm4, self.fpm5 = FPM(64), FPM(128), FPM(256)  # 深度分支各层FPM
        self.cfrm = CFRM(512)  # 最终特征细化

        # 3. Decoder模块（U-Net结构，简化实现）
        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(512, 256, 1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(256, 128, 1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(128, 64, 1), nn.ReLU()),
        ])
        self.sgm1, self.sgm2, self.sgm3 = SGM(64), SGM(128), SGM(256)
        self.edge_conv = nn.Conv2d(64, 1, 1)  # 边界预测
        self.final_conv = nn.Conv2d(64, 1, 1)  # 显著图预测

        # 4. 深度可分离卷积（边界监督用）
        self.depthwise_conv = nn.Conv2d(128, 128, 3, 1, 1, groups=128)
        self.pointwise_conv = nn.Conv2d(128, 64, 1, 1, 0)

    def _build_backbone(self, resnet_func, in_channels):
        # 修改ResNet第一层卷积以适配输入通道数
        backbone = resnet_func(pretrained=True)
        conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        if in_channels != 3:
            conv1.weight.data = backbone.conv1.weight.data.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
        backbone.conv1 = conv1
        # 提取layer1-layer5特征（对应论文i=1-5层）
        features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        )
        return features

    def forward(self, rgb, depth, rgbd):
        # Encoder: 提取三分支特征
        # RGB分支（layer1-layer4对应i=1-4，额外添加layer5简化实现）
        f_r1 = self.rgb_backbone[:6](rgb)  # layer1
        f_r2 = self.rgb_backbone[6](f_r1)  # layer2
        f_r3 = self.rgb_backbone[7](f_r2)  # layer3 → DGM输入
        f_r4 = self.rgb_backbone[8](f_r3)  # layer4 → DGM输入
        f_r5 = nn.AdaptiveAvgPool2d(1)(f_r4)  # 简化layer5

        # 深度分支
        f_d1 = self.depth_backbone[:6](depth)
        f_d2 = self.depth_backbone[6](f_d1)
        f_d3 = self.depth_backbone[7](f_d2)  # FPM输入
        f_d4 = self.depth_backbone[8](f_d3)  # FPM输入
        f_d5 = nn.AdaptiveAvgPool2d(1)(f_d4)

        # RGB-D分支
        f_m1 = self.rgbd_backbone[:6](rgbd)
        f_m2 = self.rgbd_backbone[6](f_m1)
        f_m3 = self.rgbd_backbone[7](f_m2)  # DGM/FPM参考
        f_m4 = self.rgbd_backbone[8](f_m3)
        f_m5 = nn.AdaptiveAvgPool2d(1)(f_m4)

        # Encoder模块：DGM + FPM + CFRM
        f_r3 = self.dgm3(f_r3, f_m3)
        f_r4 = self.dgm4(f_r4, f_m4)
        f_d3 = self.fpm3(f_m3, f_d3)
        f_d4 = self.fpm4(f_m4, f_d4)
        f_r5, f_d5, f_m5 = self.cfrm(f_r5, f_d5, f_m5)

        # Decoder: 上采样 + 边界监督 + SGM
        # 上采样到低分辨率（i=4,5）
        f_m4_up = F.interpolate(f_m5, size=f_m4.shape[2:], mode='bilinear')
        f_m3_up = F.interpolate(f_m4_up, size=f_m3.shape[2:], mode='bilinear')

        # 边界监督（i=4,5融合）
        f_r_d_fuse = torch.cat([f_r4, f_d4], dim=1)  # 128通道
        f_edge = self.depthwise_conv(f_r_d_fuse)
        f_edge = self.pointwise_conv(f_edge)
        edge_pred = self.edge_conv(f_edge)  # 边界预测图S_i

        # SGM语义引导（逐层引导）
        f_m3_enhanced = self.sgm3(f_m4_up, f_m3)
        f_m2_enhanced = self.sgm2(f_m3_enhanced, f_m2)
        f_m1_enhanced = self.sgm1(f_m2_enhanced, f_m1)

        # 最终显著图预测
        final_pred = self.final_conv(f_m1_enhanced)

        return edge_pred, torch.sigmoid(final_pred)


# 损失函数（BCE + IoU + Dice）
class TCI_Loss(nn.Module):
    def __init__(self):
        super(TCI_Loss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def iou_loss(self, pred, gt):
        intersection = (pred * gt).sum()
        union = pred.sum() + gt.sum() - intersection
        return 1 - (intersection + 1e-6) / (union + 1e-6)

    def dice_loss(self, pred, gt):
        intersection = (pred * gt).sum()
        return 1 - (2 * intersection + 1e-6) / (pred.sum() + gt.sum() + 1e-6)

    def forward(self, edge_pred, final_pred, gt_edge, gt):
        # 边界损失（BCE + IoU）
        edge_bce = self.bce(edge_pred, gt_edge)
        edge_iou = self.iou_loss(torch.sigmoid(edge_pred), gt_edge)
        # 显著图损失（Dice）
        final_dice = self.dice_loss(final_pred, gt)
        # 总损失（公式18）
        total_loss = edge_bce + edge_iou + final_dice
        return total_loss