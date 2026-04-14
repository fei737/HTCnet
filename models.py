import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


# --- 工具函数 ---
def get_spatial_gradient(hha):
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=hha.dtype, device=hha.device).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                            dtype=hha.dtype, device=hha.device).view(1, 1, 3, 3)
    # 假设 HHA 的 A 通道在索引 2
    a_channel = hha[:, 2:3, :, :]
    grad_x = F.conv2d(a_channel, kernel_x, padding=1)
    grad_y = F.conv2d(a_channel, kernel_y, padding=1)
    return torch.abs(grad_x) + torch.abs(grad_y)


# --- 几何感知模块 ---
class DualPerceptionGeoBlock(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.angle_path = nn.Sequential(
            nn.Conv2d(1, out_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.grad_path = nn.Sequential(
            nn.Conv2d(1, out_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.compress = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, hha):
        angle = hha[:, 2:3, :, :]
        grad = get_spatial_gradient(hha)
        feat_angle = self.angle_path(angle)
        feat_grad = self.grad_path(grad)
        return self.compress(torch.cat([feat_angle, feat_grad], dim=1))


# --- 最终融合层 ---
class ACMFFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 门控权重生成
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )
        # 新增：通道感知，借鉴 DFormerv2 的全局对齐
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        # 特征精炼层
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb, hha, geo_feat):
        # 1. 产生自适应权重
        weights = self.gate(geo_feat)
        w_rgb = weights[:, 0:1, :, :]
        w_hha = weights[:, 1:2, :, :]

        # 1. 动态融合
        fused_dynamic = (rgb * 0.5 + rgb * 0.5 * w_rgb) + (hha * w_hha)

        # 2. 通道重加权：利用几何特征指导哪些通道（语义）更重要
        fused_dynamic = fused_dynamic * self.channel_attn(geo_feat)

        # 3. 注入几何引导残差
        return self.refine(fused_dynamic + geo_feat)


class TriModalFusion(nn.Module):
    """RGB + HD + BE 三分支融合."""
    def __init__(self, channels):
        super().__init__()
        self.rgb_gate = nn.Conv2d(channels, 1, kernel_size=1)
        self.hd_gate = nn.Conv2d(channels, 1, kernel_size=1)
        self.be_gate = nn.Conv2d(channels, 1, kernel_size=1)
        self.project = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb_feat, hd_feat, be_feat):
        # 用 BE 特征生成空间门控，显式加强边界区域
        g_rgb = torch.sigmoid(self.rgb_gate(be_feat))
        g_hd = torch.sigmoid(self.hd_gate(be_feat))
        g_be = torch.sigmoid(self.be_gate(be_feat))
        fused = torch.cat([rgb_feat * (1.0 + g_rgb), hd_feat * (1.0 + g_hd), be_feat * (1.0 + g_be)], dim=1)
        return self.project(fused)


# --- 主模型（三分支） ---
class PFNet(nn.Module):
    def __init__(self, n_classes=14, pretrained_path=None, return_aux=False):
        super(PFNet, self).__init__()
        self.return_aux = return_aux

        # 三分支编码器：RGB / HHA(H,D) / HHA(A,Grad)
        self.rgb_encoder = smp.encoders.get_encoder("mit_b0", in_channels=3, weights=None)
        self.hd_encoder = smp.encoders.get_encoder("mit_b0", in_channels=2, weights=None)
        self.be_encoder = smp.encoders.get_encoder("mit_b0", in_channels=2, weights=None)

        if pretrained_path and os.path.exists(pretrained_path):
            print(f"正在从本地加载预训练权重: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location='cpu')
            self.rgb_encoder.load_state_dict(state_dict)
            self.hd_encoder.load_state_dict(state_dict, strict=False)
            self.be_encoder.load_state_dict(state_dict, strict=False)
            print("✅ 预训练权重加载成功！")
        else:
            print("⚠️ 未发现预训练权重，将使用随机初始化训练")

        # 每个尺度分配一个融合模块
        self.fusion_layers = nn.ModuleList()
        self.valid_indices = []
        encoder_channels = self.rgb_encoder.out_channels  # [3, 32, 64, 160, 256]

        for i in range(1, len(encoder_channels)):
            ch = encoder_channels[i]
            if ch > 0:
                self.fusion_layers.append(TriModalFusion(ch))
                self.valid_indices.append(i)

        # Decoder
        temp_model = smp.Segformer(encoder_name="mit_b0", in_channels=3, classes=n_classes, encoder_weights=None)
        self.decoder = temp_model.decoder
        # 边界辅助头（来自高分辨率融合特征）
        self.edge_head = nn.Sequential(
            nn.Conv2d(encoder_channels[1], encoder_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_channels[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(encoder_channels[1], 1, kernel_size=1)
        )

    def forward(self, rgb, hha):
        input_size = rgb.shape[2:]
        # 构建两路几何输入
        hd = hha[:, 0:2, :, :]
        angle = hha[:, 2:3, :, :]
        grad = get_spatial_gradient(hha)
        be = torch.cat([angle, grad], dim=1)

        # 提取三分支特征
        features_rgb = self.rgb_encoder(rgb)
        features_hd = self.hd_encoder(hd)
        features_be = self.be_encoder(be)

        fused_list = []
        fusion_idx = 0

        for i in range(1, len(features_rgb)):
            if i in self.valid_indices:
                fused = self.fusion_layers[fusion_idx](features_rgb[i], features_hd[i], features_be[i])
                fused_list.append(fused)
                fusion_idx += 1
            else:
                fused_list.append(features_rgb[i])

        final_fused_features = [features_rgb[0]] + fused_list
        seg_logits = self.decoder(final_fused_features)
        seg_logits = F.interpolate(seg_logits, size=input_size, mode='bilinear', align_corners=False)
        if not self.return_aux:
            return seg_logits

        edge_logits = self.edge_head(fused_list[0])
        edge_logits = F.interpolate(edge_logits, size=input_size, mode='bilinear', align_corners=False)
        return seg_logits, edge_logits
