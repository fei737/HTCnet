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

# --- 主模型 ---
class PFNet(nn.Module):
    def __init__(self, n_classes=14, pretrained_path=None):
        super(PFNet, self).__init__()

        # 1. 先初始化 Encoders (这必须在第一步，否则拿不到通道数)
        self.rgb_encoder = smp.encoders.get_encoder("mit_b0", in_channels=3,weights=None)
        self.hha_encoder = smp.encoders.get_encoder("mit_b0", in_channels=3,weights=None)

        if pretrained_path and os.path.exists(pretrained_path):
            print(f"正在从本地加载预训练权重: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location='cpu')
            self.rgb_encoder.load_state_dict(state_dict)
            self.hha_encoder.load_state_dict(state_dict)
            print("✅ 预训练权重加载成功！")
        else:
            print("⚠️ 未发现预训练权重，将使用随机初始化训练")

        # 2. 定义几何引导和融合模块
        self.geo_blocks = nn.ModuleList()
        self.fusion_layers = nn.ModuleList()
        self.valid_indices = []

        encoder_channels = self.rgb_encoder.out_channels  # mit_b0: [3, 32, 64, 160, 256]

        for i in range(1, len(encoder_channels)):
            ch = encoder_channels[i]
            if ch > 0:
                # 每个尺度分配一个双感知块和一个引导融合层
                self.geo_blocks.append(DualPerceptionGeoBlock(ch))
                self.fusion_layers.append(ACMFFusion(ch))
                self.valid_indices.append(i)

        # 3. 初始化 Decoder
        temp_model = smp.Segformer(encoder_name="mit_b0", in_channels=3, classes=n_classes, encoder_weights=None)
        self.decoder = temp_model.decoder

    def forward(self, rgb, hha):
        input_size = rgb.shape[2:]

        # 提取主干特征
        features_rgb = self.rgb_encoder(rgb)
        features_hha = self.hha_encoder(hha)

        fused_list = []
        fusion_idx = 0

        for i in range(1, len(features_rgb)):
            if i in self.valid_indices:
                # 针对当前分辨率下采样 HHA
                curr_hha = F.interpolate(hha, size=features_rgb[i].shape[2:], mode='bilinear', align_corners=False)

                # 1. 结合 A通道 与 梯度 得到几何引导特征
                geo_guidance = self.geo_blocks[fusion_idx](curr_hha)

                # 2. 执行三路融合 (RGB + HHA + Geo)
                fused = self.fusion_layers[fusion_idx](features_rgb[i], features_hha[i], geo_guidance)

                fused_list.append(fused)
                fusion_idx += 1
            else:
                # 如果是跳层或其他情况
                fused_list.append(features_rgb[i])

        # Segformer Decoder 需要第一个特征图(通常是原始输入缩小版)加上融合后的特征
        final_fused_features = [features_rgb[0]] + fused_list

        output = self.decoder(final_fused_features)
        return F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)
