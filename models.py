import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


# --- 工具函数：计算 HHA 梯度 ---
def get_spatial_gradient(hha):
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=hha.dtype, device=hha.device).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                            dtype=hha.dtype, device=hha.device).view(1, 1, 3, 3)
    a_channel = hha[:, 2:3, :, :]
    grad_x = F.conv2d(a_channel, kernel_x, padding=1)
    grad_y = F.conv2d(a_channel, kernel_y, padding=1)
    return torch.abs(grad_x) + torch.abs(grad_y)


# =========================================================================
# 🚀 创新 1：DGSA 深度引导空间校准 (用于高分辨率局部特征)
# =========================================================================
class DepthGuidedSpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spatial_extractor = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, kernel_size=1, bias=False),
            nn.Sigmoid() 
        )
        self.channel_fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb_feat, hd_feat, be_feat):
        geo_feat = torch.cat([hd_feat, be_feat], dim=1)
        spatial_mask = self.spatial_extractor(geo_feat)
        calibrated_rgb = rgb_feat * (1.0 + spatial_mask)
        fused_feat = torch.cat([calibrated_rgb, hd_feat, be_feat], dim=1)
        out = self.channel_fusion(fused_feat)
        return out, spatial_mask


# =========================================================================
# 🚀 创新 2 (魔改 DFormerV2)：MHTA 多头拓扑先验注意力 (用于全局最深层)
# =========================================================================
# =========================================================================
# 🚀 创新 2 (极限魔改 DFormerV2)：MHTA 多头拓扑先验注意力
# + 植入 LEPE 局部位置增强
# + 植入 LayerScale 动态残差缩放
# =========================================================================
class MultiHeadTopologicalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, init_value=2, heads_range=4, layer_init_values=1e-5):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        
        # 🌟 DFormerV2 暗器核心：对数分布的多头拓扑衰减 (Multi-head Decay)
        decay = torch.log(
            1 - 2 ** (-init_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads)
        )
        self.register_buffer("decay", decay)
        self.gamma = nn.Parameter(torch.ones(1, num_heads, 1, 1) * 2.0)

        # 🗡️ DFormerV2 暗器 1：LEPE (5x5 深度可分离卷积，强化局部边缘)
        self.lepe = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)

        # 🗡️ DFormerV2 暗器 2：LayerScale (稳定新模块的梯度流)
        self.layer_scale = nn.Parameter(layer_init_values * torch.ones(1, dim, 1, 1))

    def forward(self, rgb_feat, hha_feat, be_feat):
        B, C, H, W = rgb_feat.shape
        N = H * W
        
        x = rgb_feat + hha_feat 
        x_flat = x.flatten(2).transpose(1, 2)  # [B, N, C]

        # 1. 压缩物理边界特征，避免显存溢出
        be_spatial = torch.mean(be_feat, dim=1, keepdim=True) 
        be_flat = be_spatial.flatten(2) # [B, 1, N]
        
        # 2. 计算拓扑差异矩阵 [B, N, N]
        topo_diff = torch.abs(be_flat.transpose(1, 2) - be_flat)
        multi_head_topo = topo_diff.unsqueeze(1) * torch.abs(self.decay.view(1, self.num_heads, 1, 1))

        # 3. 计算 Q K V
        qkv = self.qkv(x_flat).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 🗡️ 发动暗器 1：提取 V 并并行计算 LEPE
        v_2d = v.transpose(1, 2).reshape(B, C, H, W).contiguous()
        lepe_out = self.lepe(v_2d) # 获取极强的局部纹理偏置 [B, C, H, W]

        # 4. 全局注意力计算与拓扑干预
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn - self.gamma * multi_head_topo 
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        
        # 🌟 将全局注意力输出与局部的 LEPE 进行加和融合
        out = out + lepe_out

        # 5. 线性投影输出
        out = out.flatten(2).transpose(1, 2)
        out = self.proj(out)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        
        # 🗡️ 发动暗器 2：乘上 LayerScale (动态放开特征权重)
        out = out * self.layer_scale
        
        return out, attn


# =========================================================================
# 🚀 创新 3：CARAFE 内容感知重组算子 
# =========================================================================
class CARAFE(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=4, kernel_size=3, encoder_dim=64):
        super(CARAFE, self).__init__()
        self.scale_factor = scale_factor
        self.kernel_size = kernel_size
        self.channel_compressor = nn.Sequential(
            nn.Conv2d(in_channels, encoder_dim, kernel_size=1),
            nn.BatchNorm2d(encoder_dim),
            nn.ReLU(inplace=True)
        )
        self.kernel_generator = nn.Conv2d(
            encoder_dim, (kernel_size ** 2) * (scale_factor ** 2), kernel_size=3, padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.out_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        x = x.contiguous()
        b, c, h, w = x.shape
        up_h, up_w = h * self.scale_factor, w * self.scale_factor

        compressed_x = self.channel_compressor(x)
        kernels = self.kernel_generator(compressed_x)
        kernels = self.pixel_shuffle(kernels)  
        kernels = F.softmax(kernels, dim=1)

        x_up = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        x_unfold = F.unfold(x_up, kernel_size=self.kernel_size, padding=self.kernel_size // 2)

        x_unfold = x_unfold.view(b, c, self.kernel_size ** 2, up_h, up_w)
        kernels = kernels.unsqueeze(1)  
        out = (x_unfold * kernels).sum(dim=2)  

        return self.out_conv(out)


# --- 轻量级空洞空间金字塔池化 (ASPP) ---
class LiteASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.branch2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.branch3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.project(torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1))


# =========================================================================
# 👑 主模型 PFNet
# =========================================================================
class PFNet(nn.Module):
    def __init__(self, n_classes=14, pretrained_path=None, return_aux=False, encoder_name="mit_b2"):
        super(PFNet, self).__init__()
        self.return_aux = return_aux

        # 1. 初始化 Encoder
        self.rgb_encoder = smp.encoders.get_encoder(encoder_name, in_channels=3, weights=None)
        self.hd_encoder = smp.encoders.get_encoder(encoder_name, in_channels=2, weights=None)
        self.be_encoder = smp.encoders.get_encoder(encoder_name, in_channels=2, weights=None)

        self.geo_denoise = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=3, padding=1, groups=2),
            nn.BatchNorm2d(2),
            nn.Sigmoid()
        )

        # 2. 权重修复
        if pretrained_path and os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location='cpu')
            torch.nn.Module.load_state_dict(self.rgb_encoder, state_dict, strict=False)
            target_key_w = 'patch_embed1.proj.weight'
            if target_key_w in state_dict:
                weight_2ch = state_dict[target_key_w][:, :2, :, :]
                new_state_dict = state_dict.copy()
                new_state_dict[target_key_w] = weight_2ch
                torch.nn.Module.load_state_dict(self.hd_encoder, new_state_dict, strict=False)
                torch.nn.Module.load_state_dict(self.be_encoder, new_state_dict, strict=False)

        encoder_channels = self.rgb_encoder.out_channels
        self.fusion_layers = nn.ModuleList()
        self.valid_indices = []

        # 🌟 动态架构：自动分配 DGSA (局部) 和 MHTA (全局)
        for i in range(len(encoder_channels)):
            ch = encoder_channels[i]
            if ch > 0 and i > 1:
                # 如果是最深层，挂载 DFormerV2 的 MHTA 注意力
                if i == len(encoder_channels) - 1:
                    self.fusion_layers.append(MultiHeadTopologicalAttention(dim=ch))
                # 其他高分辨率深层，挂载 DGSA 卷积融合防止爆显存
                else:
                    self.fusion_layers.append(DepthGuidedSpatialAttention(ch))
                self.valid_indices.append(i)

        deepest_ch = encoder_channels[-1]
        self.aspp = LiteASPP(deepest_ch, deepest_ch)
        
        # 🌟 自顶向下的门控 (Top-Down Gating)，利用深层语义清洗浅层噪声
        self.gate_conv = nn.Sequential(nn.Conv2d(deepest_ch, 1, kernel_size=1))

        # 3. 解码器
        temp_model = smp.Segformer(encoder_name=encoder_name, in_channels=3, classes=n_classes, encoder_weights=None)
        self.decoder = temp_model.decoder
        self.segmentation_head = temp_model.segmentation_head

        # 4. 边缘头与精炼层
        edge_in_channels = encoder_channels[self.valid_indices[0]]

        self.edge_head = nn.Sequential(
            nn.Conv2d(edge_in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

        self.boundary_refinement = nn.Sequential(
            nn.Conv2d(n_classes + 1, n_classes, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_classes),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(n_classes, n_classes, kernel_size=1)
        )

        # 5. CARAFE
        self.carafe_seg = CARAFE(in_channels=n_classes, out_channels=n_classes, scale_factor=4, kernel_size=3)
        self.carafe_edge = CARAFE(in_channels=1, out_channels=1, scale_factor=4, kernel_size=3)


    def forward(self, rgb, hha):
        hd = hha[:, 0:2, :, :]
        grad = get_spatial_gradient(hha)
        raw_be = torch.cat([hha[:, 2:3, :, :], grad], dim=1)
        be = raw_be * self.geo_denoise(raw_be)

        feats_rgb = self.rgb_encoder(rgb)
        feats_hd = self.hd_encoder(hd)
        feats_be = self.be_encoder(be)

        fused_list = []
        fusion_ptr = 0
        deepest_fused_feat = None

        # 逐层融合
        for i in range(len(feats_rgb)):
            if i in self.valid_indices:
                fused_layer, _ = self.fusion_layers[fusion_ptr](feats_rgb[i], feats_hd[i], feats_be[i])

                if i == len(feats_rgb) - 1:
                    fused_layer = self.aspp(fused_layer)
                    deepest_fused_feat = fused_layer # 记录最深层特征

                fused_list.append(fused_layer)
                fusion_ptr += 1
            elif i > 1:
                fused_list.append(feats_rgb[i])

        # 🌟 Top-Down 清洗：生成门控 Mask 去过滤第 0 和第 1 层的光影噪声
        gate_mask = torch.sigmoid(self.gate_conv(deepest_fused_feat))
        gate_mask_0 = F.interpolate(gate_mask, size=feats_rgb[0].shape[2:], mode='bilinear', align_corners=False)
        gate_mask_1 = F.interpolate(gate_mask, size=feats_rgb[1].shape[2:], mode='bilinear', align_corners=False)

        clean_feat_0 = feats_rgb[0] * gate_mask_0
        clean_feat_1 = feats_rgb[1] * gate_mask_1

        # 将清洗后的浅层与融合后的深层拼装
        final_features = [clean_feat_0.contiguous(), clean_feat_1.contiguous()] + [f.contiguous() for f in fused_list]
        decoder_features = self.decoder(final_features)

        # 尺寸拦截，获取 1/4 尺寸 (120x120)
        seg_base = self.segmentation_head[0](decoder_features) 

        # 边缘融合
        edge_feat = self.edge_head(fused_list[0]) 
        edge_feat_120 = F.interpolate(edge_feat, size=seg_base.shape[2:], mode='bilinear', align_corners=False)
        edge_prob_120 = torch.sigmoid(edge_feat_120)

        concat_feat = torch.cat([seg_base, edge_prob_120], dim=1)
        refine_feat = self.boundary_refinement(concat_feat)

        # 软门控
        seg_refined = (seg_base + refine_feat * (edge_prob_120 + 0.2)).contiguous()

        # CARAFE 上采样
        seg_logits = self.carafe_seg(seg_refined)
        edge_logits = self.carafe_edge(edge_feat_120.contiguous())

        return (seg_logits, edge_logits) if self.return_aux else seg_logits
