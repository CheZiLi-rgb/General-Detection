import torch
import torch.nn as nn
from functools import partial
import torch.fft
from transformers import XCLIPVisionModel
import torch.nn.init as init

from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math

from einops import rearrange

try:
    from .ss2d import SS2D
    from .csms6s import CrossScan_1, CrossScan_2
    from .csms6s import CrossMerge_1, CrossMerge_2
except:
    from ss2d import SS2D
    from csms6s import CrossScan_1, CrossScan_2
    from csms6s import CrossMerge_1, CrossMerge_2

try:
    from ss2d import SS2D
    from csms6s import CrossScan_1, CrossScan_2
    from csms6s import CrossMerge_1, CrossMerge_2


    class PVT2FFN(nn.Module):
        def __init__(self, in_features, hidden_features):
            super().__init__()
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden_features, in_features)

        def forward(self, x, H, W):
            B, N, C = x.shape
            x = self.fc1(x)
            # 兼容 W=1 的情况 (1D 序列)
            x_spatial = x.transpose(1, 2).view(B, -1, H, W)
            x_spatial = self.dwconv(x_spatial)
            x = x_spatial.flatten(2).transpose(1, 2)
            x = self.act(x)
            x = self.fc2(x)
            return x
except ImportError:
    pass


class DynamicChannelSorter(nn.Module):
    def __init__(self, dim, reduction=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.scorer = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, N, C = x.shape

        x_pool = x.mean(dim=1)  # (B, C) Global Average Pooling
        scores = self.scorer(x_pool)  # (B, C)
        sort_idx = torch.argsort(scores, dim=1, descending=True)  # (B, C)

        inverse_sort_idx = torch.argsort(sort_idx, dim=1)

        return sort_idx, inverse_sort_idx, scores


class DynamicGroupMambaLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=1, d_conv=3, expand=1, reduction=16, num_groups=4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_groups = num_groups

        self.norm = nn.LayerNorm(input_dim)

        self.router = DynamicChannelSorter(dim=input_dim)

        dim_per_group = input_dim // 2
        self.mamba_g1 = SS2D(d_model=dim_per_group, d_state=d_state, ssm_ratio=expand, d_conv=d_conv)
        self.mamba_g2 = SS2D(d_model=dim_per_group, d_state=d_state, ssm_ratio=expand, d_conv=d_conv)

        # 通道交互模块 (Channel Affinity / Modulation)
        self.channel_interaction = nn.Sequential(
            nn.Linear(input_dim, input_dim // reduction),
            nn.ReLU(),
            nn.Linear(input_dim // reduction, input_dim),
            nn.Sigmoid()
        )

        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x, H, W):
        # x: (B, N, C) where N = H*W
        B, N, C = x.shape
        residual = x
        x = self.norm(x)

        sort_idx, inv_idx, scores = self.router(x)  # sort_idx: (B, C)

        # x: (B, N, C) -> sort_idx: (B, 1, C) -> expand -> (B, N, C)
        idx_expanded = sort_idx.unsqueeze(1).expand(-1, N, -1)
        x_sorted = torch.gather(x, dim=2, index=idx_expanded)

        x_spatial = x_sorted.view(B, H, W, C)
        x1, x2 = torch.chunk(x_spatial, 2, dim=-1)

        y1 = self.mamba_g1(x1, CrossScan=CrossScan_1, CrossMerge=CrossMerge_1)
        y2 = self.mamba_g2(x2, CrossScan=CrossScan_2, CrossMerge=CrossMerge_2)

        y_cat = torch.cat([y1, y2], dim=-1)  # (B, H, W, C)
        y_cat = y_cat.view(B, N, C)

        inv_idx_expanded = inv_idx.unsqueeze(1).expand(-1, N, -1)
        y_restored = torch.gather(y_cat, dim=2, index=inv_idx_expanded)

        mod_gate = self.channel_interaction(x)
        y_final = y_restored * mod_gate

        y_final = self.proj(y_final)

        return residual + y_final * self.skip_scale


class XCLIP_StemGroupMamba(nn.Module):
    def __init__(self, channel_size=768, class_num=1):
        super(XCLIP_StemGroupMamba, self).__init__()
        self.encoder = XCLIPVisionModel.from_pretrained(
            "/data1/lic/DeMamba-main/xclip-base-patch16",
            local_files_only=True
        )
        self.channel = channel_size
        self.h_patches = 14
        self.w_patches = 14
        self.patch_nums = self.h_patches * self.w_patches

        self.fc1 = nn.Linear((self.patch_nums + 1) * self.channel, class_num)
        self.fc_norm = nn.LayerNorm(self.patch_nums * self.channel)
        self.fc_norm2 = nn.LayerNorm(768)

        self.dropout = nn.Dropout(p=0.0)

        self.group_mamba_blocks = nn.ModuleList([
            DynamicGroupMambaLayer(
                input_dim=768,
                output_dim=768,
                d_state=1,
                d_conv=3,
                expand=1,
                reduction=16,
                num_groups=4
            )
            for _ in range(4)
        ])

        self.initialize_weights(self.fc1)

    def initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, T, 3, 224, 224)
        b, t, _, h, w = x.shape
        images = x.view(b * t, 3, h, w)

        # Encoder Forward
        outputs = self.encoder(images, output_hidden_states=True)

        sequence_output = outputs['last_hidden_state'][:, 1:, :]  # (B*T, 196, 768)
        _, N_patches, c = sequence_output.shape

        global_feat = outputs['pooler_output'].view(b, t, -1) # (4, 8, 768)
        global_feat = global_feat.mean(1)  # (4, 8, 768) -> (4, 768)
        global_feat = self.fc_norm2(global_feat) # (4, 768)

        f_w = int(math.sqrt(N_patches))  # 14
        f_h = f_w

        x_spatial = sequence_output  # (B*T, 196, 768)

        # (Left->Right, Right->Left, Top->Bottom, Bottom->Top)
        for block in self.group_mamba_blocks:
            x_spatial = block(x_spatial, H=self.h_patches, W=self.w_patches)

        video_features = x_spatial.view(b, t, self.patch_nums, -1) # x_spatial: (32, 196, 768)
        video_level_features = video_features.mean(1)  # Temporal Mean

        # (b*s*s, C) -> (b, s*s*C)
        video_level_features = video_level_features.view(b, -1)

        video_level_features = self.fc_norm(video_level_features)
        video_level_features = torch.cat((global_feat, video_level_features), dim=1)

        pred = self.fc1(video_level_features)
        pred = self.dropout(pred)

        return pred


if __name__ == '__main__':
    model = XCLIP_StemGroupMamba()
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    model = model.to(device)
    dummy_input = torch.randn(4, 8, 3, 224, 224)
    dummy_input = dummy_input.to(device)
    output = model(dummy_input)
    print(output.shape)  # torch.Size([4, 1])

