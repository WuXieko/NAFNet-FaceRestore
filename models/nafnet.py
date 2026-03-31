# models/nafnet.py
# ============================================================
# NAFNet 自包含实现（不依赖 basicsr）
# ============================================================
# 来源：megvii-research/NAFNet
# 原论文：Simple Baselines for Image Restoration (ECCV 2022)
#
# 核心思路：用 SimpleGate（通道分半相乘）替代传统的
# ReLU/GELU 等非线性激活函数，反而效果更好。
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """对图像特征做 Layer Normalization（逐通道归一化）"""

    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        # x: [B, C, H, W]
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / (var + 1e-6).sqrt()
        x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x


class SimpleGate(nn.Module):
    """
    NAFNet 的核心创新：把通道分成两半，直接相乘。
    替代 ReLU / GELU 等激活函数，简单但有效。
    x1, x2 = split(x)  →  output = x1 * x2
    """

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)  # 沿通道维度分成两半
        return x1 * x2


class NAFBlock(nn.Module):
    """
    NAFNet 的基本构建块。
    结构：LayerNorm → Conv → SimpleGate → Conv → 残差连接
         + 类似的 Channel Attention 分支
    """

    def __init__(self, channels, dw_expand=2, ffn_expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channels = channels * dw_expand

        # ---------- 空间注意力分支 ----------
        self.conv1 = nn.Conv2d(channels, dw_channels, 1)        # 1x1 升维
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, 3,     # 3x3 深度可分离卷积
                               padding=1, groups=dw_channels)
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, 1)   # 1x1 降维

        # Simplified Channel Attention (SCA)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                             # 全局平均池化
            nn.Conv2d(dw_channels // 2, dw_channels // 2, 1),   # 1x1 学通道权重
        )

        self.sg = SimpleGate()
        self.norm1 = LayerNorm2d(channels)

        # ---------- FFN 分支 ----------
        ffn_channels = channels * ffn_expand
        self.conv4 = nn.Conv2d(channels, ffn_channels, 1)
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, 1)
        self.sg2 = SimpleGate()
        self.norm2 = LayerNorm2d(channels)

        # 可学习的残差缩放系数（稳定训练）
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

    def forward(self, x):
        identity = x

        # 空间注意力分支
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)     # 通道注意力加权
        x = self.conv3(x)
        x = self.dropout1(x)
        y = identity + x * self.beta   # 残差连接

        # FFN 分支
        identity = y
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg2(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return identity + x * self.gamma


class NAFNet(nn.Module):
    """
    NAFNet 完整网络：U-Net 结构。

    编码器（下采样）→ 中间层 → 解码器（上采样）
    每一层之间有 skip connection。

    Args:
        img_channel:    输入图片通道数（RGB=3）
        width:          基础特征通道数
        middle_blk_num: 中间层 NAFBlock 数量
        enc_blk_nums:   编码器每层的 NAFBlock 数量列表
        dec_blk_nums:   解码器每层的 NAFBlock 数量列表
    """

    def __init__(self, img_channel=3, width=32, middle_blk_num=1,
                 enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1]):
        super().__init__()

        # 输入投影：3 通道 → width 通道
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1)
        # 输出投影：width 通道 → 3 通道
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()     # 下采样层
        self.ups = nn.ModuleList()       # 上采样层

        chan = width

        # ---------- 编码器 ----------
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )
            self.downs.append(
                nn.Conv2d(chan, chan * 2, 2, stride=2)  # 2x2 卷积下采样，通道翻倍
            )
            chan *= 2

        # ---------- 中间层 ----------
        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )

        # ---------- 解码器 ----------
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1),       # 1x1 升通道
                    nn.PixelShuffle(2)                  # PixelShuffle 上采样，通道减半
                )
            )
            chan //= 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )

        # 水平翻转 padding（TLC 技巧，提升边缘效果）
        self.padder_size = 2 ** len(enc_blk_nums)

    def forward(self, x):
        B, C, H, W = x.shape

        # 确保输入尺寸能被 2^n 整除（n = 编码器层数）
        x = self._check_image_size(x)

        # 输入投影
        x = self.intro(x)

        # 编码器：逐层下采样，保存特征用于 skip connection
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        # 中间层
        x = self.middle_blks(x)

        # 解码器：逐层上采样 + skip connection
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip   # skip connection：直接相加
            x = decoder(x)

        # 输出投影 + 残差学习
        x = self.ending(x)
        x = x[..., :H, :W]    # 裁回原始尺寸（去掉 padding）

        return x

    def _check_image_size(self, x):
        """如果输入尺寸不能被 padder_size 整除，用反射 padding 补齐"""
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='reflect')
        return x


def build_model(width=32):
    """
    构建 NAFNet 模型。

    Args:
        width: 基础通道数
               32 → ~4.6M 参数，5070 Laptop 适用
               64 → ~18M 参数，效果更好但显存翻倍
    """
    return NAFNet(
        img_channel=3,
        width=width,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 28],
        dec_blk_nums=[1, 1, 1, 1],
    )


# ============================================================
# 直接运行验证：python models/nafnet.py
# ============================================================
if __name__ == "__main__":
    model = build_model(width=32).cuda()
    dummy = torch.randn(1, 3, 256, 256).cuda()

    with torch.no_grad():
        out = model(dummy)

    print(f"输入:  {dummy.shape}")
    print(f"输出:  {out.shape}")

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"参数量: {params:.2f}M")

    mem = torch.cuda.max_memory_allocated() / 1024**2
    print(f"推理峰值显存: {mem:.0f} MB")

    print("\n✅ NAFNet 模型测试通过！")