# losses/losses.py
# ============================================================
# 组合损失函数
# ============================================================
# L1 Loss        → 像素级准确性（主力）
# Perceptual Loss → 语义/感知相似度（让结果自然）
# FFT Loss       → 频域高频细节（恢复锐度）
#
# 三者互补：L1 管"对不对"，Perceptual 管"像不像"，FFT 管"清不清晰"
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """
    感知损失：用预训练 VGG19 提取特征，比较特征空间的距离。

    为什么不直接比像素？
    因为人眼对"语义相似"更敏感。两张图像素差很大但看起来一样
    （比如平移 1 像素），L1 会给高惩罚，但 Perceptual 不会。
    """

    def __init__(self):
        super().__init__()
        # 加载 VGG19，只取前 16 层（到 relu3_3）
        # 浅层捕捉边缘/纹理，中层捕捉结构/语义
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slice = nn.Sequential(*list(vgg.children())[:16]).eval()

        # VGG 不参与训练，冻结参数
        for p in self.slice.parameters():
            p.requires_grad = False

        # VGG 的标准化参数（ImageNet 均值和标准差）
        self.register_buffer('mean',
                             torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',
                             torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        # 标准化到 ImageNet 分布（VGG 训练时用的）
        pred_n = (pred - self.mean) / self.std
        target_n = (target - self.mean) / self.std
        # 比较 VGG 特征的 L1 距离
        return F.l1_loss(self.slice(pred_n), self.slice(target_n))


class FFTLoss(nn.Module):
    """
    频域损失：在傅里叶变换后的频域比较两张图。
    """

    def forward(self, pred, target):
        # FFT 不支持 bfloat16，先转 float32
        pred_f = torch.fft.fft2(pred.float(), norm='ortho')
        target_f = torch.fft.fft2(target.float(), norm='ortho')
        # 比较幅度谱
        return F.l1_loss(torch.abs(pred_f), torch.abs(target_f))
class CombinedLoss(nn.Module):
    """
    组合损失 = w1 * L1 + w2 * Perceptual + w3 * FFT

    权重说明：
        L1   = 1.0   主力损失
        Perc = 0.1   太大会让图像过度平滑
        FFT  = 0.05  太大会产生伪纹理

    这些是经验起点，不是最优值。训练时根据 TensorBoard 调整：
    - 图像细节不够 → 调大 FFT 权重到 0.1
    - 图像不自然   → 调大 Perceptual 权重到 0.2
    """

    def __init__(self, w_l1=1.0, w_perc=0.1, w_fft=0.05):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.perc = PerceptualLoss()
        self.fft = FFTLoss()
        self.w_l1 = w_l1
        self.w_perc = w_perc
        self.w_fft = w_fft

    def forward(self, pred, target):
        loss_l1 = self.l1(pred, target)
        loss_perc = self.perc(pred, target)
        loss_fft = self.fft(pred, target)

        total = (self.w_l1 * loss_l1 +
                 self.w_perc * loss_perc +
                 self.w_fft * loss_fft)

        # 返回总 loss + 分项字典（方便 TensorBoard 监控）
        loss_dict = {
            "l1": loss_l1.item(),
            "perc": loss_perc.item(),
            "fft": loss_fft.item(),
        }
        return total, loss_dict


# ============================================================
# 直接运行验证：python losses/losses.py
# ============================================================
if __name__ == "__main__":
    criterion = CombinedLoss().cuda()

    # 模拟一对预测图和目标图
    pred = torch.randn(2, 3, 256, 256).cuda().clamp(0, 1).requires_grad_(True)
    target = torch.randn(2, 3, 256, 256).cuda().clamp(0, 1)

    loss, loss_dict = criterion(pred, target)

    print(f"Total Loss:      {loss.item():.4f}")
    print(f"  L1 Loss:       {loss_dict['l1']:.4f}")
    print(f"  Perceptual:    {loss_dict['perc']:.4f}")
    print(f"  FFT Loss:      {loss_dict['fft']:.4f}")

    # 验证梯度能回传
    loss.backward()
    print("\n✅ 损失函数测试通过，梯度正常！")