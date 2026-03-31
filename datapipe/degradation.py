# datasets/degradation.py
# ============================================================
# 在线盲退化 Pipeline
# ============================================================
# 作用：模拟真实世界中照片可能遇到的各种画质损伤
# 每次调用会随机组合多种退化，让模型学会应对各种情况
# ============================================================

import cv2
import random
import numpy as np
from PIL import Image


class BlindDegradation:
    """
    盲退化生成器。
    "盲"的意思是：模型不知道输入图经历了哪些退化，
    它必须自己判断并修复——这正是真实场景的情况。

    四种退化类型：
    1. 高斯模糊   → 模拟失焦、运动模糊
    2. 高斯噪声   → 模拟高 ISO、老照片颗粒
    3. 下采样+上采样 → 模拟低分辨率
    4. JPEG 压缩  → 模拟微信传图、截图
    """

    def __init__(self,
                 blur_prob=0.7,
                 noise_prob=0.5,
                 downsample_prob=0.4,
                 jpeg_prob=0.6):
        """
        Args:
            blur_prob:       施加模糊的概率
            noise_prob:      施加噪声的概率
            downsample_prob: 施加下采样的概率
            jpeg_prob:       施加 JPEG 压缩的概率

        初期可以把概率调低（比如都设成 0.3），
        让模型先学简单退化，后期再加大难度。
        """
        self.blur_prob = blur_prob
        self.noise_prob = noise_prob
        self.downsample_prob = downsample_prob
        self.jpeg_prob = jpeg_prob

    def __call__(self, img_pil):
        """
        输入：PIL Image（干净的 HQ 图）
        输出：PIL Image（退化后的 LQ 图）
        """
        # PIL → numpy float32 [0, 1]
        img = np.array(img_pil).astype(np.float32) / 255.0

        # ---------- 1. 随机高斯模糊 ----------
        if random.random() < self.blur_prob:
            # kernel size 必须是奇数，越大越模糊
            k = random.choice([7, 9, 11, 13, 15, 17, 19, 21])
            # sigma 控制模糊程度，越大越模糊
            sigma = random.uniform(0.5, 8.0)
            img = cv2.GaussianBlur(img, (k, k), sigma)

        # ---------- 2. 随机高斯噪声 ----------
        if random.random() < self.noise_prob:
            # noise_level: 噪声标准差，范围 [1/255, 50/255]
            noise_level = random.uniform(1, 50) / 255.0
            noise = np.random.normal(0, noise_level, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 1)

        # ---------- 3. 随机下采样再上采样 ----------
        if random.random() < self.downsample_prob:
            h, w = img.shape[:2]
            # scale < 1 表示先缩小再放大，模拟低分辨率
            scale = random.uniform(0.3, 0.9)
            small = cv2.resize(img, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (w, h),
                             interpolation=cv2.INTER_LINEAR)

        # ---------- 4. 随机 JPEG 压缩 ----------
        if random.random() < self.jpeg_prob:
            quality = random.randint(30, 90)  # 越低画质越差
            img_uint8 = (img * 255).astype(np.uint8)
            # 编码为 JPEG 再解码，模拟压缩损失
            _, enc = cv2.imencode('.jpg', img_uint8,
                                 [cv2.IMWRITE_JPEG_QUALITY, quality])
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

        # numpy → PIL
        return Image.fromarray((img * 255).astype(np.uint8))