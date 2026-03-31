# datapipe/dataset.py
# ============================================================
# 数据集加载器
# ============================================================
# 作用：扫描指定文件夹的所有图片，每次取一张时：
#   1. 读取原图（作为 HQ / GT）
#   2. 随机裁剪到固定大小（patch_size × patch_size）
#   3. 用 BlindDegradation 生成退化版本（作为 LQ / 输入）
#   4. 返回 (LQ_tensor, HQ_tensor) 这一对
# ============================================================

import os
import glob
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .degradation import BlindDegradation


class FaceDataset(Dataset):
    """
    通用图片数据集，支持 FFHQ 和 WIDER FACE 两种目录结构。

    FFHQ 结构：   data/ffhq/00001.png, 00002.png, ...
    WIDER 结构：   WIDER_train/0--Parade/xxx.jpg, 1--Handshaking/xxx.jpg, ...

    两种都能自动识别，只要传入根目录路径即可。
    """

    def __init__(self, root_dir, patch_size=256, min_face_size=256):
        """
        Args:
            root_dir:      图片根目录（如 'WIDER_train' 或 'data/ffhq'）
            patch_size:    训练时裁剪的 patch 大小，默认 256
            min_face_size: 图片短边小于这个值的会被跳过（太小没法裁剪）
        """
        super().__init__()
        self.patch_size = patch_size
        self.degradation = BlindDegradation()

        # 递归扫描所有图片（支持子文件夹）
        extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(
                glob.glob(os.path.join(root_dir, '**', ext), recursive=True)
            )
        self.image_paths.sort()

        # 过滤掉太小的图片（可选，加快首次加载）
        self.min_face_size = min_face_size

        print(f"[FaceDataset] 找到 {len(self.image_paths)} 张图片，"
              f"来源：{root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # ---------- 1. 读取图片 ----------
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            # 如果某张图损坏，随机换一张
            print(f"[警告] 无法读取 {img_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))

        w, h = img.size

        # ---------- 2. 跳过太小的图 ----------
        if min(w, h) < self.patch_size:
            # 如果图片比 patch_size 小，先放大到 patch_size
            scale = self.patch_size / min(w, h)
            new_w, new_h = int(w * scale) + 1, int(h * scale) + 1
            img = img.resize((new_w, new_h), Image.BICUBIC)
            w, h = new_w, new_h

        # ---------- 3. 随机裁剪 patch ----------
        # 从大图里随机切一块 patch_size × patch_size 的区域
        left = random.randint(0, w - self.patch_size)
        top = random.randint(0, h - self.patch_size)
        hq_patch = img.crop((left, top,
                             left + self.patch_size,
                             top + self.patch_size))

        # ---------- 4. 随机数据增强（翻转/旋转） ----------
        if random.random() < 0.5:
            hq_patch = hq_patch.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            hq_patch = hq_patch.transpose(Image.FLIP_TOP_BOTTOM)

        # ---------- 5. 生成退化版本 ----------
        lq_patch = self.degradation(hq_patch)

        # ---------- 6. PIL → Tensor [0, 1] ----------
        to_tensor = transforms.ToTensor()  # 自动 /255 并转 [C, H, W]
        hq_tensor = to_tensor(hq_patch)
        lq_tensor = to_tensor(lq_patch)

        return lq_tensor, hq_tensor