# NAFNet-FaceRestore

基于 NAFNet 的人脸盲复原（Blind Face Restoration）项目。

针对真实场景中常见的人脸图像退化问题（模糊、噪声、低分辨率、JPEG 压缩等），训练一个端到端的复原模型，输入一张退化人脸图片，输出清晰的修复结果。

## 效果展示

| 原图 (GT) | 退化图 (Input) | 复原图 (Output) |
|:---------:|:--------------:|:---------------:|
| ![gt](assets/demo_gt.png) | ![deg](assets/demo_degraded.png) | ![res](assets/demo_restored.png) |

> 训练完成后，用 `test.py` 生成对比图替换上方示例图片。

## 方法概述

### 模型架构

采用 **NAFNet**（Nonlinear Activation Free Network, ECCV 2022）作为骨干网络。NAFNet 的核心创新是用 SimpleGate 机制（通道分半相乘）替代传统的 ReLU/GELU 等非线性激活函数，以更简洁的结构实现了更优的图像复原性能。

网络整体为 U-Net 编码器-解码器结构：

```
输入图像 → 编码器（逐层下采样提取特征）
        → 中间层（最深层语义处理）
        → 解码器（逐层上采样恢复细节，配合 Skip Connection）
        → 输出图像（与输入同尺寸）
```

本项目的模型配置：

| 参数 | 值 |
|------|----|
| 基础通道数 (width) | 32 |
| 编码器块数 | [1, 1, 1, 28] |
| 解码器块数 | [1, 1, 1, 1] |
| 参数量 | ~17.11M |

### 在线盲退化 Pipeline

训练时不使用预先生成的退化数据，而是**在线实时随机退化**。每张图片在送入模型前，随机组合以下四种退化操作：

| 退化类型 | 模拟场景 | 概率 |
|----------|---------|------|
| 高斯模糊 | 失焦、运动模糊 | 70% |
| 高斯噪声 | 高 ISO、老照片颗粒 | 50% |
| 下采样+上采样 | 低分辨率 | 40% |
| JPEG 压缩 | 微信传图、截图 | 60% |

由于每个 epoch 每张图的退化都不同，模型能见到几乎无穷的退化组合，泛化能力更强。

### 损失函数

采用三种损失函数组合，各自解决不同层面的问题：

| 损失函数 | 作用 | 权重 |
|---------|------|------|
| L1 Loss | 像素级准确性 | 1.0 |
| Perceptual Loss (VGG19) | 语义感知相似度 | 0.1 |
| FFT Loss | 频域高频细节恢复 | 0.05 |

### 训练策略

- 优化器：AdamW（lr=2e-4, weight_decay=1e-4）
- 学习率调度：CosineAnnealingLR（eta_min=1e-6）
- 混合精度训练：bfloat16（节省 ~30% 显存）
- 梯度裁剪：max_norm=1.0
- 支持梯度累积和断点续训

## 项目结构

```
NAFNet-FaceRestore/
├── models/
│   ├── __init__.py
│   └── nafnet.py          # NAFNet 模型（自包含，无 basicsr 依赖）
├── losses/
│   ├── __init__.py
│   └── losses.py          # L1 + Perceptual + FFT 组合损失
├── datapipe/
│   ├── __init__.py
│   ├── degradation.py     # 在线盲退化 Pipeline
│   └── dataset.py         # 数据集加载器
├── train.py               # 训练主脚本（支持断点续训）
├── test_compare.py                # 推理脚本（三图对比 + PSNR）
├── .gitignore
└── README.md
```

## 环境配置

### 硬件要求

- GPU：NVIDIA GPU（显存 ≥ 6GB），项目在 RTX 5070 Laptop GPU 上开发测试
- CUDA：≥ 12.1

### 安装步骤

```bash
# 1. 创建 conda 环境
conda create -n facerest python=3.10
conda activate facerest

# 2. 安装 PyTorch（根据你的 CUDA 版本选择）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. 安装依赖
pip install opencv-python pillow tqdm tensorboard
```

## 数据准备

### FFHQ 数据集

下载 FFHQ 数据集（推荐 256×256 或 512×512 版本），放入 `data/ffhq/` 目录：

```
data/ffhq/
├── 00000.png
├── 00001.png
├── ...
└── 69999.png
```

可选的下载途径：
- [Kaggle - FFHQ](https://www.kaggle.com/datasets/rahulbhalley/ffhq-256x256)（256×256，约 6GB）
- [Kaggle - FFHQ 512](https://www.kaggle.com/datasets/denislukovnikov/ffhq512)（512×512，约 21GB）
- [NVIDIA 官方](https://github.com/NVlabs/ffhq-dataset)（1024×1024，约 89GB）

### WIDER FACE（可选，用于预验证）

下载 [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) 训练集，解压后放入项目根目录的 `WIDER_train/` 中。

## 使用方法

### 训练

```bash
# 首次训练
python train.py

# 断点续训（自动从最新 checkpoint 恢复）
python train.py --resume

# 指定 checkpoint 恢复
python train.py --resume --ckpt checkpoints/epoch_30.pth
```

训练过程中可以用 TensorBoard 监控：

```bash
tensorboard --logdir runs/
# 浏览器打开 http://localhost:6006
```

### 推理测试

```bash
# 测试单张图片
python test.py --input your_blurry_face.jpg

# 测试整个文件夹
python test.py --input test_images/

# 指定模型
python test.py --input your_image.jpg --ckpt checkpoints/epoch_80.pth
```

输出结果保存在 `results/` 目录：
- `xxx_restored.png`：复原图
- `xxx_degraded.png`：退化图
- `xxx_compare.png`：三图对比（原图 | 退化 | 复原）
- 终端输出 PSNR 指标

### 训练参数调整

在 `train.py` 顶部的超参数区域修改：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BATCH_SIZE` | 8 | 显存不够改为 4 |
| `ACCUMULATE_STEPS` | 1 | 配合小 batch 使用，=2 时等效 batch=16 |
| `EPOCHS` | 80 | FFHQ 建议 80~100 |
| `LR` | 2e-4 | 学习率 |
| `WIDTH` | 32 | NAFNet 通道数，改 64 效果更好但更慢 |
| `PATCH_SIZE` | 256 | 训练裁剪大小 |

## 评估指标

| 指标 | 说明 | 参考基准 |
|------|------|---------|
| PSNR | 峰值信噪比，越高越好 | > 28dB 良好，> 30dB 优秀 |
| SSIM | 结构相似性（待集成） | > 0.85 良好 |

## 参考文献

```
@inproceedings{chen2022nafnet,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

## License

本项目仅用于学术研究和学习交流。